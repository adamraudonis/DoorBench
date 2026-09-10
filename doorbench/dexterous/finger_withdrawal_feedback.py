"""Track one screened distal material point through original finger motors only."""
import mujoco
import numpy as np
from .operation_teacher import smooth_phase


class FingerWithdrawalFeedback:
    def __init__(self,teacher,local_point,*,digit,segment='distal'):
        if digit not in ('ff','mf','rf','lf','th'):raise ValueError('Original Shadow digit required')
        if segment not in ('distal','middle') or (segment=='middle' and digit=='th'):raise ValueError('Original finger tracking segment required')
        self.teacher=teacher;self.local=np.asarray(local_point,float)
        if self.local.shape!=(3,) or not np.isfinite(self.local).all() or np.linalg.norm(self.local)>.06:raise ValueError('Original distal finger material point required')
        self.body=teacher.m.body('rh_'+digit+segment).id
        self.jac=np.zeros((3,teacher.m.nv));self.previous=None
        self.reference_velocity=np.zeros(3);self.reference_handoff=None

    def force(self,forces,t,goal,elapsed,*,start_handoff=False):
        teacher=self.teacher;m,d=teacher.m,teacher.d;goal=np.asarray(goal,float)
        if goal.shape!=(3,) or not np.isfinite(np.r_[goal,t,elapsed]).all() or elapsed<0:raise ValueError('Finite screened finger target required')
        if start_handoff:
            if self.previous is None or not 0<t-self.previous[0]<=.05:
                raise ValueError('Material handoff requires the preceding target')
            if np.linalg.norm(self.reference_velocity)>1e-4:
                raise ValueError('Material handoff requires a stationary outgoing target')
            if self.reference_handoff is not None and t-self.reference_handoff[0]<1.:
                raise ValueError('Material handoff already active')
            offset=self.previous[1]-goal
            if np.linalg.norm(offset)>.03:
                raise ValueError('Material handoff exceeds 3cm target offset')
            # Bridge nominal-to-attained target differences after a supported
            # stationary hold. Keep previous history and the original speed
            # guard; the quintic offset has zero endpoint speed/acceleration.
            self.reference_handoff=(float(t),offset.copy())
        if self.reference_handoff is not None:
            started,offset=self.reference_handoff
            if t<started:raise ValueError('Monotonic material handoff required')
            goal=goal+(1.-float(smooth_phase(t-started)))*offset
        velocity=np.zeros(3)
        if self.previous is not None:
            oldt,oldgoal=self.previous;dt=t-oldt
            if not 0<dt<=.05:raise ValueError('Monotonic finger feedback clock required')
            velocity=(goal-oldgoal)/dt
            if np.linalg.norm(velocity)>.5:raise ValueError('Finger reference exceeds 0.5 m/s')
        point=d.xpos[self.body]+d.xmat[self.body].reshape(3,3)@self.local
        mujoco.mj_jac(m,d,self.jac,None,point,self.body)
        error=goal-point;force=1200.*error+3.*(velocity-self.jac@d.qvel)
        force*=min(1.,6./max(np.linalg.norm(force),1e-12))*float(smooth_phase(elapsed))
        generalized=self.jac.T@force
        result=np.asarray(forces,float).copy()
        result[teacher.fingers]+=teacher.finger_inverse@generalized[teacher.va]
        result=np.clip(result,teacher.caps[:,0],teacher.caps[:,1]);self.previous=(float(t),goal.copy())
        self.reference_velocity=velocity.copy()
        info=dict(material_tracking_error_m=float(np.linalg.norm(error)),
            feedback_force_N=float(np.linalg.norm(force)),reference_speed_m_s=float(np.linalg.norm(velocity)))
        if self.reference_handoff is not None:
            started,offset=self.reference_handoff
            info['reference_handoff']=dict(started_s=started,duration_s=1.,
                offset_m=offset.tolist(),fraction=float(smooth_phase(t-started)))
        return result,info
