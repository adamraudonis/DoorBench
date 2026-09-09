"""Track one screened thumb material point through original finger motors only."""
import mujoco
import numpy as np
from .operation_teacher import smooth_phase


class ThumbWithdrawalFeedback:
    def __init__(self,teacher,local_point):
        self.teacher=teacher;self.local=np.asarray(local_point,float)
        if self.local.shape!=(3,) or not np.isfinite(self.local).all() or np.linalg.norm(self.local)>.06:raise ValueError('Original distal thumb material point required')
        self.body=teacher.m.body('rh_thdistal').id
        self.jac=np.zeros((3,teacher.m.nv));self.previous=None

    def force(self,forces,t,goal,elapsed):
        teacher=self.teacher;m,d=teacher.m,teacher.d;goal=np.asarray(goal,float)
        if goal.shape!=(3,) or not np.isfinite(np.r_[goal,t,elapsed]).all() or elapsed<0:raise ValueError('Finite screened thumb target required')
        velocity=np.zeros(3)
        if self.previous is not None:
            oldt,oldgoal=self.previous;dt=t-oldt
            if not 0<dt<=.05:raise ValueError('Monotonic thumb feedback clock required')
            velocity=(goal-oldgoal)/dt
            if np.linalg.norm(velocity)>.5:raise ValueError('Thumb reference exceeds 0.5 m/s')
        point=d.xpos[self.body]+d.xmat[self.body].reshape(3,3)@self.local
        mujoco.mj_jac(m,d,self.jac,None,point,self.body)
        error=goal-point;force=1200.*error+3.*(velocity-self.jac@d.qvel)
        force*=min(1.,6./max(np.linalg.norm(force),1e-12))*float(smooth_phase(elapsed))
        generalized=self.jac.T@force
        result=np.asarray(forces,float).copy()
        result[teacher.fingers]+=teacher.finger_inverse@generalized[teacher.va]
        result=np.clip(result,teacher.caps[:,0],teacher.caps[:,1]);self.previous=(float(t),goal.copy())
        return result,dict(thumb_material_tracking_error_m=float(np.linalg.norm(error)),
            thumb_feedback_force_N=float(np.linalg.norm(force)),thumb_reference_speed_m_s=float(np.linalg.norm(velocity)))
