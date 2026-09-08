"""Bounded Cartesian correction using scalar proprioception in the robot frame.

The caller supplies a commanded palm pose relative to the actual base. The
helper accepts no door state, world base pose, contacts, or active plant handle.
It returns joint targets for the existing original-capped force adapter. The
Controlled arm FK uses the prior commanded target, with actual measured
noncontrolled joints. Its residual is commanded-FK error, not measured palm
tracking error; original joint PD must still track the returned target. A failed
update is terminal and must not be retried on this stateful helper. The
current DoorBench caller is a privileged teacher; this helper is not a learned
vision/tactile policy or a physical qualification by itself.
"""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation


def bounded_next_velocity(position,velocity,lower,upper,dt,speed,acceleration):
    """Velocity box including a complete stop before either position limit."""
    q,v,lo,hi=[np.asarray(x,float) for x in (position,velocity,lower,upper)]
    if (q.ndim!=1 or any(x.shape!=q.shape for x in (v,lo,hi))
            or not np.isfinite(np.r_[q,v,lo,hi,dt,speed,acceleration]).all()
            or not 0<dt<=.00200001 or not 0<speed<=1.2 or not 0<acceleration<=3.
            or np.any(lo>=hi) or np.any(q<lo-1e-10) or np.any(q>hi+1e-10)
            or np.any(abs(v)>speed+1e-10)):
        raise ValueError('Require a finite original-bounded joint state and consecutive2ms interval')
    adt=acceleration*dt
    def stopping_distance(velocity):
        positive=np.maximum(velocity,0.)
        steps=np.ceil(positive/adt)
        return dt*((steps-.5)*positive-.5*adt*steps*(steps-1))
    if np.any(stopping_distance(v)>hi-q+1e-10) or np.any(stopping_distance(-v)>q-lo+1e-10):
        raise ValueError('The supplied target state has no bounded joint-limit stopping path')
    def stop_limit(distance,direction_velocity):
        # Invert the exact piecewise-linear stopping distance for trapezoidal
        # target integration. A final partial deceleration occupies a full
        # update interval; continuous v^2/(2a) would underestimate its travel.
        available=np.maximum(0.,distance/dt-.5*direction_velocity)
        steps=np.maximum(1.,np.ceil((np.sqrt(1+8*available/adt)-1)/2))
        return (available+.5*adt*steps*(steps-1))/steps
    low=np.maximum.reduce([np.full_like(v,-speed),v-adt,-stop_limit(q-lo,-v)])
    high=np.minimum.reduce([np.full_like(v,speed),v+adt,stop_limit(hi-q,v)])
    if np.any(low>high+1e-9):raise ValueError('No bounded next target velocity exists')
    return low,high


class ActualBasePalmCorrection:
    def __init__(self,model,joint_names,palm_site,*,maximum_speed=1.2,maximum_acceleration=3.,joint_margin=.0001):
        if not 0<maximum_speed<=1.2 or not 0<maximum_acceleration<=3. or not 0<=joint_margin<=.001:
            raise ValueError('The correction may only retain or tighten the original motion bounds')
        self.model=model;self.data=mujoco.MjData(model)
        roots=[j for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
        if len(roots)!=1:raise ValueError('Require one robot free base in a robot-only model')
        self.root_address=int(model.jnt_qposadr[roots[0]])
        self.names=list(joint_names);self.joints=np.array([model.joint(n).id for n in self.names])
        if len(set(self.names))!=len(self.names) or not len(self.names) or not np.all(model.jnt_limited[self.joints]):
            raise ValueError('Require unique original limited controlled joints')
        self.qa=model.jnt_qposadr[self.joints];self.va=model.jnt_dofadr[self.joints]
        self.lower=model.jnt_range[self.joints,0]+joint_margin;self.upper=model.jnt_range[self.joints,1]-joint_margin
        self.scalar={model.joint(j).name:int(model.jnt_qposadr[j]) for j in range(model.njnt) if model.jnt_type[j] in (2,3)}
        self.palm=model.site(palm_site).id;self.speed=float(maximum_speed);self.acceleration=float(maximum_acceleration)
        self.jp=np.zeros((3,model.nv));self.jr=np.zeros((3,model.nv));self.time=None;self.target=None;self.velocity=None;self.previous_goal=None;self.previous_pose=None

    def begin(self,time_s,target,velocity):
        q,v=np.asarray(target,float),np.asarray(velocity,float)
        if self.time is not None or q.shape!=self.lower.shape or v.shape!=q.shape or not np.isfinite(np.r_[time_s,q,v]).all():
            raise ValueError('Require a finite initial actually commanded arm target')
        bounded_next_velocity(q,v,self.lower,self.upper,.002,self.speed,self.acceleration)
        self.time=float(time_s);self.target=q.copy();self.velocity=v.copy()

    def update(self,time_s,joints,goal_position_base,goal_rotation_base,nominal_target):
        p=np.asarray(goal_position_base,float);r=np.asarray(goal_rotation_base,float);nominal=np.asarray(nominal_target,float)
        if (self.time is None or set(joints)!=set(self.scalar) or p.shape!=(3,) or r.shape!=(3,3)
                or nominal.shape!=self.target.shape or not np.isfinite(np.r_[time_s,list(joints.values()),p,r.ravel(),nominal]).all()
                or not np.allclose(r@r.T,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(r),1.,atol=1e-7)):
            raise ValueError('Require complete finite scalar proprioception and a proper base-frame palm command')
        dt=float(time_s-self.time)
        if dt<0 or dt>.00200001:raise ValueError('Require consecutive measured2ms target updates')
        model,data=self.model,self.data;data.qpos[:]=model.qpos0
        data.qpos[self.root_address:self.root_address+7]=[0.,0.,0.,1.,0.,0.,0.]
        for name,address in self.scalar.items():data.qpos[address]=joints[name]
        data.qpos[self.qa]=self.target;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        if dt:
            mujoco.mj_jacSite(model,data,self.jp,self.jr,self.palm)
            a=np.vstack([100*self.jp[:,self.va],10*self.jr[:,self.va],.02*np.eye(len(self.names))])
            error=np.r_[p-data.site_xpos[self.palm],Rotation.from_matrix(r@data.site_xmat[self.palm].reshape(3,3).T).as_rotvec()]
            feedforward=np.zeros(6)
            if self.previous_goal is not None:
                previous_p,previous_r=self.previous_goal;prior_p,prior_r=self.previous_pose
                goal_delta=np.r_[p-previous_p,Rotation.from_matrix(r@previous_r.T).as_rotvec()]
                passive_delta=np.r_[data.site_xpos[self.palm]-prior_p,Rotation.from_matrix(data.site_xmat[self.palm].reshape(3,3)@prior_r.T).as_rotvec()]
                feedforward=(goal_delta-passive_delta)/dt
            task_velocity=6.*(error-dt*feedforward)+feedforward
            b=np.r_[100*task_velocity[:3],10*task_velocity[3:],.12*(nominal-self.target)]
            low,high=bounded_next_velocity(self.target,self.velocity,self.lower,self.upper,dt,self.speed,self.acceleration)
            fixed=high-low<1e-12;next_velocity=.5*(low+high);free=~fixed
            matrix=a;rhs=b-matrix[:,fixed]@next_velocity[fixed]
            if np.any(free):
                fit=lsq_linear(matrix[:,free],rhs,bounds=(low[free],high[free]),method='bvls',tol=1e-12,max_iter=100)
                if not fit.success or not np.isfinite(fit.x).all():raise ValueError('Bounded Cartesian target solve failed')
                next_velocity[free]=fit.x
            acceleration=(next_velocity-self.velocity)/dt
            self.target=self.target+.5*dt*(self.velocity+next_velocity);self.velocity=next_velocity;self.time=float(time_s)
        else:acceleration=np.zeros_like(self.target)
        data.qpos[self.qa]=self.target;mujoco.mj_kinematics(model,data)
        self.previous_goal=(p.copy(),r.copy());self.previous_pose=(data.site_xpos[self.palm].copy(),data.site_xmat[self.palm].reshape(3,3).copy())
        info=dict(time_s=float(time_s),commanded_fk_position_residual_m=float(np.linalg.norm(p-data.site_xpos[self.palm])),commanded_fk_rotation_residual_rad=float(np.linalg.norm(Rotation.from_matrix(r@data.site_xmat[self.palm].reshape(3,3).T).as_rotvec())),maximum_target_speed_rad_s=float(np.max(abs(self.velocity))),maximum_target_acceleration_rad_s2=float(np.max(abs(acceleration))),minimum_joint_margin_rad=float(np.min(np.minimum(self.target-self.lower,self.upper-self.target))))
        if info['minimum_joint_margin_rad']< -1e-9 or info['maximum_target_speed_rad_s']>self.speed+1e-9 or info['maximum_target_acceleration_rad_s2']>self.acceleration+1e-9:
            raise ValueError('The correction violated its original joint or motion bounds')
        return self.target.copy(),self.velocity.copy(),info
