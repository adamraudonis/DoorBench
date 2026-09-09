"""Robot-only leg posture path preserving inferred supported foot frames.

This plans motor/stance targets in an unstepped calculator. It does not infer
world geometry or constrain the active plant's feet. Actual contact, heading,
joint limits and attained height require separate physical qualification.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def plan_lowering(balance,height):
    m=balance.m;d=mujoco.MjData(m);d.qpos[:]=balance.d.qpos
    stance=balance.stance;qa=stance.qa;initial=stance.target_root.copy()
    yaw=Rotation.from_matrix(stance.target_rotation).as_euler('xyz')[2]
    angles=Rotation.from_quat(d.qpos[[4,5,6,3]]).as_euler('xyz')[:2]
    lower=np.r_[m.jnt_range[stance.joints,0],initial[:2]-.06,[-.06,-.06]]
    upper=np.r_[m.jnt_range[stance.joints,1],initial[:2]+.06,[.06,.06]]
    x=np.r_[d.qpos[qa],initial[:2],angles];nodes=[];max_pos=max_angle=0.;solver_status=[]
    for fraction in np.linspace(0.,1.,41):
        z=initial[2]+fraction*(height-initial[2]);prior=x.copy()
        def residual(values,regularize=True):
            d.qpos[qa]=values[:10];d.qpos[:3]=[*values[10:12],z];d.qpos[3:7]=Rotation.from_euler('xyz',[*values[12:14],yaw]).as_quat()[[3,0,1,2]];mujoco.mj_kinematics(m,d)
            error=[]
            for body,position,orientation in zip(stance.feet,stance.foot_positions,stance.foot_rotations):
                error.extend(10*(d.xpos[body]-position))
                error.extend(Rotation.from_matrix(d.xmat[body].reshape(3,3)@orientation.T).as_rotvec())
            if regularize:error.extend(.001*(values-prior))
            return np.asarray(error)
        fit=least_squares(residual,np.clip(x,lower+1e-8,upper-1e-8),bounds=(lower,upper),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        x=fit.x;errors=residual(x,False).reshape(2,6)
        position=float(np.max(np.linalg.norm(errors[:,:3]/10,axis=1)));angle=float(np.max(np.linalg.norm(errors[:,3:],axis=1)))
        max_pos=max(max_pos,position);max_angle=max(max_angle,angle)
        solver_status.append(dict(converged=bool(fit.success),status=int(fit.status),evaluations=int(fit.nfev),optimality=float(fit.optimality)))
        # A budget-limited optimizer iterate can still be a geometrically valid
        # motor target. Admission uses unchanged measured pose/joint bounds.
        if position>.002 or angle>.01:raise ValueError(f'Unsupported robot-only foot path at {fraction:.3f}: {position:.6g}m / {angle:.6g}rad')
        nodes.append(np.r_[x[:10],x[10:12],z,x[12:14]])
    return np.asarray(nodes),dict(nodes=41,maximum_foot_position_error_m=max_pos,maximum_foot_rotation_error_rad=max_angle,private_physics_steps=0,solver_status=solver_status,admission='Actual foot residuals and original joint bounds; optimizer convergence disclosed separately')
