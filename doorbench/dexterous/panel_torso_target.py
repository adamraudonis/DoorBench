"""Original-capped torso effort for a continuously corrected panel target."""
import numpy as np
import mujoco


def validate_unit_joint_motor(model,actuator,joint):
    if (model.actuator_trntype[actuator]!=mujoco.mjtTrn.mjTRN_JOINT
            or model.actuator_trnid[actuator,0]!=joint
            or not np.array_equal(model.actuator_gear[actuator],[1.,0.,0.,0.,0.,0.])):
        raise ValueError("Joint-coordinate torso target requires the original direct unit transmission")


def original_torso_target_force(forces,index,target,target_velocity,position,velocity,*,kp,bias,gain,damping,caps):
    """Evaluate the existing torso PD law with the new q/dq target only.

    Target velocity uses the existing damping coefficient. Plant mass, passive
    damping, gains, transmissions and force caps are never changed. No active
    simulator or support-force interface is accepted.
    """
    scalars=(target,target_velocity,position,velocity,kp,gain,damping)
    if not all(isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)) for v in scalars):
        raise ValueError("Require scalar nonboolean motor targets and original gains")
    force=np.asarray(forces,float);bias=np.asarray(bias,float);caps=np.asarray(caps,float)
    if (force.shape!=(61,) or bias.shape!=(3,) or caps.shape!=(61,2)
            or not isinstance(index,(int,np.integer)) or isinstance(index,(bool,np.bool_)) or not 0<=index<61
            or not np.isfinite(np.r_[force,bias,caps.ravel(),target,target_velocity,position,velocity,kp,gain,damping]).all()
            or kp<=0 or gain<0 or damping<0 or np.any(caps[:,0]>caps[:,1])):
        raise ValueError('Require a complete finite original torso motor contract')
    requested=kp*target+bias[0]+bias[1]*position+bias[2]*velocity+kp*gain*(target-position)-damping*(velocity-target_velocity)
    result=force.copy();result[index]=np.clip(requested,caps[index,0],caps[index,1])
    return result,dict(motor_index=int(index),target_position_rad=float(target),target_velocity_rad_s=float(target_velocity),
                       requested_torso_effort_Nm=float(requested),consumed_torso_effort_Nm=float(result[index]))
