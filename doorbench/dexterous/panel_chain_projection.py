"""Opt-in hybrid normal control over the actual waist and left-arm motor chain.

The inherited controller already projects seven arm motors. The panel IK also
moves the waist, so this experiment projects the combined eight-motor command
after ordinary force assembly. Only original capped motor forces are returned.
"""
import numpy as np
import mujoco
from .bimanual_transfer import replace_normal_acceleration


def project_panel_chain(forces, indices, mass, normal_jacobian, bias,
                        requested_normal_force, blend, caps):
    forces=np.asarray(forces,float);indices=np.asarray(indices,int)
    mass=np.asarray(mass,float);jac=np.asarray(normal_jacobian,float)
    bias=np.asarray(bias,float);caps=np.asarray(caps,float)
    if (forces.shape!=(61,) or indices.shape!=(8,) or len(set(indices))!=8
            or min(indices)<0 or max(indices)>=61 or mass.shape!=(8,8)
            or jac.shape!=(8,) or bias.shape!=(8,) or caps.shape!=(61,2)
            or not np.isfinite(np.r_[forces,mass.ravel(),jac,bias,caps.ravel(),requested_normal_force,blend]).all()
            or not 0<=requested_normal_force<=12 or not 0<=blend<=1
            or not np.all(caps[:,0]<=caps[:,1])):
        raise ValueError('Invalid complete eight-motor force projection contract')
    projected=replace_normal_acceleration(forces[indices]-bias,mass,jac,
                                           requested_normal_force,gravity=bias)
    result=forces.copy()
    result[indices]+=blend*(projected-forces[indices])
    result=np.clip(result,caps[:,0],caps[:,1])
    inverse=np.linalg.solve(mass,jac)
    info=dict(normal_acceleration_before=float(inverse@(forces[indices]-bias)),
              normal_acceleration_after=float(inverse@(result[indices]-bias)),
              full_projection_acceleration_goal=float(requested_normal_force*(jac@inverse)),
              projected_unclipped_forces=projected.tolist(),
              actual_capped_chain_forces=result[indices].tolist(),
              active_original_caps=caps[indices].tolist(),blend=float(blend))
    return result,info


def measured_panel_chain(opening):
    """Read only the teacher's current unstepped calculator; never a plant."""
    teacher,left=opening.acquisition,opening.left
    m,d=teacher.m,teacher.d
    torso=m.joint('torso').id
    waist_motor=list(teacher.act).index(m.actuator('torso').id)
    indices=np.r_[waist_motor,left.act]
    coordinates=np.r_[m.jnt_dofadr[torso],left.va]
    if tuple(left.names[:1])!=('torso',):raise ValueError('Expected waist plus left arm IK')
    point=d.site_xpos[left.palm].copy()
    if hasattr(left,'normal_contact_point_local'):
        point+=d.site_xmat[left.palm].reshape(3,3)@left.normal_contact_point_local
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    mujoco.mj_jac(m,d,jp,jr,point,m.site_bodyid[left.palm])
    mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,mass)
    weighted=np.vstack([100*jp[:,coordinates],10*jr[:,coordinates]])
    return dict(indices=indices,coordinates=coordinates,mass=mass[np.ix_(coordinates,coordinates)],
                normal_jacobian=left.normal@jp[:,coordinates],bias=d.qfrc_bias[coordinates].copy(),
                weighted_task_jacobian_singular_values=np.linalg.svd(weighted,compute_uv=False),
                position_jacobian=jp[:,coordinates],rotation_jacobian=jr[:,coordinates],
                support_point_world=point)


def apply_waist_arm_projection(opening,forces):
    if opening.push.started is None:return np.asarray(forces).copy(),None
    if opening.panel_profile!='hybrid-surface-v2':raise ValueError('Require explicitly selected hybrid profile')
    left=opening.left
    chain=measured_panel_chain(opening)
    result,info=project_panel_chain(forces,chain['indices'],chain['mass'],
        chain['normal_jacobian'],chain['bias'],left.info['requested_normal_force_N'],
        left.hybrid_blend,opening.caps)
    info.update(chain_joint_names=list(left.names),chain_motor_indices=chain['indices'].tolist(),
                mass_matrix=chain['mass'].tolist(),normal_jacobian=chain['normal_jacobian'].tolist(),
                weighted_task_jacobian_singular_values=chain['weighted_task_jacobian_singular_values'].tolist(),
                position_jacobian=chain['position_jacobian'].tolist(),rotation_jacobian=chain['rotation_jacobian'].tolist(),
                support_point_world=chain['support_point_world'].tolist(),
                requested_normal_force_N=left.info['requested_normal_force_N'])
    return result,info
