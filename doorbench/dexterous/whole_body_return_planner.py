"""Unstepped whole-body return planning from copied attained-state evidence.

Only a static model and numeric state enter this module. It creates its own
MjData, runs kinematics/collision queries, and never steps physics or receives
an active simulator/controller. This is the first planner extraction; destination
collision-model qualification and runtime release composition remain separate.
"""
import hashlib
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def copy_attained_return_state(model, *, qpos, qvel, time_s, geometry_time_s,
                               body_ids, body_positions_world_m, body_rotations_world,
                               observation_scope="all-required-bodies"):
    """Validate same-epoch observations, then bind a private numeric state.

    The original native FK tolerances remain1e-9. A future imported-state adapter
    must explicitly qualify its own mapping rather than weakening this check.
    No force buffer or predicted endpoint can stand in for current observations.
    Historical native archives record only contact-body poses. Their explicit
    archived-native-contact-bodies mode preserves that original CLI comparison
    and does not qualify a destination-engine observation mapping.
    """
    if observation_scope not in ("all-required-bodies", "archived-native-contact-bodies"):
        raise ValueError("Unknown attained-state observation coverage")
    q = np.array(qpos, dtype=float, copy=True)
    v = np.array(qvel, dtype=float, copy=True)
    ids = np.array(body_ids, copy=True)
    positions = np.array(body_positions_world_m, dtype=float, copy=True)
    rotations = np.array(body_rotations_world, dtype=float, copy=True)
    if (q.shape != (model.nq,) or v.shape != (model.nv,)
            or ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or not len(ids) or len(set(ids.tolist())) != len(ids)
            or (ids < 0).any() or (ids >= model.nbody).any()
            or positions.shape != (len(ids), 3)
            or rotations.shape != (len(ids), 3, 3)):
        raise ValueError('Require complete numeric state and unique actual body observations')
    if (not np.isfinite(np.r_[time_s, geometry_time_s, q, v, positions.ravel(), rotations.ravel()]).all()
            or time_s < 0 or abs(time_s-geometry_time_s) > 1e-10):
        raise ValueError('Require finite same-epoch attained state')
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError('Require one free humanoid root')
    rq = int(model.jnt_qposadr[free[0]])
    if not np.isclose(np.linalg.norm(q[rq+3:rq+7]), 1., atol=1e-10, rtol=0):
        raise ValueError('Require the actual normalized root quaternion')
    required = {model.body(n).id for n in ('leaf_handle', 'robot/left_ankle_link',
                'robot/right_ankle_link', 'robot/torso_link')}
    required.update(int(model.site_bodyid[model.site('robot/'+side+'_palm_touch').id]) for side in ('rh','lh'))
    if observation_scope == 'all-required-bodies' and not required.issubset(set(ids.tolist())):
        raise ValueError('Require handle, palm, foot and torso body observations')
    d = mujoco.MjData(model)
    d.qpos[:] = q; d.qvel[:] = v; d.time = float(time_s)
    mujoco.mj_kinematics(model, d)
    if (not np.allclose(d.xpos[ids], positions, atol=1e-9, rtol=0)
            or not np.allclose(d.xmat[ids].reshape(-1,3,3), rotations, atol=1e-9, rtol=0)):
        raise ValueError('Actual body poses disagree with attained-state FK')
    packet = dict(time_s=float(time_s), geometry_time_s=float(geometry_time_s),
                  observation_scope=observation_scope,
                  qpos=q.tolist(), qvel=v.tolist(), body_ids=ids.tolist(),
                  body_positions_world_m=positions.tolist(), body_rotations_world=rotations.tolist())
    identity = hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return d, identity


def iter_whole_body_return(model, **attained_state):
    """Yield the original41-node geometric solve; does not qualify dynamics.

    All source observations are copied and validated before the first solve.
    The20mm root bounds, original joint margins and planted-foot objectives
    are unchanged. Only the private planner data may be modified.
    """
    m = model
    d, identity = copy_attained_return_state(m, **attained_state)
    base = d.qpos.copy()
    hb=m.body('leaf_handle').id;rh=m.site('robot/rh_palm_touch').id;lh=m.site('robot/lh_palm_touch').id;H=d.xmat[hb].reshape(3,3);pr=H.T@(d.site_xpos[rh]-d.xpos[hb]);rr=H.T@d.site_xmat[rh].reshape(3,3);PL=d.site_xpos[lh].copy();RL=d.site_xmat[lh].reshape(3,3).copy()
    hj=m.joint('leaf_handle_hinge').id;d.qpos[m.jnt_qposadr[hj]]=0.;mujoco.mj_kinematics(m,d);PR=d.xpos[hb]+d.xmat[hb].reshape(3,3)@pr;RR=d.xmat[hb].reshape(3,3)@rr;target_base=d.qpos.copy()
    right=['right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['rh_WRJ2','rh_WRJ1'];left=['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1'];reports=[]

    from scipy.spatial.transform import Rotation
    # base is exact attained state; target_base sets only unstepped operator to rest.
    d.qpos[:]=base;mujoco.mj_kinematics(m,d);feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')];FP=d.xpos[feet].copy();FR=d.xmat[feet].copy().reshape(2,3,3);free_ids=np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE);assert len(free_ids)==1;rq=int(m.jnt_qposadr[free_ids[0]]);rootR=Rotation.from_quat([*base[rq+4:rq+7],base[rq+3]]);rootP=base[rq:rq+3].copy()
    legs=[side+'_'+j for side in ('left','right') for j in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')];names=legs+['torso']+right+left;js=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[js];start=base[qa].copy();results=[]
    initial_h=float(base[m.jnt_qposadr[hj]]);previous=np.r_[np.zeros(6),start];extent=.02
    for blend in np.linspace(0,1,41):
     target_base=base.copy();target_base[m.jnt_qposadr[hj]]=initial_h*(1-blend);d.qpos[:]=target_base;mujoco.mj_kinematics(m,d);PR=d.xpos[hb]+d.xmat[hb].reshape(3,3)@pr;RR=d.xmat[hb].reshape(3,3)@rr
     lo=np.r_[[-extent]*3,[-.06,-.06,-.08],m.jnt_range[js,0]+.01];hi=np.r_[[extent]*3,[.06,.06,.08],m.jnt_range[js,1]-.01];x0=previous.copy()
     def fun(x):
      d.qpos[:]=target_base;d.qpos[rq:rq+3]=rootP+x[:3];rot=(Rotation.from_rotvec(x[3:6])*rootR).as_quat();d.qpos[rq+3:rq+7]=np.r_[rot[3],rot[:3]];d.qpos[qa]=x[6:];mujoco.mj_kinematics(m,d)
      hand=np.r_[100*(d.site_xpos[rh]-PR),10*Rotation.from_matrix(RR@d.site_xmat[rh].reshape(3,3).T).as_rotvec(),100*(d.site_xpos[lh]-PL),10*Rotation.from_matrix(RL@d.site_xmat[lh].reshape(3,3).T).as_rotvec()]
      foot=np.concatenate([np.r_[100*(d.xpos[b]-FP[i]),10*Rotation.from_matrix(FR[i]@d.xmat[b].reshape(3,3).T).as_rotvec()] for i,b in enumerate(feet)])
      return np.r_[hand,foot,.01*(x[6:]-start),.02*x[:6]]
     fit=least_squares(fun,np.clip(x0,lo,hi),bounds=(lo,hi),max_nfev=2000,ftol=1e-12,xtol=1e-12,gtol=1e-12);previous=fit.x.copy();res=fun(fit.x);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d);cols=[]
     for c in d.contact[:d.ncon]:
      bs=[m.body(m.geom_bodyid[g]).name for g in c.geom];gs=[m.geom(int(g)).name for g in c.geom]
      if any(b.startswith('robot/') for b in bs) and not ('floor' in gs and any(b.endswith('_ankle_link') for b in bs)) and c.dist<-.003:cols.append(dict(depth=-float(c.dist),bodies=bs))
     torso=m.body('robot/torso_link').id;root_body=m.jnt_bodyid[free_ids[0]];robot_mass=m.body_subtreemass[root_body];com=d.subtree_com[root_body].copy();up=d.xmat[torso].reshape(3,3)[:,2]
     result=dict(progress=float(blend),goal_operator_rad=initial_h*(1-float(blend)),root_qpos_address=rq,torso_tilt_deg=float(np.degrees(np.arccos(np.clip(up[2],-1,1)))),robot_com_world_m=com.tolist(),robot_mass_kg=float(robot_mass),root_translation_bound_m=extent,right_position_error_m=float(np.linalg.norm(res[:3])/100),right_rotation_error_rad=float(np.linalg.norm(res[3:6])/10),left_position_error_m=float(np.linalg.norm(res[6:9])/100),left_rotation_error_rad=float(np.linalg.norm(res[9:12])/10),foot_position_errors_m=[float(np.linalg.norm(res[12+6*i:15+6*i])/100) for i in range(2)],foot_rotation_errors_rad=[float(np.linalg.norm(res[15+6*i:18+6*i])/10) for i in range(2)],root_delta_xyz_m=fit.x[:3].tolist(),root_delta_rotvec_rad=fit.x[3:6].tolist(),joints=dict(zip(names,fit.x[6:].tolist())),joint_margins_rad=dict(zip(names,np.minimum(fit.x[6:]-(m.jnt_range[js,0]+.01),(m.jnt_range[js,1]-.01)-fit.x[6:]).tolist())),forbidden_collisions=cols,nfev=fit.nfev,qpos=d.qpos.tolist());yield result
