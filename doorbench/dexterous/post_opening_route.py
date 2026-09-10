"""Additional screened stow waypoints for a newly attained opening state."""
import mujoco
import numpy as np
from .post_opening import plan_stow, compact_posture, screen_state


def plan_sequential_stow(sim, reset, *, inward_roll=.07, retreat_m=.14,
                         retreat_normal_world=None, left_style='fingers_first',
                         right_style='yaw_first', samples=101,finger_profile='original-v1'):
    allowed=('fingers_first','yaw_first','lift_yaw','roll_yaw')
    if left_style not in allowed or right_style not in allowed:
        raise ValueError('Unknown explicit stow waypoint style')
    original=plan_stow(sim,reset,inward_roll=inward_roll,retreat_m=retreat_m,
                       retreat_normal_world=retreat_normal_world,samples=samples)
    m=sim.m;d=mujoco.MjData(m);start=sim.d.qpos.copy()
    target=compact_posture(m,start,reset,inward_roll=inward_roll,finger_profile=finger_profile)
    path=[np.asarray(q) for q,stage in zip(original['path_qpos'],original['stage']) if stage=='release']
    stages=['release']*len(path);bad=[r for r in original['bad_samples'] if r['stage']=='release']
    # Original release can have more failures than its compact bad-sample list.
    for i,q in enumerate(path):
        audit=screen_state(m,d,q)
        if not audit['passed']:bad.append(dict(stage='release',sample=i,audit=audit))
    q=path[-1].copy()
    waypoints=[]
    def add(stage,destination,label):
        nonlocal q
        before=q.copy()
        for i,u in enumerate(np.linspace(0,1,samples)):
            q=before*(1-u)+destination*u
            audit=screen_state(m,d,q)
            if not audit['passed']:bad.append(dict(stage=stage,waypoint=label,sample=i,audit=audit))
            path.append(q.copy());stages.append(stage)
        waypoints.append(dict(stage=stage,label=label))
    for side,hand,style in [('left','lh',left_style),('right','rh',right_style)]:
        destination=q.copy()
        for j in sim.joints:
            name=m.joint(j).name.removeprefix('robot/')
            if name.startswith(hand+'_') and not name.startswith(hand+'_WRJ'):
                destination[m.jnt_qposadr[j]]=target[m.jnt_qposadr[j]]
        add(side,destination,'close_fingers')
        if style in ('lift_yaw','roll_yaw'):
            destination=q.copy()
            name=side+('_shoulder_pitch' if style=='lift_yaw' else '_shoulder_roll')
            destination[m.jnt_qposadr[m.joint('robot/'+name).id]]=(-1.2 if style=='lift_yaw' else (.7 if side=='left' else -.7))
            add(side,destination,style)
        if style!='fingers_first':
            destination=q.copy();index=m.jnt_qposadr[m.joint('robot/'+side+'_shoulder_yaw').id]
            destination[index]=target[index];add(side,destination,'orient_before_lowering')
        destination=q.copy()
        for j in sim.joints:
            name=m.joint(j).name.removeprefix('robot/')
            if name.startswith((side+'_',hand+'_')):destination[m.jnt_qposadr[j]]=target[m.jnt_qposadr[j]]
        add(side,destination,'lower_compact_arm')
    destination=q.copy();index=m.jnt_qposadr[m.joint('robot/torso').id];destination[index]=target[index]
    add('torso',destination,'return_torso')
    return dict(passed=not bad,path_qpos=np.asarray(path).tolist(),stage=stages,
        bad_samples=bad[:50],bad_sample_count=len(bad),sample_count=len(path),waypoints=waypoints,
        retreat_normal_world=original['retreat_normal_world'],retreat_distance_m=retreat_m,
        inward_shoulder_roll_rad=inward_roll,profile='sequential-stow-v2',
        left_style=left_style,right_style=right_style,finger_profile=finger_profile,
        scope='Static measured terminal leaf/root; no dynamic or passage proof')
