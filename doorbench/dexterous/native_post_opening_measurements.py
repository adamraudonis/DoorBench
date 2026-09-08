"""Read-only native measurements for the privileged portable continuation.

These helpers never step or write a plant. Current poses must already have had
their kinematics refreshed; raw contacts belong to the named preceding physical
interval. This evaluator/teacher adapter must not be given to a sensor actor.
"""
import numpy as np


def measured_contacts(m,raw,*,physics_qualified):
    if not isinstance(physics_qualified,(bool,np.bool_)):raise ValueError('Explicit physical audit result required')
    forces={m.body(b).name.removeprefix('robot/'):np.zeros(3) for b in range(m.nbody) if m.body(b).name.startswith(('robot/rh_','robot/lh_'))}
    feet=np.zeros(2);left_count=right_count=0;left_load=0.
    for c in raw['contacts']:
        frame=np.asarray(c['frame_world']);force=frame.T@np.asarray(c['wrench_contact_frame'][:3]);full_names=[m.body(b).name for b in c['body']];names=[n.removeprefix('robot/') for n in full_names]
        for name,sign in zip(names,(-1,1)):
            if name in forces:forces[name]+=sign*force
            if name in ('left_ankle_link','right_ankle_link'):feet[('left_ankle_link','right_ankle_link').index(name)]+=sign*force[2]
        if any(n.startswith('lh_') for n in names):
            left_count+=int(c['distance_m']<=0 or c['wrench_contact_frame'][0]>1e-8);left_load+=max(0.,float(c['wrench_contact_frame'][0]))
        if any(n.startswith('robot/rh_') for n in full_names) and not all(n.startswith('robot/') for n in full_names):
            right_count+=int(c['distance_m']<=0 or c['wrench_contact_frame'][0]>1e-8)
    return feet,forces,dict(physics_qualified=bool(physics_qualified),left_hand_contacts=left_count,left_hand_load_N=left_load,right_environment_contacts=right_count)


def qualified_physics_row(row):
    """Existing native continuation bounds, evaluated on detached audited data."""
    if not all(isinstance(row[k],(bool,np.bool_)) for k in ('finite','native_motor_limits')):return False
    return bool(row['finite'] and row['native_motor_limits'] and row['numerical_warnings']==0
        and 0<=row['torso_tilt_deg']<12 and row['root_height_m']>.7
        and 0<=row['max_joint_limit_violation_rad']<=.02
        and 0<=row.get('all_joint_violation',row['max_joint_limit_violation_rad'])<=.02
        and 0<=row['max_nonfoot_penetration_m']<=.003
        and 0<=row['max_shadow_loopback_violation_rad']<=.02
        and row['external_wrench_max']==0 and row['applied_generalized_force_max']==0
        and row.get('motor_delivery_error_Nm',0.)<1e-5)


def measured_state(s,names,door_names,pose_names):
    m,d=s.m,s.d;q=d.qpos;v=d.qvel;r=s.root_qadr;rv=s.root_vadr;rotation=d.xmat[s.pelvis].reshape(3,3)
    root=np.r_[q[r:r+7],v[rv:rv+3],rotation@v[rv+3:rv+6]]
    joints={n:float(q[m.jnt_qposadr[m.joint('robot/'+n).id]]) for n in names}
    velocities={n:float(v[m.jnt_dofadr[m.joint('robot/'+n).id]]) for n in names}
    door={n:float(q[m.jnt_qposadr[m.joint(n).id]]) for n in door_names}
    door_v={n:float(v[m.jnt_dofadr[m.joint(n).id]]) for n in door_names}
    bodies={n:m.body(n if n.startswith('leaf') else 'robot/'+n).id for n in pose_names}
    poses={n:np.r_[d.xpos[b],d.xquat[b]].tolist() for n,b in bodies.items()}
    return root,joints,velocities,dict(door_positions=door,door_velocities=door_v,body_poses=poses)


def outward_release_normal(m,raw):
    normals=[]
    for c in raw['contacts']:
        names=[m.body(b).name for b in c['body']]
        if 'leaf' in names and any(n.startswith('robot/lh_') for n in names):normals.append((1 if names[1].startswith('robot/lh_') else -1)*np.asarray(c['frame_world'])[0])
    if not normals:raise ValueError('An actual attained left-panel contact normal is required')
    normal=np.mean(normals,axis=0)
    if not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-6:raise ValueError('Ambiguous measured release direction')
    return normal/np.linalg.norm(normal)
