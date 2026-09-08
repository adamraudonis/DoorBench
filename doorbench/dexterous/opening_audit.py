"""Evidence checks for one initialized, privileged Isaac opening demonstration.

These checks cannot establish sensor-only control, traversal or visual quality.
"""
import numpy as np


def audit_opening(rows, configuration, motors, *, minimum_opening_rad=.7):
    if not rows:
        return {'passed': False, 'checks': {'nonempty_trace': False}}
    times=np.array([r['time_s'] for r in rows])
    leaf=np.array([r['door']['leaf_hinge'] for r in rows])
    handle=np.array([r['door']['leaf_handle_hinge'] for r in rows])
    bolt=np.array([r['door']['leaf_latch_bolt_slide'] for r in rows])
    forces=np.asarray([r['motor_forces'] for r in rows])
    limits=np.asarray([m['force_range'] for m in motors['actuators']])
    opposed=np.array([r['grasp_opposition']['opposed'] for r in rows],bool)
    manipulation=(handle>.2)&(leaf<.03)
    clock_error=max(abs(r['sim_time_s']-r['time_s']) for r in rows)
    delivery_error=max(np.max(np.abs(np.asarray(r['joint_torque_command'])-r['joint_torque_sent'])) for r in rows)
    free=[i for i,x in enumerate(leaf) if x>.03]
    first_free=free[0] if free else len(rows)-1
    held=longest=0.
    for i,good in enumerate(opposed & (np.arange(len(rows))<=first_free)):
        held=held+(times[i]-times[i-1] if i else 0.) if good else 0.
        longest=max(longest,held)
    finite=all(np.isfinite(np.r_[r['root'],r['joints'],r['motor_forces'],r['joint_torque_sent'],
                                 r['sim_time_s'],r['time_s'],r['torso_tilt_deg'],list(r['door'].values())]).all() for r in rows)
    checks={
        'finite_trace':bool(finite),
        'monotonic_time':bool(np.all(np.diff(times)>0)),
        'sufficient_duration':bool(times[-1]>=3.),
        'correct_physics_clock':bool(clock_error<.0001),
        'no_runtime_pose_writes':configuration.get('runtime_pose_writes')==0,
        'no_direct_door_commands':configuration.get('direct_door_commands') is False,
        'solver_contact_coefficients':configuration.get('contact_material_audit',{}).get('backend_values_verified') is True,
        'solver_collision_offsets':configuration.get('contact_material_audit',{}).get('backend_offsets_verified') is True,
        'native_motor_force_limits':bool(forces.shape[1]==len(limits) and np.all(forces>=limits[:,0]-1e-5) and np.all(forces<=limits[:,1]+1e-5)),
        'motor_commands_delivered':bool(delivery_error<1e-4),
        'upright':bool(max(r['torso_tilt_deg'] for r in rows)<12.),
        'standing_height':bool(min(r['root'][2] for r in rows)>.7),
        'closed_start':bool(abs(leaf[0])<.01 and abs(handle[0])<.02 and abs(bolt[0])<.001),
        'latch_retracted_before_opening':bool(handle[:first_free+1].max()>.8 and bolt[:first_free+1].max()>.011),
        'opposed_loaded_grasp_before_opening':bool(longest>=.1),
        'opposed_grasp_while_turning':bool(np.count_nonzero(opposed & manipulation)>=5),
        'door_remains_open':bool(leaf[-1]>=minimum_opening_rad),
    }
    return dict(passed=all(checks.values()),checks=checks,
        scope='Initialized privileged motor teacher on one door. Not approach, traversal, a sensor-only policy, or a benchmark score. Close-up and wide visual review is still required.',
        metrics=dict(duration_s=float(times[-1]),max_handle_rad=float(handle.max()),max_bolt_m=float(bolt.max()),
            max_leaf_rad=float(leaf.max()),final_leaf_rad=float(leaf[-1]),minimum_opening_rad=minimum_opening_rad,
            max_torso_tilt_deg=max(r['torso_tilt_deg'] for r in rows),min_root_height_m=min(r['root'][2] for r in rows),
            longest_opposed_grasp_before_opening_s=float(longest),
            opposed_fraction_while_turning=float(opposed[manipulation].mean()) if manipulation.any() else 0.,
            max_clock_error_s=clock_error,max_motor_delivery_error_Nm=float(delivery_error)))
