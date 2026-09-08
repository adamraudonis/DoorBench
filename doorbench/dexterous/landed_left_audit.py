"""Independent dense geometric audit of a frozen-state left-palm plan.

Both nominal-joint interpolation and the controller's Cartesian interpolation
with bounded IK are screened. Collision forces are never inferred from FK.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares


def static_pose_check(m, d, *, coordinate, contact_start=.98):
    """Inspect all native collision pairs, including receiving-only geometries."""
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    contacts = []
    maximum = 0.
    left_touch = 0
    for c in d.contact[:d.ncon]:
        bodies = [m.body(m.geom_bodyid[g]).name or '' for g in c.geom]
        geoms = [m.geom(g).name or '' for g in c.geom]
        robot = [b.startswith('robot/') for b in bodies]
        depth = max(0., -float(c.dist))
        if not any(robot):
            continue
        foot_floor = ('floor' in geoms and any(b in
            ('robot/left_ankle_link', 'robot/right_ankle_link') for b in bodies))
        if not foot_floor:
            maximum = max(maximum, depth)
        lh = [b.startswith('robot/lh_') for b in bodies]
        left_environment = any(lh) and not all(robot)
        touching = c.dist <= 0.
        if left_environment and touching:
            left_touch += 1
        invalid_left = left_environment and touching and (coordinate < contact_start or
            bodies[robot.index(False)] != 'leaf')
        if invalid_left or (depth > .003 and not foot_floor):
            contacts.append(dict(bodies=bodies, geoms=geoms, distance_m=float(c.dist),
                reason='Premature/off-panel left touch' if invalid_left else 'Nonfoot penetration'))
    maximum_joint = 0.
    for j in range(m.njnt):
        if not m.jnt_limited[j]:
            continue
        if m.jnt_type[j] not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError('Explicit ball-joint bound audit required')
        value = d.qpos[m.jnt_qposadr[j]]
        maximum_joint = max(maximum_joint, m.jnt_range[j, 0]-value, value-m.jnt_range[j, 1])
    maximum_loopback = 0.
    for i in range(m.ntendon):
        if not m.tendon_limited[i] or not (m.tendon(i).name or '').startswith('robot/'):
            continue
        length = 0.
        for k in range(m.tendon_adr[i], m.tendon_adr[i]+m.tendon_num[i]):
            if m.wrap_type[k] != int(mujoco.mjtWrap.mjWRAP_JOINT):
                raise ValueError('Explicit spatial-tendon audit required')
            length += m.wrap_prm[k]*d.qpos[m.jnt_qposadr[int(m.wrap_objid[k])]]
        maximum_loopback = max(maximum_loopback, m.tendon_range[i, 0]-length,
                               length-m.tendon_range[i, 1])
    finite = bool(np.isfinite(d.qpos).all())
    return dict(passed=finite and not contacts and maximum_joint <= .02 and maximum_loopback <= .02,
                contacts=contacts, left_environment_touches=left_touch,
                maximum_nonfoot_penetration_m=float(maximum), maximum_joint_violation=float(maximum_joint),
                maximum_loopback_violation_rad=float(maximum_loopback), finite=finite)


def audit_landed_left_path(scene, config, state, *, subdivisions=10):
    """Recompute FK, then audit densely without reusing the planner's solver."""
    from .landed_left_planner import read_targets
    if type(subdivisions) is not int or not 2 <= subdivisions <= 100:
        raise ValueError('Use 2..100 independent samples per original segment')
    provisional = dict(config, passed=True)
    source = read_targets(provisional)
    m = scene.m
    d = mujoco.MjData(m)
    frozen = scene.freeze(state)
    qa = np.array([m.jnt_qposadr[m.joint('robot/'+n).id] for n in config['joint_names']])
    palm, leaf = m.site('robot/lh_palm_touch').id, m.body('leaf').id
    d.qpos[:] = frozen
    mujoco.mj_kinematics(m, d)
    leaf_rotation, leaf_position = d.xmat[leaf].reshape(3, 3).copy(), d.xpos[leaf].copy()
    maximum_position = maximum_normal = maximum_leaf_metadata = maximum_torso = 0.
    for row in config['targets']:
        d.qpos[:] = frozen
        d.qpos[qa] = row['nominal']
        mujoco.mj_kinematics(m, d)
        position = leaf_rotation.T@(d.site_xpos[palm]-leaf_position)
        normal = leaf_rotation.T@d.site_xmat[palm].reshape(3, 3)[:, 2]
        maximum_position = max(maximum_position, float(np.linalg.norm(position-row['position'])))
        maximum_normal = max(maximum_normal, float(np.linalg.norm(normal-row['normal'])))
        maximum_torso = max(maximum_torso, float(abs(row['nominal'][0]-frozen[qa[0]])))
        maximum_leaf_metadata = max(maximum_leaf_metadata, abs(row['leaf_rad']-state['door_positions']['leaf_hinge']))
    initial_error = float(np.max(abs(source['nominal'][0]-frozen[qa])))
    samples = (len(source['nominal'])-1)*subdivisions+1
    arm = np.array([m.joint('robot/'+n).id for n in config['joint_names'][1:]])
    lower, upper = m.jnt_range[arm, 0]+.01, m.jnt_range[arm, 1]-.01
    reports = {}
    for mode in ('nominal_joint_interpolation', 'cartesian_target_interpolation_ik'):
        bad, maximum_depth, maximum_joint, maximum_loopback = [], 0., 0., 0.
        previous = frozen[qa[1:]].copy()
        maximum_ik_position = maximum_ik_normal = maximum_ik_delta = 0.
        first_left_touch = None
        solver_failures = 0
        for index, u in enumerate(np.linspace(0., 1., samples)):
            coordinate = u*(len(source['nominal'])-1)
            i = min(int(coordinate), len(source['nominal'])-2)
            f = coordinate-i
            nominal = (1-f)*source['nominal'][i]+f*source['nominal'][i+1]
            d.qpos[:] = frozen
            d.qpos[qa] = nominal
            position_error = normal_error = 0.
            if mode == 'cartesian_target_interpolation_ik':
                local = (1-f)*source['position'][i]+f*source['position'][i+1]
                normal = (1-f)*source['normal'][i]+f*source['normal'][i+1]
                normal /= np.linalg.norm(normal)
                goal, desired_normal = leaf_position+leaf_rotation@local, leaf_rotation@normal

                def residual(q):
                    d.qpos[qa[1:]] = q
                    mujoco.mj_kinematics(m, d)
                    return np.r_[100*(d.site_xpos[palm]-goal),
                        10*(d.site_xmat[palm].reshape(3, 3)[:, 2]-desired_normal), .03*(q-nominal[1:])]

                fit = least_squares(residual, np.clip(previous, lower, upper),
                                    bounds=(lower, upper), max_nfev=140)
                d.qpos[qa[1:]] = fit.x
                if not fit.success:
                    solver_failures += 1
                delta = float(np.max(abs(fit.x-previous)))
                maximum_ik_delta = max(maximum_ik_delta, delta)
                previous = fit.x.copy()
                mujoco.mj_kinematics(m, d)
                position_error = float(np.linalg.norm(d.site_xpos[palm]-goal))
                normal_error = float(np.rad2deg(np.arccos(np.clip(
                    desired_normal@d.site_xmat[palm].reshape(3, 3)[:, 2], -1., 1.))))
                maximum_ik_position = max(maximum_ik_position, position_error)
                maximum_ik_normal = max(maximum_ik_normal, normal_error)
            result = static_pose_check(m, d, coordinate=u)
            maximum_depth = max(maximum_depth, result['maximum_nonfoot_penetration_m'])
            maximum_joint = max(maximum_joint, result['maximum_joint_violation'])
            maximum_loopback = max(maximum_loopback, result['maximum_loopback_violation_rad'])
            if result['left_environment_touches'] and first_left_touch is None:
                first_left_touch = float(u)
            if not result['passed'] or position_error > .001 or normal_error > .5:
                bad.append(dict(sample=index, path_coordinate=float(u), **result,
                                ik_position_error_m=position_error, ik_normal_error_deg=normal_error))
        reports[mode] = dict(passed=not bad and solver_failures == 0, samples=samples,
            failed_samples=len(bad), bad_samples=bad, solver_failures=solver_failures,
            maximum_nonfoot_penetration_m=maximum_depth, maximum_joint_violation=maximum_joint,
            maximum_loopback_violation_rad=maximum_loopback,
            maximum_ik_position_error_m=maximum_ik_position, maximum_ik_normal_error_deg=maximum_ik_normal,
            maximum_ik_joint_sample_delta_rad=maximum_ik_delta, first_left_environment_touch=first_left_touch)
    checks = dict(exact_initial_arm=initial_error < 1e-10,
        fixed_actual_torso=maximum_torso < 1e-10, target_fk_positions=maximum_position < 1e-8,
        target_fk_normals=maximum_normal < 1e-8, actual_leaf_metadata=maximum_leaf_metadata < 1e-10,
        nominal_path=reports['nominal_joint_interpolation']['passed'],
        cartesian_ik_path=reports['cartesian_target_interpolation_ik']['passed'])
    return dict(passed=all(checks.values()), checks=checks, paths=reports,
        initial_arm_error_rad=initial_error, maximum_target_fk_error_m=maximum_position,
        maximum_normal_vector_error=maximum_normal, maximum_leaf_metadata_error_rad=maximum_leaf_metadata,
        physics_steps=0, active_state_writes=0, contact_force_claim=False,
        scope='Frozen actual root, door, opposite arm, legs and fingers. Dense sampled static geometry only; '
              'no between-sample, moving-base, contact-offset, motor, balance, or live-plant guarantee')
