"""Independent original-model geometry audit of an actual-Isaac release plan.

No native episode metadata is created. The source is independently re-admitted;
the candidate is checked, never modified/promoted. Geometry does not measure
motor force, contact load, balance dynamics, or successful PhysX release.
"""
import json
from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .landed_left_audit import static_pose_check
from .grasp_verification import shadow_surface_qualified
from .qualified_isaac_grasp import digest
from scripts.dexterous.audit_standing_ungrip import release_surface_scores


SAMPLES = 2001
ORIGINAL_LIMITS = dict(position_error_m=.001, rotation_error_rad=.01,
    torso_tilt_deg=4., nonfoot_penetration_m=.003, joint_violation_rad=.02,
    loopback_violation_rad=.02, joint_reference_velocity_rad_s=2.,
    final_right_hand_clearance_m=.04, lever_axial_margin_m=.001,
    lever_normal_alignment_strict_minimum=.8)


def validate_candidate_binding(candidate, context):
    """An old audit label can neither provide source identity nor skip checks."""
    if (candidate.get('schema') != 'doorbench.isaac-profiled-release-candidate.v1'
            or candidate.get('source_engine') != 'isaac-physx'
            or candidate.get('physics_steps') != 0
            or candidate.get('source_sample_playback') != 0
            or candidate.get('grasp_profile') != context.admission['grasp_profile']
            or candidate.get('source_context_sha256') != context.sha256
            or candidate.get('source_admission') != context.admission
            or candidate.get('initial_time_s') != context.terminal_time_s
            or not np.array_equal(np.asarray(candidate.get('initial_qpos')), context.qpos)):
        raise ValueError('Candidate must bind the exact newly admitted Isaac source')
    hashes = candidate.get('input_sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Candidate source and planner input hashes required')
    for name, expected in context.admission['input_sha256'].items():
        if hashes.get(name) != expected:
            raise ValueError('Candidate omits or changes actual-source evidence: '+name)
    archive = str(context.state_archive_path)
    if hashes.get(archive) != digest(archive):
        raise ValueError('Candidate must bind original actual Isaac physics NPZ')
    for name, expected in hashes.items():
        if not Path(name).is_absolute() or digest(name) != expected:
            raise ValueError('Candidate input changed: '+name)
    context.verify_inputs()


def _path_arrays(scene, candidate, actual):
    """Validate every row independently before computing dense interpolation."""
    m, d = scene.m, scene.d
    trials = candidate.get('trials')
    if not isinstance(trials, list) or len(trials) != 1:
        raise ValueError('Exactly one explicit release path required')
    nodes = trials[0].get('rows')
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError('Complete sampled release path required')
    times = np.asarray([row['time_s'] for row in nodes], float)
    qs = np.asarray([row['qpos'] for row in nodes], float)
    positions = np.asarray([row['palm_position'] for row in nodes], float)
    rotations = np.asarray([row['palm_rotation'] for row in nodes], float)
    if (times.shape != (len(nodes),) or times[0] != 0. or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0) or qs.shape != (len(nodes), m.nq)
            or not np.isfinite(qs).all() or not np.array_equal(qs[0], actual)
            or positions.shape != (len(nodes), 3) or not np.isfinite(positions).all()
            or rotations.shape != (len(nodes), 3, 3) or not np.isfinite(rotations).all()):
        raise ValueError('Finite complete monotonic path and exact first normalized qpos required')
    rq = scene.root
    if not np.allclose(np.linalg.norm(qs[:, rq+3:rq+7], axis=1), 1., atol=1e-8, rtol=0):
        raise ValueError('Already-normalized path root quaternions required')
    if (not np.allclose(rotations @ np.transpose(rotations, (0, 2, 1)), np.eye(3), atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-8, rtol=0)):
        raise ValueError('Explicit proper palm rotation matrices required')
    door = [m.jnt_qposadr[j] for j in range(m.njnt)
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            and not m.joint(j).name.startswith('robot/')]
    if not door or not np.array_equal(qs[:, door], np.broadcast_to(actual[door], (len(qs), len(door)))):
        raise ValueError('This audit requires fixed measured source door coordinates; coupled motion needs its own envelope')
    # Named reference fields are later consumed by controllers: they must encode
    # these exact screened coordinates, not an independent hidden target stream.
    robot_names = [m.joint(j).name.removeprefix('robot/') for j in range(m.njnt)
        if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        and m.joint(j).name.startswith('robot/')]
    expected_names = dict(
        joints={name for name in robot_names if not re.fullmatch(r'[rl]h_(FF|MF|RF|LF|TH)J[1-5]', name)},
        finger_joints={name for name in robot_names if re.fullmatch(r'rh_(FF|MF|RF|LF|TH)J[1-5]', name)})
    for row in nodes:
        for group in ('joints', 'finger_joints'):
            values = row.get(group)
            if not isinstance(values, dict) or not values or set(values) != expected_names[group]:
                raise ValueError('Complete named body and finger reference fields required')
            for name, value in values.items():
                if (type(value) not in (int, float) or not np.isfinite(value)
                        or float(row['qpos'][m.joint('robot/'+name).qposadr[0]]) != value):
                    raise ValueError('Named release targets differ from screened qpos')
    # Preserve the existing runtime's initial half-second source-to-first-row
    # reference segment, including exact source geometry at its first sample.
    d.qpos[:] = actual
    mujoco.mj_kinematics(m, d)
    rh = m.site('robot/rh_palm_touch').id
    return (np.r_[0., times+.5], np.vstack([actual, qs]),
            np.vstack([d.site_xpos[rh], positions]),
            np.concatenate([d.site_xmat[rh].reshape(1, 3, 3), rotations]))


def audit_geometry(scene, candidate, actual, *, duration_s):
    """Recompute 2,001 configurations with the unchanged original primitives."""
    if type(duration_s) not in (int, float) or not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError('Finite positive release duration required')
    m, d, rq = scene.m, scene.d, scene.root
    actual = np.asarray(actual, float)
    if actual.shape != (m.nq,) or not np.isfinite(actual).all():
        raise ValueError('Complete finite admitted source coordinates required')
    times, qs, positions, palm_matrices = _path_arrays(scene, candidate, actual)
    root_rotations = Slerp(times, Rotation.from_quat(qs[:,rq+3:rq+7][:,[1,2,3,0]]))
    palms = Slerp(times, Rotation.from_matrix(palm_matrices))
    rh, lh = [m.site('robot/'+name+'_palm_touch').id for name in ('rh','lh')]
    feet = [m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    torso, lever = m.body('robot/torso_link').id, m.geom('leaf_handle_lever_col_n').id
    d.qpos[:] = actual
    mujoco.mj_kinematics(m,d)
    lp, lr = d.site_xpos[lh].copy(), d.site_xmat[lh].reshape(3,3).copy()
    fp, fr = d.xpos[feet].copy(), d.xmat[feet].reshape(2,3,3).copy()
    failures, sampled = [], []
    maxima = dict(position_error_m=0., rotation_error_rad=0., torso_tilt_deg=0.,
                  nonfoot_penetration_m=0., joint_violation_rad=0., loopback_violation_rad=0.)
    distal_samples = distal_patches = 0
    checks = dict(exact_initial_normalized_qpos=True, complete_2001_samples=True,
        fixed_source_mechanism=True, finite_joint_and_tendon_limits=True,
        foot_and_palm_position=True, foot_and_palm_orientation=True,
        upright_torso=True, environment_penetration=True, selected_right_anatomy=True,
        whole_handle_geometry=True)
    for index, elapsed in enumerate(np.linspace(0, duration_s, SAMPLES)):
        u = elapsed/duration_s
        clock = times[-1]*u**3*(10+u*(-15+6*u))
        i = min(len(times)-2, max(0, int(np.searchsorted(times,clock,side='right')-1)))
        f = (clock-times[i])/(times[i+1]-times[i])
        q = (1-f)*qs[i]+f*qs[i+1]
        q[rq+3:rq+7] = root_rotations(clock).as_quat()[[3,0,1,2]]
        if index == 0: q = actual.copy()
        d.qpos[:] = q
        static = static_pose_check(m,d,coordinate=1.)
        pe = max(np.linalg.norm(d.site_xpos[rh]-((1-f)*positions[i]+f*positions[i+1])),
                 np.linalg.norm(d.site_xpos[lh]-lp),
                 max(np.linalg.norm(d.xpos[b]-p) for b,p in zip(feet,fp)))
        pairs = [(palms(clock).as_matrix(),d.site_xmat[rh].reshape(3,3)),
                 (lr,d.site_xmat[lh].reshape(3,3))]+list(zip(fr,d.xmat[feet].reshape(2,3,3)))
        er = max(np.linalg.norm(Rotation.from_matrix(a@b.T).as_rotvec()) for a,b in pairs)
        tilt = float(np.degrees(np.arccos(np.clip(d.xmat[torso].reshape(3,3)[2,2],-1,1))))
        invalid, outside, distal_bad = [], [], []
        for contact in d.contact[:d.ncon]:
            bodies = [m.body(m.geom_bodyid[g]).name or '' for g in contact.geom]
            if (contact.dist < 0 and 'leaf_handle' in bodies
                    and any(body.startswith('robot/rh_') for body in bodies) and lever not in contact.geom):
                outside.append(dict(bodies=bodies,distance_m=float(contact.dist)))
            if lever not in contact.geom or contact.dist >= 0: continue
            side = 0 if contact.geom[1] == lever else 1
            body = int(m.geom_bodyid[contact.geom[side]])
            name = m.body(body).name or ''
            if not name.startswith('robot/rh_'): continue
            match = re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(distal|middle|proximal)',name)
            matrix = d.xmat[body].reshape(3,3)
            local = matrix.T@(contact.pos-d.xpos[body])
            normal = matrix.T@(contact.frame[:3]*(1 if side == 0 else -1))
            axis = d.geom_xmat[lever].reshape(3,3)[:,2]
            relative = contact.pos-d.geom_xpos[lever]
            axial = float(relative@axis)
            radial = relative-axial*axis
            alignment = float((matrix@normal)@(-radial/max(np.linalg.norm(radial),1e-12)))
            selected, distal = (release_surface_scores(*match.groups(), local, normal,
                m.geom_size[lever,1]-abs(axial), alignment, profile=candidate['grasp_profile'])
                if match else (False,False))
            if not selected:
                invalid.append(dict(body=name, body_position_m=local.tolist(),
                    axial_clearance_m=float(m.geom_size[lever,1]-abs(axial)), alignment=alignment))
            if not distal: distal_bad.append(name)
        distal_samples += bool(distal_bad)
        distal_patches += len(distal_bad)
        values = dict(position_error_m=float(pe), rotation_error_rad=float(er), torso_tilt_deg=tilt,
            nonfoot_penetration_m=static['maximum_nonfoot_penetration_m'],
            joint_violation_rad=static['maximum_joint_violation'],
            loopback_violation_rad=static['maximum_loopback_violation_rad'])
        for key,value in values.items(): maxima[key] = max(maxima[key],value)
        checks['finite_joint_and_tendon_limits'] &= bool(static['finite'] and
            static['maximum_joint_violation'] <= .02 and static['maximum_loopback_violation_rad'] <= .02)
        checks['environment_penetration'] &= bool(not static['contacts'])
        checks['foot_and_palm_position'] &= bool(pe <= .001)
        checks['foot_and_palm_orientation'] &= bool(er <= .01)
        checks['upright_torso'] &= bool(tilt <= 4.)
        checks['selected_right_anatomy'] &= not invalid
        checks['whole_handle_geometry'] &= not outside
        sampled.append(q)
        if not static['passed'] or invalid or outside or pe > .001 or er > .01 or tilt > 4.:
            failures.append(dict(sample=index, elapsed_s=float(elapsed), **values,
                invalid_surfaces=invalid, outside_grasped_lever=outside, collision=static))
    hand = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
    environment = [g for g in range(m.ngeom) if m.geom_contype[g] and not m.body(m.geom_bodyid[g]).name.startswith('robot/')]
    if not hand or not environment: raise ValueError('Original hand and environment collision geometry required')
    clearance = min(float(mujoco.mj_geomDistance(m,d,g,h,.5,None)) for g in hand for h in environment)
    scalar = [m.jnt_qposadr[j] for j in range(m.njnt)
              if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.joint(j).name.startswith('robot/')]
    if not scalar: raise ValueError('Original robot scalar joint inventory required')
    speed = float(np.max(abs(np.diff(np.asarray(sampled)[:,scalar],axis=0)/(duration_s/2000))))
    checks.update(final_hand_environment_clearance=clearance >= .04, joint_reference_speed=speed <= 2.)
    return dict(passed=all(checks.values()) and not failures, checks=checks, samples=SAMPLES,
        duration_s=duration_s, maximum_errors=maxima, failures=failures,
        maximum_joint_reference_velocity_rad_s=speed, final_rh_environment_clearance_m=clearance,
        original_distal_counter_score=dict(invalid_anatomy_samples=distal_samples, invalid_anatomy_patches=distal_patches),
        original_thresholds=ORIGINAL_LIMITS.copy(), physics_steps=0, active_state_writes=0,
        physical_contact_qualification=False, delivered_motor_force_checked=False,
        motor_force_feasibility_inferred=False)


def audit_isaac_release_candidate(candidate_path, *, source, robot, door_xml,
                                 door_usd, duration_s=16., profile='volar-phalange-v1'):
    from .isaac_release_planning import admit_isaac_release_context
    candidate_path = Path(candidate_path).resolve()
    initial_sha = digest(candidate_path)
    primitives = [Path(__file__).resolve(), Path(static_pose_check.__code__.co_filename).resolve(),
                  Path(release_surface_scores.__code__.co_filename).resolve(),
                  Path(shadow_surface_qualified.__code__.co_filename).resolve()]
    primitive_hashes = {str(path):digest(path) for path in primitives}
    candidate = json.loads(candidate_path.read_text())
    context = admit_isaac_release_context(source,robot=robot,door_xml=door_xml,door_usd=door_usd,profile=profile)
    validate_candidate_binding(candidate, context)
    scene = context.scene()
    result = audit_geometry(scene, candidate, context.qpos, duration_s=duration_s)
    context.verify_inputs()
    validate_candidate_binding(candidate, context)
    if digest(candidate_path) != initial_sha: raise ValueError('Candidate changed during independent audit')
    if any(digest(name) != expected for name,expected in primitive_hashes.items()):
        raise ValueError('Independent geometric primitive source changed during audit')
    result.update(schema='doorbench.isaac-release-dense-geometry-audit.v1',
        source_engine='isaac-physx', source_admission=context.admission,
        source_context_sha256=context.sha256, initial_episode_time_s=context.terminal_time_s,
        grasp_profile=profile, motor_contract_identity_bound=True,
        source_physics_archive=str(context.state_archive_path),
        input_sha256={**candidate['input_sha256'],str(candidate_path):initial_sha,
                      **primitive_hashes},
        runtime_route_exported=False, source_sample_playback=0,
        scope='Independent 2001-sample original-model fixed-mechanism geometry/anatomy/reference-speed audit from an actual qualified PhysX source. No torque, load, dynamic balance, moving-leaf envelope, physical release, or traversal qualification; no candidate promotion.')
    return result
