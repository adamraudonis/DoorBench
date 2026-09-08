"""Static left-palm replanning at an actual attained H1/Shadow stance.

This privileged planner owns an unstepped geometry calculator. It cannot move
the active robot. A passed plan is a sampled geometric candidate, not a force,
contact, balance, or continuous opening qualification.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .robot_design_identity import robot_design_identity, verify_robot_design_identity
from .robot_identity import robot_file_identity

JOINT_NAMES = ['torso', 'left_shoulder_pitch', 'left_shoulder_roll',
               'left_shoulder_yaw', 'left_elbow', 'left_wrist_yaw', 'lh_WRJ2', 'lh_WRJ1']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_targets(config):
    if (config.get('schema') != 'doorbench.left-palm-targets.v1' or
            config.get('passed') is not True or config.get('fixed_waist') is not True or
            config.get('joint_names') != JOINT_NAMES):
        raise ValueError('A passed, fixed-waist left-palm target contract is required')
    rows = config.get('targets', [])
    if len(rows) < 2 or any(row.get('phase') != 'left_reach' for row in rows):
        raise ValueError('This planner only adapts one complete left_reach path')
    values = {}
    for key, width in [('position', 3), ('normal', 3), ('nominal', 8)]:
        a = np.asarray([row[key] for row in rows], float)
        if a.shape != (len(rows), width) or not np.isfinite(a).all():
            raise ValueError('Invalid finite target ' + key)
        values[key] = a
    if not np.allclose(np.linalg.norm(values['normal'], axis=1), 1., atol=1e-6, rtol=0.):
        raise ValueError('Left-palm normals must be unit vectors')
    if not np.isfinite([r['leaf_rad'] for r in rows]).all():
        raise ValueError('Invalid target leaf angle')
    return values


class LandedLeftScene:
    """Canonical DoorBench attachment frames, complete named measured state."""
    def __init__(self, robot_xml, door_xml):
        self.robot_xml, self.door_xml = Path(robot_xml), Path(door_xml)
        spec = mujoco.MjSpec.from_file(str(self.door_xml))
        spec.memory = 128 * 1024 * 1024
        spec.worldbody.add_site(name='robot_attach', pos=[0., -1.5, 0.])
        frame = spec.worldbody.add_frame(pos=[0., -1.5, 0.])
        spec.attach(mujoco.MjSpec.from_file(str(self.robot_xml)), prefix='robot/', frame=frame)
        self.m = m = spec.compile()
        self.d = mujoco.MjData(m)
        self.root = int(m.jnt_qposadr[m.joint('robot/free_base').id])
        scalar = [j for j in range(m.njnt) if m.jnt_type[j] in
                  (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
        if m.nq != len(scalar) + 7:
            raise ValueError('Expected one free robot root and otherwise scalar scene joints')
        self.robot_names = [m.joint(j).name.removeprefix('robot/') for j in scalar
                            if m.joint(j).name.startswith('robot/')]
        self.door_names = [m.joint(j).name for j in scalar
                           if not m.joint(j).name.startswith('robot/')]
        self.arm = np.array([m.joint('robot/' + n).id for n in JOINT_NAMES])
        self.qa = m.jnt_qposadr[self.arm]
        self.palm, self.leaf = m.site('robot/lh_palm_touch').id, m.body('leaf').id

    def freeze(self, state):
        """Copy exact numeric actual state; never install a nominal root/legs."""
        root = np.asarray(state['root'], float)
        if root.shape != (7,) or not np.isfinite(root).all() or not np.isclose(
                np.linalg.norm(root[3:]), 1., atol=1e-6, rtol=0.):
            raise ValueError('Expected finite root xyz+wxyz with a unit quaternion')
        if not np.isfinite(state['pose_time_s']) or state['pose_time_s'] < 0:
            raise ValueError('An actual nonnegative pose timestamp is required')
        q = self.m.qpos0.copy()
        q[self.root:self.root+7] = root
        for key, names, prefix in [('joints', self.robot_names, 'robot/'),
                                   ('door_positions', self.door_names, '')]:
            if set(state[key]) != set(names):
                raise ValueError('Every actual named ' + key + ' coordinate is required')
            for name in names:
                value = float(state[key][name])
                if not np.isfinite(value):
                    raise ValueError('Nonfinite ' + key)
                q[self.m.jnt_qposadr[self.m.joint(prefix+name).id]] = value
        return q

    def state_from_qpos(self, qpos, *, pose_time_s):
        q = np.asarray(qpos, float)
        if q.shape != (self.m.nq,) or not np.isfinite(q).all():
            raise ValueError('Recorded qpos does not match the complete compiled scene')
        result = dict(pose_time_s=float(pose_time_s), root=q[self.root:self.root+7].tolist())
        for key, names, prefix in [('joints', self.robot_names, 'robot/'),
                                   ('door_positions', self.door_names, '')]:
            result[key] = {n: float(q[self.m.jnt_qposadr[self.m.joint(prefix+n).id]]) for n in names}
        if not np.array_equal(self.freeze(result), q):
            raise ValueError('Named state does not exactly cover recorded qpos')
        return result

    def palm_local(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        rotation = self.d.xmat[self.leaf].reshape(3, 3)
        return (rotation.T @ (self.d.site_xpos[self.palm] - self.d.xpos[self.leaf]),
                rotation.T @ self.d.site_xmat[self.palm].reshape(3, 3)[:, 2])


def fit_landed_left_targets(scene, original, state, *, tangent_limit_m=.15):
    """Preserve panel-local Y and palm normal; allow a bounded X/Z contact shift.

    The torso, legs, opposite arm, fingers, door and free root stay exactly at
    their measured coordinates. Only seven left arm/wrist joints are optimized.
    Authored bounds have a 25 mrad planning margin; no model limits are changed.
    """
    if not np.isfinite(tangent_limit_m) or not 0 < tangent_limit_m <= .15:
        raise ValueError('Contact-plane tangential freedom must be in (0, .15] metres')
    source = read_targets(original)
    frozen = scene.freeze(state)
    m, qa = scene.m, scene.qa
    lower = m.jnt_range[scene.arm[1:], 0] + .025
    upper = m.jnt_range[scene.arm[1:], 1] - .025
    if not np.all(lower < upper):
        raise ValueError('Authored arm bounds cannot support the planning margin')
    old_nominal = np.clip(source['nominal'][-1, 1:], lower, upper)
    old_pos, old_normal = source['position'][-1], source['normal'][-1]
    query = frozen.copy()

    def residual(q):
        query[qa[1:]] = q
        pos, normal = scene.palm_local(query)
        tangent_excess = np.maximum(np.abs(pos[[0, 2]] - old_pos[[0, 2]]) - tangent_limit_m, 0.)
        return np.r_[100 * (pos[1] - old_pos[1]), 10 * (normal-old_normal),
                     .15 * (q-old_nominal), 100*tangent_excess]

    fit = least_squares(residual, old_nominal, bounds=(lower, upper), max_nfev=500)
    query[qa[1:]] = fit.x
    goal_pos, goal_normal = scene.palm_local(query)
    goal = np.r_[frozen[qa[0]], fit.x]
    rows = []
    for i, nominal in enumerate(source['nominal']):
        u = i / (len(source['nominal']) - 1)
        blend = u*u*(3.-2.*u)
        target = (nominal + (1-blend)*(frozen[qa]-source['nominal'][0]) +
                  blend*(goal-source['nominal'][-1]))
        target[0] = frozen[qa[0]]
        query[:] = frozen
        query[qa] = target
        pos, normal = scene.palm_local(query)
        rows.append(dict(phase='left_reach', leaf_rad=state['door_positions']['leaf_hinge'],
                         position=pos.tolist(), normal=normal.tolist(), nominal=target.tolist()))
    normal_angle = float(np.rad2deg(np.arccos(np.clip(goal_normal@old_normal, -1., 1.))))
    checks = dict(solver_converged=bool(fit.success), panel_plane_error=bool(abs(goal_pos[1]-old_pos[1]) <= .001),
                  palm_normal_error=normal_angle <= .5,
                  bounded_panel_tangent=bool(np.all(abs(goal_pos[[0, 2]]-old_pos[[0, 2]]) <= tangent_limit_m)),
                  interior_joint_solution=bool(np.all(fit.x >= lower) and np.all(fit.x <= upper)))
    diagnostics = dict(passed=all(checks.values()), checks=checks, solver_nfev=fit.nfev,
        objective_residual=float(np.linalg.norm(fit.fun)),
        geometric_residual=float(np.linalg.norm(fit.fun[:4])),
        panel_plane_error_m=float(abs(goal_pos[1]-old_pos[1])), normal_error_deg=normal_angle,
        tangent_shift_m=(goal_pos[[0, 2]]-old_pos[[0, 2]]).tolist(),
        minimum_authored_arm_margin_rad=float(np.min(np.minimum(fit.x-m.jnt_range[scene.arm[1:], 0],
                                                               m.jnt_range[scene.arm[1:], 1]-fit.x))),
        goal_position=goal_pos.tolist(), goal_normal=goal_normal.tolist(), goal_nominal=goal.tolist(),
        declared_tangent_limit_m=float(tangent_limit_m), frozen_state_sha256=digest(state),
        physics_steps=0, scope='Static geometric endpoint fit; force and contact feasibility unqualified')
    return rows, diagnostics


def make_landed_left_plan(robot_xml, door_xml, original, state, *, subdivisions=10):
    """Fit and independently audit the complete actual-state candidate."""
    from .landed_left_audit import audit_landed_left_path
    scene = LandedLeftScene(robot_xml, door_xml)
    if 'source_design_identity' not in original:
        raise ValueError('Source targets require an authored robot design identity')
    verify_robot_design_identity(robot_xml, original['source_design_identity'])
    rows, fit = fit_landed_left_targets(scene, original, state)
    config = copy.deepcopy(original)
    config.update(passed=False, targets=rows,
        robot_sha256=hashlib.sha256(Path(robot_xml).read_bytes()).hexdigest(),
        compiled_robot_identity=robot_file_identity(robot_xml),
        source_design_identity=robot_design_identity(robot_xml),
        scope='Actual-state sampled FK/IK candidate only; no physical opening qualification',
        source=dict(parent_target_content_sha256=digest(original), frozen_state=copy.deepcopy(state),
                    frozen_state_sha256=digest(state), door_xml_sha256=hashlib.sha256(Path(door_xml).read_bytes()).hexdigest(),
                    door_source_design_identity=robot_design_identity(door_xml)))
    config.pop('runtime_rescreen_trajectory_sha256', None)
    # The audit intentionally does not trust a planner's passed flag.
    audit = audit_landed_left_path(scene, config, state, subdivisions=subdivisions)
    config['passed'] = fit['passed'] and audit['passed']
    return config, dict(passed=config['passed'], fit=fit, audit=audit)
