"""Detached geometry at explicit panel reference and measured mechanism states.

This fills the gap between a single-lag static screen and a future per-update
guard. It never reads a live simulator, advances time, writes plant state, or
grants a controller authority. A passed point is not a proved operating domain.
"""
import copy
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .isaac_panel_geometry_audit import validate_panel_candidate
from .isaac_panel_planning import IsaacPanelPlanningContext, JOINT_NAMES, clearance_candidate_pairs
from .palm_panel_geometry import original_palm_vertices, flatten_palm_goal, twist_palm_goal
from .qualified_isaac_grasp import digest


SCHEMA = 'doorbench.isaac-panel-geometry-point.v1'
STATIC_LIMITS = dict(left_position_m=.0001, left_rotation_rad=.001,
    right_position_m=.0001, right_rotation_rad=.001, foot_position_m=.0001,
    foot_rotation_rad=.001, joint_violation_increase_rad=.000001,
    torso_tilt_deg=4., root_translation_m=.03, root_rotation_rad=.05,
    com_xy_displacement_m=.015)
MECHANISM_JOINTS = dict(leaf='leaf_hinge', operator='leaf_handle_hinge',
    latch='leaf_latch_bolt_slide')


def _rotation_error(target, actual):
    return float(Rotation.from_matrix(target @ actual.T).magnitude())


class IsaacPanelGeometryProbe:
    """Re-admitted source/candidate plus an isolated, unstepped FK calculator.

    ``coordinates`` uses the original plan convention: world root displacement,
    world-relative root rotation vector, then the 25 named scalar joints.
    The reference aperture defines the requested LH pose. The three explicit
    measured mechanism coordinates define collision geometry. This separation
    retains lead/lag instead of silently replacing the actual leaf by its goal.
    No target correction, extrapolation, clipping, or rate projection is done.
    """
    def __init__(self, context, candidate):
        if not isinstance(context, IsaacPanelPlanningContext):
            raise ValueError('Freshly admitted actual released-source context required')
        self.context = context
        self.scene = context.scene()
        validate_panel_candidate(candidate, context, self.scene)
        self.report = copy.deepcopy(candidate)
        self.hashes = dict(candidate['input_sha256'])
        self.hashes[str(Path(__file__).resolve())] = digest(__file__)
        self.m, self.d = self.scene.m, self.scene.d
        m, d = self.m, self.d
        self.initial = context.qpos.copy()
        self.rq = int(m.joint('robot/free_base').qposadr[0])
        self.qa = np.array([m.joint('robot/'+n).qposadr[0] for n in JOINT_NAMES])
        self.robot_joints = {m.joint(j).name.removeprefix('robot/'): j for j in range(m.njnt)
            if m.joint(j).name.startswith('robot/') and m.jnt_type[j] in (2, 3)}
        self.mechanism = {role: m.joint(name).id for role, name in MECHANISM_JOINTS.items()}
        self.r0 = Rotation.from_quat(self.initial[self.rq+3:self.rq+7][[1, 2, 3, 0]])
        self.start_angle = float(self.initial[m.jnt_qposadr[self.mechanism['leaf']]])
        self.final_angle = float(candidate['target_aperture_rad'])
        self.feet = [m.body('robot/'+side+'_ankle_link').id for side in ('left', 'right')]
        self.feet_p = d.xpos[self.feet].copy()
        self.feet_r = d.xmat[self.feet].reshape(2, 3, 3).copy()
        self.rh, self.lh = [m.site('robot/'+side+'_palm_touch').id for side in ('rh', 'lh')]
        self.rh_p, self.rh_r = d.site_xpos[self.rh].copy(), d.site_xmat[self.rh].reshape(3, 3).copy()
        self.leaf = m.body('leaf').id
        lr, lp = d.xmat[self.leaf].reshape(3, 3), d.xpos[self.leaf]
        self.lh_p = lr.T @ (d.site_xpos[self.lh]-lp)
        self.lh_r = lr.T @ d.site_xmat[self.lh].reshape(3, 3)
        self.palm_vertices = (original_palm_vertices(m, d, self.lh)
            if candidate['configuration']['flatten_palm'] else None)
        self.root_body = int(m.jnt_bodyid[m.joint('robot/free_base').id])
        self.com = d.subtree_com[self.root_body].copy()
        self.torso, self.floor = m.body('robot/torso_link').id, m.geom('floor').id
        self.scalar = np.array([j for j in range(m.njnt)
            if m.jnt_limited[j] and m.jnt_type[j] in (2, 3)])
        self.sq = m.jnt_qposadr[self.scalar]
        self.original_violation = np.maximum(0., np.maximum(
            m.jnt_range[self.scalar, 0]-self.initial[self.sq],
            self.initial[self.sq]-m.jnt_range[self.scalar, 1]))
        active = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.hand = np.array([g for g in active if m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')])
        self.environment = np.array([g for g in active if not m.body(m.geom_bodyid[g]).name.startswith('robot/')])
        self.elbows = np.array([g for g in active if m.body(m.geom_bodyid[g]).name
            in ('robot/left_elbow_link', 'robot/right_elbow_link')])
        if not all(len(a) for a in (self.hand, self.environment, self.elbows)):
            raise ValueError('Original hand, elbow and environment colliders required')
        self.radii = np.zeros(m.ngeom)
        for g in set(np.r_[self.hand, self.environment, self.elbows].tolist()):
            kind, size = int(m.geom_type[g]), m.geom_size[g]
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = m.geom_dataid[g]; begin = m.mesh_vertadr[mesh]; count = m.mesh_vertnum[mesh]
                radius = np.linalg.norm(m.mesh_vert[begin:begin+count], axis=1).max()
            elif kind == int(mujoco.mjtGeom.mjGEOM_BOX): radius = np.linalg.norm(size)
            elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE): radius = size[0]
            elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE): radius = size[0]+size[1]
            elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER): radius = np.hypot(size[0], size[1])
            elif kind == int(mujoco.mjtGeom.mjGEOM_PLANE): radius = np.inf
            else: raise ValueError('Unsupported original collision shape')
            self.radii[g] = radius
        self.verify_inputs()

    def verify_inputs(self):
        self.context.verify_inputs()
        if any(digest(p) != expected for p, expected in self.hashes.items()):
            raise ValueError('Panel probe input changed')

    def initial_coordinates(self):
        return np.r_[np.zeros(6), self.initial[self.qa]]

    def _clearance(self, shapes, cap):
        m, d = self.m, self.d
        planes = [(column, d.geom_xmat[g].reshape(3, 3)[:, 2])
            for column, g in enumerate(self.environment)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
        candidates = clearance_candidate_pairs(d.geom_xpos[shapes], self.radii[shapes],
            d.geom_xpos[self.environment], self.radii[self.environment], planes, cap)
        nearest, pair = cap, None
        for i, j in candidates:
            g, h = int(shapes[i]), int(self.environment[j])
            distance = float(mujoco.mj_geomDistance(m, d, g, h, cap, None))
            if not np.isfinite(distance): raise ValueError('Nonfinite original geometry distance')
            if distance < nearest: nearest, pair = distance, [m.geom(g).name, m.geom(h).name]
        return nearest, pair, len(candidates)

    def evaluate(self, coordinates, *, reference_aperture_rad, measured_angles,
                 robot_joint_targets=None):
        x = np.array(coordinates, dtype=float, copy=True)
        if (x.shape != (6+len(JOINT_NAMES),) or not np.isfinite(x).all()
                or type(reference_aperture_rad) not in (int, float)
                or not np.isfinite(reference_aperture_rad)
                or not self.start_angle <= reference_aperture_rad <= self.final_angle):
            raise ValueError('Finite complete coordinates and in-range reference aperture required')
        if type(measured_angles) is not dict or set(measured_angles) != set(MECHANISM_JOINTS):
            raise ValueError('Explicit complete measured leaf/operator/latch coordinates required')
        for name, value in measured_angles.items():
            joint = self.mechanism[name]
            source_value = self.initial[self.m.jnt_qposadr[joint]]
            original_low, original_high = map(float, self.m.jnt_range[joint])
            baseline = max(original_low-float(source_value), float(source_value)-original_high, 0.)
            low, high = original_low-baseline-.000001, original_high+baseline+.000001
            if (type(value) not in (int, float) or not np.isfinite(value)
                    or not low <= value <= high):
                raise ValueError('Finite mechanism coordinate within the original source-relative joint envelope required: '+name)
        if robot_joint_targets is not None:
            if (type(robot_joint_targets) is not dict or set(robot_joint_targets) != set(self.robot_joints)
                    or any(type(v) not in (int, float) or not np.isfinite(v) for v in robot_joint_targets.values())
                    or not np.array_equal([robot_joint_targets[n] for n in JOINT_NAMES], x[6:])):
                raise ValueError('Complete finite robot joint targets must agree with the explicit panel coordinates')
        m, d = self.m, self.d
        d.qpos[:] = self.initial
        d.qpos[self.rq:self.rq+3] += x[:3]
        d.qpos[self.rq+3:self.rq+7] = (Rotation.from_rotvec(x[3:6])*self.r0).as_quat()[[3, 0, 1, 2]]
        d.qpos[self.qa] = x[6:]
        if robot_joint_targets is not None:
            for name, value in robot_joint_targets.items(): d.qpos[m.jnt_qposadr[self.robot_joints[name]]] = value
        for name, value in measured_angles.items(): d.qpos[m.jnt_qposadr[self.mechanism[name]]] = value
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_collision(m, d)
        if not all(np.isfinite(a).all() for a in (d.qpos, d.xpos, d.xmat, d.site_xpos,
                d.site_xmat, d.geom_xpos, d.geom_xmat, d.subtree_com)):
            raise ValueError('Nonfinite private geometry')
        lag = reference_aperture_rad-measured_angles['leaf']
        # Rotate only the intended LH goal frame. Actual mechanism geometry in
        # d remains at the supplied measurement for all collision queries.
        joint = self.mechanism['leaf']; anchor = d.xanchor[joint]
        lead_rotation = Rotation.from_rotvec(d.xaxis[joint]*lag).as_matrix()
        leaf_p = anchor+lead_rotation@(d.xpos[self.leaf]-anchor)
        leaf_r = lead_rotation@d.xmat[self.leaf].reshape(3, 3)
        u = float(np.clip((reference_aperture_rad-self.start_angle)/.35, 0., 1.))
        blend = u**3*(10.+u*(-15.+6.*u))
        config = self.report['configuration']; local = self.lh_p.copy(); local_r = self.lh_r.copy()
        local[0] += config['radius_shift_m']*blend
        local[2] -= config['height_drop_m']*blend
        if self.palm_vertices is not None:
            f = float(np.clip((reference_aperture_rad-self.start_angle)/config['flatten_over_rad'], 0., 1.))
            local, local_r, _ = flatten_palm_goal(local, local_r, self.palm_vertices, f**3*(10.+f*(-15.+6.*f)))
        if config['palm_twist_rad']: local_r = twist_palm_goal(local_r, config['palm_twist_rad']*blend)
        target_p, target_r = leaf_p+leaf_r@local, leaf_r@local_r
        increase = np.maximum(m.jnt_range[self.scalar, 0]-d.qpos[self.sq],
            d.qpos[self.sq]-m.jnt_range[self.scalar, 1])-self.original_violation
        values = dict(left_position_m=float(np.linalg.norm(d.site_xpos[self.lh]-target_p)),
            left_rotation_rad=_rotation_error(target_r, d.site_xmat[self.lh].reshape(3, 3)),
            right_position_m=float(np.linalg.norm(d.site_xpos[self.rh]-self.rh_p)),
            right_rotation_rad=_rotation_error(self.rh_r, d.site_xmat[self.rh].reshape(3, 3)),
            foot_position_m=max(float(np.linalg.norm(d.xpos[b]-self.feet_p[k])) for k,b in enumerate(self.feet)),
            foot_rotation_rad=max(_rotation_error(self.feet_r[k],d.xmat[b].reshape(3,3)) for k,b in enumerate(self.feet)),
            joint_violation_increase_rad=max(0., float(increase.max())),
            torso_tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[self.torso].reshape(3,3)[2,2],-1.,1.)))),
            root_translation_m=float(np.linalg.norm(x[:3])), root_rotation_rad=float(np.linalg.norm(x[3:6])),
            com_xy_displacement_m=float(np.linalg.norm(d.subtree_com[self.root_body,:2]-self.com[:2])))
        collisions = []
        for contact in d.contact[:d.ncon]:
            if not np.isfinite(contact.dist): raise ValueError('Nonfinite collision gap')
            bodies = [m.body(m.geom_bodyid[g]).name for g in contact.geom]
            allowed = self.floor in contact.geom and any(b.endswith('_ankle_link') for b in bodies)
            if not allowed and any(b.startswith('robot/') for b in bodies) and contact.dist < -.003:
                collisions.append(dict(bodies=bodies, depth_m=-float(contact.dist)))
        hand, hand_pair, hand_count = self._clearance(self.hand, .040001)
        elbow, elbow_pair, elbow_count = self._clearance(self.elbows, .003001)
        violations = {name:value for name,value in values.items() if value > STATIC_LIMITS[name]}
        if hand < .04: violations['right_scene_clearance_m'] = hand
        if elbow < .003: violations['elbow_scene_clearance_m'] = elbow
        return dict(schema=SCHEMA, passed=not violations and not collisions,
            source_context_sha256=self.context.sha256, source_time_s=self.report['source_time_s'],
            coordinates=x.tolist(), reference_aperture_rad=float(reference_aperture_rad),
            robot_joint_targets={n:float(d.qpos[m.jnt_qposadr[j]]) for n,j in self.robot_joints.items()},
            robot_target_scope='Complete explicit target bundle' if robot_joint_targets is not None else 'Panel subset plus measured source values for all remaining scalar joints',
            measured_angles=dict(measured_angles), reference_minus_measured_leaf_rad=float(lag),
            limits=dict(STATIC_LIMITS), values=values, violations=violations, collisions=collisions,
            right_scene_clearance_capped_m=hand, right_scene_closest_pair=hand_pair,
            elbow_scene_clearance_capped_m=elbow, elbow_scene_closest_pair=elbow_pair,
            exact_distance_pairs=hand_count+elbow_count,
            requested_left_palm_position=target_p.tolist(), requested_left_palm_rotation=target_r.tolist(),
            mechanism_qpos={name:float(d.qpos[m.jnt_qposadr[j]]) for name,j in self.mechanism.items()},
            authorized_stages=0, physics_steps=0, active_state_writes=0,
            physical_admission=False, runtime_route_exported=False,
            scope='One unstepped geometry point with explicit mechanism measurements; no domain, motion-rate, support, tracking, handoff or physical qualification.')
