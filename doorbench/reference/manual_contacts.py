"""Conservative, explicit hasp-to-pull contact schedules for a new revision.

This is a proposal module, not an acceptance gate. It supports one unpowered
sliding leaf, an already-unlocked hasp, and a real negative-side pull collider.
Native poses remain prescribed. Small continued hasp effort is an explicit
unverified hold assumption; grip orientation, force closure and causality are
not inferred from source generalized efforts or the recorder's one XYZ target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

EFFORT_THRESHOLD = .001
SUPPORTED_PULLS = frozenset({'pull_d', 'pull_bar_offset'})
SCHEMA = 'doorbench.manual-contact-schedule.v1'


class UnsupportedManualContact(ValueError):
    """No supported contact proposal; never a claim of physical impossibility."""
    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        super().__init__('Manual contact schedule rejected: '+', '.join(self.reasons))


@dataclass(frozen=True)
class ContactRole:
    id: str
    joint_name: str
    joint_role: str
    body_name: str
    site_name: str
    geom_name: str


@dataclass(frozen=True)
class ContactSegment:
    phase: str
    role_id: str | None
    source_start_index: int
    source_end_index: int
    native_frozen: bool


@dataclass(frozen=True)
class ContactPlan:
    roles: tuple[ContactRole, ...]
    segments: tuple[ContactSegment, ...]
    first_active_index: int
    first_contact_index: int
    transfer_index: int
    stop_index: int
    effort_threshold: float
    residual_hasp_effort_max: float
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ManualGuide:
    """Wrapper so the frozen MotionGuide schema need not be edited here."""
    guide: Any
    contact_role_ids: tuple[str | None, ...]
    plan: ContactPlan

    @property
    def roles(self):
        return {role.id: role for role in self.plan.roles}


def _require(ok, reason):
    if not ok:
        raise UnsupportedManualContact((reason,))


def _source_arrays(source):
    try:
        arrays = {key: np.asarray(source[key], dtype=float) for key in ('time', 'qpos', 'tau', 'target', 'base')}
        n = len(arrays['time'])
        _require(n >= 2 and arrays['time'].shape == (n,) and arrays['time'][0] == 0 and
                 np.all(np.diff(arrays['time']) > 0), 'invalid_native_time')
        _require(arrays['qpos'].shape == arrays['tau'].shape == (n, 2) and
                 arrays['target'].shape == arrays['base'].shape == (n, 3), 'unsupported_native_dimensions')
        _require(all(np.isfinite(a).all() for a in arrays.values()), 'nonfinite_native_arrays')
        return arrays
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        if isinstance(exc, UnsupportedManualContact): raise
        raise UnsupportedManualContact(('invalid_native_arrays',)) from exc


def _role(body, joint_role, site_suffix, geom_suffix, geom_type, role_id):
    joint = body['joint']
    sites = [s for s in body.get('sites', []) if s.get('name', '').endswith(site_suffix) and s.get('role') == 'grip']
    _require(len(sites) == 1, role_id+'_site_missing_or_ambiguous')
    site = sites[0]; stem = site['name'][:-len(site_suffix)]
    geoms = [g for g in body.get('geoms', []) if g.get('name') == stem+geom_suffix]
    _require(len(geoms) == 1, role_id+'_geometry_missing_or_ambiguous')
    geom = geoms[0]
    _require(geom.get('collision') is True and geom.get('type') == geom_type and
             geom.get('semantic') in ('operator', 'lock', 'handle'), role_id+'_requires_exact_collision_primitive')
    _require(np.shape(geom.get('size')) == ((2,) if geom_type == 'capsule' else (3,)) and
             np.shape(geom.get('pos')) == (3,) and np.shape(geom.get('quat')) == (4,) and
             np.shape(site.get('pos')) == (3,) and
             all(np.isfinite(geom.get(k, [])).all() for k in ('size', 'pos', 'quat')) and
             np.isfinite(site['pos']).all() and min(geom['size']) > 0 and
             abs(np.linalg.norm(geom['quat'])-1) < 1e-5, role_id+'_invalid_geometry')
    return ContactRole(role_id, joint['name'], joint_role, body['name'], site['name'], geom['name'])


def plan_manual_contacts(spec, model_ir, native_clip, source_arrays):
    """Plan only a verified semantic shape; pure inputs are never modified.

    Authentication belongs to build_manual_guide's index-bound loader. This pure
    function uses meaningful effort >=0.001 in each joint's native unit, requires
    sequential controls, and rejects all other joints/control combinations.
    """
    _require(spec.get('family') == 'sliding_single', 'unsupported_family')
    _require(spec.get('operator', {}).get('model') in SUPPORTED_PULLS, 'unsupported_pull_operator')
    _require(spec.get('lock', {}).get('model') == 'padlock' and spec['lock'].get('engaged') is False,
             'requires_already_unlocked_hasp')
    _require(spec.get('latch', {}).get('model') == 'none' and spec.get('closer', {}).get('model') == 'none',
             'unsupported_latch_or_closer')
    _require(spec.get('kinematics', {}).get('actuator', {}).get('powered') is not True, 'powered_leaf_requires_other_schedule')
    outcome = native_clip.get('outcome', {}); labels = outcome.get('labels', {})
    _require(native_clip.get('scenario') == 'open_and_traverse' and outcome.get('success') is True and
             outcome.get('outcome') == 'success' and outcome.get('error') is None, 'unsupported_or_failed_source_scenario')
    _require(spec.get('benchmark', {}).get('primary_scenario') == native_clip['scenario'], 'source_primary_scenario_mismatch')
    _require(spec.get('id') == native_clip.get('door_id') == outcome.get('door_id') and
             outcome.get('scenario') == native_clip['scenario'], 'source_identity_mismatch')
    _require(outcome.get('damage') is False and outcome.get('env_damage') is False and
             labels.get('door_damaged') is False and labels.get('lock_released') is False,
             'damage_or_unmodeled_lock_release')
    _require(not any(e[0] in ('badge', 'declare_locked') for e in outcome.get('events', [])), 'unsupported_source_api_action')
    scenarios = [s for s in spec.get('benchmark', {}).get('scenarios', []) if s.get('name') == native_clip['scenario']]
    _require(len(scenarios) == 1 and scenarios[0].get('goal') and scenarios[0].get('pass_plane'), 'missing_traversal_scenario')
    bodies = [b for b in model_ir.get('bodies', []) if b.get('joint')]
    primary = [b for b in bodies if b['joint'].get('role') == 'primary' and b['joint'].get('type') == 'slide']
    locks = [b for b in bodies if b['joint'].get('role') == 'lock' and b['joint'].get('type') == 'hinge']
    _require(len(bodies) == 2 and len(primary) == len(locks) == 1, 'unsupported_extra_or_concurrent_controls')
    primary, lock = primary[0], locks[0]
    _require(lock.get('parent') == primary['name'], 'hasp_not_attached_to_primary_leaf')
    roles = (_role(lock, 'lock', '_hasp_grip', '_hasp_strap', 'box', 'hasp'),
             _role(primary, 'primary', '_pull_grip_n', '_pull_col_n', 'capsule', 'pull'))
    source = _source_arrays(source_arrays)
    try:
        native = native_clip['native']; names = native['joint_names']; qa = native['qpos_addresses']; va = native['qvel_addresses']
        _require(len(names) == 2 and len(set(names)) == 2 and set(names) == {r.joint_name for r in roles} and
                 all(type(i) is int for i in qa+va) and sorted(qa) == sorted(va) == [0, 1], 'native_joint_mapping_mismatch')
        mapping = dict(zip(names, va))
        hasp_effort = abs(source['tau'][:, mapping[roles[0].joint_name]])
        pull_effort = abs(source['tau'][:, mapping[roles[1].joint_name]])
    except (KeyError, TypeError) as exc:
        raise UnsupportedManualContact(('native_joint_mapping_mismatch',)) from exc
    hasp = np.flatnonzero(hasp_effort >= EFFORT_THRESHOLD); pull = np.flatnonzero(pull_effort >= EFFORT_THRESHOLD)
    _require(len(hasp) > 0 and len(pull) > 0, 'missing_sequential_hasp_and_pull_effort')
    _require(not np.any((hasp_effort >= EFFORT_THRESHOLD) & (pull_effort >= EFFORT_THRESHOLD)), 'unsupported_concurrent_efforts')
    _require(hasp[-1] < pull[0], 'unsupported_repeated_or_reversed_roles')
    first = max(0, int(hasp[0])-1); transfer = int(pull[0])-1
    plane = scenarios[0]['pass_plane']; center = np.asarray(plane['center'], float); normal = np.asarray(plane['normal'], float)
    _require(center.shape == normal.shape == (3,) and np.isfinite([center, normal]).all() and
             np.allclose(normal, [0, 1, 0], rtol=0, atol=1e-8), 'requires_vertical_forward_passage')
    crossed = np.flatnonzero((source['base']-center)@normal > .05)
    _require(len(crossed) > 0 and (source['base'][0]-center)@normal < 0, 'source_did_not_cross_passage')
    stop = int(crossed[0]); _require(first < transfer < stop, 'unsupported_contact_timeline')
    residual = float(hasp_effort[transfer:stop+1].max())
    _require(residual < EFFORT_THRESHOLD, 'unsupported_continued_hasp_hold')
    segments = tuple(ContactSegment(*s) for s in (
        ('source_wait', None, 0, first, False), ('reach_hasp', 'hasp', first, first, True),
        ('operate_hasp', 'hasp', first, transfer, False), ('withdraw_hasp', 'hasp', transfer, transfer, True),
        ('transfer', None, transfer, transfer, True), ('reach_pull', 'pull', transfer, transfer, True),
        ('operate_pull', 'pull', transfer, stop, False), ('release', 'pull', stop, stop, True),
        ('traverse', None, stop, stop, True)))
    assumptions = (
        'Native generalized effort is not a contact wrench. Meaningful effort is >=0.001 in each joint native unit.',
        'Continued hasp effort below that threshold is not represented by another hand; its mechanical necessity is unverified.',
        'The verified padlock is already unengaged. The small recorded hasp movement is followed without claiming necessary unlocking.',
        'The pull grip is selected from primary-joint body metadata after primary effort starts, even if residual-command priority keeps the recorded XYZ target on the hasp.',
        'Role changes withdraw to rest and re-reach while native qpos is held. No nearest-surface fallback or unrelated contact exemption.',
        'No grasp orientation, articulated fingers, force closure, dynamic balance, motor/sensor causality or original time-budget certificate.')
    return ContactPlan(roles, segments, int(hasp[0]), first, transfer, stop, EFFORT_THRESHOLD, residual, assumptions)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inputs(door_dir, recording_dir):
    """Read through the common new-revision index-bound source loader.

    The loader performs no powered eligibility test; reuse avoids two different
    authentication contracts. This dependency is on the new powered module only.
    """
    from .powered import _inputs as authenticated_inputs
    clip, spec, source, bindings, paths, digests = authenticated_inputs(door_dir, recording_dir)
    model = json.loads(paths['source/model.json'].read_bytes())
    return clip, spec, model, source, bindings, paths, digests


def build_manual_guide(door_dir, recording_dir, *, fps=60, gait_profile='smooth'):
    """Return ManualGuide; caller must solve and independently validate it."""
    _require(type(fps) is int and 10 <= fps <= 60, 'unsupported_fps')
    _require(gait_profile in ('controlled', 'wide_turns', 'compact', 'smooth'), 'unsupported_gait_profile')
    clip, spec, ir, source, bindings, paths, digests = _inputs(door_dir, recording_dir)
    plan = plan_manual_contacts(spec, ir, clip, source)
    # Deferred imports permit a dispatcher inside guidance without a module cycle.
    import mujoco
    from .guidance import GuideBuilder, stationary, smooth_body_guidance, smoothstep
    from .gait import plan_walk
    from .planning import SceneNavigator
    nav = SceneNavigator(door_dir); model, data = nav.model, nav.data; roles = {r.id: r for r in plan.roles}
    _require(model.nq == 2 and model.nv == 2, 'compiled_joint_dimensions_mismatch')
    native = clip['native']
    for name, qa, va in zip(native['joint_names'], native['qpos_addresses'], native['qvel_addresses']):
        joint = model.joint(name)
        _require(int(joint.qposadr[0]) == qa and int(joint.dofadr[0]) == va, 'compiled_joint_mapping_mismatch')
    for role in plan.roles:
        bid = int(model.body(role.body_name).id)
        _require(int(model.site(role.site_name).bodyid[0]) == int(model.geom(role.geom_name).bodyid[0]) ==
                 int(model.joint(role.joint_name).bodyid[0]) == bid, 'compiled_contact_body_mismatch')
        geom = model.geom(role.geom_name)
        _require(geom.contype[0] or geom.conaffinity[0], 'compiled_contact_is_not_collidable')

    def anchors(times, role_id):
        sid = int(model.site(roles[role_id].site_name).id); values = []
        for time in np.atleast_1d(times):
            data.qpos[:] = [np.interp(time, source['time'], source['qpos'][:, j]) for j in range(model.nq)]
            mujoco.mj_kinematics(model, data); values.append(data.site_xpos[sid].copy())
        return np.asarray(values)

    # The first meaningful effort must actually identify this source hasp site.
    onset = plan.first_active_index
    _require(np.linalg.norm(anchors([source['time'][onset]], 'hasp')[0]-source['target'][onset]) < 1e-5,
             'first_active_target_does_not_match_hasp_site')
    options = dict(fps=fps, step_length=.42, step_duration=.65, stance_width=.21, blend_turns=True,
                   max_step_yaw_deg=45. if gait_profile == 'wide_turns' else 20.,
                   pelvis_acceleration_m_s2=None if gait_profile in ('compact', 'smooth') else 1.5)
    builder = GuideBuilder(source, fps); role_parts = []

    def add(gait, native_time, phase, *, role=None, hand=None, weight=None, height=None):
        builder.add(gait, native_time, phase, pelvis_height=height)
        part = builder.parts[-1]; n = len(part['time']); role_parts.append([role]*n)
        if role is not None and hand is not None:
            side = int(hand == 'right_hand'); w = np.ones(n) if weight is None else np.asarray(weight)
            target = anchors(part['native_time'], role)
            part['hands'][:, side] = part['hands'][:, side]*(1-w[:, None])+target*w[:, None]
            part['hand_weight'][:, side] = w; part['hand_contact'][:, side] = w >= 1-1e-8

    first, transfer, stop = plan.first_contact_index, plan.transfer_index, plan.stop_index
    start = source['base'][0, :2].copy(); initial = plan_walk(start, 0., [start], **options)
    wait = stationary(initial, max(.5, float(source['time'][first])), fps)
    add(wait, np.linspace(source['time'][0], source['time'][first], len(wait['time'])), 'source_wait')
    nav.update(source['qpos'][first]); stance = nav.stance(anchors([source['time'][first]], 'hasp')[0], start)
    route = nav.route(start, stance.xy); approach = plan_walk(start, 0., route[1:], **options)
    add(approach, source['time'][first], 'approach')
    turn = plan_walk(route[-1], float(approach['pelvis_yaw'][-1]), [route[-1]], **options, waypoint_yaws=[stance.yaw])
    add(turn, source['time'][first], 'face_hardware')
    reach = stationary(turn, .9, fps); w = smoothstep(reach['time']/reach['time'][-1])
    add(reach, source['time'][first], 'reach_hasp', role='hasp', hand=stance.hand, weight=w,
        height=.94*(1-w)+stance.pelvis_height*w)
    current = stance.xy.copy(); yaw = stance.yaw; hand = stance.hand; height = stance.pelvis_height; last = reach

    def operate(a, b, role):
        nonlocal current, yaw, hand, height, last
        q = source['qpos']; targets = anchors(source['time'][a:b+1], role)
        scale = np.maximum(np.ptp(q, axis=0), .05); keys = [a]
        for i in range(a+1, b+1):
            j = keys[-1]
            if np.max(np.abs(q[i]-q[j])/scale) > .14 or np.linalg.norm(targets[i-a]-targets[j-a]) > .10:
                keys.append(i)
        if keys[-1] != b: keys.append(b)
        for x, y in zip(keys, keys[1:]):
            nav.update(q[y]); nxt = nav.stance(targets[y-a], current, preferred_hand=hand, previous_yaw=yaw, previous_height=height)
            if np.linalg.norm(nxt.xy-current) < .09:
                nxt = type(nxt)(current.copy(), yaw, nxt.pelvis_height, hand, nxt.clearance)
            travel = plan_walk(current, yaw, [nxt.xy], **options, waypoint_yaws=[nxt.yaw])
            joint = np.max(np.abs(np.diff(q[x:y+1], axis=0)), axis=1)/.45
            handpath = np.linalg.norm(np.diff(targets[x-a:y-a+1], axis=0), axis=1)/.18
            arc = np.r_[0., np.cumsum(np.maximum(np.maximum(joint, handpath), 1e-9))]
            duration = max(.35, float(arc[-1])*1.875)
            if travel['time'][-1] < duration:
                if np.linalg.norm(nxt.xy-current) < 1e-8 and abs(nxt.yaw-yaw) < 1e-8:
                    travel = stationary(last, duration, fps)
                else:
                    travel['time'] *= math.ceil(duration/max(travel['time'][-1], 1e-8))
            blend = smoothstep(travel['time']/max(travel['time'][-1], 1e-8))
            nt = np.interp(blend*arc[-1], arc, source['time'][x:y+1])
            add(travel, nt, 'operate_'+role, role=role, hand=hand, height=height*(1-blend)+nxt.pelvis_height*blend)
            current = nxt.xy.copy(); yaw = nxt.yaw; last = travel; height = nxt.pelvis_height
        return len(keys)

    nkeys = operate(first, transfer, 'hasp')
    release = stationary(last, .9, fps); w = 1-smoothstep(release['time']/release['time'][-1])
    add(release, source['time'][transfer], 'withdraw_hasp', role='hasp', hand=hand, weight=w, height=.94*(1-w)+height*w)
    nav.update(source['qpos'][transfer])
    nxt = nav.stance(anchors([source['time'][transfer]], 'pull')[0], current, preferred_hand=hand, previous_yaw=yaw, previous_height=.94)
    route = nav.route(current, nxt.xy); move = plan_walk(current, yaw, route[1:], **options)
    add(move, source['time'][transfer], 'transfer_reposition')
    turn = plan_walk(nxt.xy, float(move['pelvis_yaw'][-1]), [nxt.xy], **options, waypoint_yaws=[nxt.yaw])
    add(turn, source['time'][transfer], 'transfer_face_pull')
    current = nxt.xy.copy(); yaw = nxt.yaw; hand = nxt.hand; height = nxt.pelvis_height
    reach = stationary(turn, 1.1, fps); w = smoothstep(reach['time']/reach['time'][-1])
    add(reach, source['time'][transfer], 'reach_pull', role='pull', hand=hand, weight=w, height=.94*(1-w)+height*w)
    last = reach; nkeys += operate(transfer, stop, 'pull')
    release = stationary(last, .9, fps); w = 1-smoothstep(release['time']/release['time'][-1])
    add(release, source['time'][stop], 'release', role='pull', hand=hand, weight=w, height=.94*(1-w)+height*w)
    scenario = next(s for s in spec['benchmark']['scenarios'] if s['name'] == clip['scenario'])
    nav.update(source['qpos'][stop]); route = nav.passage_route(current, np.asarray(scenario['goal']['center'])[:2], scenario['pass_plane'])
    walk = plan_walk(current, yaw, route[1:], **options); add(walk, source['time'][stop], 'traverse')
    add(stationary(walk, .5, fps), source['time'][stop], 'settle')
    role_ids = tuple(role for i, rows in enumerate(role_parts) for role in (rows[1:] if i else rows))
    guide = builder.finish({'door_id': spec['id'], 'scenario': clip['scenario'], 'source_outcome': clip['outcome'],
        'source_sha256': clip['source_sha256'], 'traversal': 'proposed', 'traversal_reason': None, 'hand': hand,
        'native_keyframes': nkeys, 'gait_profile': gait_profile,
        'scope': 'Explicit contact schedule proposal; independent kinematic and task validation still required.',
        'manual_contact_schedule': {'schema': SCHEMA, **asdict(plan), 'input_bindings': bindings,
                                    'contact_role_ids': role_ids}})
    if gait_profile == 'smooth': guide = smooth_body_guidance(guide, fps)
    _require(len(role_ids) == len(guide.time) and guide.hand_contact.sum(axis=1).max() <= 1, 'invalid_authored_contact_schedule')
    expected = np.stack([np.interp(guide.native_time, source['time'], source['qpos'][:, j]) for j in range(model.nq)], axis=1)
    _require(np.array_equal(guide.native_qpos, expected) and np.all(np.diff(guide.native_time) >= 0), 'native_source_path_changed')
    _require(all(_sha(path) == digests[name] for name, path in paths.items()), 'source_changed_during_contact_planning')
    return ManualGuide(guide, role_ids, plan)
