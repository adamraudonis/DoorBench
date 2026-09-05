"""Guarded non-manual wait/traverse proposals for recorded powered doors.

This removes invented manual reach/contact from eligible source episodes. It
does not model an actor triggering a sensor/button or supply a power, force,
balance, safety-sensor or original task-clock certificate. Native qpos always
comes from the untouched recording at the explicit retained source clock map.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

POWERED_FAMILIES = frozenset({'automatic_sliding', 'automatic_swing', 'elevator'})
SOURCE_FILES = frozenset({'model.json', 'spec.json', 'door.xml'})


class PoweredIneligible(ValueError):
    """Explicit schedule exclusion, not a physical infeasibility verdict."""
    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        super().__init__('Powered non-manual schedule rejected: '+', '.join(self.reasons))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _file(root, relative):
    relative = Path(relative); root = Path(root).resolve()
    _require(not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() not in ('', '.'), 'Unsafe powered input path')
    path = root/relative
    _require(path.is_file() and not any(p.is_symlink() for p in [path, *path.parents] if p != root and p.is_relative_to(root))
             and path.resolve().is_relative_to(root), 'Powered input is missing, a symlink, or outside its source tree')
    return path


def _scenario(spec, name):
    rows = [item for item in spec.get('benchmark', {}).get('scenarios', []) if item.get('name') == name]
    return rows[0] if len(rows) == 1 else None


def _crossing(source, scenario):
    plane = scenario['pass_plane']; center = np.asarray(plane['center'], float); normal = np.asarray(plane['normal'], float)
    _require(center.shape == (3,) and normal.shape == (3,) and np.isfinite([center, normal]).all() and
             np.allclose(normal, [0, 1, 0], rtol=0, atol=1e-8), 'Powered schedule requires a +Y vertical passage')
    distance = (source['base']-center) @ normal
    _require(distance[0] < 0, 'Powered source must start on the approach side')
    crossed = np.flatnonzero(distance > .05)
    _require(len(crossed) > 0, 'Powered source base did not cross its declared passage')
    return int(crossed[0])


def powered_eligibility(clip, spec, source):
    """Return stable exclusion codes; absent/ambiguous evidence fails closed.

    This pure guard does not authenticate hashes; make_powered_guide also binds
    the inputs to their recording index and current native geometry first.
    Exactly zero recorded tau is required, including tiny residual commands.
    """
    reasons = []
    if spec.get('family') not in POWERED_FAMILIES:
        reasons.append('not_powered_family')
    if spec.get('kinematics', {}).get('actuator', {}).get('powered') is not True:
        reasons.append('actuator_not_explicitly_powered')
    outcome = clip.get('outcome', {}); labels = outcome.get('labels', {})
    if outcome.get('success') is not True or outcome.get('outcome') != 'success' or outcome.get('error') is not None:
        reasons.append('source_failed')
    for name, code in [('touched_door', 'source_manual_door_touch'), ('touched_operator', 'source_manual_operator_touch'),
                       ('operator_actuated', 'source_operator_actuated'), ('lock_released', 'source_lock_released')]:
        if labels.get(name) is not False:
            reasons.append(code if labels.get(name) is True else 'missing_'+name+'_evidence')
    if outcome.get('damage') is not False or outcome.get('env_damage') is not False or labels.get('door_damaged') is not False:
        reasons.append('source_damage_or_missing_damage_evidence')
    if labels.get('robot_passed_through') is not True or labels.get('door_open_clear') is not True:
        reasons.append('source_traversal_evidence_missing')
    if clip.get('scenario') != 'open_and_traverse':
        reasons.append('unsupported_scenario')
    scenario = _scenario(spec, clip.get('scenario'))
    if scenario is None or not scenario.get('goal') or not scenario.get('pass_plane'):
        reasons.append('missing_traversal_scenario')
    try:
        time = np.asarray(source['time']); qpos = np.asarray(source['qpos']); tau = np.asarray(source['tau'])
        n = len(time)
        valid = (time.shape == (n,) and n >= 2 and time[0] == 0 and np.all(np.diff(time) > 0)
                 and qpos.ndim == 2 and qpos.shape[0] == n and qpos.shape[1] > 0 and tau.shape == qpos.shape
                 and all(np.shape(source[k]) == (n, 3) for k in ('base', 'target'))
                 and all(np.isfinite(source[k]).all() for k in ('time', 'qpos', 'tau', 'base', 'target')))
        if not valid:
            reasons.append('invalid_native_arrays')
        else:
            if np.any(tau != 0): reasons.append('nonzero_native_manual_effort')
            if scenario and scenario.get('pass_plane'):
                try: _crossing(source, scenario)
                except (ValueError, KeyError, TypeError): reasons.append('unsupported_or_missing_source_crossing')
    except (ValueError, KeyError, TypeError, IndexError):
        reasons.append('invalid_native_arrays')
    return tuple(reasons)


def _inputs(door_dir, recording_dir):
    """Load immutable index-bound source data; retain private paths only locally."""
    door_dir = Path(door_dir).resolve(); recording_dir = Path(recording_dir).resolve(); door_id = door_dir.name
    _require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', door_id), 'Unsafe powered door identity')
    paths = {}; digests = {}
    def bind(label, path, expected=None):
        digest = _sha(path)
        _require(expected is None or digest == expected, 'Powered input hash mismatch: '+label)
        paths[label] = path; digests[label] = digest
        return path
    index_path = bind('recording_index', _file(recording_dir, 'index.json'))
    index = json.loads(index_path.read_bytes())
    _require(index.get('schema') == 'doorbench.reference-motion.v1', 'Unsupported source recording index')
    manifest = bind('dataset_manifest', _file(door_dir.parent.parent, 'manifest.json'), index['manifest_sha256'])
    _require(sum(row.get('id') == door_id for row in json.loads(manifest.read_bytes())['doors']) == 1, 'Powered door missing or duplicated in dataset')
    matches = [row for row in index['clips'] if row.get('door_id') == door_id]
    _require(len(matches) == 1, 'Powered recording index requires one matching door')
    row = matches[0]
    clip_path = bind('recording_clip', _file(recording_dir, row['clip']), row['clip_sha256'])
    trajectory = bind('recording_trajectory', _file(recording_dir, row['trajectory']), row['trajectory_sha256'])
    clip = json.loads(clip_path.read_bytes())
    _require(clip.get('schema') == 'doorbench.reference-motion.v1' and clip.get('door_id') == door_id and
             clip.get('scenario') == row.get('scenario'), 'Powered clip/index identity mismatch')
    _require(clip.get('units') == 'metres/radians/seconds' and clip.get('up_axis') == 'Z', 'Unsupported powered source coordinates')
    outcome = clip['outcome']
    _require(outcome.get('door_id') == door_id and outcome.get('scenario') == clip['scenario'] and
             outcome.get('success') is row.get('success') and outcome.get('outcome') == row.get('outcome'), 'Powered source outcome/index mismatch')
    _require(set(row['source_sha256']) == SOURCE_FILES and clip.get('source_sha256') == row['source_sha256'], 'Powered source bindings disagree')
    for name, digest in row['source_sha256'].items(): bind('source/'+name, _file(door_dir, name), digest)
    spec = json.loads(paths['source/spec.json'].read_bytes()); model = json.loads(paths['source/model.json'].read_bytes())
    _require(spec.get('id') == door_id and spec.get('family') == row.get('family') == outcome.get('family'), 'Powered spec/source family mismatch')
    resources = {}
    for body in model.get('bodies', []):
        for geom in body.get('geoms', []):
            if geom.get('type') == 'mesh':
                mesh = geom['mesh_name']; _require(re.fullmatch(r'[A-Za-z0-9_.-]+', mesh), 'Unsafe powered mesh name')
                name = 'hardware/'+mesh+'.obj'; path = bind('resource/'+name, _file(door_dir.parent.parent, name)); resources[name] = _sha(path)
    with np.load(trajectory, allow_pickle=False) as arrays:
        source = {key: np.asarray(arrays[key], dtype=float).copy() for key in ('time', 'qpos', 'target', 'base', 'tau')}
    public = {'recording_index_sha256': digests['recording_index'], 'manifest_sha256': digests['dataset_manifest'],
              'recording_clip': {'path': row['clip'], 'sha256': digests['recording_clip']},
              'recording_trajectory': {'path': row['trajectory'], 'sha256': digests['recording_trajectory']},
              'source_sha256': row['source_sha256'], 'native_resources_sha256': resources}
    return clip, spec, source, public, paths, digests


def make_powered_guide(door_dir, recording_dir, *, fps=60, gait_profile='smooth'):
    """Propose a non-manual powered wait and aperture traversal; never approve it."""
    _require(type(fps) in (int, float) and np.isfinite(fps) and fps > 0, 'fps must be positive and finite')
    _require(gait_profile in ('controlled', 'wide_turns', 'compact', 'smooth'), 'Unknown gait profile')
    clip, spec, source, bindings, paths, digests = _inputs(door_dir, recording_dir)
    reasons = powered_eligibility(clip, spec, source)
    if reasons: raise PoweredIneligible(reasons)
    # Deferred imports allow guidance.make_guide to select this module without a
    # module cycle. Ineligible source episodes never compile native geometry.
    from .guidance import GuideBuilder, stationary, smooth_body_guidance
    from .gait import plan_walk
    from .planning import SceneNavigator, NoRoute
    scenario = _scenario(spec, clip['scenario']); stop = _crossing(source, scenario)
    options = dict(fps=fps, step_length=.42, step_duration=.65, stance_width=.21, blend_turns=True,
                   max_step_yaw_deg=45. if gait_profile == 'wide_turns' else 20.,
                   pelvis_acceleration_m_s2=None if gait_profile in ('compact', 'smooth') else 1.5)
    start = source['base'][0, :2].copy(); navigator = SceneNavigator(door_dir)
    navigator.update(source['qpos'][0])
    if not navigator.clear(start): raise NoRoute('Powered wait footprint intersects initial scene geometry')
    initial = plan_walk(start, 0., [start], **options)
    wait = stationary(initial, max(1., float(source['time'][stop])), fps)
    builder = GuideBuilder(source, fps)
    builder.add(wait, np.linspace(source['time'][0], source['time'][stop], len(wait['time'])), 'powered_wait')
    navigator.update(source['qpos'][stop])
    route = navigator.passage_route(start, np.asarray(scenario['goal']['center'])[:2], scenario['pass_plane'])
    walk = plan_walk(start, 0., route[1:], **options)
    builder.add(walk, source['time'][stop], 'traverse')
    builder.add(stationary(walk, .5, fps), source['time'][stop], 'settle')
    guide = builder.finish({'door_id': spec['id'], 'scenario': clip['scenario'], 'source_outcome': clip['outcome'],
        'source_sha256': clip['source_sha256'], 'traversal': 'proposed', 'traversal_reason': None, 'hand': None,
        'native_keyframes': stop+1, 'gait_profile': gait_profile,
        'scope': 'Experimental non-manual powered motion proposal; constrained IK and independent acceptance still required.',
        'powered_schedule': {'schema': 'doorbench.powered-reference-schedule.v1', 'source_manual_effort_max': 0.,
            'source_touched_door': False, 'source_touched_operator': False, 'both_hands_inactive': True,
            'source_crossing_index': stop, 'source_stop_time': float(source['time'][stop]), 'input_bindings': bindings,
            'source_sensor': spec['kinematics']['actuator'].get('sensor'), 'power_and_trigger_causality': 'unverified',
            'native_clock': 'Explicit nondecreasing source-time map. Wait replays the source prefix; traversal and settle hold its crossing pose.',
            'limitations': ['Native source poses are prescribed, not caused by the actor or its sensor/button interaction.',
                           'Source progression stops at the original base crossing; later native closure is not replayed.',
                           'No force, balance, sensor safety, original elapsed task-time, or personal visual certification.']}})
    if gait_profile == 'smooth': guide = smooth_body_guidance(guide, fps)
    _require(not guide.hand_contact.any() and not guide.hand_weight.any(), 'Powered schedule unexpectedly introduced manual contact')
    expected = np.stack([np.interp(guide.native_time, source['time'], source['qpos'][:, j]) for j in range(source['qpos'].shape[1])], axis=1)
    _require(np.array_equal(guide.native_qpos, expected) and np.all(np.diff(guide.native_time) >= 0), 'Powered schedule altered native pose/source-clock mapping')
    _require(all(_sha(path) == digests[name] for name, path in paths.items()), 'Powered inputs changed during guide preparation')
    return guide
