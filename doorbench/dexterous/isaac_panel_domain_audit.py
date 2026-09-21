"""Finite mechanism-grid geometry screening; no live/runtime admission.

Every declared Cartesian point is counted, including rejected mechanism inputs.
No physics, target projection, clipping or interpolation-domain extrapolation is
performed. Passing finite samples does not prove the continuous volume between
them, dynamic support, tracking, handoff, or physical opening.
"""
import copy
import itertools
import json
from pathlib import Path

import numpy as np

from .isaac_panel_geometry_audit import AUDIT_SCHEMA, PLAN_SCHEMA
from .isaac_panel_geometry_probe import IsaacPanelGeometryProbe, STATIC_LIMITS
from .isaac_panel_planning import JOINT_NAMES, admit_isaac_panel_context
from .qualified_isaac_grasp import digest
from .screened_panel_path import ScreenedPanelPath

SCHEMA = 'doorbench.isaac-panel-sampled-mechanism-domain-audit.v1'
SAMPLING_SCHEMA = 'doorbench.isaac-panel-domain-sampling.v1'
MAXIMUM_SAMPLES = 250000
RATE_LIMITS = dict(joint_velocity_rad_s=1.2, joint_acceleration_rad_s2=3.,
    root_velocity_m_s=.02, root_rotvec_velocity_rad_s=.03)
ORIGINAL_LIMITS = {**STATIC_LIMITS, **RATE_LIMITS}
SAMPLING_KEYS = frozenset(('schema', 'reference_aperture_rad', 'reference_samples',
    'leaf_lag_rad', 'leaf_lag_samples', 'operator_rad', 'operator_samples',
    'latch_m', 'latch_samples', 'maximum_samples'))


def _unique(pairs):
    value = {}
    for name, item in pairs:
        if name in value: raise ValueError('Duplicate domain input JSON field: '+name)
        value[name] = item
    return value


def _bad_constant(value):
    raise ValueError('Nonfinite domain JSON input: '+value)


def _finite_tree(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError('Nonfinite geometry/domain evidence')
    if isinstance(value, dict):
        for item in value.values(): _finite_tree(item)
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in value: _finite_tree(item)


def _read(path):
    result = json.loads(Path(path).read_text(), object_pairs_hook=_unique, parse_constant=_bad_constant)
    _finite_tree(result)
    return result


def _number(value):
    return type(value) in (int, float) and np.isfinite(value)


def _range(value, name):
    if (type(value) is not list or len(value) != 2
            or not all(_number(x) for x in value) or value[0] > value[1]):
        raise ValueError('Explicit finite ordered two-endpoint '+name+' required')
    return [float(x) for x in value]


def _axis(bounds, count, *, name, include=()):
    bounds = _range(bounds, name)
    if (type(count) is not int or not 1 <= count <= MAXIMUM_SAMPLES
            or (bounds[0] == bounds[1] and count != 1)
            or (bounds[0] < bounds[1] and count < 2)):
        raise ValueError('Explicit nonduplicated grid count required: '+name)
    if any(not _number(x) or not bounds[0] <= x <= bounds[1] for x in include):
        raise ValueError('Required source/grid node is outside declared '+name)
    uniform = np.linspace(*bounds, count)
    if count > 1 and not np.all(np.diff(uniform) > 0):
        raise ValueError('Requested grid has no distinct floating-point nodes: '+name)
    return np.unique(np.r_[uniform, include])


def sampling_grid(spec, *, initial_aperture, final_aperture, knot_progress, source_angles):
    """Validate all axes before work; no requested point is clipped or dropped."""
    if type(spec) is not dict or set(spec) != SAMPLING_KEYS or spec['schema'] != SAMPLING_SCHEMA:
        raise ValueError('Exact explicit mechanism-domain sampling schema required')
    lo, hi = _range(spec['reference_aperture_rad'], 'reference aperture')
    n = spec['reference_samples']; maximum = spec['maximum_samples']
    if (type(n) is not int or not 2001 <= n <= 20001
            or not initial_aperture <= lo < hi <= final_aperture
            or type(maximum) is not int or not 1 <= maximum <= MAXIMUM_SAMPLES):
        raise ValueError('In-range aperture segment, >=2001 reference nodes and bounded explicit budget required')
    knots = np.asarray(knot_progress, float)
    if (knots.ndim != 1 or len(knots) < 4 or not np.isfinite(knots).all()
            or knots[0] != 0 or knots[-1] != 1 or not np.all(np.diff(knots) > 0)):
        raise ValueError('Complete original spline-knot progress required')
    # Work in progress so the exact spline knot is sampled, not a round-tripped
    # aperture approximation. Store requested aperture values separately.
    delta = final_aperture-initial_aperture
    reference = np.linspace(lo, hi, n)
    if not np.all(np.diff(reference) > 0):
        raise ValueError('Requested reference grid requires >=2001 distinct floating-point nodes')
    pairs = {float((angle-initial_aperture)/delta): float(angle) for angle in reference}
    if len(pairs) != n: raise ValueError('Distinct reference progress nodes required')
    for value in knots:
        angle = float(initial_aperture if value == 0 else final_aperture if value == 1 else initial_aperture+value*delta)
        if lo <= angle <= hi: pairs.setdefault(float(value), angle)
    progress = np.array(sorted(pairs))
    angles = np.array([pairs[p] for p in progress])
    if np.any(progress < 0) or np.any(progress > 1):
        raise ValueError('No spline progress extrapolation allowed')
    lag_range = _range(spec['leaf_lag_rad'], 'leaf lag')
    lag = _axis(lag_range, spec['leaf_lag_samples'], name='leaf lag',
        include=(0.,) if lag_range[0] <= 0 <= lag_range[1] else ())
    operator = _axis(spec['operator_rad'], spec['operator_samples'], name='operator',
        include=(source_angles['operator'],))
    latch = _axis(spec['latch_m'], spec['latch_samples'], name='latch',
        include=(source_angles['latch'],))
    total = 1+len(progress)*len(lag)*len(operator)*len(latch)
    if total > maximum:
        raise ValueError(f'Full sample denominator {total} exceeds declared budget {maximum}')
    return dict(progress=progress, reference_aperture_rad=angles, leaf_lag_rad=lag,
        operator_rad=operator, latch_m=latch, domain_samples=total-1,
        source_point_samples=1, total_samples=total)


def reference_rate_envelope(path, *, aperture_delta, speed, acceleration):
    """Same cubic extrema and chain-rule arithmetic as the original auditor."""
    if (not _number(speed) or not 0 < speed <= .149 or not _number(acceleration)
            or not 0 < acceleration <= .08 or not _number(aperture_delta) or aperture_delta <= 0):
        raise ValueError('Original bounded finite aperture-rate envelope required')
    bounds = path.derivative_bounds()
    envelope = dict(
        maximum_joint_speed_rad_s=float(np.max(bounds['first'][6:])*speed/aperture_delta),
        maximum_joint_acceleration_rad_s2=float(np.max(bounds['second'][6:]*(speed/aperture_delta)**2
            + bounds['first'][6:]*acceleration/aperture_delta)),
        maximum_root_speed_m_s=float(bounds['root_translation_first_norm']*speed/aperture_delta),
        maximum_root_rotvec_speed_rad_s=float(bounds['root_rotvec_first_norm']*speed/aperture_delta),
        aperture_speed_limit_rad_s=speed, aperture_acceleration_limit_rad_s2=acceleration,
        method='Exact cubic-segment derivative extrema and conservative chain-rule acceleration bound')
    _finite_tree(envelope)
    envelope['passed'] = bool(envelope['maximum_joint_speed_rad_s'] <= 1.2
        and envelope['maximum_joint_acceleration_rad_s2'] <= 3.
        and envelope['maximum_root_speed_m_s'] <= .02
        and envelope['maximum_root_rotvec_speed_rad_s'] <= .03)
    return envelope


def validate_static_proof(candidate, plan, receipt, *, context, probe, candidate_path):
    """Bind an unchanged successful original static screen, not just passed=true."""
    if receipt.get('schema') != AUDIT_SCHEMA or plan.get('schema') != PLAN_SCHEMA:
        raise ValueError('Distinct original Isaac panel plan/static audit required')
    if (receipt.get('passed') is not True or receipt.get('geometric_admission') is not True
            or plan.get('geometric_admission') is not True or receipt.get('failures') != []
            or plan.get('screen_receipt') != receipt or receipt.get('limits') != ORIGINAL_LIMITS):
        raise ValueError('Passing unchanged original static audit and exact embedded receipt/limits required')
    expected_hashes = {**candidate['input_sha256'], str(candidate_path): digest(candidate_path),
        str(Path(__file__).with_name('isaac_panel_geometry_audit.py').resolve()):
            digest(Path(__file__).with_name('isaac_panel_geometry_audit.py'))}
    for value in (plan, receipt):
        if (value.get('source_admission') != context.admission or value.get('source_context_sha256') != context.sha256
                or value.get('source_engine') != 'isaac-physx' or value.get('source_time_s') != context.admission['source_time_s']
                or value.get('candidate_path') != str(candidate_path) or value.get('candidate_sha256') != digest(candidate_path)
                or value.get('input_sha256') != expected_hashes):
            raise ValueError('Static proof source/candidate/current input binding differs')
        for key in ('authorized_stages','physics_steps','source_sample_playback','active_state_writes'):
            if type(value.get(key)) is not int or value[key] != 0: raise ValueError('Static proof cannot claim stage authority')
        if value.get('physical_admission') is not False or value.get('runtime_route_exported') is not False:
            raise ValueError('Static proof cannot claim runtime or physical authority')
    if any(digest(p) != expected for p, expected in expected_hashes.items()):
        raise ValueError('Bound static proof input changed')
    rows = candidate['rows']; angles = np.array([r['leaf_angle_rad'] for r in rows])
    progress = (angles-angles[0])/(angles[-1]-angles[0])
    coordinates = np.array([r['root_delta']+[r['joint_targets'][n] for n in JOINT_NAMES] for r in rows])
    if (not np.array_equal(plan.get('progress'), progress) or not np.array_equal(plan.get('coordinates'), coordinates)
            or plan.get('joint_names') != JOINT_NAMES or plan.get('joint_qpos_addresses') != probe.qa.tolist()
            or plan.get('root_qpos_address') != probe.rq
            or not np.array_equal(plan.get('initial_qpos'), context.qpos)
            or not np.array_equal(plan.get('initial_qvel'), context.qvel)
            or plan.get('initial_episode_time_s') != context.admission['source_time_s']
            or plan.get('initial_leaf_angle_rad') != float(angles[0])
            or plan.get('final_leaf_angle_rad') != float(angles[-1])
            or receipt.get('target_aperture_rad') != float(angles[-1])):
        raise ValueError('Static plan coordinates/epoch differ from exact candidate')
    leaf_joint = probe.mechanism['leaf']
    if plan.get('initial_leaf_velocity_rad_s') != float(context.qvel[probe.m.jnt_dofadr[leaf_joint]]):
        raise ValueError('Initial measured leaf velocity differs')
    duration = receipt.get('duration_s'); samples = receipt.get('samples')
    if (not _number(duration) or not 3 <= duration <= 600 or plan.get('duration_s') != duration
            or type(samples) is not int or not 2001 <= samples <= 20001
            or receipt.get('exact_initial_state') is not True
            or receipt.get('optimizer_initial_coordinate_adjustment_removed') != 0.):
        raise ValueError('Original complete exact-source static duration/density required')
    lag = receipt.get('actual_leaf_lag_rad'); onset = receipt.get('lag_start_angle_rad')
    if (not _number(lag) or not 0 <= lag <= .02 or (onset is not None
            and (not _number(onset) or not angles[0] <= onset <= angles[-1]))):
        raise ValueError('Original static lag slice metadata required')
    maxima = receipt.get('maxima', {})
    if (set(maxima) != set(ORIGINAL_LIMITS) or any(not _number(v) or v > ORIGINAL_LIMITS[k] for k,v in maxima.items())
            or not _number(receipt.get('minimum_all_rh_scene_clearance_capped_m'))
            or receipt['minimum_all_rh_scene_clearance_capped_m'] < .04
            or not _number(receipt.get('minimum_elbow_scene_clearance_capped_m'))
            or receipt['minimum_elbow_scene_clearance_capped_m'] < .003
            or receipt.get('required_elbow_clearance_m') != .003):
        raise ValueError('Original successful static maxima/clearance evidence required')
    path = ScreenedPanelPath(progress, coordinates, duration)
    existing = receipt['reference_phase_envelope']
    envelope = reference_rate_envelope(path, aperture_delta=float(angles[-1]-angles[0]),
        speed=existing['aperture_speed_limit_rad_s'], acceleration=existing['aperture_acceleration_limit_rad_s2'])
    if envelope != existing or not envelope['passed']:
        raise ValueError('Original analytic reference-rate envelope does not reproduce')
    return path, envelope


def audit_panel_domain(candidate_path, plan_path, static_audit_path, sampling_spec, *, progress=None):
    """Fresh source admission followed by every declared detached geometry point."""
    paths = tuple(Path(p).resolve() for p in (candidate_path, plan_path, static_audit_path))
    consumers = (Path(__file__).resolve(), Path(__file__).with_name('isaac_panel_geometry_probe.py').resolve())
    initial_hashes = {str(p): digest(p) for p in (*paths, *consumers)}
    candidate, plan, static = [_read(p) for p in paths]
    source = candidate['source_admission']
    context = admit_isaac_panel_context(source['source_run'], robot=source['robot_path'],
        door_xml=source['door_xml_path'], door_usd=source['door_usd_path'])
    probe = IsaacPanelGeometryProbe(context, candidate)
    path, rate = validate_static_proof(candidate, plan, static, context=context, probe=probe, candidate_path=paths[0])
    source_angles = {name: float(context.qpos[probe.m.jnt_qposadr[j]]) for name,j in probe.mechanism.items()}
    spec = copy.deepcopy(sampling_spec)
    grid = sampling_grid(spec, initial_aperture=probe.start_angle, final_aperture=probe.final_angle,
        knot_progress=plan['progress'], source_angles=source_angles)
    hashes = dict(initial_hashes)
    for name, expected in probe.hashes.items():
        if name in hashes and hashes[name] != expected:
            raise ValueError('Panel domain consumer changed during admission')
        hashes[name] = expected
    failures = []; maxima = {}; minima = {}; count = 0; rejected = 0; source_point = None

    def inspect(x, reference, measured, location):
        nonlocal count, rejected
        count += 1
        try:
            point = probe.evaluate(x, reference_aperture_rad=float(reference), measured_angles=measured)
            _finite_tree(point)
            if (point.get('limits') != STATIC_LIMITS or type(point.get('passed')) is not bool
                    or point.get('source_context_sha256') != context.sha256
                    or point.get('measured_angles') != measured or set(point.get('values', {})) != set(STATIC_LIMITS)):
                raise ValueError('Point evaluator contract differs')
            for key,value in point['values'].items(): maxima[key] = max(maxima.get(key, -float('inf')), value)
            for key in ('right_scene_clearance_capped_m', 'elbow_scene_clearance_capped_m'):
                minima[key] = min(minima.get(key, float('inf')), point[key])
            if not point['passed']:
                failures.append(dict(sample_index=count-1, **location,
                    reference_aperture_rad=float(reference), measured_angles=dict(measured),
                    violations=point['violations'], collisions=point['collisions'],
                    right_scene_clearance_capped_m=point['right_scene_clearance_capped_m'],
                    right_scene_closest_pair=point['right_scene_closest_pair'],
                    elbow_scene_clearance_capped_m=point['elbow_scene_clearance_capped_m'],
                    elbow_scene_closest_pair=point['elbow_scene_closest_pair']))
        except Exception as error:
            rejected += 1
            point = dict(passed=False, input_rejected=True, error_type=type(error).__name__, error=str(error))
            failures.append(dict(sample_index=count-1, **location, reference_aperture_rad=float(reference),
                measured_angles=dict(measured), **point))
        if progress is not None and (count == 1 or count % 200 == 0 or count == grid['total_samples']):
            progress(dict(attempted_samples=count, total_samples=grid['total_samples'],
                failed_samples=len(failures), rejected_samples=rejected))
        return point

    source_point = inspect(probe.initial_coordinates(), probe.start_angle, source_angles, dict(kind='exact_source'))
    for i, (s, reference) in enumerate(zip(grid['progress'], grid['reference_aperture_rad'])):
        # Direct cubic progress evaluation; no time warp, clipping or sampling
        # outside the proven interpolation interval is hidden in this call.
        x = np.asarray(path.spline(float(s)), float)
        for j, (lag, operator, latch) in enumerate(itertools.product(grid['leaf_lag_rad'],grid['operator_rad'],grid['latch_m'])):
            measured = dict(leaf=float(reference-lag), operator=float(operator), latch=float(latch))
            inspect(x, reference, measured, dict(kind='domain', reference_index=i, mechanism_index=j,
                progress=float(s), reference_minus_measured_leaf_rad=float(lag)))
    if count != grid['total_samples']: raise RuntimeError('Declared domain denominator changed')
    probe.verify_inputs()
    if any(digest(p) != expected for p,expected in hashes.items()):
        raise ValueError('Panel domain input changed during sampling')
    result = dict(schema=SCHEMA, passed=not failures and rate['passed'], sampled_points_passed=not failures,
        source_admission=context.admission, source_context_sha256=context.sha256,
        source_time_s=context.admission['source_time_s'], source_engine='isaac-physx',
        candidate_path=str(paths[0]), plan_path=str(paths[1]), static_audit_path=str(paths[2]),
        input_sha256=hashes, sampling_spec=spec,
        grid={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in grid.items()},
        source_point=source_point, source_point_is_separate=True,
        attempted_samples=count, failed_samples=len(failures), rejected_samples=rejected,
        maxima=maxima, minima=minima, limits=dict(ORIGINAL_LIMITS), reference_phase_envelope=rate,
        failures=failures, input_hashes_unchanged=True,
        robot_target_scope='Original panel subset plus fixed actual-source values for every remaining scalar joint; no extra full-target route supplied',
        lag_definition='Reference leaf aperture minus supplied measured leaf aperture; negative lag means measured leaf leads',
        authorized_stages=0, physics_steps=0, source_sample_playback=0, active_state_writes=0,
        geometric_admission=False, physical_admission=False, runtime_route_exported=False,
        continuous_domain_proved=False,
        scope='Finite sampled authored geometry at explicit mechanism coordinates, plus original analytic nominal-reference rate bounds; no continuous-volume, dynamic-support, tracking, handoff, live-state or runtime qualification.')
    _finite_tree(result)
    return result
