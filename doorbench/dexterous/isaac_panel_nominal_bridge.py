"""Detached nominal-coordinate interpolation, without geometry or authority.

The caller supplies the reviewed, freshly prepared bridge seed. This numerical
stage checks its current input identities but does not repeat physical-source
admission. It never evaluates a model, changes a controller, or submits forces.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .isaac_panel_bridge_seed import (
    SCHEMA as SEED_SCHEMA, CONVENTION, JOINT_NAMES, _merge_hashes,
    panel_coordinates_in_withdrawal_chart,
)
from .isaac_standing_continuation_audit import _json
from .qualified_isaac_grasp import digest

SCHEMA = 'doorbench.isaac-panel-nominal-bridge.v1'
DT = .002
MAXIMUM_STEPS = 30000
# Exactly the existing CoupledReferenceMotion post-projection checks. The
# acceleration tolerance applies to a velocity difference, not acceleration.
LIMITS = dict(joint_speed_rad_s=1.2, joint_velocity_change_per_step_rad_s=3.*DT,
    joint_acceleration_rad_s2=3., root_speed_m_s=.02,
    root_rotvec_speed_rad_s=.03, arithmetic_tolerance=1e-12)
AUTHORITY = dict(authorized_stages=0, physics_steps=0, active_state_writes=0,
    source_sample_playback=0, runtime_route_exported=False,
    controller_state_restoration_supported=False, geometric_admission=False,
    bridge_feasibility_qualified=False, physical_task_qualification=False,
    upstream_admission_reexecuted=False)


def content_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _vector(value, length, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError('Complete finite '+label+' required')
    return result.copy()


def _steps(duration):
    if type(duration) not in (int, float) or not np.isfinite(duration):
        raise ValueError('Finite explicit bridge duration required')
    steps = round(duration/DT)
    if not 1 <= steps <= MAXIMUM_STEPS or abs(duration-steps*DT) > 1e-12:
        raise ValueError('Bridge duration must be 1..30000 complete 500 Hz intervals')
    return steps


def _seed_inputs(seed):
    """Validate the numerical seam and upstream identities, not physical truth."""
    if type(seed) is not dict or seed.get('schema') != SEED_SCHEMA:
        raise ValueError('Reviewed detached bridge seed required')
    # Enforce JSON finiteness and detach before using nested caller data.
    seed = json.loads(json.dumps(seed, allow_nan=False))
    for key in ('authorized_stages', 'physics_steps', 'source_sample_playback', 'active_state_writes'):
        if type(seed.get(key)) is not int or seed[key] != 0:
            raise ValueError('Seed must retain zero authority')
    for key in ('runtime_route_exported', 'controller_state_restoration_supported',
                'bridge_feasibility_qualified', 'physical_task_qualification'):
        if seed.get(key) is not False:raise ValueError('Seed must retain zero authority')
    previous = seed['previous_reference']; names = previous['joint_names']
    if (type(names) is not list or len(names) != 69 or len(set(names)) != 69
            or any(type(n) is not str or not n for n in names)
            or seed['panel_joint_names'] != JOINT_NAMES or not set(JOINT_NAMES) <= set(names)
            or previous['coordinate_convention'] != CONVENTION):
        raise ValueError('Exact original named 75-coordinate convention required')
    p0 = _vector(previous['value'], 75, 'previous nominal')
    v0 = _vector(previous['velocity'], 75, 'stored prior nominal velocity')
    terminal, start = seed['source_time_s'], previous['command_time_s']
    if (type(terminal) not in (int, float) or type(start) not in (int, float)
            or not np.isfinite([terminal, start]).all() or terminal <= 0 or start < 0
            or abs((terminal-start)-DT) > 1e-10
            or previous['paired_post_step_time_s'] != terminal
            or seed['measured_terminal']['time_s'] != terminal
            or abs(terminal/DT-round(terminal/DT)) > 1e-7):
        raise ValueError('Exact previous command at source T-0.002 required')
    tail = seed['previous_three_references']
    if (type(tail) is not list or len(tail) != 3
            or tail[-1]['command_time_s'] != start
            or tail[-1]['post_step_time_s'] != terminal
            or tail[-1]['coupled']['value'] != previous['value']
            or tail[-1]['coupled']['velocity'] != previous['velocity']
            or tail[-1]['coupled']['joint_names'] != names
            or tail[-1]['coupled']['root_coordinate_origin_xyz_wxyz'] != previous['root_coordinate_origin_xyz_wxyz']):
        raise ValueError('Previous nominal must match the bound reference tail')
    preserved_names = [n for n in names if n not in JOINT_NAMES]
    preserved = [6+names.index(n) for n in preserved_names]
    active = list(range(6))+[6+names.index(n) for n in JOINT_NAMES]
    if (len(preserved) != 44 or seed['preserved_nominal_joint_names'] != preserved_names
            or seed['preserved_nominal_joint_values'] != p0[preserved].tolist()):
        raise ValueError('Exact 44 previous non-panel nominal joints required')
    allowed = LIMITS['joint_velocity_change_per_step_rad_s']+LIMITS['arithmetic_tolerance']
    bad = [dict(joint_name=names[i-6], stored_velocity_rad_s=float(v0[i]))
        for i in preserved if abs(v0[i]) > allowed]
    if bad:
        raise ValueError('Constant preserved joints cannot stop within original discrete acceleration gate: '+json.dumps(bad))
    hashes = _merge_hashes(seed['input_sha256'])
    candidate_path = Path(seed['candidate_path'])
    if not candidate_path.is_absolute():raise ValueError('Absolute bound panel candidate required')
    candidate_path = candidate_path.resolve()
    if hashes.get(str(candidate_path)) != seed['candidate_sha256']:
        raise ValueError('Exact panel candidate must be a bound seed input')
    panel = _json(candidate_path)
    if (panel.get('schema') != 'doorbench.isaac-panel-candidate.v1'
            or panel['source_run'] != seed['source_run']
            or panel['source_time_s'] != terminal
            or panel['source_context_sha256'] != seed['source_context_sha256']
            or panel['names'] != JOINT_NAMES):
        raise ValueError('Panel first point must belong to the same actual-source seed')
    first = panel['rows'][0]
    x = _vector(first['root_delta']+[first['joint_targets'][n] for n in JOINT_NAMES],
        31, 'panel first coordinates')
    p1 = panel_coordinates_in_withdrawal_chart(seed, x)
    if not np.array_equal(p1, _vector(seed['panel_first_target_in_old_chart'], 75, 'mapped panel first target')):
        raise ValueError('Saved panel first mapping differs from current original chart mapping')
    # This comparison is intentionally exact; no measured-T remainder enters.
    if not np.array_equal(p1[preserved], p0[preserved]):
        raise ValueError('Mapped endpoint changed preserved nominal joints')
    _merge_hashes(hashes)
    return seed, p0, v0, p1, active, preserved, hashes


def build_panel_nominal_bridge(seed, *, duration_s):
    """Produce a candidate even if its later complete rate audit will fail.

    The 31 free coordinates have the captured initial analytic velocity. The
    44 constant coordinates have polynomial derivative zero; their stored old
    limiter velocity remains separately recorded for the first discrete step.
    """
    count = _steps(duration_s)
    seed, p0, v0, p1, active, preserved, hashes = _seed_inputs(seed)
    # Use the declared integer motor grid at the endpoint too; retain the
    # caller's float so a sub-tolerance representation difference is visible.
    duration = count*DT
    helper = Path(__file__).resolve()
    hashes = _merge_hashes(hashes, {str(helper):digest(helper),
        str(helper.with_name('coupled_release_motion.py')):digest(helper.with_name('coupled_release_motion.py'))})
    # Normalized-time quintic Hermite coefficients. Existing repository
    # helpers are rest-to-rest and cannot preserve the nonzero old velocity.
    coefficients = np.zeros((6, 75)); coefficients[0] = p0
    delta = p1[active]-p0[active]; scaled_velocity = duration*v0[active]
    coefficients[1, active] = scaled_velocity
    coefficients[3, active] = 10*delta-6*scaled_velocity
    coefficients[4, active] = -15*delta+8*scaled_velocity
    coefficients[5, active] = 6*delta-3*scaled_velocity
    if not np.isfinite(coefficients).all():raise ValueError('Finite bridge coefficients required')
    polynomial_v0 = v0.copy(); polynomial_v0[preserved] = 0.
    result = dict(schema=SCHEMA, seed_sha256=content_sha256(seed), seed=seed,
        source_run=seed['source_run'], source_time_s=seed['source_time_s'],
        source_context_sha256=seed['source_context_sha256'],
        source_physics_sha256=seed['source_physics_sha256'],
        panel_candidate_path=seed['candidate_path'], panel_candidate_sha256=seed['candidate_sha256'],
        input_sha256=hashes, dt_s=DT, requested_duration_s=float(duration_s),
        duration_s=duration, new_command_samples=count,
        start_command_time_s=seed['previous_reference']['command_time_s'],
        first_new_command_time_s=seed['source_time_s'],
        final_command_time_s=seed['previous_reference']['command_time_s']+duration,
        joint_names=copy.deepcopy(seed['previous_reference']['joint_names']),
        coordinate_convention=CONVENTION,
        root_coordinate_origin_xyz_wxyz=copy.deepcopy(seed['previous_reference']['root_coordinate_origin_xyz_wxyz']),
        active_coordinate_indices=active, preserved_coordinate_indices=preserved,
        preserved_nominal_joint_names=copy.deepcopy(seed['preserved_nominal_joint_names']),
        coefficients_normalized_time_ascending=coefficients.tolist(),
        initial_position=p0.tolist(), stored_prior_velocity=v0.tolist(),
        initial_polynomial_velocity=polynomial_v0.tolist(),
        initial_polynomial_acceleration=[0.]*75,
        initial_acceleration_choice='Prospective zero polynomial acceleration; no prior acceleration state was restored',
        final_position=p1.tolist(), final_polynomial_velocity=[0.]*75,
        final_polynomial_acceleration=[0.]*75,
        derivative_semantics='Polynomial derivatives in the original withdrawal chart. The 44 constant coordinates are not analytically C1 with a nonzero stored prior limiter velocity; compare that stored velocity at the first discrete step.',
        endpoint_evaluation='Exact declared position/derivative boundary values; interior uses stored polynomial coefficients',
        limits=copy.deepcopy(LIMITS), rate_audit_performed=False,
        scope='Bound upstream seed identities and detached coordinates only. No fresh source admission, geometry, motor calculation or physical qualification.',
        **AUTHORITY)
    _merge_hashes(hashes)
    return result


def sample_panel_nominal_bridge(candidate, elapsed_s):
    """Read a candidate at an in-domain clock; return detached references only."""
    if candidate.get('schema') != SCHEMA or candidate.get('dt_s') != DT:
        raise ValueError('Explicit detached nominal bridge candidate required')
    _steps(candidate['duration_s'])
    if (type(elapsed_s) not in (int, float) or not np.isfinite(elapsed_s)
            or not 0 <= elapsed_s <= candidate['duration_s']):
        raise ValueError('Bridge sampling outside declared duration is forbidden')
    coefficients = np.asarray(candidate['coefficients_normalized_time_ascending'], float)
    if coefficients.shape != (6, 75) or not np.isfinite(coefficients).all():
        raise ValueError('Complete finite quintic coefficients required')
    duration = candidate['duration_s']; u = elapsed_s/duration
    position = coefficients[5].copy(); velocity = 5*coefficients[5]; acceleration = 20*coefficients[5]
    for i in range(4, -1, -1):position = position*u+coefficients[i]
    for i in range(4, 0, -1):velocity = velocity*u+i*coefficients[i]
    for i in range(4, 1, -1):acceleration = acceleration*u+i*(i-1)*coefficients[i]
    velocity = velocity/duration; acceleration = acceleration/duration**2
    if elapsed_s == 0:
        position = _vector(candidate['initial_position'], 75, 'initial position')
        velocity = _vector(candidate['initial_polynomial_velocity'], 75, 'initial polynomial velocity')
        acceleration = _vector(candidate['initial_polynomial_acceleration'], 75, 'initial polynomial acceleration')
    elif elapsed_s == duration:
        position = _vector(candidate['final_position'], 75, 'final position')
        velocity = _vector(candidate['final_polynomial_velocity'], 75, 'final polynomial velocity')
        acceleration = _vector(candidate['final_polynomial_acceleration'], 75, 'final polynomial acceleration')
    if not np.isfinite(np.r_[position, velocity, acceleration]).all():
        raise ValueError('Bridge evaluation is nonfinite')
    return dict(command_time_s=candidate['start_command_time_s']+elapsed_s,
        elapsed_s=float(elapsed_s), position=position.copy(),
        polynomial_velocity=velocity.copy(), polynomial_acceleration=acceleration.copy(),
        stored_prior_velocity=(np.asarray(candidate['stored_prior_velocity'], float).copy() if elapsed_s == 0 else None),
        authorized_stages=0)
