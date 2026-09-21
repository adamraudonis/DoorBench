"""Independent complete 500 Hz rate accounting for a detached bridge.

Stored coefficients are evaluated directly, without calling the builder or
sampler. Passing these finite-difference checks is not continuous derivative,
geometry, contact, motor, controller-restoration or physical qualification.
"""
import copy
from pathlib import Path

import numpy as np

from .isaac_panel_nominal_bridge import (
    SCHEMA as CANDIDATE_SCHEMA, DT, LIMITS, AUTHORITY, _steps, _seed_inputs,
    content_sha256,
)
from .isaac_panel_bridge_seed import _merge_hashes
from .qualified_isaac_grasp import digest

SCHEMA = 'doorbench.isaac-panel-nominal-bridge-rate-audit.v1'


def _polynomial(coefficients, u, duration):
    # Separate evaluator from the sampler's Horner loop.
    p = np.polynomial.polynomial.polyval(u, coefficients)
    v = np.polynomial.polynomial.polyval(u, coefficients[1:]*np.arange(1, 6)[:, None])/duration
    a = np.polynomial.polynomial.polyval(u, coefficients[2:]*np.array([2, 6, 12, 20])[:, None])/duration**2
    return p, v, a


def audit_panel_nominal_bridge(candidate, *, progress=None):
    """Inspect every command, including the old boundary and terminal hold.

    Invalid candidate structure/identity raises. Numerical rate violations are
    retained as a failed complete receipt, without truncating the denominator.
    """
    original_sha = content_sha256(candidate); document = copy.deepcopy(candidate)
    if document.get('schema') != CANDIDATE_SCHEMA or document.get('dt_s') != DT:
        raise ValueError('Explicit original 500 Hz bridge candidate required')
    if document.get('limits') != LIMITS or any(type(document.get(k)) is not type(v) or document[k] != v for k, v in AUTHORITY.items()):
        raise ValueError('Original rate limits and zero bridge authority required')
    count = _steps(document['duration_s']); duration = document['duration_s']
    if duration != count*DT or _steps(document['requested_duration_s']) != count:
        raise ValueError('Exact integer 500 Hz duration grid required')
    seed, p0, v0, p1, active, preserved, seed_hashes = _seed_inputs(document['seed'])
    if document['seed_sha256'] != content_sha256(seed):raise ValueError('Bridge seed content changed')
    hashes = _merge_hashes(document['input_sha256'])
    if any(hashes.get(k) != v for k, v in seed_hashes.items()):raise ValueError('Bridge lost source input bindings')
    for name in ('isaac_panel_nominal_bridge.py', 'coupled_release_motion.py'):
        path = Path(__file__).resolve().with_name(name)
        if hashes.get(str(path)) != digest(path):raise ValueError('Bridge lost current numerical helper binding: '+name)
    for key in ('source_run', 'source_time_s', 'source_context_sha256', 'source_physics_sha256'):
        if document[key] != seed[key]:raise ValueError('Bridge changed source identity: '+key)
    if (document['panel_candidate_path'] != seed['candidate_path']
            or document['panel_candidate_sha256'] != seed['candidate_sha256']
            or document['joint_names'] != seed['previous_reference']['joint_names']
            or document['coordinate_convention'] != seed['previous_reference']['coordinate_convention']
            or document['root_coordinate_origin_xyz_wxyz'] != seed['previous_reference']['root_coordinate_origin_xyz_wxyz']
            or document['active_coordinate_indices'] != active or document['preserved_coordinate_indices'] != preserved
            or document['preserved_nominal_joint_names'] != seed['preserved_nominal_joint_names']
            or document['start_command_time_s'] != seed['previous_reference']['command_time_s']
            or document['first_new_command_time_s'] != seed['source_time_s']
            or document['new_command_samples'] != count
            or document['final_command_time_s'] != document['start_command_time_s']+duration):
        raise ValueError('Bridge coordinate order, source clock or endpoint identity changed')
    expected_v0 = v0.copy(); expected_v0[preserved] = 0.
    for key, expected in (('initial_position', p0), ('stored_prior_velocity', v0),
            ('initial_polynomial_velocity', expected_v0), ('initial_polynomial_acceleration', np.zeros(75)),
            ('final_position', p1), ('final_polynomial_velocity', np.zeros(75)),
            ('final_polynomial_acceleration', np.zeros(75))):
        if not np.array_equal(np.asarray(document[key]), expected):
            raise ValueError('Bridge boundary differs from bound source or declared Hermite conditions: '+key)
    coefficients = np.asarray(document['coefficients_normalized_time_ascending'], float)
    if coefficients.shape != (6, 75) or not np.isfinite(coefficients).all():raise ValueError('Complete finite quintic coefficients required')
    tolerance = LIMITS['arithmetic_tolerance']
    if (not np.array_equal(coefficients[0], p0)
            or not np.array_equal(coefficients[1:, preserved], np.zeros((5, 44)))):
        raise ValueError('Exact original position and 44 constant polynomial coordinates required')
    # Check coefficient constraints in normalized time before endpoint
    # anchoring; do not amplify floating cancellation by a short duration.
    # Actual 500 Hz rate gates below retain their unchanged 1e-12 tolerance.
    for u, expected in ((0., (p0, expected_v0*duration, np.zeros(75))), (1., (p1, np.zeros(75), np.zeros(75)))):
        actual = _polynomial(coefficients, u, 1.)
        if any(not np.allclose(a, e, atol=tolerance, rtol=0) for a, e in zip(actual, expected)):
            raise ValueError('Stored polynomial violates declared Hermite boundary conditions')
    helper = Path(__file__).resolve()
    hashes = _merge_hashes(hashes, {str(helper):digest(helper)})
    failures = []; maxima = dict(joint_speed_rad_s=0., joint_acceleration_rad_s2=0.,
        root_speed_m_s=0., root_rotvec_speed_rad_s=0.)
    analytic_maxima = dict(joint_speed_rad_s=float(np.max(abs(expected_v0[6:]))),
        joint_acceleration_rad_s2=0., root_speed_m_s=float(np.linalg.norm(expected_v0[:3])),
        root_rotvec_speed_rad_s=float(np.linalg.norm(expected_v0[3:6])))
    previous_position = p0.copy(); previous_velocity = v0.copy(); first = None

    def inspect(index, elapsed, velocity, acceleration, kind):
        metrics = dict(joint_speed_rad_s=float(np.max(abs(velocity[6:]))),
            joint_acceleration_rad_s2=None if acceleration is None else float(np.max(abs(acceleration))),
            root_speed_m_s=float(np.linalg.norm(velocity[:3])), root_rotvec_speed_rad_s=float(np.linalg.norm(velocity[3:6])))
        if any(value is not None and not np.isfinite(value) for value in metrics.values()):
            raise ValueError('Derived bridge rate is nonfinite')
        bad = []
        for key in ('joint_speed_rad_s', 'root_speed_m_s', 'root_rotvec_speed_rad_s'):
            if metrics[key] > LIMITS[key]+tolerance:bad.append(key)
        if acceleration is not None and np.max(abs(velocity[6:]-previous_velocity[6:])) > 3.*DT+tolerance:
            bad.append('joint_velocity_change_per_step_rad_s')
        for key, value in metrics.items():
            if value is not None:maxima[key] = max(maxima[key], value)
        if bad:
            failures.append(dict(sample_index=index, kind=kind, elapsed_s=elapsed,
                command_time_s=document['start_command_time_s']+elapsed,
                failed_checks=bad, metrics=metrics,
                maximum_speed_joint=document['joint_names'][int(np.argmax(abs(velocity[6:])))],
                maximum_acceleration_joint=(None if acceleration is None else document['joint_names'][int(np.argmax(abs(acceleration)))])))
        return metrics

    inspect(0, 0., v0, None, 'stored_prior_boundary')
    for index in range(1, count+1):
        elapsed = index*DT
        position, analytic_v, analytic_a = _polynomial(coefficients, elapsed/duration, duration)
        # The sampler defines exact endpoint anchoring; audit those submitted
        # candidate positions, after separately checking the polynomial above.
        if index == count:position = p1.copy(); analytic_v = np.zeros(75); analytic_a = np.zeros(75)
        if not np.isfinite(np.r_[position, analytic_v, analytic_a]).all():
            raise ValueError('Nonfinite bridge polynomial evaluation')
        if not np.array_equal(position[preserved], p0[preserved]):
            raise ValueError('Polynomial changed a preserved nominal joint')
        velocity = (position-previous_position)/DT
        acceleration = (velocity[6:]-previous_velocity[6:])/DT
        metrics = inspect(index, elapsed, velocity, acceleration, 'new_command')
        if first is None:first = dict(command_time_s=document['first_new_command_time_s'],
            position=position.tolist(), finite_difference_velocity=velocity.tolist(),
            stored_prior_velocity=v0.tolist(), joint_acceleration_rad_s2=acceleration.tolist(), metrics=metrics)
        for key, value in (('joint_speed_rad_s', np.max(abs(analytic_v[6:]))),
                ('joint_acceleration_rad_s2', np.max(abs(analytic_a[6:]))),
                ('root_speed_m_s', np.linalg.norm(analytic_v[:3])),
                ('root_rotvec_speed_rad_s', np.linalg.norm(analytic_v[3:6]))):
            analytic_maxima[key] = max(analytic_maxima[key], float(value))
        previous_position = position; previous_velocity = velocity
        if progress is not None and (index % 1000 == 0 or index == count):
            progress(dict(new_command_samples_checked=index, expected_new_command_samples=count,
                failed_checks_count=len(failures)))
    # A stationary panel start follows. This is an explicit one-step numerical
    # compatibility check, not extrapolation or an additional physical sample.
    inspect(count+1, duration+DT, np.zeros(75), -previous_velocity[6:]/DT, 'prospective_terminal_hold')
    _merge_hashes(hashes)
    if content_sha256(candidate) != original_sha:raise ValueError('Bridge candidate changed during audit')
    return dict(schema=SCHEMA, candidate_content_sha256=original_sha,
        source_run=seed['source_run'], source_time_s=seed['source_time_s'],
        source_context_sha256=seed['source_context_sha256'], seed_sha256=document['seed_sha256'],
        input_sha256=hashes, limits=copy.deepcopy(LIMITS), dt_s=DT, duration_s=duration,
        complete=True, expected_new_command_samples=count, checked_new_command_samples=count,
        boundary_checks=1, prospective_terminal_hold_checks=1, total_checked_samples=count+2,
        failed_samples=len(failures), failures=failures, passed=not failures,
        discrete_rate_checks_passed=not failures, first_new_command=first,
        maximum_discrete_rates=maxima, sampled_analytic_derivative_maxima=analytic_maxima,
        continuous_derivative_bound_proved=False, geometry_samples=0,
        preserved_polynomial_derivative_is_zero=True,
        stored_prior_preserved_velocities=v0[preserved].tolist(),
        scope='Complete finite-difference 500 Hz candidate accounting plus one prospective stationary-terminal check. Analytic maxima are sampled diagnostics only. No geometry, motor or physical qualification.',
        **AUTHORITY)
