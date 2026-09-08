"""Read-only archive verification of the opt-in passive-joint adapter.

Reconstruct original coefficients by name, independently of the runtime profile
builder. A persistence receipt summarizes runtime queries; it is not a second
measurement of the simulator, nor a claim of cross-engine dynamic equivalence.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

PROPERTY_ORDER = ['static_friction_effort_Nm', 'dynamic_friction_effort_Nm',
                  'viscous_friction_Nm_s_per_rad']


def audit_joint_passive_evidence(run, steps, declared):
    run = Path(run)
    errors = []
    hashes = {}

    def read(name):
        data = (run / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    profile = None
    complete = None
    receipt = None
    try:
        config = read('configuration.json')
        selected = config.get('args', {}).get('joint_passive_profile')
        reported = declared.get('joint_passive_profile')
        has_start = (run / 'joint-passive-profile.json').exists()
        has_guard = (run / 'joint-passive-invariants.json').exists()
        start = read('joint-passive-profile.json') if has_start else None
        labels = [p for p in (selected, reported, None if start is None else start.get('profile')) if p is not None]
        require(all(p in ('legacy-tanh-v1', 'backend-dry-v2') for p in labels), 'Unknown passive profile label')
        require(len(set(labels)) <= 1, 'Configuration, report and startup passive profiles disagree')
        profile = labels[0] if labels else 'legacy-unversioned'
        if profile != 'backend-dry-v2':
            require(not has_guard, 'Persistence evidence cannot be attributed to a legacy profile')
            # Historical captures predate both the option and its receipt. Their
            # original qualification is still auditable, with no new claim.
            return dict(verification_passed=True, profile=profile,
                persistence_complete=None, errors=[], source_sha256=hashes,
                limitation='Legacy archive: backend passive persistence was not recorded or independently established.')

        require(selected == reported == 'backend-dry-v2' and has_start and has_guard,
                'Opt-in dry profile requires explicit configuration, report, startup and invariant receipts')
        motors = read('motor-contract.json')
        provenance = read('provenance.json')
        require(provenance.get('files', {}).get(config['args']['motors']) == hashes['motor-contract.json'],
                'Original motor contract is not bound to pre-step provenance')
        names = config.get('robot_joint_names')
        require(type(names) is list and len(names) == 69 and all(type(n) is str for n in names) and len(set(names)) == 69,
                'Require exact 69-name backend joint order')
        require(type(motors.get('joint_names')) is list and len(motors['joint_names']) == 69 and
                len(set(motors['joint_names'])) == 69 and set(names) == set(motors['joint_names']) == set(motors['passive']),
                'Backend order must cover every original named passive joint exactly once')
        values = np.asarray([[motors['passive'][n][k] for k in
                             ('friction', 'damping', 'armature', 'stiffness', 'springref')] for n in names], float)
        require(values.shape == (69, 5) and np.isfinite(values).all() and np.all(values[:, :4] >= 0) and
                np.all(values[:, 3] == 0), 'Invalid original passive coefficients or unsupported spring')
        expected = np.column_stack([values[:, 0], values[:, 0], values[:, 1]]).astype(np.float32).astype(float)
        armature = values[:, 2].astype(np.float32).astype(float)

        def exact(value, target, name):
            actual = np.asarray(value, float)
            require(actual.shape == target.shape and np.isfinite(actual).all() and np.array_equal(actual, target),
                    name + ' differs from exact original float32 coefficients')

        require(start['joint_names'] == names and start['property_order'] == PROPERTY_ORDER,
                'Startup joint/property order mismatch')
        exact(start['backend_friction_properties'], expected, 'Startup friction properties')
        exact(start['native_armature_readback_kg_m2'], armature, 'Startup armature')
        exact(start['explicit_damping'], np.zeros(69), 'Duplicate explicit damping')
        exact(start['explicit_friction'], np.zeros(69), 'Duplicate explicit friction')
        receipt = read('joint-passive-invariants.json')
        require(receipt.get('schema') == 'doorbench.passive-property-invariant.v1' and
                receipt.get('profile') == 'backend-dry-v2', 'Wrong passive invariant schema/profile')
        require(receipt['joint_names'] == names and receipt['property_order'] == PROPERTY_ORDER,
                'Invariant joint/property order mismatch')
        exact(receipt['expected_float32_properties'], expected, 'Invariant friction properties')
        exact(receipt['expected_float32_armature_kg_m2'], armature, 'Invariant armature')
        require(receipt.get('physics_dt_s') == .002 and config.get('dt') == .002,
                'Passive persistence requires original 2ms intervals')
        counts = [receipt.get(k) for k in ('attempted_intervals', 'checked_intervals', 'valid_intervals')]
        require(all(type(n) is int and n >= 0 for n in counts), 'Malformed invariant interval counts')
        require(receipt.get('passed') is True and receipt.get('first_failure') is None,
                'Actual passive-property persistence failed; retain original failed prefix')
        exact(receipt['maximum_absolute_property_difference'], np.zeros(3), 'Per-step friction drift')
        require(type(receipt.get('maximum_armature_difference_kg_m2')) in (int, float) and
                receipt['maximum_armature_difference_kg_m2'] == 0., 'Per-step armature changed')
        duration = declared.get('expected_duration_s')
        require(type(duration) in (int, float) and np.isfinite(duration) and duration > 0 and
                abs(duration / .002 - round(duration / .002)) < 1e-8, 'Missing declared physical duration')
        total = round(duration / .002)
        times = np.asarray([r['time_s'] for r in steps], float)
        require(counts == [total] * 3 and len(steps) == total and total > 0 and
                np.isfinite(times).all() and np.allclose(times, np.arange(1, total + 1) * .002, atol=1e-8, rtol=0),
                'Passive receipt does not cover every actual and requested 2ms interval')
        require(type(receipt.get('last_checked_interval_end_s')) in (int, float) and
                abs(receipt['last_checked_interval_end_s'] - duration) <= 1e-8,
                'Passive last checked endpoint differs from complete actual duration')
        # Bind the implementation responsible for these query counts to both
        # its immutable run-local copy and the pre-step source manifest.
        source_name = 'source-isaac_joint_passive.py'
        source_bytes = (run / source_name).read_bytes()
        hashes[source_name] = hashlib.sha256(source_bytes).hexdigest()
        source_hashes = [v for k, v in provenance['files'].items()
                         if k.endswith('/doorbench/dexterous/isaac_joint_passive.py')]
        require(source_hashes == [hashes[source_name]], 'Passive guard source is not uniquely bound to pre-step provenance')
        complete = True
    except (ValueError, KeyError, TypeError, OverflowError, OSError) as exc:
        errors.append(type(exc).__name__ + ': ' + str(exc))
        if profile == 'backend-dry-v2':
            complete = False
    return dict(verification_passed=not errors, profile=profile,
        persistence_complete=complete, interval_receipt=receipt, errors=errors, source_sha256=hashes,
        limitation='Checks archived original coefficients, readbacks and full interval guard receipt; no simulator stepping or new contact/dynamic equivalence claim.')
