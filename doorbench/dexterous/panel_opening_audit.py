"""Independent evaluator for the opt-in physical panel recontact experiment."""
import numpy as np


def audit_panel_opening(rows, opening, mechanical):
    """Reject mere post-release coasting, incomplete mechanics, and missing data."""
    contacts = []
    for row in rows:
        phase = row.get('teacher', {}).get('panel_phase')
        forces = row.get('hand_forces_panel_N', {})
        load = sum(float(np.linalg.norm(f)) for f in forces.values())
        if phase in ('reach', 'push', 'complete') and np.isfinite(load) and load > 1.:
            contacts.append((float(row['time_s']), float(row['door']['leaf_hinge']), load))
    loaded_angles = [c[1] for c in contacts]
    loaded_travel = 0.
    prior_minimum = float('inf')
    for angle in loaded_angles:
        loaded_travel = max(loaded_travel, angle-prior_minimum)
        prior_minimum = min(prior_minimum, angle)
    first = loaded_angles[0] if loaded_angles else None
    maximum = max(loaded_angles) if loaded_angles else None
    checks = {
        'original_opening_gate': bool(opening.get('passed', False)),
        'complete_mechanical_gate': bool(mechanical.get('passed', False)),
        'panel_recontact_recorded': len(contacts) >= 3,
        'contact_before_usable_aperture': first is not None and first < .7,
        'usable_aperture_under_contact': maximum is not None and maximum >= .7,
        'substantial_loaded_panel_travel': loaded_travel >= .3,
    }
    return dict(scope='Initialized privileged motor teacher; physical panel recontact, not approach/traversal or a sensor-only policy',
        passed=all(checks.values()), checks=checks, loaded_panel_samples=len(contacts),
        first_loaded_panel_angle_rad=first, max_loaded_panel_angle_rad=maximum,
        loaded_panel_travel_rad=loaded_travel)
