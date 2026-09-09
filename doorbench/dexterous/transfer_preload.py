"""Explicit experimental contact preloads; motor and contact limits stay fixed."""
import math
from .operation_teacher import smooth_phase


PROFILES = {
    'maintain': None,
    'balanced-4n': dict(ff=4., mf=4., rf=4., lf=4., th=8.),
    'index-6n': dict(ff=6., mf=4., rf=4., lf=4., th=8.),
}


def transfer_preload(initial, elapsed, profile):
    if profile not in PROFILES:
        raise ValueError('Unknown standing-transfer preload profile')
    if set(initial) != {'ff', 'mf', 'rf', 'lf', 'th'} or not all(
        math.isfinite(v) and 0 <= v <= 8 for v in initial.values()
    ) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Complete finite bounded preload and transfer clock required')
    target = PROFILES[profile]
    if target is None:
        return dict(initial)
    blend = float(smooth_phase(elapsed))
    return {digit: force + blend * (target[digit] - force)
            for digit, force in initial.items()}
