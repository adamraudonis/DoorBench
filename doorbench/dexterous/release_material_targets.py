"""Source-measured phalange targets for a prospective geometric release.

Force only weights measured locations within a segment. No future force or
successful release is inferred from these material points.
"""
import numpy as np


def loaded_segment_targets(contacts, digit):
    if digit not in ('ff', 'mf', 'rf', 'lf', 'th'):
        raise ValueError('Known measured digit required')
    grouped = {}
    for contact in contacts:
        if contact['digit'] != digit:
            continue
        force = float(contact['normal_force_N'])
        point = np.asarray(contact['body_position_m'], dtype=float)
        if not np.isfinite(force) or force < 0 or point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Finite measured material contact required')
        if force <= 1e-6:
            continue
        if contact['pad_qualified'] is not True:
            raise ValueError('Release source contains an invalid loaded patch')
        body = contact['body']
        if not body.startswith('robot/rh_' + digit):
            raise ValueError('Measured body and digit disagree')
        grouped.setdefault(body, []).append((force, point))
    if not grouped:
        raise ValueError('Every release digit needs an actual loaded material segment')
    result = []
    for body, patches in sorted(grouped.items()):
        total = sum(force for force, _ in patches)
        point = sum(force * point for force, point in patches) / total
        result.append(dict(body=body, body_position_m=point.tolist(), source_force_N=total))
    return result
