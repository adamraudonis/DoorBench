"""Detached actual PhysX contacts for privileged post-opening control.

The actor never receives these identity-bearing evaluator measurements. Normal
contacts and friction patches remain separate buffers; no index pairing between
them is assumed. Hand force vectors come from contact_force_pairs separately.
"""
import numpy as np


def continuation_contact_summary(sensor_paths, filter_paths, normal_forces,
                                 normals, distances, counts, starts, *, capacity,
                                 physics_qualified):
    if not isinstance(physics_qualified, (bool, np.bool_)):
        raise ValueError('Explicit physical qualification is required')
    paths = tuple(sensor_paths)
    filters = np.asarray(filter_paths, str)
    counts, starts = np.asarray(counts), np.asarray(starts)
    forces, normals, distances = (np.asarray(v, float) for v in
                                  (normal_forces, normals, distances))
    if (len(paths) != len(set(paths)) or filters.ndim != 2 or
            filters.shape[0] != len(paths) or counts.shape != filters.shape or
            starts.shape != filters.shape or counts.dtype.kind not in 'iu' or
            starts.dtype.kind not in 'iu' or np.any(counts < 0) or
            np.any(starts < 0) or forces.shape != (capacity, 1) or
            normals.shape != (capacity, 3) or distances.shape != (capacity, 1)):
        raise ValueError('Malformed actual normal-contact buffer')
    if counts.sum() >= capacity:
        raise ValueError('Potentially truncated actual contact buffer')
    names = [p.rsplit('/', 1)[-1] for p in paths]
    feet_names = ('left_ankle_link', 'right_ankle_link')
    if any(names.count(n) != 1 for n in (*feet_names, 'lh_palm', 'rh_palm')):
        raise ValueError('Actual feet and palm sensor rows are required')
    feet = np.zeros(2)
    left_count = right_count = 0
    left_load = 0.
    panel_normals = []
    occupied = set()
    for i, j in np.ndindex(counts.shape):
        count, start = int(counts[i, j]), int(starts[i, j])
        if not count:
            continue
        if start + count > capacity:
            raise ValueError('Normal contact slice exceeds capacity')
        slots = set(range(start, start + count))
        if slots & occupied:
            raise ValueError('Overlapping normal contacts would double-count load')
        occupied.update(slots)
        name, other = names[i], filters[i, j]
        for k in range(start, start + count):
            load, normal, gap = float(forces[k, 0]), normals[k], float(distances[k, 0])
            if (not np.isfinite([load, gap, *normal]).all() or load < 0 or
                    not np.isclose(np.linalg.norm(normal), 1., atol=1e-5, rtol=0)):
                raise ValueError('Invalid measured normal contact')
            if name in feet_names and other.rsplit('/', 1)[-1] == 'floor':
                feet[feet_names.index(name)] += load * normal[2]
            if name.startswith('lh_'):
                # Both sensors observe a self-contact. Count/load it only once,
                # while retaining separate physical forces on each hand body.
                duplicate = (other.rsplit('/', 1)[-1].startswith('lh_') and
                             other in paths and paths[i] > other)
                if not duplicate:
                    left_count += int(gap <= 0 or load > 1e-8)
                    left_load += load
                if (other == '/World/Door/Articulation/leaf' and
                        (gap <= 0 or load > 1e-8)):
                    panel_normals.append(normal.copy())
            if (name.startswith('rh_') and not other.startswith('/World/H1/') and
                    (gap <= 0 or load > 1e-8)):
                right_count += 1
    normal = None
    if panel_normals:
        mean = np.mean(panel_normals, axis=0)
        if np.linalg.norm(mean) < 1e-6:
            raise ValueError('Ambiguous actual panel release direction')
        normal = mean / np.linalg.norm(mean)
    return dict(foot_loads=feet, evidence=dict(
        physics_qualified=bool(physics_qualified), left_hand_contacts=left_count,
        left_hand_load_N=left_load, right_environment_contacts=right_count),
        release_normal_world=normal)
