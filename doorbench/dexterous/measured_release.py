"""Relocate a measured release without inventing a new finger-opening pose.

These transforms describe privileged teacher targets. The caller must use
bounded robot motors and re-qualify the resulting physical motion.
"""
import numpy as np


def relocate_release(source_position, source_rotation, initial_position,
                     initial_rotation, target_position, target_rotation):
    """Preserve motion in the source initial palm frame at a new grasp pose.

    The source positions/rotations must share one frame, usually the handle
    body frame. Target pose can be in any frame, usually the actual leaf frame
    captured when release begins. All rotation matrices map local to parent.
    """
    vectors = [np.asarray(value, dtype=float) for value in
        (source_position, initial_position, target_position)]
    rotations = [np.asarray(value, dtype=float) for value in
        (source_rotation, initial_rotation, target_rotation)]
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in vectors):
        raise ValueError("Expected finite release positions")
    if any(value.shape != (3, 3) or not np.isfinite(value).all() or
           not np.allclose(value.T @ value, np.eye(3), atol=1e-6) or
           abs(np.linalg.det(value) - 1.) > 1e-6 for value in rotations):
        raise ValueError("Expected proper local-to-parent release rotations")
    source, initial, target = vectors
    source_r, initial_r, target_r = rotations
    delta_position = initial_r.T @ (source - initial)
    delta_rotation = initial_r.T @ source_r
    return target + target_r @ delta_position, target_r @ delta_rotation
