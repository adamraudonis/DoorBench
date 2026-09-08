"""Conservative sampled configuration-path connectivity for release screening.

Rows advance hand-ungrip progress; columns increase lever angle. A return path
may advance one row or decrease one column at a time. Disconnected individually
feasible frames are rejected. A returned grid path still needs swept-edge
refinement and an actual force-controlled physical trial.
"""
import numpy as np


def monotone_release_path(free):
    """Return adjacent (hand row, lever column) nodes or None; no corner jumps.

    Starts at the first hand pose with the maximally pressed lever and ends at
    the last hand pose with the resting lever. Inputs must be a literal finite
    Boolean matrix. Numeric costs or NaNs must be thresholded/handled explicitly
    by the caller, rather than silently interpreted as feasibility.
    """
    free = np.asarray(free)
    if free.ndim != 2 or min(free.shape) < 2 or free.dtype != np.bool_:
        raise ValueError('Expected a Boolean hand-progress/lever-angle grid')
    nr, nc = free.shape
    if not free[0, -1] or not free[-1, 0]:
        return None
    parent = np.full((nr, nc, 2), -1, dtype=np.int64)
    reached = np.zeros_like(free)
    reached[0, -1] = True
    for row in range(nr):
        for col in range(nc-1, -1, -1):
            if not free[row, col] or reached[row, col]:
                continue
            if row > 0 and reached[row-1, col]:
                reached[row, col] = True
                parent[row, col] = row-1, col
            elif col < nc-1 and reached[row, col+1]:
                reached[row, col] = True
                parent[row, col] = row, col+1
    if not reached[-1, 0]:
        return None
    node = (nr-1, 0)
    path = [node]
    while node != (0, nc-1):
        node = tuple(int(x) for x in parent[node])
        path.append(node)
    path.reverse()
    return path
