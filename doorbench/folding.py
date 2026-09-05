"""Folding doors (bifold / accordion): mechanism constants and the closed-state clearances the zigzag needs.

Shared by the spec generator (which sizes the opening around the panel set) and the geometry builder (which places
the panels in it), so the two cannot drift apart.

The mechanism: panel 0 turns on a jamb pivot FOLD_PIVOT_IN inside its edge; every further panel hangs on a piano /
butt hinge along the previous panel's edge with the axis on the pair of faces that close onto each other - the axis
alternates between the two faces of the door from hinge to hinge, which is what lets the panels stack face to face.
Every second hinge line rides in the top track on a glide, and for equal panels the track constraint makes the hinge
angles q_k = fold_coupling(k) * q_pivot (-2 for odd k, +2 for even k).

Because every panel link (hinge axis to hinge axis) is tilted by its thickness, the chain first LENGTHENS as it starts
to fold: the lead edge moves out along the track by fold_lead_excursion() before the W (1 - cos q) shortening wins.
The closed lead gap has to swallow that, or the lead edge jams on the strike jamb a few degrees into the travel (the
10-panel 18 mm doors need ~17 mm).
"""
from __future__ import annotations

import math

FOLD_TRACK_H = 0.03        # top track channel height (m), mounted under the head jamb
FOLD_TRACK_GAP = 0.005     # panel tops hang this far below the track underside (glide pins are inside the channel)
FOLD_FLOOR_GAP = 0.02      # panel bottoms above the floor
FOLD_HINGE_GAP = 0.002     # panel edge set back from the hinge axis on each side (the knuckle sits in the gap)
FOLD_PIVOT_MAX_DEG = 85.0  # pivot travel at the stack: adjacent panels fold to 170 deg (knuckles / edges stop them short of flat)
FOLD_PIVOT_IN = 0.035      # jamb pivot pin inside the pivot panel's edge
FOLD_JAMB_GAP = 0.005      # pivot panel edge to the pivot jamb (closed)
FOLD_LEAD_GAP = 0.006      # minimum lead edge to strike jamb gap (closed); half of it per leaf when two stacks meet
FOLD_LEAD_MARGIN = 0.003   # clearance kept beyond the lead excursion


def fold_coupling(k: int) -> float:
    """Track constraint for equal panels: hinge k (between panel k-1 and panel k) turns q_k = fold_coupling(k) * q_pivot."""
    return -2.0 if k % 2 == 1 else 2.0


def fold_hinge_range(k: int) -> tuple:
    """Half-turn range of a driven panel hinge on the side its coupling drives it to (never fights the equality)."""
    return (-math.pi, 0.0) if fold_coupling(k) < 0 else (0.0, math.pi)


def fold_lead_excursion(n: int, W: float, t: float, pivot_in: float = FOLD_PIVOT_IN) -> float:
    """Largest outward travel (m) of the lead edge of an n-panel face-hinged stack while it folds.

    Lead edge x from the pivot = (nW - pivot_in) cos q + (n - 1/2) t sin q: the alternating face offsets lengthen the
    chain by (n - 1/2) t sin q before the W (1 - cos q) shortening wins.  Maximum at tan q* = A / B."""
    A, B = (n - 0.5) * t, n * W - pivot_in
    return max(0.0, math.hypot(A, B) - B)


def fold_lead_gap(n: int, W: float, t: float) -> float:
    """Closed gap between the lead edge of an n-panel stack and what it closes against."""
    return max(FOLD_LEAD_GAP, fold_lead_excursion(n, W, t) + FOLD_LEAD_MARGIN)


def fold_meeting_gap(n_per_stack: int, W: float, t: float) -> float:
    """Closed gap between the meeting line of two stacks and each lead edge (one half of the meeting gap)."""
    return max(FOLD_LEAD_GAP / 2, fold_lead_gap(n_per_stack, W, t) - 0.001)


def fold_groups(n: int, accordion: bool) -> int:
    """Number of stacks: an accordion or a 2-panel bifold folds to one jamb, a 4-panel bifold to both."""
    return 1 if (accordion or n == 2) else 2


def fold_opening_width(W: float, n: int, n_groups: int, t: float) -> float:
    """Clear opening width for n panels of width W in n_groups stacks: a pivot-jamb gap per stack plus the lead gap
    (one stack: at the strike jamb; two stacks: the meeting gap at the centre)."""
    per = n // n_groups
    if n_groups == 1:
        return n * W + FOLD_JAMB_GAP + fold_lead_gap(per, W, t)
    return n * W + 2 * FOLD_JAMB_GAP + 2 * fold_meeting_gap(per, W, t)


def fold_opening_height(Hh: float) -> float:
    """Clear opening height: panels + floor gap + track clearance + the track under the head jamb."""
    return Hh + FOLD_FLOOR_GAP + FOLD_TRACK_GAP + FOLD_TRACK_H
