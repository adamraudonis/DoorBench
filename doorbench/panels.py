"""Panel & glazing layouts shared by the spec sampler (mass) and the geometry
builder (meshes).  Leaf-local coordinates: x across the leaf from the hinge
edge (0..W), z up from the leaf bottom (0..H).  Rectangles are (x0, z0, w, h).
"""
from __future__ import annotations

from typing import List, Tuple

Rect = Tuple[float, float, float, float]


def _grid(x0, z0, w, h, cols, rows, gap) -> List[Rect]:
    cw = (w - (cols - 1) * gap) / cols
    rh = (h - (rows - 1) * gap) / rows
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append((x0 + c * (cw + gap), z0 + r * (rh + gap), cw, rh))
    return out


def glazing_layout(style: str, W: float, H: float) -> List[Rect]:
    """Glass rectangles for a panel style (empty list if no glazing)."""
    stile = min(max(0.09, 0.11 * W), 0.28 * W)      # vertical frame member width
    rail_b = max(0.18, 0.11 * H)     # bottom rail
    rail_t = max(0.09, 0.05 * H)     # top rail
    inner_w = W - 2 * stile
    inner_h = H - rail_b - rail_t
    if style in ("glass_full", "glass_sidelite_style"):
        return [(stile, rail_b, inner_w, inner_h)]
    if style == "glass_frameless":
        return [(0.0, 0.0, W, H)]
    if style == "glass_half":
        return [(stile, H * 0.52, inner_w, H - rail_t - H * 0.52)]
    if style == "glass_vision":
        vw = min(0.15, W * 0.2)
        return [(W - stile - vw - 0.03, H * 0.42, vw, H * 0.40)]
    if style == "steel_vision":
        vw = min(0.25, W * 0.3)
        return [(W / 2 - vw / 2, H * 0.55, vw, H * 0.30)]
    if style == "steel_half_glass":
        return [(stile, H * 0.5, inner_w, H * 0.42)]
    if style == "glass_15_lite":
        return _grid(stile, rail_b, inner_w, inner_h, 3, 5, 0.03)
    if style == "glass_10_lite":
        return _grid(stile, rail_b, inner_w, inner_h, 2, 5, 0.03)
    if style == "glass_6_lite":
        return _grid(stile, rail_b, inner_w, inner_h, 2, 3, 0.03)
    if style == "glass_9_lite":
        gh = inner_h * 0.5
        return _grid(stile, H - rail_t - gh, inner_w, gh, 3, 3, 0.03)
    if style == "glass_1_lite_top":
        gh = inner_h * 0.35
        return [(stile, H - rail_t - gh, inner_w, gh)]
    if style == "glass_oval":
        gw, gh = inner_w * 0.55, inner_h * 0.55
        return [(W / 2 - gw / 2, H * 0.30, gw, gh)]
    if style == "glass_fan":
        gw, gh = inner_w * 0.8, inner_h * 0.22
        return [(W / 2 - gw / 2, H - rail_t - gh, gw, gh)]
    if style == "porthole":
        d = min(0.25, W * 0.35)
        return [(W / 2 - d / 2, H * 0.6, d, d)]
    if style == "sectional_long_windows":
        return _grid(0.05, H * 0.78, W - 0.1, H * 0.16, 4, 1, 0.04)
    if style == "louver_half":
        return []
    return []


def glazing_area_fraction(rects: List[Rect], W: float, H: float) -> float:
    a = sum(r[2] * r[3] for r in rects)
    if any(r[2] >= W - 1e-6 for r in rects):
        return min(a / (W * H), 0.95)
    return min(a / (W * H), 0.9)


def raised_panel_layout(style: str, W: float, H: float) -> List[Rect]:
    """Recessed/raised panel rectangles (decorative relief), leaf-local."""
    stile = min(max(0.09, 0.11 * W), 0.28 * W)
    rail_b = max(0.18, 0.11 * H)
    rail_t = max(0.09, 0.05 * H)
    inner_w = W - 2 * stile
    inner_h = H - rail_b - rail_t
    mid = 0.09
    if style in ("6_panel", "steel_embossed_6"):
        rows = [0.30, 0.30, 0.40]  # bottom, middle, top fractions (top row taller in colonial)
        out = []
        z = rail_b
        for i, f in enumerate(rows):
            rh = (inner_h - 2 * mid) * f
            out += _grid(stile, z, inner_w, rh, 2, 1, mid)
            z += rh + mid
        return out
    if style == "4_panel":
        return _grid(stile, rail_b, inner_w, inner_h, 2, 2, mid)
    if style in ("2_panel", "shaker_2"):
        return _grid(stile, rail_b, inner_w, inner_h, 1, 2, mid)
    if style == "2_panel_arch":
        return _grid(stile, rail_b, inner_w, inner_h, 1, 2, mid)
    if style in ("3_panel", "shaker_3"):
        return _grid(stile, rail_b, inner_w, inner_h, 1, 3, mid)
    if style == "5_panel_horizontal":
        return _grid(stile, rail_b, inner_w, inner_h, 1, 5, mid)
    if style == "shaker_1":
        return [(stile, rail_b, inner_w, inner_h)]
    if style in ("carved_ornate",):
        return _grid(stile, rail_b, inner_w, inner_h, 1, 2, mid)
    if style == "raised_carriage":
        return _grid(0.08, 0.08, W - 0.16, H - 0.16, max(2, int(W / 0.6)), 2, 0.06)
    if style in ("sectional_raised_short",):
        return _grid(0.05, 0.03, W - 0.1, H - 0.06, max(4, int(W / 0.55)), 4, 0.04)
    if style == "sectional_flush":
        return []
    if style == "padded_diamond":
        return _grid(0.06, 0.06, W - 0.12, H - 0.12, 3, 6, 0.02)
    return []


def louver_slats(style: str, W: float, H: float):
    """(x0, z0, w, h) region for louvered infill, plus slat count."""
    stile = min(max(0.09, 0.11 * W), 0.28 * W)
    rail_b = max(0.18, 0.11 * H)
    rail_t = max(0.09, 0.05 * H)
    if style == "louver_full":
        r = (stile, rail_b, W - 2 * stile, H - rail_b - rail_t)
    elif style == "louver_half":
        r = (stile, H * 0.52, W - 2 * stile, H - rail_t - H * 0.52)
    elif style == "steel_louvered":
        r = (W * 0.25, 0.25, W * 0.5, 0.45)
    else:
        return None, 0
    n = max(4, int(r[3] / 0.05))
    return r, n
