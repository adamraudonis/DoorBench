"""Shared authored construction dimensions, in metres (not OEM tolerances)."""

# Source multipoint case depth measured inward from its latch edge. Glazing
# leaves ten additional millimetres of intact stock behind this preparation.
MULTIPOINT_CASE_DEPTH_M = .135
MULTIPOINT_GLAZING_STOCK_WEB_M = .010

# Paired butt-hinged leaves retain structural jamb clearance under the small
# native closing-stop compliance. Meeting gaps are specified independently.
PAIRED_JAMB_GAP_M = .004

FLOOR_STRIKE_TOP_M = {"none":0., "saddle":.013, "sill":.013, "sill_step":.045, "coaming":0.}
