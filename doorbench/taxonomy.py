"""The DoorBench taxonomy: every family of human/animal-passable door we model,
with quotas and the option spaces the sampler draws from.

The sampler (spec.py) performs *balanced* coverage: for each family, every
discrete dimension cycles through independent shuffled permutations of its
levels, so all levels appear ~equally and combinations vary.  Continuous
parameters are drawn from the family's ranges with a fixed seed.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Kinematic families
# ---------------------------------------------------------------------------
FAMILIES = {
    # name: (quota, kinematics, description)
    "swing_single":        (440, "hinge",     "Single-leaf hinged door (residential, commercial, fire, institutional, industrial, security, storefront)"),
    "swing_double":        (76,  "hinge",     "Pair of hinged leaves (french, commercial pair w/ panic hardware, double egress, storefront pair)"),
    "dutch":               (12,  "hinge",     "Dutch door: independently hinged upper & lower halves with joining bolt"),
    "saloon":              (12,  "hinge",     "Double-acting spring-hinged leaves swinging both ways (cafe / saloon / kitchen pass)"),
    "pivot":               (20,  "hinge",     "Pivot door: center or offset floor pivot, often oversized / heavy"),
    "sliding_single":      (100, "slide",     "Single sliding leaf: pocket, barn (surface track), patio glass, shoji/fusuma, cell, industrial"),
    "sliding_bypass":      (35,  "slide",     "Two or three overlapping leaves on parallel tracks (closet, mirrored, shoji pair)"),
    "bifold":              (30,  "hinge",     "Bi-fold closet doors (2 or 4 panels) pivoting with guided free edge"),
    "accordion":           (12,  "hinge",     "Accordion / concertina folding partition door"),
    "revolving":           (15,  "hinge",     "Revolving door: 3 or 4 wings on a central rotor inside a drum"),
    "turnstile_tripod":    (10,  "hinge",     "Waist-high tripod turnstile (ratcheting rotor)"),
    "turnstile_fullheight":(10,  "hinge",     "Full-height rotating turnstile (3-4 wing rotor in cage)"),
    "garage_sectional":    (18,  "slide",     "Overhead sectional garage door (vertical lift approximated)"),
    "garage_tiltup":       (7,   "hinge",     "One-piece tilt-up garage door (offset pivot)"),
    "rollup":              (15,  "slide",     "Roll-up / coiling steel curtain or grille"),
    "pet_door":            (15,  "hinge",     "Standalone pet door flap in a wall/door panel (dog & cat sizes)"),
    "hatch_floor":         (10,  "hinge",     "Floor hatch / cellar trapdoor (horizontal hinge, lift up)"),
    "hatch_ceiling":       (8,   "hinge",     "Ceiling / attic hatch, roof scuttle (push up)"),
    "ship_watertight":     (10,  "hinge",     "Marine watertight door with dogging levers or central wheel"),
    "vault":               (8,   "hinge",     "Vault / safe-room door with handwheel boltwork"),
    "blast":               (6,   "hinge",     "Blast door (very heavy, multi-hinge, lever bolts)"),
    "gate_swing":          (40,  "hinge",     "Outdoor swing gates: picket, chain-link, wrought iron, pool, ranch"),
    "gate_sliding":        (10,  "slide",     "Cantilever / track sliding vehicle & pedestrian gates"),
    "baby_gate":           (10,  "hinge",     "Pressure- or hardware-mounted child safety gate"),
    "stall":               (15,  "hinge",     "Toilet partition stall door (gravity hinge, slide latch)"),
    "strip_curtain":       (8,   "hinge",     "PVC strip curtain doorway (many hinged strips)"),
    "cold_storage":        (15,  "hinge",     "Walk-in cooler / freezer doors (cam-lift hinges, gasket, inside release)"),
    "automatic_sliding":   (15,  "slide",     "Sensor-activated sliding doors (single / bi-parting) with manual breakout"),
    "automatic_swing":     (10,  "hinge",     "Automatic swing operators (low-energy push-to-open / full-energy)"),
    "elevator":            (8,   "slide",     "Elevator landing doors (center or side opening, interlocked)"),
}
assert sum(v[0] for v in FAMILIES.values()) == 1000, sum(v[0] for v in FAMILIES.values())

# Sub-contexts for swing_single (weights inside the family quota)
SWING_SINGLE_CONTEXTS = {
    "residential_interior": 130,
    "residential_exterior": 60,
    "commercial_office": 60,
    "fire_egress": 55,
    "institutional": 45,     # hospital, school, lab, hotel
    "industrial_utility": 25,
    "security_detention": 15,
    "storefront_glass": 25,
    "heritage_rustic": 25,
}
assert sum(SWING_SINGLE_CONTEXTS.values()) == 440

SWING_DOUBLE_CONTEXTS = {"french": 20, "commercial_pair_panic": 26, "double_egress": 10, "storefront_pair": 12, "barn_pair": 8}
assert sum(SWING_DOUBLE_CONTEXTS.values()) == 76

SLIDING_SINGLE_CONTEXTS = {"pocket": 22, "barn": 26, "patio_glass": 22, "shoji_fusuma": 16, "cell_industrial": 14}
assert sum(SLIDING_SINGLE_CONTEXTS.values()) == 100

GATE_SWING_CONTEXTS = {"garden_picket": 12, "chain_link": 8, "wrought_iron": 8, "pool_safety": 6, "ranch_tube": 6}
assert sum(GATE_SWING_CONTEXTS.values()) == 40

# ---------------------------------------------------------------------------
# Option spaces per context (slab ids, operators, latches, locks, closers, hinges, finishes ...)
# ---------------------------------------------------------------------------
PANEL_STYLES = {
    "flush": "Flush slab", "2_panel": "2-panel", "2_panel_arch": "2-panel arch top", "3_panel": "3-panel mission",
    "4_panel": "4-panel", "5_panel_horizontal": "5-panel horizontal", "6_panel": "6-panel colonial", "shaker_1": "Shaker single panel",
    "shaker_2": "Shaker 2-panel", "shaker_3": "Shaker 3-panel", "louver_full": "Full louver", "louver_half": "Half louver / half panel",
    "plank_vertical": "Vertical planks", "plank_z_brace": "Planks with Z-brace", "plank_x_brace": "Planks with X-brace", "beadboard": "Beadboard",
    "carved_ornate": "Carved / ornate raised panels", "arched_top": "Arched-top plank door", "board_batten": "Board & batten",
    "glass_full": "Full glass", "glass_half": "Half lite", "glass_vision": "Vision lite (narrow)", "glass_15_lite": "15-lite french",
    "glass_10_lite": "10-lite french", "glass_6_lite": "6-lite", "glass_9_lite": "9-lite", "glass_1_lite_top": "Single top lite (craftsman)",
    "glass_oval": "Oval lite (entry)", "glass_fan": "Fan lite", "glass_sidelite_style": "Full lite w/ grilles", "steel_flush": "Flush steel",
    "steel_embossed_6": "Embossed 6-panel steel", "steel_vision": "Steel w/ vision lite", "steel_louvered": "Steel w/ louver vent",
    "steel_half_glass": "Steel half glass (wired)", "mesh_panel": "Mesh infill", "bar_grille": "Bar grille", "pickets": "Picket infill",
    "ornamental_scroll": "Ornamental scroll infill", "lattice_shoji": "Shoji lattice (kumiko)", "raised_carriage": "Carriage-house panels",
    "sectional_raised_short": "Short raised sectional panels", "sectional_flush": "Flush sectional", "sectional_long_windows": "Long panels w/ window row",
    "corrugated_slats": "Corrugated slats", "grille_rollup": "Rolling grille pattern", "planks_diagonal": "Diagonal planks",
    "padded_diamond": "Diamond-tufted upholstery", "riveted_steel": "Riveted steel plate", "porthole": "Porthole window", "dished_plate": "Dished plate",
    "hpl_flat": "Flat laminate panel", "glass_frameless": "Frameless glass", "acrylic_flap": "Acrylic flap", "strips": "PVC strips", "canvas_flap": "Canvas flap",
}

FINISH_KINDS = ["paint", "stain", "natural", "metal_bare", "powder_coat", "anodized", "galvanized", "weathered", "laminate", "glass", "mirror", "paper"]

# Conditions modify friction/backlash/damage
CONDITIONS = {
    "new": {"friction_mult": 1.0, "backlash_add": 0.0, "damping_mult": 1.0, "sag_deg": 0.0, "stick_torque": 0.0, "label": "New / well maintained"},
    "normal": {"friction_mult": 1.3, "backlash_add": 0.01, "damping_mult": 1.0, "sag_deg": 0.0, "stick_torque": 0.0, "label": "Normal wear"},
    "worn": {"friction_mult": 2.0, "backlash_add": 0.04, "damping_mult": 0.9, "sag_deg": 0.3, "stick_torque": 0.5, "label": "Worn, loose hardware"},
    "old_dry": {"friction_mult": 3.5, "backlash_add": 0.06, "damping_mult": 0.8, "sag_deg": 0.5, "stick_torque": 1.5, "label": "Old, dry hinges (squeaks)"},
    "rusty": {"friction_mult": 6.0, "backlash_add": 0.03, "damping_mult": 1.5, "sag_deg": 0.8, "stick_torque": 4.0, "label": "Rusty / corroded"},
    "swollen": {"friction_mult": 1.6, "backlash_add": 0.02, "damping_mult": 1.0, "sag_deg": 0.4, "stick_torque": 12.0, "label": "Swollen wood, rubs frame (sticks)"},
    "sagging": {"friction_mult": 2.5, "backlash_add": 0.05, "damping_mult": 1.0, "sag_deg": 1.2, "stick_torque": 6.0, "label": "Sagging, drags on floor/frame"},
    "damaged": {"friction_mult": 2.0, "backlash_add": 0.10, "damping_mult": 0.7, "sag_deg": 0.6, "stick_torque": 2.0, "label": "Damaged (dented / cracked)"},
    "well_oiled": {"friction_mult": 0.7, "backlash_add": 0.0, "damping_mult": 1.0, "sag_deg": 0.0, "stick_torque": 0.0, "label": "Freshly lubricated"},
}

# Robot-facing benchmark tasks
TASKS = {
    "open_and_traverse": "Approach, operate hardware, open, pass through to the far side",
    "open_only": "Operate hardware and open the door past the clearance angle/travel",
    "traverse_open": "Door starts open; walk through without touching it",
    "close": "Door starts open; close and latch it",
    "unlock_open_traverse": "Door is locked with a robot-side release (thumbturn, keypad code, REX button, slide bolt); unlock, open, traverse",
    "locked_recognize": "Door is locked with no robot-side release; robot should try, recognize, and stop without damage",
    "push_through": "Free-swinging / spring-return door (saloon, strip curtain, pet flap, turnstile, revolving): walk through",
    "hold_and_pass": "Self-closing door: open, hold, pass through before it closes",
    "peek": "Open door partially (e.g. 10-20 deg) and hold, without letting it swing further",
}

# Extras (attachments) and which families they may appear on
EXTRAS = {
    "kick_plate": ["swing_single", "swing_double", "cold_storage", "automatic_swing"],
    "armor_plate": ["swing_single"],
    "push_plate": ["swing_single", "swing_double"],
    "peephole": ["swing_single"],
    "mail_slot": ["swing_single"],
    "knocker": ["swing_single"],
    "house_number": ["swing_single"],
    "pet_flap": ["swing_single"],
    "chain_lock": ["swing_single"],
    "swing_bar_guard": ["swing_single"],
    "exit_sign": ["swing_single", "swing_double", "automatic_swing", "automatic_sliding"],
    "push_pull_sign": ["swing_single", "swing_double"],
    "vision_lite_grille": ["swing_single"],
    "door_stop_floor": ["swing_single", "swing_double", "pivot"],
    # "door_stop_wall" was here and no builder ever drew one.  It cannot be drawn either: a wall bumper only
    # reaches a leaf that folds back flat against its own wall, and every hinged door in this dataset is capped at
    # 135-140 deg by its casing (spec.make_specs).  The 21 doors that declared it now declare "door_stop_floor",
    # which is the stop those doors really take and which is drawn.  See doorbench/spec_realized.py.
    "hold_open_kickdown": ["swing_single"],
    "wreath": ["swing_single"],
    "keypad_reader_wall": ["swing_single", "swing_double", "automatic_sliding", "gate_swing", "gate_sliding", "elevator"],
    "rex_button": ["swing_single", "swing_double"],
    "wave_sensor": ["automatic_swing", "automatic_sliding"],
    "call_button": ["elevator"],
    "threshold_saddle": ["swing_single", "swing_double", "cold_storage", "sliding_single"],
    "weather_drip_cap": ["swing_single"],
    "door_viewer_camera": ["swing_single"],
    "coat_hook": ["swing_single", "stall"],
    "bumper_rail": ["swing_single", "swing_double"],
    "louver_vent": ["swing_single"],
    "transom_window": ["swing_single", "swing_double"],
    "sidelite": ["swing_single", "swing_double", "pivot"],
    "warning_placard": ["cold_storage", "ship_watertight", "vault", "blast", "industrial_utility"],
    "floor_guide": ["sliding_single", "sliding_bypass"],
    "soft_close_damper": ["sliding_single", "sliding_bypass"],
}
