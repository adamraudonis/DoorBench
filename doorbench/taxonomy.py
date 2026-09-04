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

# Robot-facing benchmark tasks (legacy vocabulary: `spec.task` / manifest `task`).  The benchmark itself uses the
# scenario names in doorbench/benchmark/scenarios.py (core: open_and_traverse, open_then_close, close_only,
# unlock_and_traverse, locked_recognize; human: hold_open_for_human, wait_for_human, knock_and_wait); see
# docs/TAXONOMY.md "Tasks vs scenarios" for the mapping.
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

# Extras (attachments) and which families they appear on in the generated dataset (documentation; the family
# generators in spec.py decide what is attached).  `automatic_swing` inherits every swing_single extra because it is
# generated through gen_swing_single.  Entries flagged "unused" are defined but never attached by the current sampler.
EXTRAS = {
    "kick_plate": ["swing_single", "swing_double", "cold_storage", "automatic_swing", "saloon"],
    "armor_plate": ["swing_single", "automatic_swing"],
    "push_plate": [],                      # unused: push plates are the `push_plate` operator, not an extra
    "peephole": ["swing_single", "automatic_swing"],
    "mail_slot": ["swing_single"],
    "knocker": ["swing_single"],
    "house_number": ["swing_single"],
    "pet_flap": ["swing_single"],
    "chain_lock": [],                      # unused: modelled as the `chain` lock
    "swing_bar_guard": [],                 # unused: modelled as the `swing_bar_guard` lock
    "exit_sign": ["swing_single", "swing_double", "automatic_swing"],
    "push_pull_sign": ["swing_single", "swing_double", "automatic_swing", "revolving", "automatic_sliding"],
    "vision_lite_grille": [],              # unused
    "door_stop_floor": [],                 # unused: floor stops are the `floor_dome` stop in kinematics.stop
    "door_stop_wall": ["swing_single", "automatic_swing"],
    "hold_open_kickdown": ["swing_single", "automatic_swing"],
    "wreath": ["swing_single", "automatic_swing"],
    "keypad_reader_wall": ["swing_single", "swing_double", "automatic_swing", "automatic_sliding", "gate_swing", "gate_sliding", "elevator", "turnstile_tripod", "turnstile_fullheight"],
    "rex_button": ["swing_single", "swing_double", "automatic_swing"],
    "wave_sensor": ["automatic_swing", "automatic_sliding"],
    "call_button": ["elevator"],
    "threshold_saddle": ["swing_single", "swing_double", "automatic_swing", "cold_storage", "sliding_single"],
    "weather_drip_cap": ["swing_single"],
    "door_viewer_camera": ["swing_single"],
    "coat_hook": ["swing_single", "automatic_swing", "stall"],
    "bumper_rail": ["swing_single", "swing_double", "automatic_swing"],
    "louver_vent": ["swing_single", "automatic_swing"],
    "transom_window": ["swing_single", "swing_double"],
    "sidelite": ["swing_single", "swing_double", "pivot"],
    "warning_placard": ["swing_single", "automatic_swing", "sliding_single", "cold_storage", "ship_watertight", "vault", "blast"],
    "floor_guide": ["sliding_single", "sliding_bypass"],
    "soft_close_damper": ["sliding_single", "sliding_bypass"],
}
assert all(f in FAMILIES for fams in EXTRAS.values() for f in fams)


# ===========================================================================
# Hierarchy metadata (documentation, taxonomy report and the site's Hierarchy
# view).  Nothing below is read by the sampler: FAMILIES, the *_CONTEXTS dicts
# and CONDITIONS above are the only inputs to spec.generate_all(), so editing
# this section never changes a generated door.
# ===========================================================================

# Precise kinematic type per family (kinematics.type written into every spec).
# FAMILIES[...] only carries the coarse "hinge" / "slide" tag used for grouping.
KINEMATICS_TYPES = {
    "hinge_vertical":   "Rotation about a vertical axis (hinged / pivoted leaf)",
    "hinge_horizontal": "Rotation about a horizontal axis (hatch, flap, tilt-up)",
    "slide_horizontal": "Translation along the wall (sliding leaf)",
    "slide_vertical":   "Vertical lift (sectional, coiling)",
    "rotor":            "Continuous rotation of a compartmented rotor (revolving, turnstile)",
}

# Motion classes: how the leaf moves (the top of the hierarchy).  Every family
# belongs to exactly one class; the class counts add up to the 1000 doors.
MOTION_CLASSES = {
    # id: (label, description, families)
    "hinged_swing": ("Hinged swing", "Leaf rotates about a vertical axis at its edge: butt, continuous, strap, spring, cam-lift or crane hinges.  The default door of every building code (ANSI/BHMA A156.1 / EN 1935 hinges).",
                     ["swing_single", "swing_double", "dutch", "saloon", "automatic_swing", "cold_storage", "stall", "gate_swing", "baby_gate", "ship_watertight", "vault", "blast"]),
    "pivot":        ("Pivot", "Leaf rotates about a vertical axis inset from its edge on floor / head pivots (ANSI/BHMA A156.4 floor closers, A156.17 pivots); oversized architectural leaves.",
                     ["pivot"]),
    "sliding":      ("Sliding", "Leaf translates along the wall on rollers or hangers: pocket, surface (barn), patio, shoji, bypass, automatic, hoistway.",
                     ["sliding_single", "sliding_bypass", "automatic_sliding", "gate_sliding", "elevator"]),
    "folding":      ("Folding", "Several hinged panels fold against each other while a guided free edge translates in a track (bifold, accordion / concertina).",
                     ["bifold", "accordion"]),
    "overhead":     ("Overhead", "Large leaf that lifts overhead: sectional on vertical + horizontal tracks with a torsion-spring counterbalance, or one-piece tilt-up on offset pivots (DASMA 102, EN 13241).",
                     ["garage_sectional", "garage_tiltup"]),
    "rollup":       ("Roll-up", "Coiling curtain of interlocking slats or a grille winding onto a barrel above the opening (ANSI/DASMA 108, EN 13241).",
                     ["rollup"]),
    "rotary":       ("Rotary", "Compartmented rotor turning continuously about a vertical axis: revolving doors (ANSI/BHMA A156.27), tripod and full-height turnstiles (EN 17352, IBC 1010.3).",
                     ["revolving", "turnstile_tripod", "turnstile_fullheight"]),
    "hatch_flap":   ("Hatches & flaps", "Leaf rotates about a horizontal axis: floor and ceiling hatches lifted against gravity, pet flaps swinging both ways on a top pin.",
                     ["hatch_floor", "hatch_ceiling", "pet_door"]),
    "flexible":     ("Flexible", "No rigid leaf: overlapping PVC strips hanging from a rail, pushed aside by the body.",
                     ["strip_curtain"]),
}
assert sorted(f for _, _, fams in MOTION_CLASSES.values() for f in fams) == sorted(FAMILIES), "every family in exactly one motion class"


def motion_class_of(family: str) -> str:
    for cid, (_, _, fams) in MOTION_CLASSES.items():
        if family in fams:
            return cid
    raise KeyError(family)


# Per-family reference card: what it is in the real world, how it moves, what
# standards describe it and what makes it hard for a robot.  Hardware, sizes and
# counts are *derived* from the generated dataset (see build_hierarchy), not
# hard-coded here.
FAMILY_INFO = {
    "swing_single": dict(label="Swing (single)", kinematics="hinge_vertical", leaves="1",
        examples=["bedroom / office door", "front entry door", "stairwell fire door", "hospital patient-room door", "storefront glass door", "detention cell door", "cottage plank door"],
        standards=["ANSI/BHMA A156.1 hinges", "A156.2 bored locks", "A156.3 exit devices", "A156.4 closers", "A156.13 mortise locks", "EN 12519 terminology", "EN 1154 / EN 1155 / EN 1125 / EN 179", "NFPA 80 / UL 10C fire doors", "ADA 2010 §404", "IBC §1010"],
        robot="The reference problem: locate and operate the operator (lever, knob, panic bar, pull), overcome latch preload and closer spring, keep the body clear of the swing, hold a self-closing leaf while passing, re-latch on close.  Handing (left/right), push/pull side and lock state change the whole plan."),
    "swing_double": dict(label="Swing (pair)", kinematics="hinge_vertical", leaves="2 (active + inactive, or both active)",
        examples=["french patio doors", "auditorium exit pair with vertical rods", "hospital double-egress cross-corridor pair", "mall entrance storefront pair", "barn / carriage-house double doors"],
        standards=["ANSI/BHMA A156.3 (exit devices, removable mullions)", "A156.16 flush bolts / coordinators", "EN 1125 panic pairs", "IBC §1010.1.9 (inactive leaves)", "NFPA 80 §6.4 pairs (astragals, coordinators)"],
        robot="Which leaf is active?  Inactive leaves are held by flush bolts, cane bolts or a removable mullion; double-egress leaves swing opposite ways; the clear opening may need both leaves; vertical-rod devices latch top and bottom."),
    "dutch": dict(label="Dutch", kinematics="hinge_vertical", leaves="1 leaf split into 2 half-leaves",
        examples=["kitchen / nursery dutch door", "stable half-door", "daycare reception door"],
        standards=["ANSI/BHMA A156.16 (joining / dutch-door bolt)"],
        robot="Two independently hinged halves joined by a bolt: with the bolt thrown the door behaves as one leaf, otherwise only the upper half opens; the lower half must be opened separately to pass."),
    "saloon": dict(label="Saloon", kinematics="hinge_vertical", leaves="1 or 2 (double-acting)",
        examples=["cafe kitchen pass doors", "restaurant kitchen swing door", "hospital utility double-acting door", "saloon bar doors"],
        standards=["ANSI/BHMA A156.17 (double-acting spring hinges)", "EN 1935 (hinge grades)"],
        robot="No latch and no operator: push through in either direction against the spring hinges and stay clear of the return swing, which can strike the robot from behind."),
    "pivot": dict(label="Pivot (architectural)", kinematics="hinge_vertical", leaves="1 (oversized, 0.9-1.8 m)",
        examples=["modern residence pivot entry", "hotel lobby / museum pivot door", "boutique frameless-glass pivot"],
        standards=["ANSI/BHMA A156.4 (floor closers / pivots)", "A156.17 pivots", "EN 1154 floor springs"],
        robot="Heavy leaf (56-262 kg) on a centre or offset pivot: part of the leaf swings towards the robot while the rest swings away; the floor spring's hold-open at 90 deg and large inertia dominate."),
    "sliding_single": dict(label="Sliding", kinematics="slide_horizontal", leaves="1 (+ fixed panel for patio / shoji)",
        examples=["bathroom pocket door", "barn door on a flat track", "patio sliding glass door", "shoji / fusuma", "detention cell slider", "industrial sliding fire door"],
        standards=["ANSI/BHMA A156.14 sliding & folding door hardware", "AAMA/WDMA/CSA 101 (patio doors)", "NFPA 80 (sliding fire doors)", "IBC §1010.1.2 (sliding doors not in egress except auto)"],
        robot="Grasp a flush pull or edge and translate the leaf along its track against roller friction; a hook lock or teardrop latch must be lifted first; the leaf disappears into a pocket or behind a fixed panel so the grasp point moves."),
    "sliding_bypass": dict(label="Bypass closet", kinematics="slide_horizontal", leaves="2 or 3 on parallel tracks",
        examples=["bedroom closet bypass doors", "mirrored wardrobe doors", "shoji closet (oshiire)"],
        standards=["ANSI/BHMA A156.14"],
        robot="Leaves overlap: only one track's leaf can be moved from a given side, and moving one leaf can uncover or cover the other; finger cups give little purchase."),
    "bifold": dict(label="Bifold", kinematics="hinge_vertical", leaves="2 or 4 panels (coupled)",
        examples=["bedroom closet bifold", "louvered utility-closet bifold"],
        standards=["ANSI/BHMA A156.14 (bifold hardware)"],
        robot="Pulling the knob rotates the pivot panel while the guide panel's free edge slides in the head track: a closed kinematic chain whose panels fold towards the robot."),
    "accordion": dict(label="Accordion", kinematics="hinge_vertical", leaves="6-10 narrow panels (coupled)",
        examples=["room-divider accordion", "laundry-nook accordion", "office partition"],
        standards=["ANSI/BHMA A156.14"],
        robot="Many light panels on piano hinges concertina together; the pull travels the whole opening width and the stack can jam."),
    "revolving": dict(label="Revolving", kinematics="rotor", leaves="3 or 4 wings",
        examples=["office tower / hotel / department store revolving door", "airport and hospital lobby revolving doors"],
        standards=["ANSI/BHMA A156.27 (power & manual revolving doors)", "IBC §1010.1.4.1 (breakout, speed, adjacent swing door)", "EN 16005 (power-operated)"],
        robot="Enter a moving compartment, keep pace with the wing (speed governor), and exit on the far side without touching the drum; breakout wings and a possible electric bolt at night."),
    "turnstile_tripod": dict(label="Tripod turnstile", kinematics="rotor", leaves="3 arms (tripod)",
        examples=["metro / subway turnstile", "office lobby tripod turnstile", "gym or stadium entrance"],
        standards=["EN 17352 (pedestrian entrance control)", "IBC §1010.3 turnstiles", "ADA §404 (turnstiles are not accessible routes)"],
        robot="Present a credential, then push a waist-high arm that ratchets one way; the next arm rises into the path.  One-way and drop-arm variants."),
    "turnstile_fullheight": dict(label="Full-height turnstile", kinematics="rotor", leaves="3-4 wings of 8 bars",
        examples=["stadium / factory / metro full-height turnstile", "parking-garage pedestrian turnstile"],
        standards=["EN 17352", "IBC §1010.3"],
        robot="Walk inside a rotating cage compartment while pushing the bars; the rotor indexes 90-120 deg per passage and cannot be reversed."),
    "garage_sectional": dict(label="Garage (sectional)", kinematics="slide_vertical", leaves="1 door of 4-5 hinged sections (vertical lift)",
        examples=["residential single / double garage door", "townhouse garage door"],
        standards=["ANSI/DASMA 102 (sectional garage doors)", "UL 325 (operators)", "EN 13241 / EN 12604"],
        robot="Lift a 2.4-5.5 m wide, 55-190 kg leaf from a low handle: the torsion-spring counterbalance carries most of the weight but a slack or disengaged opener changes the force; the leaf moves overhead towards the robot."),
    "garage_tiltup": dict(label="Garage (tilt-up)", kinematics="hinge_horizontal", leaves="1 one-piece panel",
        examples=["1960s tilt-up garage door", "carport tilt-up door"],
        standards=["ANSI/DASMA 102", "EN 13241"],
        robot="The whole panel swings out at the bottom before it rises overhead, sweeping the approach area; extension springs counterbalance."),
    "rollup": dict(label="Roll-up", kinematics="slide_vertical", leaves="1 coiling curtain / grille",
        examples=["self-storage unit door", "shop-front security shutter", "loading-dock coiling door", "parking-garage grille"],
        standards=["ANSI/DASMA 108 (rolling doors)", "EN 13241", "UL 325 (motorised)"],
        robot="Lift the bottom bar of a curtain that coils overhead; manual curtains need a strong pull to start (counterbalance, slat friction) and chain hoists need many turns."),
    "pet_door": dict(label="Pet door", kinematics="hinge_horizontal", leaves="1 flap (cat to XL dog)",
        examples=["cat flap in a back door", "large-dog flap in a wall or garage door"],
        standards=["no code; manufacturer size classes (small / medium / large / XL)"],
        robot="Too small for a humanoid: a flap swinging both ways on a top pin with a weak magnet, sometimes closed by a slide-in locking panel.  Relevant for quadrupeds and for recognising a non-passable opening."),
    "hatch_floor": dict(label="Floor hatch", kinematics="hinge_horizontal", leaves="1 (lift up)",
        examples=["cellar trapdoor", "utility floor hatch", "ship deck hatch", "stage trapdoor", "storm-shelter hatch"],
        standards=["IBC §1011.12 (roof / floor access)", "manufacturer data (Bilco floor doors)"],
        robot="Lift a horizontal leaf against gravity from a ring pull, hold it past its balance point or onto a prop arm; gas struts assist; the robot stands next to the leaf's own edge."),
    "hatch_ceiling": dict(label="Ceiling hatch", kinematics="hinge_horizontal", leaves="1 (push up)",
        examples=["attic access hatch", "roof scuttle", "ceiling maintenance hatch"],
        standards=["IBC §1011.12"],
        robot="Overhead push at 2.4 m; passing through needs a ladder, so the realistic task is open / close only."),
    "ship_watertight": dict(label="Watertight (marine)", kinematics="hinge_vertical", leaves="1 (dogged)",
        examples=["ship bulkhead WT door", "engine-room WT door", "offshore weathertight door", "submarine bulkhead hatch"],
        standards=["SOLAS II-1 Reg. 13 (watertight doors)", "ISO 6042 (weathertight steel doors)", "class rules (ABS / DNV / LR)"],
        robot="Release 4-8 wedge dogs one by one (or spin a central handwheel on quick-acting doors), pull a 64-120 kg leaf off a compressed gasket, step over a 150-450 mm coaming, re-dog behind."),
    "vault": dict(label="Vault", kinematics="hinge_vertical", leaves="1 (0.8-1.5 t)",
        examples=["bank vault door", "safe-room door", "data-centre vault", "gun vault"],
        standards=["UL 608 (burglary-resistant vault doors)", "EN 1143-1 (secure storage units)"],
        robot="Turn a handwheel one to two full turns to retract 4-8 bolts, then move a tonne of steel on crane hinges: high inertia, very low friction, a step sill."),
    "blast": dict(label="Blast door", kinematics="hinge_vertical", leaves="1 (0.7-1.2 t)",
        examples=["bunker / shelter blast door", "test-cell blast door"],
        standards=["ASTM F2247 (blast-resistant doors)", "UFC 4-010-01"],
        robot="Like a vault door but latched by lever dogs or a wheel; heavy gaskets and a raised sill."),
    "gate_swing": dict(label="Gate (swing)", kinematics="hinge_vertical", leaves="1 (outdoor, 0.9-4.8 m)",
        examples=["garden picket gate", "schoolyard chain-link gate", "estate wrought-iron gate", "pool safety gate", "ranch tube gate"],
        standards=["ISPSC §305 / ASTM F1908 pool barriers (self-closing, self-latching, latch at 1.37 m)", "EN 12209 / BS 3621 gate locks", "ASTM F2200 (automated gates)"],
        robot="Outdoor: uneven ground clearance, sagging hinges, gravity latches that must be lifted, hasps and padlocks, pool latches mounted at 1.5 m out of a child's reach; wide farm gates swing through large arcs."),
    "gate_sliding": dict(label="Gate (sliding)", kinematics="slide_horizontal", leaves="1 (cantilever or bottom rail)",
        examples=["cantilever driveway gate (manual)", "pedestrian sliding gate", "warehouse yard gate"],
        standards=["ASTM F2200", "EN 13241 / EN 12453 (power-operated gates)"],
        robot="Long, heavy leaves (up to 330 kg) on cantilever rollers: large start force, long travel, pinch zones at the posts."),
    "baby_gate": dict(label="Baby gate", kinematics="hinge_vertical", leaves="1 (0.75-1.1 m wide, waist high)",
        examples=["stair-top gate", "kitchen doorway gate", "hallway pet gate"],
        standards=["ASTM F1004 (expansion gates and expandable enclosures)", "EN 1930 (child safety barriers)"],
        robot="A lift-and-swing latch designed to defeat toddlers, a trip bar at floor level, spring return; the robot can also step over it."),
    "stall": dict(label="Toilet stall", kinematics="hinge_vertical", leaves="1 (partition door, 0.6-0.86 m)",
        examples=["public restroom stall", "ADA outswing stall", "locker-room changing stall"],
        standards=["ADA §604.8 (toilet compartments)", "manufacturer hardware (Bobrick gravity hinges, slide latches)"],
        robot="Gravity hinges hold the door ajar when vacant; an occupied stall is latched from inside with a slide latch and must be recognised as such; narrow leaf and pilaster gap."),
    "strip_curtain": dict(label="Strip curtain", kinematics="hinge_horizontal", leaves="5-18 overlapping PVC strips",
        examples=["walk-in cooler strip curtain", "warehouse dock strip door", "food-processing strip curtain"],
        standards=["no code; OSHA / food-safety guidance"],
        robot="Deformable strips that wrap around the body and obscure vision; no mechanism, but contact along the whole body."),
    "cold_storage": dict(label="Cold storage", kinematics="hinge_vertical", leaves="1 (100-150 mm insulated)",
        examples=["walk-in cooler door", "walk-in freezer door", "lab cold-storage door", "florist cooler"],
        standards=["NSF/ANSI 7 (commercial refrigerators)", "industry hardware: Kason 1245 cam-lift hinges, Kason 58 SafeGuard latch"],
        robot="Cam-lift hinges raise the leaf as it opens (self-closing by gravity), a magnetic gasket holds it shut, the SafeGuard handle needs a pull-and-lift; an inside release must always work."),
    "automatic_sliding": dict(label="Automatic sliding", kinematics="slide_horizontal", leaves="1 or 2 (bi-parting) + fixed sidelites",
        examples=["supermarket / pharmacy entrance", "hospital entrance", "airport and office-lobby sliders"],
        standards=["ANSI/BHMA A156.10 (full-energy power-operated doors)", "EN 16005", "IBC §1010.1.4.3 (breakout for egress)"],
        robot="The door opens itself when the sensor fires: approach into the detection zone, wait, pass before hold-open time expires; if the power is off, break out the leaf manually (220 N)."),
    "automatic_swing": dict(label="Automatic swing", kinematics="hinge_vertical", leaves="1",
        examples=["low-energy push-to-open office / hotel door", "full-energy hospital corridor door"],
        standards=["ANSI/BHMA A156.19 (low-energy / power-assist)", "A156.10 (full-energy)", "EN 16005"],
        robot="Press the wall button or wave, or start the swing by hand (push-and-go); the operator then swings and holds the leaf; unpowered it behaves as a heavy closer.  Card readers and maglocks sit on the same doors."),
    "elevator": dict(label="Elevator", kinematics="slide_horizontal", leaves="1 (side) or 2 (centre-opening) hoistway panels",
        examples=["office / residential-tower landing doors", "hospital and freight elevator doors"],
        standards=["ASME A17.1 / CSA B44 (hoistway door interlocks)", "EN 81-20 / EN 81-50"],
        robot="The robot cannot open a hoistway door: it presses the call button, waits for the car, and passes during the door-open dwell; doors reopen on obstruction."),
}
assert set(FAMILY_INFO) == set(FAMILIES)

# Human-readable labels + which setting a context belongs to.  Contexts come
# from two places in spec.py: the four *_CONTEXTS dicts above and the fixed
# context string each single-context family generator writes.
CONTEXT_INFO = {
    # swing_single
    "residential_interior": ("Residential interior", "residential", "Bedroom, bathroom, closet and hallway doors: hollow-core slabs, knobs and levers, privacy locks, no closer."),
    "residential_exterior": ("Residential exterior", "residential", "Entry, back and side doors incl. screen / storm doors: deadbolts, handlesets, keypads, weather seals."),
    "commercial_office":    ("Commercial office", "commercial", "Office, conference, corridor and restroom doors: grade-1 levers, deadlatches, surface closers, access control."),
    "fire_egress":          ("Fire / egress", "commercial", "Rated hollow-metal doors in the means of egress: exit devices, closers, smoke seals, delayed egress."),
    "institutional":        ("Institutional", "institutional", "Hospital, school, lab, hotel, radiology and behavioural-health doors."),
    "industrial_utility":   ("Industrial / utility", "industrial", "Mechanical, electrical, warehouse and roof-access doors: padlocks, louvers, worn hardware."),
    "security_detention":   ("Security / detention", "security", "Cell, sally-port, evidence and armory doors: 14 ga steel, bar grilles, heavy hinges."),
    "storefront_glass":     ("Storefront glass", "commercial", "Aluminium storefront and frameless glass doors on pivots with floor springs and pull bars."),
    "heritage_rustic":      ("Heritage / rustic", "residential", "Plank, brace and carved doors with Suffolk latches, rim locks, strap hinges."),
    # swing_double
    "french":                ("French pair", "residential", "Glazed residential pairs: one active leaf, flush bolts on the inactive leaf, optional cremone bolt."),
    "commercial_pair_panic": ("Commercial panic pair", "commercial", "Egress pairs with rim / vertical-rod exit devices, closers, removable mullions."),
    "double_egress":         ("Double egress", "institutional", "Cross-corridor pairs whose leaves swing in opposite directions (hospital, airport, mall smoke doors)."),
    "storefront_pair":       ("Storefront pair", "commercial", "Entrance pairs in aluminium / frameless glass with floor springs and ladder pulls."),
    "barn_pair":             ("Barn pair", "outdoor", "Hinged barn and carriage-house pairs on strap hinges with cane bolts and slide bolts."),
    # sliding_single
    "pocket":         ("Pocket", "residential", "Leaf disappears into the wall; flush pulls and hook locks."),
    "barn":           ("Barn (surface track)", "residential", "Surface-mounted flat track with hangers; floor guide, teardrop privacy latch."),
    "patio_glass":    ("Patio glass", "residential", "Bottom-rolling insulated glass leaf beside a fixed panel; hook lock with thumb latch."),
    "shoji_fusuma":   ("Shoji / fusuma", "residential", "Paper-and-lattice panels sliding wood on wood in kamoi / shikii grooves."),
    "cell_industrial": ("Cell / industrial", "industrial", "Detention cell sliders, industrial sliding fire doors, cold-room sliders, freight-elevator manual gates."),
    # gate_swing
    "garden_picket": ("Garden picket", "outdoor", "Cedar picket gates with Suffolk or fork latches and gate springs."),
    "chain_link":    ("Chain-link", "outdoor", "Galvanised frame + mesh with fork latches, hasps and padlocks."),
    "wrought_iron":  ("Wrought iron", "outdoor", "Ornamental gates with mortise locks, deadbolts or electric strikes."),
    "pool_safety":   ("Pool safety", "outdoor", "Self-closing, self-latching gates with a magnetic latch at 1.5 m (pool code)."),
    "ranch_tube":    ("Ranch tube", "outdoor", "Wide galvanised tube gates with slide bolts, chains and padlocks."),
    # fixed single-context families
    "residential":        ("Residential", "residential", "Home setting."),
    "hospitality":        ("Hospitality", "commercial", "Cafes, restaurants, bars, supermarkets."),
    "architectural":      ("Architectural", "commercial", "Statement entrances in residences, hotels, museums, offices."),
    "closet":             ("Closet", "residential", "Wardrobe and storage closets."),
    "partition":          ("Partition", "residential", "Room dividers and nook closures."),
    "commercial_entry":   ("Commercial entry", "commercial", "Main entrances of offices, hotels, stores, airports, hospitals."),
    "access_control":     ("Access control", "security", "Credential-gated pedestrian lanes."),
    "garage":             ("Garage / vehicle", "residential", "Vehicle doors: garages, carports, docks, shutters."),
    "utility":            ("Utility", "industrial", "Cellars, attics, roofs, service spaces."),
    "marine":             ("Marine", "marine", "Ships, submarines, offshore platforms."),
    "security":           ("Security", "security", "Vaults, safe rooms, bunkers, shelters."),
    "outdoor":            ("Outdoor", "outdoor", "Yards, driveways, fields."),
    "restroom":           ("Restroom", "commercial", "Public toilet and changing rooms."),
    "industrial":         ("Industrial", "industrial", "Warehouses, docks, food processing."),
    "food_service":       ("Food service", "commercial", "Restaurant and lab cold rooms."),
    "vertical_transport": ("Vertical transport", "commercial", "Elevator landings."),
}
SETTINGS = {
    "residential": "Residential", "commercial": "Commercial", "institutional": "Institutional", "industrial": "Industrial",
    "outdoor": "Outdoor", "security": "Security", "marine": "Marine",
}
assert all(ci[1] in SETTINGS for ci in CONTEXT_INFO.values())
assert set(SWING_SINGLE_CONTEXTS) | set(SWING_DOUBLE_CONTEXTS) | set(SLIDING_SINGLE_CONTEXTS) | set(GATE_SWING_CONTEXTS) <= set(CONTEXT_INFO)

# Leaf level of the hierarchy: how each family is split into variants and how a
# variant is selected in the catalogue.  ("context", None) splits by the manifest
# context; the other rules split by one manifest field or a tag token and carry
# the labels.  Every door falls into exactly one variant (unknown values are
# reported as "other" so the counts always add up).
FAMILY_VARIANTS = {
    "swing_single":         ("context", None),
    "swing_double":         ("context", None),
    "sliding_single":       ("context", None),
    "gate_swing":           ("context", None),
    "automatic_swing":      ("context", None),
    "dutch":                ("context", None),
    "saloon":               ("context", None),
    "strip_curtain":        ("context", None),
    "accordion":            ("tag", {"6_panel": "6 panels", "8_panel": "8 panels", "10_panel": "10 panels"}),
    "bifold":               ("tag", {"2_panel": "2 panels", "4_panel": "4 panels"}),
    "sliding_bypass":       ("tag", {"closet_wood": "Wood closet", "mirror": "Mirrored wardrobe", "shoji_pair": "Shoji pair (oshiire)", "glass_frameless": "Frameless glass"}),
    "revolving":            ("tag", {"3_wing": "3 wings", "4_wing": "4 wings"}),
    "automatic_sliding":    ("tag", {"bi_parting": "Bi-parting", "single_slide": "Single slide"}),
    "elevator":             ("tag", {"center_opening": "Centre opening", "side_opening": "Side opening"}),
    "pet_door":             ("tag", {"cat": "Cat", "small_dog": "Small dog", "medium_dog": "Medium dog", "large_dog": "Large dog", "xl_dog": "XL dog"}),
    "turnstile_tripod":     ("lock", {"mag_lock": "Credential-locked", "none": "Free-spinning"}),
    "turnstile_fullheight": ("lock", {"mag_lock": "Credential-locked", "none": "Free-spinning"}),
    "pivot":                ("closer", {"floor_spring": "Floor spring (hold-open)", "floor_spring_nohold": "Floor spring", "none": "Free pivot"}),
    "baby_gate":            ("closer", {"gate_spring": "Self-closing", "none": "Manual"}),
    "ship_watertight":      ("operator", {"dog_lever": "Individually dogged", "wheel_ship_hatch": "Quick-acting (handwheel)"}),
    "vault":                ("operator", {"wheel_vault": "Handwheel boltwork", "dog_lever": "Lever dogs", "lever_straight": "Lever bolt"}),
    "blast":                ("operator", {"wheel_vault": "Handwheel boltwork", "dog_lever": "Lever dogs", "lever_straight": "Lever bolt"}),
    "garage_sectional":     ("slab", {"garage_steel_single": "Steel, non-insulated", "garage_steel_insulated": "Steel, insulated", "garage_wood_carriage": "Wood carriage-house"}),
    "garage_tiltup":        ("slab", {"garage_steel_single": "Steel", "garage_wood_carriage": "Wood carriage-house"}),
    "rollup":               ("slab", {"rollup_steel": "Steel slat curtain", "rollup_alu_grille": "Aluminium grille"}),
    "hatch_floor":          ("slab", {"cellar_trapdoor": "Oak cellar trapdoor", "steel_plate_security": "Steel plate hatch", "attic_hatch": "Plywood hatch"}),
    "hatch_ceiling":        ("slab", {"attic_hatch": "Plywood attic hatch", "hollow_metal_18ga": "Hollow-metal scuttle", "steel_plate_security": "Steel plate hatch"}),
    "gate_sliding":         ("slab", {"chain_link_gate": "Chain-link", "wrought_iron_gate": "Wrought iron", "steel_bar_grille": "Bar grille", "expanded_metal_gate": "Expanded metal"}),
    "stall":                ("slab", {"hpl_partition": "HPL partition", "phenolic_partition": "Powder-coated steel", "stainless_hollow": "Stainless"}),
    "cold_storage":         ("slab", {"cold_storage_100": "Cooler (100 mm)", "freezer_150": "Freezer (150 mm)"}),
}
assert set(FAMILY_VARIANTS) == set(FAMILIES)


def variant_of(row: dict) -> tuple[str, str, dict]:
    """Leaf node of the hierarchy for one manifest row: (variant id, label, catalogue filter).

    The filter is a dict of catalogue query parameters (family + one of context / operator / lock / closer /
    tag / slab, all exact matches) that selects exactly this variant's doors in the viewer."""
    fam = row["family"]
    rule, table = FAMILY_VARIANTS[fam]
    if rule == "context":
        ctx = row.get("context", "") or "default"
        label = CONTEXT_INFO.get(ctx, (ctx.replace("_", " ").title(),))[0]
        return ctx, label, {"family": fam, "context": ctx}
    if rule == "tag":
        for tok, label in table.items():
            if tok in row.get("tags", []):
                return tok, label, {"family": fam, "tag": tok}
        return "other", "Other", {"family": fam}
    if rule in ("lock", "operator", "closer"):
        v = row.get(rule, "none")
        return v, table.get(v, v.replace("_", " ")), {"family": fam, rule: v}
    if rule == "slab":
        v = row["leaf"]["slab"]
        return v, table.get(v, v.replace("_", " ")), {"family": fam, "slab": v}
    raise ValueError(rule)


KINEMATIC_FLAGS = ("pair", "dutch", "both_ways", "fold", "accordion", "flap", "strips", "bi_parting", "center_opening", "breakout", "one_way",
                   "self_closing", "gravity_assisted_close", "double_egress", "locked_until_credential", "interlocked", "auto_close")


def build_hierarchy(rows: list[dict], specs: dict | None = None, n_reps: int = 6) -> dict:
    """Motion class -> family -> variant tree with counts, representative doors, hardware, sizes and the
    family x mechanism-kind relationship matrices.  `rows` are manifest door rows (assets/manifest.json);
    `specs` optionally maps door id -> spec.json for kinematics flags and leaf counts."""
    from collections import Counter, defaultdict
    from . import hardware as H

    rows = [r for r in rows if not r.get("error")]
    catalogs = {"operator": H.OPERATORS, "latch": H.LATCHES, "lock": H.LOCKS, "closer": H.CLOSERS, "hinge": H.HINGES}

    def kind_of(mech: str, model: str) -> str:
        cat = catalogs[mech]
        return cat[model].kind if model in cat else model

    def rep_list(ds):
        ds = sorted(ds, key=lambda d: (not d.get("signed_off", False), not d.get("thumbs"), d.get("index", 0)))
        out, seen = [], set()
        for d in ds:                     # prefer visually different doors: distinct slab/operator first
            key = (d["leaf"]["slab"], d["operator"])
            if key in seen and len(ds) > n_reps:
                continue
            seen.add(key)
            th = next((t for t in d.get("thumbs", []) if "thumb_iso.jpg" in t), (d.get("thumbs") or [None])[0])
            out.append({"id": d["id"], "thumb": th, "use_case": d.get("use_case", ""), "mass_kg": d.get("mass_kg")})
            if len(out) >= n_reps:
                break
        return out

    def summarize(ds):
        ws = [d["leaf"]["width"] for d in ds]; hs = [d["leaf"]["height"] for d in ds]; ms = [d["mass_kg"] for d in ds]
        hw = {}
        for mech in catalogs:
            hw[mech] = dict(Counter(d[mech] for d in ds).most_common())
            hw[mech + "_kind"] = dict(Counter(kind_of(mech, d[mech]) for d in ds).most_common())
        kin, flags, leaves = Counter(), Counter(), Counter()
        if specs:
            for d in ds:
                s = specs.get(d["id"])
                if not s:
                    continue
                k = s.get("kinematics", {})
                kin[k.get("type", "?")] += 1
                leaves[s.get("leaf", {}).get("count", 1)] += 1
                for fl in KINEMATIC_FLAGS:
                    if k.get(fl):
                        flags[fl] += 1
                if k.get("actuator"):
                    flags["powered" if k["actuator"].get("powered", True) else "unpowered_operator"] += 1
        return {
            "count": len(ds),
            "signed_off": sum(1 for d in ds if d.get("signed_off")),
            "sizes": {"leaf_width_m": [min(ws), max(ws)], "leaf_height_m": [min(hs), max(hs)], "mass_kg": [round(min(ms), 1), round(max(ms), 1)],
                      "mass_median_kg": round(sorted(ms)[len(ms) // 2], 1)},
            "hardware": hw,
            "kinematics": dict(kin), "leaves": {str(k): v for k, v in sorted(leaves.items())}, "flags": dict(flags),
            "conditions": dict(Counter(d["condition"] for d in ds).most_common()),
            "tasks": dict(Counter(d.get("task") for d in ds).most_common()),
            "scenarios": dict(Counter(s for d in ds for s in ((d.get("benchmark") or {}).get("core") or [])).most_common()),
            "locked": sum(1 for d in ds if d.get("lock_engaged")),
            "locked_no_release": sum(1 for d in ds if d.get("lock_engaged") and not d.get("robot_side_release", True)),
            "difficulty_mean": round(sum(d.get("difficulty") or 0 for d in ds) / len(ds), 2),
            "reps": rep_list(ds),
        }

    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    classes = []
    for cid, (label, desc, fams) in MOTION_CLASSES.items():
        fam_nodes = []
        for fam in fams:
            ds = by_fam.get(fam, [])
            if not ds:
                continue
            info = FAMILY_INFO[fam]
            by_var, var_meta = defaultdict(list), {}
            for d in ds:
                vid, vlabel, filt = variant_of(d)
                by_var[vid].append(d)
                var_meta[vid] = (vlabel, filt)
            variants = []
            for vid, vds in sorted(by_var.items(), key=lambda kv: -len(kv[1])):
                vlabel, filt = var_meta[vid]
                node = {"id": vid, "label": vlabel, "filter": filt, "ids": sorted(d["id"] for d in vds)}
                ctx = CONTEXT_INFO.get(vid)
                if FAMILY_VARIANTS[fam][0] == "context" and ctx:
                    node["description"] = ctx[2]
                    node["setting"] = ctx[1]
                node.update(summarize(vds))
                variants.append(node)
            fnode = {"id": fam, "label": info["label"], "description": FAMILIES[fam][2], "kinematics_type": info["kinematics"],
                     "leaves_note": info["leaves"], "examples": info["examples"], "standards": info["standards"], "robot": info["robot"],
                     "quota": FAMILIES[fam][0], "variant_rule": FAMILY_VARIANTS[fam][0], "variants": variants}
            fnode.update(summarize(ds))
            fam_nodes.append(fnode)
        cds = [d for fam in fams for d in by_fam.get(fam, [])]
        if not cds:
            continue
        cnode = {"id": cid, "label": label, "description": desc, "families": fam_nodes}
        cnode.update(summarize(cds))
        classes.append(cnode)

    # relationships: family x mechanism kind (rows in hierarchy order)
    fams_order = [f for _, _, fs in MOTION_CLASSES.values() for f in fs if f in by_fam]
    relations, shared = {}, []
    for mech in catalogs:
        cols = sorted({kind_of(mech, d[mech]) for d in rows} - {"none"})
        mat = [[sum(1 for d in by_fam[fam] if kind_of(mech, d[mech]) == c) for c in cols] for fam in fams_order]
        relations[mech] = {"rows": fams_order, "cols": cols, "matrix": mat}
        for j, c in enumerate(cols):
            fams_with = [fam for fam, rowv in zip(fams_order, mat) if rowv[j] > 0]
            if len(fams_with) >= 2:
                shared.append({"mechanism": mech, "kind": c, "families": fams_with, "n_doors": sum(rowv[j] for rowv in mat)})
    shared.sort(key=lambda s: (-len(s["families"]), s["mechanism"], s["kind"]))
    kind_examples = {mech: {kd: sorted({m.name for m in cat.values() if m.kind == kd})[:3] for kd in relations[mech]["cols"]} for mech, cat in catalogs.items()}
    settings = Counter((CONTEXT_INFO.get(r.get("context", ""), (None, "other"))[1]) for r in rows)
    return {
        "n_doors": len(rows),
        "n_signed_off": sum(1 for r in rows if r.get("signed_off")),
        "motion_classes": classes,
        "kinematics_types": KINEMATICS_TYPES,
        "family_labels": {f: FAMILY_INFO[f]["label"] for f in FAMILIES},
        "motion_class_of": {f: motion_class_of(f) for f in FAMILIES},
        "context_info": {k: {"label": v[0], "setting": v[1], "description": v[2]} for k, v in CONTEXT_INFO.items()},
        "settings": {k: {"label": SETTINGS[k], "count": settings.get(k, 0)} for k in SETTINGS},
        "relations": relations,
        "kind_examples": kind_examples,
        "shared_mechanisms": shared,
    }
