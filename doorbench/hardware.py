"""Hardware catalog: operators (handles), latches, locks, exit devices,
closers, hinges, stops, seals.  All force/torque values are grounded in
codes (ADA §404, IBC §1010, EN 1154, ANSI/BHMA A156.x, UL 305) or
manufacturer catalog data; provenance noted per item.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

LBF = 4.44822   # N
IN = 0.0254     # m

# ---------------------------------------------------------------------------
# EN 1154 controlled door closers (Table 1).  closing_moment @ 0-4 deg,
# opening_moment max @ 88-92 deg; efficiency >= 50% (size 1-2), 55% (3-4), 60% (5-6), 65% (7).
# ---------------------------------------------------------------------------
@dataclass
class CloserSize:
    size: int
    max_leaf_width: float      # m
    test_mass: float           # kg
    closing_moment_min: float  # N*m between 0 and 4 deg
    opening_moment_max: float  # N*m between 88 and 92 deg
    closing_moment_min_88: float  # N*m between 88 and 92 deg


EN1154_SIZES = {
    1: CloserSize(1, 0.75, 20, 9, 26, 3),
    2: CloserSize(2, 0.85, 40, 13, 36, 4),
    3: CloserSize(3, 0.95, 60, 18, 47, 6),
    4: CloserSize(4, 1.10, 80, 26, 62, 9),
    5: CloserSize(5, 1.25, 100, 37, 83, 12),
    6: CloserSize(6, 1.40, 120, 54, 134, 18),
    7: CloserSize(7, 1.60, 160, 87, 215, 29),
}


def closer_size_for(mass_kg: float, width_m: float) -> int:
    for s in sorted(EN1154_SIZES):
        c = EN1154_SIZES[s]
        if mass_kg <= c.test_mass and width_m <= c.max_leaf_width:
            return s
    return 7


@dataclass
class CloserModel:
    id: str
    name: str
    kind: str                # surface_overhead | concealed_overhead | floor_spring | spring_hinge | pneumatic | gate | gas_strut | electromagnetic_hold | auto_operator_low_energy | auto_operator_full | none
    en_size: Optional[int]   # EN1154 power size if applicable (None -> derive from door)
    closing_damping: float   # N*m*s/rad hydraulic damping during closing (regulated sweep)
    opening_damping: float   # N*m*s/rad resistance during opening (check valve open)
    backcheck_angle: Optional[float]  # rad; extra damping beyond this angle (None = no backcheck)
    backcheck_damping: float
    latch_boost: float       # multiplier on spring torque in last 15 deg (latching action), 1.0 = none
    hold_open: Optional[float]  # rad; hold-open position (None = no hold-open)
    delayed_action: bool
    mass: float              # kg (body on door)
    mounts_on: str           # door_push | door_pull | frame | floor | hinge | none
    body_size: tuple         # m (l, w, h) of the closer body
    source: str

    def to_dict(self):
        return asdict(self)


CLOSERS: dict[str, CloserModel] = {}


def _c(m: CloserModel):
    CLOSERS[m.id] = m
    return m


_c(CloserModel("none", "No closer", "none", None, 0, 0, None, 0, 1.0, None, False, 0.0, "none", (0, 0, 0), ""))
_c(CloserModel("lcn_4040", "Surface closer, heavy duty (LCN 4040XP class), adjustable size 1-6", "surface_overhead", None, 90.0, 12.0, 1.22, 60.0, 1.3, None, False, 3.6, "door_push", (0.29, 0.07, 0.10), "LCN 4040XP: 1-6 adjustable spring, backcheck, 8 lb body"))
_c(CloserModel("lcn_4040_delayed", "Surface closer w/ delayed action", "surface_overhead", None, 90.0, 12.0, 1.22, 60.0, 1.3, None, True, 3.8, "door_push", (0.29, 0.07, 0.10), "LCN 4040XP DEL"))
_c(CloserModel("norton_1600", "Surface closer, standard duty (Norton 1600 class), size 1-6", "surface_overhead", None, 70.0, 10.0, 1.31, 45.0, 1.25, None, False, 2.7, "door_push", (0.24, 0.06, 0.08), "Norton 1600BC"))
_c(CloserModel("residential_light", "Light residential surface closer (size 1-2)", "surface_overhead", 2, 40.0, 6.0, None, 0.0, 1.2, None, False, 1.6, "door_pull", (0.20, 0.05, 0.06), "Residential closer for screen/storm/interior"))
_c(CloserModel("concealed_overhead", "Concealed overhead closer (in head frame)", "concealed_overhead", None, 80.0, 10.0, 1.31, 50.0, 1.25, None, False, 3.0, "frame", (0.45, 0.04, 0.03), "Dorma RTS88 class"))
_c(CloserModel("floor_spring", "Floor spring (pivot door / storefront)", "floor_spring", None, 100.0, 8.0, 1.40, 60.0, 1.3, 1.5708, False, 6.0, "floor", (0.30, 0.11, 0.05), "Dorma BTS80 class, 90 deg hold-open option"))
_c(CloserModel("floor_spring_nohold", "Floor spring, no hold-open", "floor_spring", None, 100.0, 8.0, 1.40, 60.0, 1.3, None, False, 6.0, "floor", (0.30, 0.11, 0.05), "Dorma BTS80"))
_c(CloserModel("spring_hinge_single", "Single-acting spring hinge (adjustable)", "spring_hinge", None, 2.0, 2.0, None, 0.0, 1.0, None, False, 0.3, "hinge", (0.114, 0.114, 0.003), "Bommer 4310 class; 2-3 hinges per door"))
_c(CloserModel("spring_hinge_double", "Double-acting spring hinge (saloon)", "spring_hinge", None, 1.5, 1.5, None, 0.0, 1.0, None, False, 0.5, "hinge", (0.15, 0.10, 0.02), "Bommer 3029 class double-acting"))
_c(CloserModel("pneumatic_screen", "Pneumatic screen door closer", "pneumatic", 1, 25.0, 4.0, None, 0.0, 1.0, 1.5708, False, 0.5, "door_pull", (0.30, 0.03, 0.03), "Wright Products V150; hold-open washer"))
_c(CloserModel("gate_spring", "Gate spring (coil, adjustable tension)", "gate", None, 1.0, 1.0, None, 0.0, 1.0, None, False, 0.4, "door_push", (0.28, 0.03, 0.03), "National Hardware gate spring 6-15 lbf"))
_c(CloserModel("gate_hydraulic", "Hydraulic gate closer (pool code)", "gate", None, 60.0, 8.0, None, 0.0, 1.3, None, False, 1.8, "door_push", (0.25, 0.05, 0.05), "Lokk-Latch / D&D self-closing hinges for pool gates"))
_c(CloserModel("gas_strut", "Gas strut (hatch lift assist)", "gas_strut", None, 30.0, 30.0, None, 0.0, 1.0, None, False, 0.6, "door_push", (0.35, 0.02, 0.02), "Gas spring 150-400 N"))
_c(CloserModel("magnetic_hold", "Closer w/ electromagnetic hold-open (fire alarm release)", "electromagnetic_hold", None, 90.0, 12.0, 1.22, 60.0, 1.3, 1.5708, False, 4.5, "door_push", (0.29, 0.07, 0.10), "LCN 4040SE / Sentronic"))
_c(CloserModel("auto_low_energy", "Low-energy automatic operator (push-to-open / knowing act)", "auto_operator_low_energy", None, 60.0, 15.0, None, 0.0, 1.0, None, False, 9.0, "door_push", (0.68, 0.11, 0.11), "LCN Senior Swing / Norton 6000; ANSI A156.19 <= 15 lbf on stall"))
_c(CloserModel("auto_full_energy", "Full-energy automatic swing operator (sensor activated)", "auto_operator_full", None, 60.0, 15.0, None, 0.0, 1.0, None, False, 12.0, "frame", (0.90, 0.13, 0.13), "ANSI A156.10 full-energy swing operator"))


# ---------------------------------------------------------------------------
# Hinges
# ---------------------------------------------------------------------------
@dataclass
class HingeModel:
    id: str
    name: str
    kind: str            # butt | continuous | pivot_offset | pivot_center | spring | strap | concealed | lift_off | rising_butt | cam_lift | double_action | gravity_pivot | flap_pin | rotor
    bearing: str         # key into HINGE_BEARING_MU
    pin_radius: float    # m
    thrust_radius: float # m (knuckle contact mean radius)
    count_default: int
    size: tuple          # leaf size (h, w) m
    mass_each: float     # kg
    max_door_mass: float # kg rated
    axis_tilt_deg: float # rising butt / gravity closing: tilt of hinge axis (deg) producing gravity return
    source: str

    def to_dict(self):
        return asdict(self)


HINGES: dict[str, HingeModel] = {}


def _h(m: HingeModel):
    HINGES[m.id] = m
    return m


_h(HingeModel("butt_35_plain", "3.5 in residential butt hinge, plain bearing", "butt", "plain_bearing_new", 0.0040, 0.0060, 3, (0.089, 0.089), 0.18, 40, 0, "Hager 1741 3.5x3.5"))
_h(HingeModel("butt_35_worn", "3.5 in residential butt hinge, worn/dry", "butt", "plain_bearing_worn", 0.0040, 0.0060, 3, (0.089, 0.089), 0.18, 40, 0, "Hager 1741 aged"))
_h(HingeModel("butt_45_plain", "4.5 in commercial butt hinge, plain bearing", "butt", "plain_bearing_new", 0.0048, 0.0075, 3, (0.114, 0.114), 0.32, 90, 0, "Hager 1279 4.5x4.5"))
_h(HingeModel("butt_45_bb", "4.5 in heavy weight ball-bearing hinge", "butt", "ball_bearing", 0.0048, 0.0080, 3, (0.114, 0.114), 0.45, 180, 0, "Hager BB1199 / Ives 5BB1HW"))
_h(HingeModel("butt_45_bb_4", "4.5 in ball-bearing hinge (4 per door, tall/heavy)", "butt", "ball_bearing", 0.0048, 0.0080, 4, (0.114, 0.114), 0.45, 250, 0, "Hager BB1199 x4"))
_h(HingeModel("butt_5_bb_heavy", "5 in heavy-weight BB hinge (fire/detention)", "butt", "ball_bearing", 0.0055, 0.0090, 4, (0.127, 0.114), 0.62, 350, 0, "Ives 5BB1HW 5x4.5"))
_h(HingeModel("butt_rusty", "Rusty exterior butt hinge (seized)", "butt", "rusty", 0.0048, 0.0075, 3, (0.114, 0.114), 0.32, 90, 0, "Aged un-lubricated steel hinge"))
_h(HingeModel("continuous_geared", "Continuous geared aluminum hinge (full height)", "continuous", "continuous_geared", 0.0060, 0.0100, 1, (2.1, 0.045), 4.5, 300, 0, "Select SL11 / Pemko CFM"))
_h(HingeModel("piano", "Continuous piano hinge (steel)", "continuous", "piano", 0.0030, 0.0040, 1, (2.0, 0.038), 2.2, 60, 0, "Steel piano hinge 1.5 in open"))
_h(HingeModel("pivot_offset", "Offset pivot set (3/4 in offset) w/ intermediate pivot", "pivot_offset", "pivot_thrust", 0.0080, 0.0150, 2, (0.10, 0.05), 1.2, 200, 0, "Rixson 117/127"))
_h(HingeModel("pivot_center", "Center-hung pivot (floor + top)", "pivot_center", "pivot_thrust", 0.0120, 0.0200, 2, (0.12, 0.06), 2.0, 500, 0, "Rixson 340 / Dorma BTS"))
_h(HingeModel("pivot_center_heavy", "Heavy center pivot (large pivot door, 400 kg)", "pivot_center", "pivot_thrust", 0.0200, 0.0350, 2, (0.15, 0.08), 4.0, 900, 0, "FritsJurgens System M class"))
_h(HingeModel("spring_single", "Single-acting spring hinge 4 in", "spring", "spring_hinge", 0.0045, 0.0070, 3, (0.102, 0.102), 0.30, 60, 0, "Bommer 4310"))
_h(HingeModel("spring_double", "Double-acting spring hinge (saloon/cafe)", "double_action", "double_action_spring", 0.0060, 0.0080, 2, (0.15, 0.10), 0.55, 40, 0, "Bommer 3029 / Rixson double acting"))
_h(HingeModel("strap_pintle", "Strap hinge on pintle (barn/gate)", "strap", "strap_pintle", 0.0080, 0.0100, 2, (0.30, 0.05), 0.9, 120, 0, "12 in strap hinge, 5/8 in pintle"))
_h(HingeModel("strap_heavy", "Heavy strap hinge (castle/warehouse)", "strap", "strap_pintle", 0.0125, 0.0160, 3, (0.50, 0.08), 3.0, 400, 0, "20 in wrought strap hinge"))
_h(HingeModel("concealed_soss", "Concealed invisible hinge (SOSS 218)", "concealed", "concealed_soss", 0.0035, 0.0060, 3, (0.117, 0.028), 0.35, 60, 0, "SOSS 218"))
_h(HingeModel("lift_off", "Lift-off flag hinge (hospital/cabinet room)", "lift_off", "lift_off", 0.0050, 0.0080, 3, (0.10, 0.06), 0.30, 80, 0, "Flag hinge 8 mm pin"))
_h(HingeModel("rising_butt", "Rising butt hinge (helical, self-closing)", "rising_butt", "rising_butt", 0.0048, 0.0075, 3, (0.102, 0.076), 0.30, 50, 3.0, "Rising butt hinge; 3 deg effective axis tilt"))
_h(HingeModel("cam_lift", "Cam-lift hinge (cold storage, self-closing)", "cam_lift", "cam_lift", 0.0080, 0.0120, 2, (0.20, 0.10), 1.5, 250, 4.0, "Kason 1245 cam-lift; rises 12 mm over 90 deg"))
_h(HingeModel("gravity_pivot", "Gravity self-closing pivot (toilet partition)", "gravity_pivot", "gravity_pivot", 0.0050, 0.0100, 2, (0.06, 0.04), 0.25, 30, 2.5, "Bobrick gravity hinge; closes from < 90 deg"))
_h(HingeModel("flap_pin", "Pet flap pivot pin", "flap_pin", "pet_flap_pin", 0.0025, 0.0030, 1, (0.02, 0.02), 0.02, 2, 0, "Plastic pin hinge"))
_h(HingeModel("rotor_bearing", "Rotor bearing (revolving door / turnstile)", "rotor", "rotor_bearing", 0.0250, 0.0400, 1, (0.10, 0.10), 5.0, 1200, 0, "Slewing/thrust bearing"))
_h(HingeModel("garden_tee", "Tee hinge (garden gate)", "strap", "strap_pintle", 0.0060, 0.0080, 2, (0.25, 0.045), 0.5, 40, 0, "10 in tee hinge"))
_h(HingeModel("chain_link_hinge", "Chain-link gate hinge (male/female post hinge)", "strap", "plain_bearing_new", 0.0080, 0.0100, 2, (0.10, 0.05), 0.6, 80, 0, "Bulldog gate hinge"))
_h(HingeModel("vault_hinge", "Vault door hinge (crane hinge)", "butt", "ball_bearing", 0.0300, 0.0500, 2, (0.40, 0.15), 20.0, 3000, 0, "Crane hinge with thrust bearing"))
_h(HingeModel("ship_hinge", "Marine door hinge (bronze bushing)", "butt", "bronze_bushing", 0.0100, 0.0150, 2, (0.15, 0.10), 1.5, 300, 0, "Watertight door hinges"))
_h(HingeModel("hatch_hinge", "Hatch hinge (heavy steel)", "butt", "plain_bearing_new", 0.0080, 0.0120, 2, (0.15, 0.10), 0.8, 150, 0, "Trapdoor hinge"))
_h(HingeModel("baby_gate", "Baby gate hinge (plastic bushing)", "butt", "nylon_bushing", 0.0040, 0.0060, 2, (0.05, 0.03), 0.08, 10, 0, "Pressure gate hinge"))


# ---------------------------------------------------------------------------
# Operators (what the robot grabs / pushes)
# ---------------------------------------------------------------------------
@dataclass
class OperatorModel:
    id: str
    name: str
    kind: str                 # lever | knob | pull | push_plate | panic_touchbar | panic_crossbar | paddle | thumb_latch | wheel | flush_pull | ring_pull | none | t_handle | cremone | handleset | slide_bolt_handle | keypad_lever | keypad_deadbolt | card_lever | push_button_screen | lift_latch | hook_lock_slider | gate_latch_fork | hasp
    motion: str               # rotate_normal (about door-normal axis) | push_in (prismatic along normal) | none | rotate_vertical | rotate_horizontal | lift
    travel: float             # rad for rotation, m for translation (full actuation)
    dead_travel: float        # rad/m before latch begins to retract (backlash)
    spring_torque_preload: float   # N*m at rest (rotational) or N (translational)
    spring_rate: float        # N*m/rad or N/m
    operable_force_limit: float    # N at grip point per code (ADA 22.2 N / IBC 66.7 N) for QA
    grip_offset: float        # m from spindle to grip point (lever length / knob radius / bar depth)
    yield_torque: float       # N*m torque that damages the operator (or N for translational)
    mass: float               # kg for the operator set (both sides)
    both_sides: bool          # is operator present on both faces
    style_params: dict        # shape family & dims for the mesh builder
    material: str
    source: str
    unlatches: bool = True    # does actuating retract a latch bolt

    def to_dict(self):
        return asdict(self)


OPERATORS: dict[str, OperatorModel] = {}


def _o(m: OperatorModel):
    OPERATORS[m.id] = m
    return m


# Levers (ADA compliant).  Return spring: typical lever set requires 0.4-1.2 N*m at start, ~2 N*m at full 55 deg.
_o(OperatorModel("lever_straight", "Straight tubular lever (commercial Grade 1)", "lever", "rotate_normal", 0.96, 0.05, 0.45, 1.2, 22.2, 0.115, 25.0, 1.4, True,
                 {"shape": "straight", "length": 0.125, "diameter": 0.019, "rose_diameter": 0.070, "return": False}, "nickel_satin", "Schlage ND series Rhodes lever; ANSI A156.2 Grade 1"))
_o(OperatorModel("lever_return", "Return-to-door lever (hospital/ADA)", "lever", "rotate_normal", 0.96, 0.05, 0.45, 1.2, 22.2, 0.110, 25.0, 1.5, True,
                 {"shape": "return", "length": 0.120, "diameter": 0.019, "rose_diameter": 0.070, "return": True}, "stainless", "Schlage ND Athens / Sparta return lever"))
_o(OperatorModel("lever_curved", "Curved wave lever (residential)", "lever", "rotate_normal", 0.87, 0.08, 0.30, 0.9, 22.2, 0.105, 15.0, 1.0, True,
                 {"shape": "wave", "length": 0.112, "diameter": 0.016, "rose_diameter": 0.066, "return": False}, "chrome", "Kwikset Halifax / Schlage Latitude"))
_o(OperatorModel("lever_l_shape", "L-shaped square lever (modern)", "lever", "rotate_normal", 0.87, 0.06, 0.35, 1.0, 22.2, 0.115, 18.0, 1.1, True,
                 {"shape": "L", "length": 0.120, "diameter": 0.014, "rose_diameter": 0.060, "return": False, "square": True}, "black_matte_metal", "Emtek Hercules / Schlage Latitude square"))
_o(OperatorModel("lever_mortise_escutcheon", "Mortise lever on long escutcheon plate", "lever", "rotate_normal", 0.70, 0.04, 0.50, 1.4, 22.2, 0.125, 30.0, 2.2, True,
                 {"shape": "straight", "length": 0.130, "diameter": 0.020, "rose_diameter": 0.0, "escutcheon": (0.24, 0.045), "return": False}, "brass", "Sargent 8200 mortise lock w/ LE1 escutcheon"))
_o(OperatorModel("lever_euro_backplate", "European lever on backplate (DIN 18255)", "lever", "rotate_normal", 0.65, 0.04, 0.40, 1.0, 22.2, 0.120, 25.0, 1.6, True,
                 {"shape": "straight", "length": 0.130, "diameter": 0.019, "rose_diameter": 0.0, "escutcheon": (0.22, 0.040), "return": False}, "stainless", "Hoppe Amsterdam lever DIN backplate"))
_o(OperatorModel("lever_loose", "Worn lever with excess play (loose spindle)", "lever", "rotate_normal", 0.96, 0.20, 0.20, 0.8, 22.2, 0.115, 12.0, 1.3, True,
                 {"shape": "straight", "length": 0.125, "diameter": 0.019, "rose_diameter": 0.070, "return": False}, "brass_antique", "Aged lever, 11 deg free play"))
_o(OperatorModel("lever_keypad", "Electronic keypad lever set (residential)", "keypad_lever", "rotate_normal", 0.87, 0.06, 0.35, 1.0, 22.2, 0.110, 18.0, 1.6, True,
                 {"shape": "wave", "length": 0.112, "diameter": 0.016, "rose_diameter": 0.070, "keypad": (0.070, 0.150), "return": False, "keys": 10}, "nickel_satin", "Schlage FE595 keypad lever"))
_o(OperatorModel("lever_card_reader", "Lever w/ integrated card reader (hotel)", "card_lever", "rotate_normal", 0.87, 0.06, 0.40, 1.1, 22.2, 0.115, 20.0, 2.0, True,
                 {"shape": "straight", "length": 0.120, "diameter": 0.019, "rose_diameter": 0.0, "escutcheon": (0.26, 0.075), "reader": True, "return": False}, "chrome", "Onity/dormakaba Saflok RFID"))

# Knobs (non-ADA).  Rotation 45-60 deg to retract latch.
_o(OperatorModel("knob_round", "Round knob (residential passage)", "knob", "rotate_normal", 0.87, 0.06, 0.25, 0.5, 22.2, 0.027, 12.0, 0.9, True,
                 {"shape": "round", "diameter": 0.054, "depth": 0.060, "rose_diameter": 0.064}, "brass", "Kwikset Polo / Schlage Georgian; ANSI A156.2 Grade 2"))
_o(OperatorModel("knob_round_large", "76 mm rounded knob with independent entry trim", "knob", "rotate_normal", 0.87, 0.06, 0.25, 0.5, 22.2, 0.0361, 12.0, 0.9, True,
                 {"shape": "round", "diameter": 0.076, "depth": 0.084, "rose_diameter": 0.070, "contact_profile": True}, "brass",
                 "Original generic solid-brass profile; size precedent Croft 6344M 76 mm knob/70 mm rose (https://croft.co.uk/products/rounded-knob). Same source return spring and per-point force cap; actual material BOM replaces catalogue allowance."))
_o(OperatorModel("knob_round_privacy", "Round knob w/ privacy push-button (bath/bedroom)", "knob", "rotate_normal", 0.87, 0.06, 0.25, 0.5, 22.2, 0.027, 12.0, 0.9, True,
                 {"shape": "round", "diameter": 0.054, "depth": 0.060, "rose_diameter": 0.064, "privacy_button": True}, "nickel_satin", "Kwikset Tylo privacy"))
_o(OperatorModel("knob_egg", "Egg/oval knob", "knob", "rotate_normal", 0.87, 0.06, 0.25, 0.5, 22.2, 0.030, 12.0, 0.95, True,
                 {"shape": "egg", "diameter": 0.056, "depth": 0.066, "rose_diameter": 0.064}, "bronze", "Baldwin egg knob"))
_o(OperatorModel("knob_glass_antique", "Antique faceted glass knob (loose)", "knob", "rotate_normal", 0.80, 0.15, 0.15, 0.35, 22.2, 0.028, 6.0, 0.6, True,
                 {"shape": "faceted", "diameter": 0.052, "depth": 0.058, "rose_diameter": 0.055}, "glass_clear", "Vintage 1920s glass knob w/ mortise lock"))
_o(OperatorModel("knob_porcelain", "Porcelain knob (vintage)", "knob", "rotate_normal", 0.85, 0.10, 0.20, 0.4, 22.2, 0.028, 8.0, 0.6, True,
                 {"shape": "round", "diameter": 0.055, "depth": 0.060, "rose_diameter": 0.055}, "hpl", "Vintage porcelain knob"))
_o(OperatorModel("knob_rim_lock", "Knob w/ surface-mounted rim lock (heritage)", "knob", "rotate_normal", 0.85, 0.08, 0.30, 0.6, 22.2, 0.028, 10.0, 1.6, True,
                 {"shape": "round", "diameter": 0.050, "depth": 0.060, "rose_diameter": 0.050, "rim_box": (0.10, 0.16, 0.03)}, "cast_iron", "Carpenter rim lock"))
_o(OperatorModel("knob_childproof", "Knob with free-spinning cover (grip through openings)", "knob", "rotate_normal", 0.87, 0.06, 0.25, 0.5, 22.2, 0.040, 8.0, 0.95, True,
                 {"shape": "round", "diameter": 0.054, "depth": 0.060, "rose_diameter": 0.064, "childproof_cover": 0.082}, "pvc", "Generic access-hole cover, informed by Safety 1st Grip 'n Twist instructions; no child-resistance certification"))
_o(OperatorModel("knob_keypad_deadbolt", "Knob + electronic keypad deadbolt above", "keypad_deadbolt", "rotate_normal", 0.87, 0.06, 0.25, 0.5, 22.2, 0.027, 12.0, 1.4, True,
                 {"shape": "round", "diameter": 0.054, "depth": 0.060, "rose_diameter": 0.064, "keypad": (0.070, 0.150), "keys": 10}, "nickel_satin", "Schlage BE365 keypad deadbolt + passage knob"))

# Pulls / push plates (no latch)
_o(OperatorModel("pull_bar_offset", "Offset pull bar 1 in dia (storefront)", "pull", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.075, 3000.0, 1.2, False,
                 {"shape": "offset_bar", "length": 0.30, "diameter": 0.025, "standoff": 0.075}, "stainless", "Rockwood BF158 offset pull", unlatches=False))
_o(OperatorModel("pull_d", "D-pull handle 200 mm", "pull", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.060, 2000.0, 0.5, False,
                 {"shape": "d_pull", "length": 0.20, "diameter": 0.019, "standoff": 0.060}, "stainless", "Ives 8103 straight pull", unlatches=False))
_o(OperatorModel("pull_ladder_full", "Full-height ladder pull (pivot/glass door)", "pull", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.060, 4000.0, 6.0, True,
                 {"shape": "ladder", "length": 1.80, "diameter": 0.032, "standoff": 0.065}, "stainless", "Back-to-back ladder pull 72 in", unlatches=False))
_o(OperatorModel("pull_ring", "Ring pull (castle/gate)", "ring_pull", "rotate_horizontal", 1.2, 0.0, 0.0, 0.0, 22.2, 0.060, 3000.0, 0.8, False,
                 {"shape": "ring", "ring_diameter": 0.12, "bar_diameter": 0.014}, "wrought_iron", "Iron ring pull on backplate", unlatches=False))
_o(OperatorModel("pull_flush_recessed", "Recessed flush pull (pocket / bypass door)", "flush_pull", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.012, 500.0, 0.2, True,
                 {"shape": "flush", "size": (0.05, 0.10), "depth": 0.012}, "nickel_satin", "Ives 221 flush pull", unlatches=False))
_o(OperatorModel("pull_finger_cup", "Round finger cup (bypass closet)", "flush_pull", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.008, 300.0, 0.05, False,
                 {"shape": "cup", "diameter": 0.055, "depth": 0.010}, "nickel_satin", "Finger pull cup 2-1/8 in", unlatches=False))
_o(OperatorModel("pull_barn_iron", "Flat-bar barn door pull (12 in)", "pull", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.045, 3000.0, 0.7, True,
                 {"shape": "flat_bar", "length": 0.30, "width": 0.032, "standoff": 0.045}, "black_matte_metal", "Rustic flat pull", unlatches=False))
_o(OperatorModel("push_plate", "Push plate 4x16 in (no latch)", "push_plate", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.0, 5000.0, 0.4, False,
                 {"shape": "plate", "size": (0.10, 0.40)}, "stainless", "Rockwood 70C push plate", unlatches=False))
_o(OperatorModel("none", "No operator (push/pull on face)", "none", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.0, 1e9, 0.0, False, {}, "steel", "", unlatches=False))
_o(OperatorModel("pull_t_handle_garage", "T-handle w/ lock (garage / tilt-up)", "t_handle", "rotate_normal", 1.5708, 0.1, 0.3, 0.4, 22.2, 0.05, 15.0, 0.4, False,
                 {"shape": "T", "length": 0.11, "diameter": 0.016}, "black_matte_metal", "Garage T-handle lock"))
_o(OperatorModel("pull_lift_garage", "Lift handle (garage exterior/interior step plate)", "pull", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.05, 2000.0, 0.5, True,
                 {"shape": "lift_handle", "length": 0.20, "width": 0.03, "standoff": 0.05}, "black_matte_metal", "Garage lift handle", unlatches=False))

# Exit devices (UL 305).  Touch bar travel ~ 12-19 mm; unlatch <= 67 N (IBC); ADA egress <= 22 N.
_o(OperatorModel("panic_touchbar_rim", "Rim exit device, touch bar (Von Duprin 99 class)", "panic_touchbar", "push_in", 0.016, 0.002, 18.0, 1200.0, 66.7, 0.0, 800.0, 5.5, False,
                 {"shape": "touchbar", "bar_length_frac": 0.65, "bar_height": 0.05, "bar_depth": 0.065, "rim_case": True}, "stainless", "Von Duprin 99 rim; 3/4 in pad travel; ~15 lbf unlatch"))
_o(OperatorModel("panic_touchbar_rim_light", "Rim exit device, touch bar, light spring (ADA-tuned)", "panic_touchbar", "push_in", 0.016, 0.002, 8.0, 700.0, 22.2, 0.0, 800.0, 5.0, False,
                 {"shape": "touchbar", "bar_length_frac": 0.65, "bar_height": 0.05, "bar_depth": 0.065, "rim_case": True}, "aluminum", "Sargent 80 series w/ reduced spring tension"))
_o(OperatorModel("panic_touchbar_svr", "Surface vertical rod exit device (top & bottom latches)", "panic_touchbar", "push_in", 0.016, 0.002, 22.0, 1400.0, 66.7, 0.0, 800.0, 8.5, False,
                 {"shape": "touchbar", "bar_length_frac": 0.65, "bar_height": 0.05, "bar_depth": 0.065, "vertical_rods": True}, "stainless", "Von Duprin 9927 SVR"))
_o(OperatorModel("panic_touchbar_mortise", "Mortise exit device w/ touch bar", "panic_touchbar", "push_in", 0.016, 0.002, 20.0, 1300.0, 66.7, 0.0, 800.0, 6.5, False,
                 {"shape": "touchbar", "bar_length_frac": 0.60, "bar_height": 0.05, "bar_depth": 0.065, "rim_case": False}, "stainless", "Von Duprin 9975 mortise"))
_o(OperatorModel("panic_touchbar_stiff", "Old exit device, stiff/sticky (poor maintenance)", "panic_touchbar", "push_in", 0.016, 0.004, 45.0, 2500.0, 66.7, 0.0, 800.0, 6.0, False,
                 {"shape": "touchbar", "bar_length_frac": 0.65, "bar_height": 0.05, "bar_depth": 0.065, "rim_case": True}, "steel_painted", "Aged exit device near code max 15 lbf"))
_o(OperatorModel("panic_crossbar", "Crossbar exit device (Von Duprin 88 class)", "panic_crossbar", "rotate_horizontal", 0.35, 0.03, 4.0, 12.0, 66.7, 0.06, 600.0, 6.0, False,
                 {"shape": "crossbar", "bar_length_frac": 0.75, "bar_diameter": 0.025, "arm_length": 0.06}, "brass", "Von Duprin 88 crossbar; 20 deg arc"))
_o(OperatorModel("panic_touchbar_alarm", "Exit device w/ alarm & delayed egress (15 s)", "panic_touchbar", "push_in", 0.016, 0.002, 18.0, 1200.0, 66.7, 0.0, 800.0, 6.5, False,
                 {"shape": "touchbar", "bar_length_frac": 0.65, "bar_height": 0.05, "bar_depth": 0.065, "rim_case": True, "alarm": True}, "stainless", "Von Duprin Chexit delayed egress"))
_o(OperatorModel("paddle_push_pull", "Push/pull paddle (hospital latch)", "paddle", "rotate_horizontal", 0.40, 0.03, 0.6, 1.5, 22.2, 0.09, 30.0, 1.8, True,
                 {"shape": "paddle", "size": (0.10, 0.18), "standoff": 0.045}, "stainless", "Glynn-Johnson HL6 hospital push/pull latch"))
_o(OperatorModel("paddle_hospital_arm", "Hospital arm-pull paddle (elbow operated)", "paddle", "rotate_horizontal", 0.40, 0.03, 0.5, 1.2, 22.2, 0.12, 30.0, 2.0, True,
                 {"shape": "paddle_arm", "size": (0.12, 0.22), "standoff": 0.06}, "stainless", "Rixson / Glynn-Johnson arm pull"))

# Gate & rustic latches
_o(OperatorModel("thumb_latch_suffolk", "Suffolk thumb latch (garden/cottage)", "thumb_latch", "rotate_horizontal", 0.30, 0.02, 0.3, 0.6, 22.2, 0.04, 20.0, 0.6, True,
                 {"shape": "suffolk", "handle_length": 0.20, "bar_length": 0.18}, "wrought_iron", "Suffolk latch; thumb press lifts latch bar"))
_o(OperatorModel("gate_latch_fork", "Fork gravity gate latch (chain-link)", "gate_latch_fork", "lift", 1.55, 0.005, 2.0, 40.0, 22.2, 0.04, 200.0, 0.6, False,
                 {"shape": "fork", "length": 0.15}, "steel_galvanized", "Chain link fork latch; lift to release"))
_o(OperatorModel("gate_latch_magnetic", "Magnetic pool gate latch (child safe, top-pull)", "lift_latch", "lift", 0.03, 0.002, 6.0, 150.0, 22.2, 0.03, 300.0, 0.9, False,
                 {"shape": "magnalatch", "height": 0.20}, "black_matte_metal", "D&D MagnaLatch; 1.5 m mounting height"))
_o(OperatorModel("hasp_padlock", "Hasp & staple w/ padlock", "hasp", "rotate_vertical", 1.5708, 0.0, 0.0, 0.0, 22.2, 0.03, 500.0, 0.5, False,
                 {"shape": "hasp", "length": 0.11}, "steel_galvanized", "4.5 in hasp; padlock must be removed first", unlatches=False))
_o(OperatorModel("slide_bolt_barrel", "Barrel bolt (slide bolt) 4 in", "slide_bolt_handle", "lift", 0.045, 0.0, 1.0, 20.0, 22.2, 0.02, 400.0, 0.2, False,
                 {"shape": "barrel_bolt", "length": 0.10, "diameter": 0.012}, "brass", "Barrel bolt; slide to release"))
_o(OperatorModel("slide_bolt_heavy", "Heavy gate slide bolt / drop bar", "slide_bolt_handle", "lift", 0.08, 0.0, 3.0, 40.0, 66.7, 0.05, 1500.0, 1.2, False,
                 {"shape": "slide_bolt_heavy", "length": 0.30, "diameter": 0.02}, "steel_galvanized", "Cane bolt / slide bolt 12 in"))
_o(OperatorModel("cane_bolt_drop", "Drop rod (cane bolt) for inactive gate leaf", "slide_bolt_handle", "lift", 0.10, 0.0, 4.0, 30.0, 66.7, 0.05, 1500.0, 1.0, False,
                 {"shape": "cane_bolt", "length": 0.45, "diameter": 0.016}, "steel_galvanized", "Cane bolt"))

# Sliding-door hardware
_o(OperatorModel("hook_lock_slider", "Patio slider handle w/ hook lock (thumb latch)", "hook_lock_slider", "rotate_normal", 1.0, 0.05, 0.3, 0.5, 22.2, 0.06, 15.0, 0.9, True,
                 {"shape": "slider_handle", "length": 0.20, "hook": True}, "black_matte_metal", "Milgard/Andersen sliding door handle set"))
_o(OperatorModel("shoji_finger_pull", "Shoji recessed finger pull (hikite)", "flush_pull", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.006, 40.0, 0.03, True,
                 {"shape": "hikite", "size": (0.03, 0.08), "depth": 0.006}, "hinoki", "Traditional hikite", unlatches=False))
_o(OperatorModel("barn_privacy_hook", "Barn door pull + teardrop privacy latch", "pull", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.045, 3000.0, 0.9, True,
                 {"shape": "flat_bar", "length": 0.30, "width": 0.032, "standoff": 0.045, "teardrop_latch": True}, "black_matte_metal", "Teardrop latch", unlatches=False))
_o(OperatorModel("bifold_knob", "Bifold door knob (small)", "knob", "none", 0.0, 0.0, 0.0, 0.0, 22.2, 0.015, 40.0, 0.05, False,
                 {"shape": "round", "diameter": 0.030, "depth": 0.030, "rose_diameter": 0.0}, "nickel_satin", "Bifold knob", unlatches=False))
_o(OperatorModel("elevator_none", "No operator (automatic)", "none", "none", 0, 0, 0, 0, 22.2, 0, 1e9, 0, False, {}, "stainless", "", unlatches=False))

# Heavy/industrial
_o(OperatorModel("wheel_vault", "Vault door handwheel (multi-bolt)", "wheel", "rotate_normal", 6.2832, 0.1, 0.0, 0.0, 66.7, 0.20, 500.0, 6.0, False,
                 {"shape": "spoked_wheel", "diameter": 0.40, "spokes": 5, "bar_diameter": 0.022}, "stainless", "Vault handwheel; 1-2 turns throws 4-8 bolts"))
_o(OperatorModel("wheel_ship_hatch", "Ship hatch handwheel (central dogging)", "wheel", "rotate_normal", 9.4248, 0.1, 0.0, 0.0, 66.7, 0.18, 500.0, 5.0, True,
                 {"shape": "spoked_wheel", "diameter": 0.36, "spokes": 4, "bar_diameter": 0.020}, "brass", "Quick-acting watertight door wheel"))
_o(OperatorModel("dog_lever", "Individual dog lever (watertight door)", "lever", "rotate_normal", 1.5708, 0.05, 1.5, 2.0, 66.7, 0.20, 200.0, 1.2, True,
                 {"shape": "dog", "length": 0.22, "diameter": 0.025}, "steel_painted", "Watertight door dogs, 6-8 per door"))
_o(OperatorModel("vault_lever", "Independent vault bolt lever (quarter turn)", "lever", "rotate_normal", 1.5708, 0.0, 0.0, 0.0, 66.7, 0.20, 200.0, 1.2, True,
                 {"shape": "dog", "length": 0.22, "diameter": 0.025, "standoff": 0.14}, "stainless", "Original supported crank/rod boltwork; actual geometry BOM replaces this legacy mass allowance; no strength rating"))
_o(OperatorModel("cold_storage_handle", "Cold storage door handle w/ inside release", "lever", "rotate_horizontal", 0.5, 0.03, 1.2, 2.5, 66.7, 0.15, 60.0, 2.5, True,
                 {"shape": "safeguard", "length": 0.25}, "chrome", "Kason 58 SafeGuard latch w/ inside release"))
_o(OperatorModel("cremone_bolt", "Cremone bolt (french door)", "cremone", "rotate_normal", 1.5708, 0.05, 0.4, 0.6, 22.2, 0.08, 20.0, 2.5, False,
                 {"shape": "cremone", "rod_length": 1.6}, "brass", "Cremone bolt; rotating knob drives top/bottom rods"))
_o(OperatorModel("handleset_thumb", "Entry handleset w/ thumb latch (exterior) + knob (interior)", "handleset", "rotate_horizontal", 0.35, 0.03, 0.5, 1.2, 22.2, 0.05, 20.0, 1.9, True,
                 {"shape": "handleset", "grip_length": 0.24, "plate": (0.30, 0.07)}, "bronze", "Kwikset Arlington / Schlage Camelot handleset"))
_o(OperatorModel("push_button_screen", "Push-button screen door latch", "push_button_screen", "push_in", 0.006, 0.001, 3.0, 400.0, 22.2, 0.0, 50.0, 0.3, True,
                 {"shape": "screen_latch", "size": (0.03, 0.08)}, "aluminum", "Wright Products push-button latch"))
_o(OperatorModel("baby_gate_latch", "Baby gate lift-and-swing latch", "lift_latch", "lift", 0.02, 0.002, 8.0, 300.0, 22.2, 0.03, 200.0, 0.3, False,
                 {"shape": "gate_latch", "size": (0.05, 0.12)}, "pvc", "Regalo / Munchkin pressure gate latch"))
_o(OperatorModel("stall_slide_latch", "Toilet stall slide latch (indicator)", "slide_bolt_handle", "lift", 0.03, 0.0, 1.5, 30.0, 22.2, 0.02, 100.0, 0.2, True,
                 {"shape": "stall_latch", "size": (0.03, 0.06)}, "stainless", "Bobrick partition slide latch"))
_o(OperatorModel("turnstile_arm", "Tripod turnstile arm (push through)", "none", "none", 0, 0, 0, 0, 66.7, 0.45, 1e4, 0, False, {"shape": "tripod"}, "stainless", "Tripod turnstile", unlatches=False))
_o(OperatorModel("hatch_ring", "Recessed hatch ring pull", "ring_pull", "rotate_horizontal", 1.5708, 0.0, 0.0, 0.0, 66.7, 0.04, 3000.0, 0.4, False,
                 {"shape": "recessed_ring", "ring_diameter": 0.08}, "steel_galvanized", "Flush hatch ring", unlatches=False))
_o(OperatorModel("mail_slot", "Mail slot flap (spring)", "none", "rotate_horizontal", 1.2, 0.0, 0.2, 0.3, 22.2, 0.05, 20.0, 0.5, False, {"shape": "mail_slot", "size": (0.30, 0.08)}, "brass", "Letterbox plate", unlatches=False))


# ---------------------------------------------------------------------------
# Latches & locks
# ---------------------------------------------------------------------------
@dataclass
class LatchModel:
    id: str
    name: str
    kind: str              # tubular_latch | deadlatch | mortise_latch | rim_latch | vertical_rods | hook | roller | magnetic | none | gravity_bar | slide_bolt | ball_catch | dogs | multi_bolt | electric_strike | mag_lock | electric_bolt
    throw: float           # m bolt extension into strike (retracts fully on actuation)
    bolt_size: tuple       # (width along door thickness, height) m
    spring_preload: float  # N holding bolt extended
    spring_rate: float     # N/m
    backset: float         # m from latch edge to spindle center
    holding_force: float   # N (for magnets/mag locks; 0 for mechanical)
    yield_force: float     # N shear force that breaks the latch (door damage label)
    source: str
    strike_type: str = "curved_lip"   # curved_lip | ASA_4_7_8 | box | keeper | T_strike | electric

    def to_dict(self):
        return asdict(self)


LATCHES: dict[str, LatchModel] = {}


def _l(m: LatchModel):
    LATCHES[m.id] = m
    return m


_l(LatchModel("none", "No latch (free swinging / friction only)", "none", 0.0, (0, 0), 0, 0, 0.0, 0, 1e9, ""))
_l(LatchModel("tubular_residential", "Tubular spring latch 1/2 in throw (Grade 3)", "tubular_latch", 0.0127, (0.0127, 0.0254), 2.5, 300, 0.060, 0, 2000, "ANSI A156.2 Grade 3; 2-3/8 in backset"))
_l(LatchModel("tubular_residential_70", "Tubular spring latch 1/2 in throw, 2-3/4 in backset", "tubular_latch", 0.0127, (0.0127, 0.0254), 2.5, 300, 0.070, 0, 2000, "ANSI A156.2 Grade 2"))
_l(LatchModel("deadlatch_grade1", "Deadlatch 3/4 in throw w/ auxiliary bolt (Grade 1)", "deadlatch", 0.019, (0.016, 0.0286), 6.0, 600, 0.070, 0, 4500, "Schlage ND / A156.2 Grade 1", strike_type="ASA_4_7_8"))
_l(LatchModel("mortise_latch", "Mortise lock latch bolt 3/4 in + deadbolt 1 in", "mortise_latch", 0.019, (0.016, 0.028), 5.5, 550, 0.070, 0, 6000, "Sargent 8200 / Schlage L9000", strike_type="ASA_4_7_8"))
_l(LatchModel("mortise_euro", "Euro mortise sashlock (DIN), 55 mm backset", "mortise_latch", 0.014, (0.010, 0.022), 4.0, 400, 0.055, 0, 4000, "DIN 18251 class 3"))
_l(LatchModel("rim_exit", "Rim exit device latch (Pullman 3/4 in)", "rim_latch", 0.019, (0.019, 0.032), 8.0, 800, 0.0, 0, 8000, "Von Duprin 99 rim strike 299", strike_type="box"))
_l(LatchModel("vertical_rods", "Surface vertical rod latches (top + bottom)", "vertical_rods", 0.016, (0.016, 0.025), 10.0, 900, 0.0, 0, 8000, "Von Duprin 9927 top/bottom", strike_type="box"))
_l(LatchModel("roller_latch", "Roller latch (holds door closed by friction)", "roller", 0.006, (0.012, 0.030), 12.0, 3000, 0.0, 0, 200, "Ives RL30 roller latch; ~12 N push to release"))
_l(LatchModel("ball_catch", "Ball catch (double door / closet)", "ball_catch", 0.004, (0.012, 0.012), 15.0, 4000, 0.0, 0, 150, "Ball catch; ~15-25 N to release"))
_l(LatchModel("magnetic_catch", "Magnetic catch (closet)", "magnetic", 0.0, (0.012, 0.030), 0, 0, 0.0, 35.0, 60, "Magnetic catch 35 N pull"))
_l(LatchModel("magnetic_gasket", "Magnetic gasket (cold storage full perimeter)", "magnetic", 0.0, (0.012, 0.030), 0, 0, 0.0, 120.0, 300, "Cold-storage magnetic gasket ~ 30-50 N/m"))
_l(LatchModel("hook_slider", "Hook lock (sliding patio)", "hook", 0.020, (0.010, 0.040), 2.0, 200, 0.030, 0, 3000, "Sliding patio hook bolt"))
_l(LatchModel("gravity_bar", "Gravity latch bar (Suffolk / barn)", "gravity_bar", 0.025, (0.008, 0.030), 0.0, 0, 0.0, 0, 1500, "Latch bar rests in keeper by gravity"))
_l(LatchModel("slide_bolt", "Slide bolt (manual)", "slide_bolt", 0.030, (0.012, 0.012), 0.0, 0, 0.0, 0, 3000, "Barrel bolt"))
_l(LatchModel("slide_bolt_heavy", "Heavy slide bolt / drop bar", "slide_bolt", 0.060, (0.020, 0.020), 0.0, 0, 0.0, 0, 15000, "Cane bolt"))
_l(LatchModel("dogs_6", "Watertight door dogs (6 wedge dogs)", "dogs", 0.030, (0.020, 0.050), 0.0, 0, 0.0, 0, 50000, "Quick-acting WT door"))
_l(LatchModel("vault_bolts_2", "Vault boltwork (2 independently driven 32 mm bolts)", "multi_bolt", 0.050, (0.032, 0.032), 0.0, 0, 0.0, 0, 50000, "Original supported boltwork; 50 kN legacy damage threshold is not a verified strength rating"))
_l(LatchModel("multi_bolt_4", "Vault multi-bolt (4 x 25 mm bolts)", "multi_bolt", 0.040, (0.025, 0.025), 0.0, 0, 0.0, 0, 200000, "Vault boltwork"))
_l(LatchModel("multi_bolt_8", "Vault multi-bolt (8 x 32 mm bolts)", "multi_bolt", 0.050, (0.032, 0.032), 0.0, 0, 0.0, 0, 400000, "Vault boltwork"))
_l(LatchModel("electric_strike", "Electric strike (fail-secure) on tubular latch", "electric_strike", 0.0127, (0.0127, 0.0254), 3.5, 400, 0.070, 0, 3000, "HES 1006; released by access control", strike_type="electric"))
_l(LatchModel("mag_lock_600", "Electromagnetic lock 600 lbf", "mag_lock", 0.0, (0.03, 0.25), 0, 0, 0.0, 2670.0, 2670, "Securitron M32 600 lbf holding"))
_l(LatchModel("mag_lock_1200", "Electromagnetic lock 1200 lbf", "mag_lock", 0.0, (0.035, 0.27), 0, 0, 0.0, 5340.0, 5340, "Securitron M62 1200 lbf"))
_l(LatchModel("electric_bolt", "Electric drop bolt (fail-safe)", "electric_bolt", 0.015, (0.016, 0.016), 0, 0, 0.0, 0, 6000, "Electric deadbolt"))
_l(LatchModel("teardrop", "Teardrop barn privacy latch", "gravity_bar", 0.020, (0.005, 0.05), 0, 0, 0.0, 0, 800, "Teardrop latch"))
_l(LatchModel("fork_gravity", "Fork gravity latch (chain-link)", "gravity_bar", 0.030, (0.012, 0.040), 0, 0, 0.0, 0, 2000, "Fork latch"))
_l(LatchModel("magnalatch", "Magnetic pool gate latch (Magna-latch)", "hook", 0.018, (0.012, 0.030), 4.0, 300, 0.0, 15.0, 2500, "D&D MagnaLatch; magnet pulls latch pin"))
_l(LatchModel("pet_flap_magnet", "Pet flap magnetic closure", "magnetic", 0.0, (0.01, 0.06), 0, 0, 0.0, 4.0, 40, "Flap magnet strip ~ 3-6 N"))
_l(LatchModel("elevator_interlock", "Elevator hoistway interlock (mechanical + electrical)", "electric_bolt", 0.020, (0.02, 0.05), 0, 0, 0.0, 0, 20000, "ASME A17.1 hoistway door interlock"))
_l(LatchModel("garage_slide_lock", "Garage inside slide lock", "slide_bolt", 0.040, (0.012, 0.025), 0, 0, 0.0, 0, 3000, "Garage slide lock"))
_l(LatchModel("stall_slide", "Toilet stall slide latch", "slide_bolt", 0.020, (0.008, 0.030), 0, 0, 0.0, 0, 500, "Partition slide latch"))
_l(LatchModel("screen_pushbutton", "Screen door push-button latch (spring)", "tubular_latch", 0.010, (0.008, 0.015), 2.0, 300, 0.045, 0, 300, "Wright Products"))


@dataclass
class LockModel:
    id: str
    name: str
    kind: str          # none | privacy_button | keyed_cylinder | deadbolt_single | deadbolt_double | mortise_deadbolt | chain | swing_bar_guard | padlock | keypad_code | card_reader | mag_lock | electric_strike | slide_bolt | hook_lock | thumbturn_only | night_latch | multipoint | dogs | vault_wheel | jam_stuck | child_lock_cover | delayed_egress | interlock
    outside_release: str   # none | key | code | card | thumbturn | button | none_locked
    inside_release: str    # thumbturn | button | lever | key | none | rex_button | slide
    handle_backlash_locked: float  # rad free play when locked ("jiggle")
    deadbolt_throw: float  # m (0 if none)
    thumbturn_travel: float  # rad
    thumbturn_torque: float  # N*m needed
    chain_slack: float     # m of opening allowed when chain/guard engaged (0 = n/a)
    mass: float
    source: str

    def to_dict(self):
        return asdict(self)


LOCKS: dict[str, LockModel] = {}


def _k(m: LockModel):
    LOCKS[m.id] = m
    return m


_k(LockModel("none", "No lock", "none", "none", "lever", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ""))
_k(LockModel("privacy_button", "Privacy lock (push-button / turn-button inside; emergency pin outside)", "privacy_button", "none_locked", "button", 0.05, 0.0, 0.0, 0.0, 0.0, 0.05, "Kwikset Tylo privacy"))
_k(LockModel("keyed_cylinder", "Keyed knob/lever cylinder (outside key, inside turn-button)", "keyed_cylinder", "key", "button", 0.05, 0.0, 0.0, 0.0, 0.0, 0.08, "Schlage F51A keyed entry"))
_k(LockModel("deadbolt_single", "Single-cylinder deadbolt (key outside, thumbturn inside), 1 in throw", "deadbolt_single", "key", "thumbturn", 0.0, 0.0254, 1.5708, 0.35, 0.0, 0.55, "ANSI/BHMA A156.5 / A156.40 Grade 1-3; Schlage B60"))
_k(LockModel("deadbolt_double", "Double-cylinder deadbolt (key both sides)", "deadbolt_double", "key", "key", 0.0, 0.0254, 1.5708, 0.35, 0.0, 0.6, "Schlage B62"))
_k(LockModel("mortise_deadbolt", "Mortise lock w/ integral deadbolt (thumbturn inside)", "mortise_deadbolt", "key", "thumbturn", 0.03, 0.0254, 1.5708, 0.40, 0.0, 1.5, "Sargent 8200 function 04"))
_k(LockModel("chain", "Door chain (engaged; limits opening ~ 8 cm)", "chain", "none_locked", "slide", 0.0, 0.0, 0.0, 0.0, 0.08, 0.15, "Door chain 4 in"))
_k(LockModel("swing_bar_guard", "Swing bar door guard (hotel), ~ 6 cm opening", "swing_bar_guard", "none_locked", "slide", 0.0, 0.0, 1.0, 0.1, 0.06, 0.25, "Generic solid bar guard; Ives 482 installation drawing is a dimensional reference, not an OEM replica"))
_k(LockModel("padlock", "Padlock on hasp", "padlock", "key", "key", 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, "Padlock 40 mm"))
_k(LockModel("keypad_code_4", "Electronic keypad (4-digit code) + lever", "keypad_code", "code", "lever", 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, "Schlage FE595"))
_k(LockModel("keypad_code_6", "Electronic keypad (6-digit code) deadbolt", "keypad_code", "code", "thumbturn", 0.0, 0.0254, 1.5708, 0.30, 0.0, 0.8, "Schlage BE365 / Yale Assure"))
_k(LockModel("keypad_mechanical", "Mechanical pushbutton lock (Simplex 1000, 5-button)", "keypad_code", "code", "lever", 0.05, 0.0, 0.0, 0.0, 0.0, 2.2, "Kaba Simplex 1000; 2-5 button combination"))
_k(LockModel("card_reader", "RFID card reader lock (hotel)", "card_reader", "card", "lever", 0.05, 0.0254, 1.5708, 0.30, 0.0, 0.6, "Saflok / Onity"))
_k(LockModel("mag_lock", "Maglock w/ REX button inside (fail-safe)", "mag_lock", "card", "rex_button", 0.0, 0.0, 0.0, 0.0, 0.0, 3.2, "Securitron M62 + REX"))
_k(LockModel("turnstile_index_bolt", "Turnstile credential index bolt (fail-secure; geometry-backed mass)", "credential_index_bolt", "card", "card", 0.0, 0.022, 0.0, 0.0, 0.0, 0.0, "Generic contact slot wheel and spring bolt; actual moving parts use the geometry BOM, no additional catalogue mass"))
_k(LockModel("electric_strike", "Electric strike w/ card reader (fail-secure)", "electric_strike", "card", "lever", 0.05, 0.0, 0.0, 0.0, 0.0, 0.9, "HES 1006 + reader"))
_k(LockModel("slide_bolt", "Slide bolt engaged (inside)", "slide_bolt", "none_locked", "slide", 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, "Barrel bolt"))
_k(LockModel("hook_lock", "Sliding door hook lock (thumb latch inside)", "hook_lock", "key", "thumbturn", 0.0, 0.020, 1.0, 0.25, 0.0, 0.3, "Patio hook lock"))
_k(LockModel("thumbturn_only", "Thumbturn-only deadbolt (interior)", "thumbturn_only", "none_locked", "thumbturn", 0.0, 0.0254, 1.5708, 0.30, 0.0, 0.4, "One-sided deadbolt"))
_k(LockModel("night_latch", "Rim night latch (Yale) w/ snib", "night_latch", "key", "thumbturn", 0.0, 0.016, 1.0, 0.25, 0.0, 0.7, "Yale 77 night latch"))
_k(LockModel("multipoint", "Multipoint lock (uPVC door, lift lever to engage 3 points)", "multipoint", "key", "thumbturn", 0.03, 0.020, 0.0, 0.0, 0.0, 2.5, "Yale/Winkhaus multipoint"))
_k(LockModel("dogs", "Dogged watertight door (6 dogs closed)", "dogs", "lever", "lever", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "WT door dogs"))
_k(LockModel("vault_lever_boltwork", "Two independent vault bolts thrown (both levers must turn)", "vault_lever_boltwork", "lever", "lever", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Original crank/rod bolt carriers; actual native geometry controls release"))
_k(LockModel("vault_wheel", "Vault boltwork thrown (wheel must be turned)", "vault_wheel", "none_locked", "lever", 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, "Vault"))
_k(LockModel("jam_stuck", "Door stuck (swollen / paint-sealed), no lock", "jam_stuck", "none", "lever", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Swollen wood door; breakaway torque added"))
_k(LockModel("child_lock_cover", "Child-proof knob cover", "child_lock_cover", "none", "lever", 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, "Knob cover"))
_k(LockModel("delayed_egress", "Delayed egress (15 s after 3 s push)", "delayed_egress", "none_locked", "lever", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "IBC 1010.1.9.7 delayed egress"))
_k(LockModel("interlock", "Elevator/airlock interlock (opens only when adjacent closed)", "interlock", "none_locked", "none", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Hoistway interlock"))
_k(LockModel("garage_slide_lock", "Garage inside slide lock engaged", "slide_bolt", "none_locked", "slide", 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, "Garage slide lock"))


# ---------------------------------------------------------------------------
# Stops, seals, misc
# ---------------------------------------------------------------------------
STOPS = {
    "none": {"name": "No stop (frame/hinge geometry limits only)", "max_open_deg": 180, "compliant": False},
    "wall_bumper": {"name": "Wall bumper (rubber)", "max_open_deg": 90, "compliant": True},
    "floor_dome": {"name": "Floor dome stop", "max_open_deg": 95, "compliant": True},
    "hinge_pin": {"name": "Hinge-pin stop", "max_open_deg": 100, "compliant": True},
    "overhead_90": {"name": "Overhead stop 90 deg", "max_open_deg": 90, "compliant": False},
    "overhead_105": {"name": "Overhead stop 105 deg", "max_open_deg": 105, "compliant": False},
    "overhead_110_hold": {"name": "Overhead stop & holder 110 deg", "max_open_deg": 110, "compliant": False},
    "wall_180": {"name": "Wall adjacent, folds back 180 deg", "max_open_deg": 175, "compliant": True},
    "corridor_wall_120": {"name": "Perpendicular wall at 120 deg", "max_open_deg": 120, "compliant": True},
    "wedge_jammed": {"name": "Door wedge jammed under leaf (holds at 0-2 deg)", "max_open_deg": 2, "compliant": False},
    "kick_down_holder": {"name": "Kick-down holder (down: holds open)", "max_open_deg": 110, "compliant": False},
}

SEALS = {
    "none": {"name": "No seal", "closing_resistance_N_per_m": 0.0, "compression_m": 0.0, "stiffness_N_per_m2": 0.0},
    "kerf_foam": {"name": "Kerf-in foam weatherstrip", "closing_resistance_N_per_m": 12.0, "compression_m": 0.006, "stiffness_N_per_m2": 2000.0},
    "brush_pile": {"name": "Brush pile seal", "closing_resistance_N_per_m": 6.0, "compression_m": 0.004, "stiffness_N_per_m2": 1500.0},
    "silicone_bulb": {"name": "Silicone bulb seal (commercial)", "closing_resistance_N_per_m": 20.0, "compression_m": 0.008, "stiffness_N_per_m2": 2500.0},
    "smoke_seal_intumescent": {"name": "Smoke seal + intumescent strip (fire door)", "closing_resistance_N_per_m": 25.0, "compression_m": 0.005, "stiffness_N_per_m2": 5000.0},
    "gasket_rubber_heavy": {"name": "Heavy EPDM compression gasket (cold storage/acoustic)", "closing_resistance_N_per_m": 60.0, "compression_m": 0.012, "stiffness_N_per_m2": 5000.0},
    "magnetic_gasket": {"name": "Magnetic gasket (cooler)", "closing_resistance_N_per_m": 40.0, "compression_m": 0.010, "stiffness_N_per_m2": 4000.0},
    "watertight_rubber": {"name": "Watertight knife-edge rubber gasket", "closing_resistance_N_per_m": 150.0, "compression_m": 0.006, "stiffness_N_per_m2": 25000.0},
    "door_sweep": {"name": "Bottom door sweep (drag on floor)", "closing_resistance_N_per_m": 8.0, "compression_m": 0.003, "stiffness_N_per_m2": 800.0},
    "automatic_drop_seal": {"name": "Automatic drop-down seal", "closing_resistance_N_per_m": 15.0, "compression_m": 0.004, "stiffness_N_per_m2": 3000.0},
}

# Standard door sizes (m): nominal US & metric leaf widths and heights
LEAF_WIDTHS_US = [0.610, 0.660, 0.711, 0.762, 0.813, 0.864, 0.914, 0.965, 1.067, 1.219]   # 24,26,28,30,32,34,36,38,42,48 in
LEAF_HEIGHTS_US = [2.032, 2.134, 2.438]   # 80, 84, 96 in
LEAF_WIDTHS_METRIC = [0.626, 0.726, 0.826, 0.926, 1.026, 1.126]   # DIN 18101
LEAF_HEIGHTS_METRIC = [1.985, 2.110, 2.235]
LEAF_THICKNESS = {"residential_interior": 0.035, "residential_exterior": 0.044, "commercial": 0.044, "metric_interior": 0.040, "heavy": 0.057}

# Handle center heights (m).  ADA: 34-48 in (0.864-1.219 m). Typical 38-42 in.
HANDLE_HEIGHTS = {"ada_low": 0.864, "typical_low": 0.914, "typical": 0.965, "typical_high": 1.016, "commercial": 1.040, "ada_high": 1.219, "euro": 1.050, "pet": 0.0}

_k(LockModel("electric_bolt", "Electric drop bolt (fail-safe) w/ card reader outside, REX button inside", "electric_strike", "card", "rex_button", 0.0, 0.015, 0.0, 0.0, 0.0, 1.2, "Electric deadbolt + REX"))
_k(LockModel("hasp", "Hasp & staple (unlocked, no padlock)", "none", "none", "lever", 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, "Hasp"))

_h(HingeModel("none", "No hinge (sliding / lifting door)", "none", "rotor_bearing", 0.0, 0.0, 0, (0, 0), 0.0, 1e6, 0, ""))
