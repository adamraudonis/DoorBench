"""Door specification sampler.

`generate_all(seed)` deterministically produces exactly 1000 DoorSpec dicts
covering every family in taxonomy.FAMILIES with balanced coverage of each
discrete design dimension.
"""
from __future__ import annotations

import math
import random
from typing import Any

from . import materials as M
from . import hardware as H
from . import taxonomy as T
from .panels import glazing_layout, glazing_area_fraction
from .folding import fold_groups, fold_opening_width, fold_opening_height, FOLD_PIVOT_MAX_DEG

IN = 0.0254


class Balanced:
    """Balanced (quota) sampler: cycles through shuffled level lists so every
    level appears proportionally to its weight within a key."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.cycles: dict[str, list] = {}

    def pick(self, key: str, levels):
        cyc = self.cycles.get(key)
        if not cyc:
            if isinstance(levels, dict):
                cyc = []
                for lv, w in levels.items():
                    if w < 0.5:
                        continue
                    cyc.extend([lv] * int(max(1, round(w))))
            else:
                cyc = list(levels)
            self.rng.shuffle(cyc)
            self.cycles[key] = cyc
        return cyc.pop()


def _u(rng, a, b):
    return a + (b - a) * rng.random()


def _round(x, q=0.001):
    return round(round(x / q) * q, 6)


# ---------------------------------------------------------------------------
# Finishes
# ---------------------------------------------------------------------------
PAINT_GRADE = {"hollow_core", "hollow_core_molded", "mdf_solid", "solid_core_pb", "solid_core_scl", "hollow_metal_20ga", "hollow_metal_18ga",
               "hollow_metal_18ga_pu", "hollow_metal_16ga", "hollow_metal_14ga", "steel_entry_24ga", "fiberglass_entry", "upvc_panel",
               "mineral_core_20", "mineral_core_45", "mineral_core_90", "garage_steel_single", "garage_steel_insulated", "steel_plate_security",
               "vault_composite", "blast_steel", "lead_lined", "hospital_solid", "acoustic_wood", "elevator_landing", "ship_watertight",
               "submarine_hatch", "phenolic_partition", "baby_gate_steel", "wrought_iron_gate", "steel_bar_grille", "louver_wood", "screen_wood", "cardboard"}
STAIN_GRADE = {"solid_wood_pine", "solid_wood_fir", "solid_wood_oak", "solid_wood_mahogany", "solid_wood_walnut", "solid_wood_maple", "solid_wood_cherry",
               "solid_wood_teak", "barn_plank", "cedar_plank", "garage_wood_carriage", "cellar_trapdoor", "attic_hatch", "bamboo_solid", "shoji", "fusuma"}

PAINT_PALETTES = {
    "residential_interior": {"white": 6, "off_white": 3, "cream": 1, "light_grey": 2, "sage": 1, "navy": 1, "black": 1, "charcoal": 1, "pink": 1, "mint": 1},
    "residential_exterior": {"black": 2, "navy": 2, "red": 2, "forest_green": 1, "white": 2, "teal": 1, "yellow": 1, "burgundy": 1, "charcoal": 1, "brown": 1},
    "commercial_office": {"grey": 3, "beige": 2, "white": 2, "charcoal": 1, "hotel_walnut": 1, "school_blue": 1},
    "fire_egress": {"grey": 3, "fire_red": 2, "beige": 2, "charcoal": 1, "safety_yellow": 1, "off_white": 1},
    "institutional": {"hospital_green": 2, "school_blue": 2, "beige": 2, "white": 2, "light_grey": 1, "hotel_walnut": 1},
    "industrial_utility": {"grey": 3, "safety_yellow": 1, "olive": 1, "charcoal": 1, "orange": 1, "safety_red": 1},
    "security_detention": {"grey": 3, "beige": 1, "charcoal": 1, "olive": 1},
    "default": {"white": 3, "grey": 2, "black": 1, "beige": 1, "navy": 1},
}


def finish_for(slab_id: str, ctx: str, B: Balanced, rng: random.Random) -> dict:
    slab = M.SLABS[slab_id]
    face = M.slab_face_material(slab)
    if slab_id in PAINT_GRADE:
        pal = PAINT_PALETTES.get(ctx, PAINT_PALETTES["default"])
        c = B.pick(f"paint:{ctx}", pal)
        rgba = M.PAINT_COLORS[c]
        kind = "powder_coat" if face.family == "metal" else "paint"
        return {"kind": kind, "color": c, "rgba": list(rgba), "texture": None,
                "roughness": _round(_u(rng, 0.35, 0.7), 0.01), "metallic": 0.05 if face.family == "metal" else 0.0}
    if slab_id in STAIN_GRADE:
        kind = B.pick(f"stain:{slab_id}", {"natural": 3, "stain": 2, "weathered": 1}) if slab.core_material not in ("reclaimed_barnwood",) else "weathered"
        rgba = list(face.base_color)
        if kind == "stain":
            k = _u(rng, 0.55, 0.9)
            rgba = [rgba[0] * k, rgba[1] * k * 0.95, rgba[2] * k * 0.9, 1.0]
        if kind == "weathered":
            rgba = [0.5 * rgba[0] + 0.25, 0.5 * rgba[1] + 0.25, 0.5 * rgba[2] + 0.25, 1.0]
        return {"kind": kind, "color": face.id, "rgba": rgba, "texture": face.texture, "roughness": face.roughness, "metallic": 0.0}
    if face.family == "glass":
        return {"kind": "glass", "color": face.id, "rgba": list(face.base_color), "texture": None, "roughness": face.roughness, "metallic": face.metallic}
    if face.family == "metal":
        return {"kind": "metal_bare", "color": face.id, "rgba": list(face.base_color), "texture": face.texture, "roughness": face.roughness, "metallic": face.metallic}
    return {"kind": "natural", "color": face.id, "rgba": list(face.base_color), "texture": face.texture, "roughness": face.roughness, "metallic": face.metallic}


def glazing_for(panel_style: str, W: float, Hh: float, glass_mat: str, thickness: float, rng: random.Random):
    rects = glazing_layout(panel_style, W, Hh)
    if not rects:
        return None
    return {"material": glass_mat, "thickness": thickness, "area_fraction": glazing_area_fraction(rects, W, Hh), "panel_style": panel_style,
            "count": len(rects)}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def base_spec(index: int, family: str, ctx: str, rng: random.Random) -> dict:
    return {
        "schema_version": "1.0",
        "id": f"db{index:04d}_{family}",
        "index": index,
        "seed": rng.randrange(2 ** 31),
        "family": family,
        "context": ctx,
        "use_case": "",
        "tags": [],
        "extras": [],
        "robot": {"approach_side": "-y", "is_push": True, "robot_outside": False, "start_distance_m": _round(_u(rng, 1.2, 2.2), 0.05)},
    }


def hinge_block(model: str, count: int, side: str, swing: str, tilt: float = 0.0):
    return {"model": model, "count": count, "side": side, "swing": swing, "axis_tilt_deg": tilt}


def keypad_code(lk: H.LockModel, rng: random.Random) -> str:
    """The lock's code.  Electronic keypads: 4 or 6 digits, repeats allowed.  A mechanical pushbutton lock
    (Kaba Simplex) has a combination *chamber*: each button can appear at most once and the order does not
    matter, so its code is a sorted set of 3-4 of its five buttons (drawn with the same number of rng draws as
    the digits it replaces, then repaired to distinct buttons, so the dataset stream is unchanged)."""
    n = 6 if "6" in lk.id else 4
    if lk.id != "keypad_mechanical":
        return "".join(str(rng.randrange(10)) for _ in range(n))
    n = rng.choice([3, 4])
    drawn = [str(rng.randrange(1, 6)) for _ in range(n)]
    out = []
    for ch in drawn:
        if ch not in out:
            out.append(ch)
    for ch in "12345":                      # a button can only be in the combination once: fill deterministically
        if len(out) >= n:
            break
        if ch not in out:
            out.append(ch)
    return "".join(sorted(out))


def choose_lock_engagement(spec: dict, B: Balanced, rng: random.Random, p_engaged: float = 0.5):
    """Decide whether a lock is engaged and if the robot can release it from its side."""
    lk = H.LOCKS[spec["lock"]["model"]]
    engaged = lk.kind not in ("none", "child_lock_cover") and rng.random() < p_engaged
    if lk.kind in ("chain", "swing_bar_guard", "slide_bolt", "padlock", "jam_stuck", "dogs", "vault_wheel", "delayed_egress", "interlock"):
        engaged = True if lk.kind in ("jam_stuck", "dogs", "vault_wheel", "delayed_egress", "interlock") else engaged
    spec["lock"]["engaged"] = bool(engaged)
    robot_outside = spec["robot"]["robot_outside"]
    if not engaged:
        spec["lock"]["robot_side_release"] = True
        if lk.kind == "keypad_code":
            # an unlocked keypad set still HAS a code (the lock is simply not thrown): drawn from a stream of its
            # own so adding it does not shift the dataset's main stream
            spec["lock"]["code"] = keypad_code(lk, random.Random(int(spec["seed"]) ^ 0x0DDC0DE))
        return
    if robot_outside:
        rel = lk.outside_release in ("code",) and "keypad" in lk.kind or lk.outside_release in ("thumbturn", "button", "lever")
        if lk.kind == "keypad_code" and lk.outside_release == "code":
            rel = True
    else:
        rel = lk.inside_release in ("thumbturn", "button", "lever", "rex_button", "slide")
    if lk.kind in ("padlock", "jam_stuck", "interlock", "delayed_egress"):
        rel = lk.kind == "delayed_egress"
    if lk.kind in ("keyed_cylinder", "slide_bolt") and spec["family"] in ("garage_sectional", "garage_tiltup", "rollup") and spec["operator"]["model"] != "pull_t_handle_garage":
        # a garage / roll-up door's cylinder or slide lock lives IN the T-handle and is withdrawn by turning it.
        # On a door with a lift handle, a pull ring or no handle at all there is nothing for a robot to turn, so the
        # lock is not released from its side and the door's task is to recognise that, not to unlock it.
        rel = False
    if lk.kind in ("chain", "swing_bar_guard"):
        # A door chain and a hotel swing bar are released by lifting the chain's ball out of its slotted track, or
        # the bar off its knob - hardware DoorBench draws but does not articulate.  The leaf's slack limit IS the
        # chain, and it is honest; claiming the robot can release it is not, because there is no part in the model
        # for a policy to move.  Declaring it unreleasable puts these doors on `locked_recognize`, which is exactly
        # the task a robot meeting a chained door should perform: push, feel the 60 mm stop, and declare it locked.
        rel = False
    # panic hardware: the bar side always exits (key cylinders only lock the outside trim)
    if spec["operator"]["model"].startswith("panic") and not robot_outside and lk.kind in ("keyed_cylinder", "privacy_button", "card_reader", "electric_strike", "keypad_code"):
        rel = True
    spec["lock"]["robot_side_release"] = bool(rel)
    if lk.kind == "keypad_code":
        spec["lock"]["code"] = keypad_code(lk, rng)


def assign_task(spec: dict, B: Balanced):
    fam = spec["family"]
    lock = spec["lock"]
    free_swing = fam in ("saloon", "strip_curtain", "pet_door", "turnstile_tripod", "turnstile_fullheight", "revolving")
    if free_swing:
        spec["task"] = B.pick(f"task:{fam}", {"push_through": 4, "traverse_open": 1})
    elif lock.get("engaged") and not lock.get("robot_side_release"):
        spec["task"] = "locked_recognize"
    elif lock.get("engaged"):
        spec["task"] = "unlock_open_traverse"
    elif spec["closer"]["model"] != "none" and spec["closer"]["model"] not in ("gas_strut",):
        spec["task"] = B.pick(f"task:{fam}:closer", {"hold_and_pass": 3, "open_and_traverse": 2, "open_only": 1, "peek": 1})
    else:
        spec["task"] = B.pick(f"task:{fam}", {"open_and_traverse": 5, "open_only": 2, "close": 2, "traverse_open": 1, "peek": 1})


def difficulty(spec: dict, phys: dict) -> int:
    d = 1.0
    m = phys["mass"]["primary_assembly_kg"]      # what the robot has to move, not what the whole door weighs
    d += min(m / 60.0, 2.0)
    if phys["closer"]["kind"] != "none":
        d += 0.8 + phys["closer"]["spring_preload_Nm"] / 40.0
    op = H.OPERATORS[spec["operator"]["model"]]
    d += {"lever": 0.3, "knob": 0.8, "pull": 0.1, "push_plate": 0.0, "panic_touchbar": 0.2, "panic_crossbar": 0.3, "paddle": 0.4, "thumb_latch": 0.9,
          "wheel": 1.5, "flush_pull": 0.5, "ring_pull": 0.5, "none": 0.0, "t_handle": 0.8, "cremone": 1.0, "handleset": 0.7, "slide_bolt_handle": 0.9,
          "keypad_lever": 1.2, "keypad_deadbolt": 1.4, "card_lever": 0.6, "push_button_screen": 0.5, "lift_latch": 0.9, "hook_lock_slider": 0.7,
          "gate_latch_fork": 0.8, "hasp": 1.2}.get(op.kind, 0.5)
    if spec["lock"].get("engaged"):
        d += 1.2 if spec["lock"].get("robot_side_release") else 0.6
    if spec["condition"] in ("swollen", "rusty", "sagging"):
        d += 0.6
    if spec.get("kinematics", {}).get("type") == "rotor":
        d += 0.5
    return int(max(1, min(5, round(d))))


def approach(spec: dict, B: Balanced, rng: random.Random, allow_pull=True):
    """Randomise robot side: push or pull, inside or outside."""
    swing = B.pick(f"swing:{spec['family']}:{spec['context']}", {"push": 1, "pull": 1}) if allow_pull else "push"
    outside = rng.random() < 0.5
    spec["robot"]["is_push"] = swing == "push"
    spec["robot"]["robot_outside"] = outside
    return swing


# ---------------------------------------------------------------------------
# Family generators
# ---------------------------------------------------------------------------
def gen_swing_single(i, ctx, B, rng):
    s = base_spec(i, "swing_single", ctx, rng)
    side = B.pick(f"side:{ctx}", {"left": 1, "right": 1})
    swing = approach(s, B, rng)
    extras = []
    if ctx == "residential_interior":
        slab = B.pick("ri:slab", {"hollow_core": 5, "hollow_core_molded": 3, "solid_core_pb": 2, "mdf_solid": 2, "solid_wood_pine": 2, "louver_wood": 1, "solid_wood_oak": 1, "solid_wood_fir": 1, "bamboo_solid": 1})
        W = B.pick("ri:w", {0.610: 1, 0.660: 1, 0.711: 2, 0.762: 3, 0.813: 3, 0.864: 1, 0.914: 1})
        Hh = B.pick("ri:h", {2.032: 5, 2.134: 1, 1.985: 1})
        t = 0.035 if slab in ("hollow_core", "hollow_core_molded", "louver_wood") else B.pick("ri:t", {0.035: 2, 0.040: 1, 0.044: 1})
        panel = B.pick("ri:panel", {"6_panel": 3, "2_panel_arch": 2, "shaker_1": 3, "shaker_3": 1, "shaker_2": 1, "flush": 4, "5_panel_horizontal": 1, "3_panel": 1, "louver_full": 1, "louver_half": 1, "glass_15_lite": 1, "glass_10_lite": 1, "beadboard": 1})
        if slab == "louver_wood":
            panel = "louver_full"
        op = B.pick("ri:op", {"knob_round": 3, "knob_round_privacy": 3, "lever_curved": 3, "lever_l_shape": 2, "knob_egg": 1, "knob_porcelain": 1, "knob_glass_antique": 1, "lever_straight": 1, "knob_childproof": 1, "lever_loose": 1})
        latch = "tubular_residential"
        lock = "privacy_button" if op == "knob_round_privacy" else B.pick("ri:lock", {"none": 7, "privacy_button": 2, "jam_stuck": 1, "thumbturn_only": 1})
        if op == "knob_childproof":
            lock = "child_lock_cover"
        closer = B.pick("ri:closer", {"none": 9, "spring_hinge_single": 1})
        hinge = B.pick("ri:hinge", {"butt_35_plain": 5, "butt_35_worn": 2, "butt_45_plain": 1, "concealed_soss": 1, "rising_butt": 1})
        hinge_count = 2 if Hh <= 2.04 and slab.startswith("hollow") and rng.random() < 0.5 else 3
        stop = B.pick("ri:stop", {"wall_bumper": 3, "floor_dome": 1, "hinge_pin": 2, "none": 2, "wall_180": 1, "corridor_wall_120": 1})
        seal = B.pick("ri:seal", {"none": 6, "brush_pile": 1, "door_sweep": 1})
        cond = B.pick("ri:cond", {"new": 3, "normal": 4, "worn": 2, "old_dry": 1, "swollen": 1, "sagging": 1, "well_oiled": 1})
        use = B.pick("ri:use", ["bedroom door", "bathroom door", "closet door", "home office door", "pantry door", "laundry room door", "hallway door", "nursery door", "basement door", "guest room door"])
        for e in ("coat_hook", "pet_flap", "door_stop_wall", "hold_open_kickdown", "wreath", "louver_vent"):
            if rng.random() < 0.10:
                extras.append(e)
        frame = {"kind": "wood_jamb_casing", "material": "pine", "casing": True, "stop_depth": 0.012, "jamb_depth": 0.115}
        handle_h = B.pick("ri:hh", {0.914: 2, 0.965: 3, 1.016: 1})
    elif ctx == "residential_exterior":
        slab = B.pick("re:slab", {"fiberglass_entry": 3, "steel_entry_24ga": 3, "solid_wood_oak": 2, "solid_wood_mahogany": 1, "upvc_panel": 2, "solid_wood_fir": 1, "screen_alu": 1, "screen_wood": 1, "storm_alu_glass": 1, "solid_wood_teak": 1})
        W = B.pick("re:w", {0.813: 2, 0.864: 1, 0.914: 5, 0.965: 1, 1.067: 1})
        Hh = B.pick("re:h", {2.032: 4, 2.134: 2, 2.438: 1})
        t = M.SLABS[slab].typical_thickness[0] if slab.startswith(("screen", "storm")) else 0.044
        panel = B.pick("re:panel", {"6_panel": 2, "steel_embossed_6": 2, "glass_half": 2, "glass_oval": 1, "glass_1_lite_top": 2, "glass_fan": 1, "2_panel_arch": 1, "plank_vertical": 1, "carved_ornate": 1, "glass_full": 1, "flush": 1, "glass_9_lite": 1})
        if slab.startswith("screen"):
            panel = "mesh_panel"
        if slab == "storm_alu_glass":
            panel = "glass_full"
        if slab.startswith(("screen", "storm")):
            op = B.pick("re:op_screen", {"push_button_screen": 3, "pull_d": 1, "lever_curved": 1})
            latch = "screen_pushbutton"
            lock = B.pick("re:lock_screen", {"none": 3, "slide_bolt": 1})
            closer = B.pick("re:closer_screen", {"pneumatic_screen": 3, "spring_hinge_single": 1, "none": 1})
            hinge = B.pick("re:hinge_screen", {"butt_35_plain": 2, "piano": 1, "spring_single": 1})
        else:
            op = B.pick("re:op", {"handleset_thumb": 3, "lever_straight": 2, "knob_round": 2, "lever_keypad": 2, "knob_keypad_deadbolt": 2, "lever_euro_backplate": 1, "lever_curved": 1, "lever_l_shape": 1})
            latch = B.pick("re:latch", {"tubular_residential_70": 3, "deadlatch_grade1": 1, "mortise_euro": 1})
            lock = B.pick("re:lock", {"deadbolt_single": 4, "keyed_cylinder": 2, "keypad_code_4": 1, "keypad_code_6": 1, "chain": 1, "swing_bar_guard": 1, "night_latch": 1, "none": 2, "deadbolt_double": 1})
            if slab == "upvc_panel":
                lock = "multipoint"
            if op in ("lever_keypad",):
                lock = "keypad_code_4"
            if op == "knob_keypad_deadbolt":
                lock = "keypad_code_6"
            closer = B.pick("re:closer", {"none": 6, "residential_light": 1, "spring_hinge_single": 1})
            hinge = B.pick("re:hinge", {"butt_45_plain": 3, "butt_45_bb": 2, "butt_rusty": 1, "butt_35_plain": 1, "butt_35_worn": 1})
        hinge_count = 3 if Hh < 2.3 else 4
        stop = B.pick("re:stop", {"wall_bumper": 2, "floor_dome": 1, "hinge_pin": 2, "none": 1, "overhead_90": 1})
        seal = B.pick("re:seal", {"kerf_foam": 3, "silicone_bulb": 1, "brush_pile": 1, "door_sweep": 1, "none": 1})
        cond = B.pick("re:cond", {"new": 3, "normal": 3, "worn": 2, "old_dry": 1, "swollen": 1, "rusty": 1, "damaged": 1})
        use = B.pick("re:use", ["front entry door", "back door to yard", "side entry / mudroom door", "garage-to-house door", "porch screen door", "patio door (hinged)", "apartment entry door", "basement walk-out door"])
        for e, p in (("peephole", 0.35), ("mail_slot", 0.15), ("knocker", 0.15), ("house_number", 0.3), ("wreath", 0.08), ("kick_plate", 0.25), ("weather_drip_cap", 0.2), ("door_viewer_camera", 0.15), ("sidelite", 0.2), ("transom_window", 0.12), ("pet_flap", 0.1), ("threshold_saddle", 0.5)):
            if rng.random() < p:
                extras.append(e)
        frame = {"kind": "wood_jamb_brickmould", "material": "pine", "casing": True, "stop_depth": 0.012, "jamb_depth": 0.125}
        handle_h = B.pick("re:hh", {0.914: 1, 0.965: 3, 1.016: 2, 1.050: 1})
    elif ctx == "commercial_office":
        slab = B.pick("co:slab", {"solid_core_pb": 4, "hollow_metal_18ga": 3, "hollow_metal_20ga": 2, "solid_core_scl": 1, "acoustic_wood": 1, "mdf_solid": 1, "solid_wood_maple": 1})
        W = B.pick("co:w", {0.813: 1, 0.914: 5, 0.965: 1, 1.067: 2})
        Hh = B.pick("co:h", {2.134: 4, 2.032: 3, 2.438: 1})
        t = 0.044 if slab != "acoustic_wood" else 0.057
        panel = B.pick("co:panel", {"flush": 5, "steel_flush": 2, "glass_vision": 3, "steel_vision": 1, "glass_half": 1, "shaker_1": 1, "glass_full": 1})
        if slab.startswith("hollow_metal") and panel in ("flush", "shaker_1"):
            panel = "steel_flush"
        op = B.pick("co:op", {"lever_straight": 4, "lever_return": 2, "lever_mortise_escutcheon": 1, "lever_euro_backplate": 1, "lever_card_reader": 1, "push_plate": 1, "pull_d": 1, "lever_l_shape": 1})
        latch = B.pick("co:latch", {"deadlatch_grade1": 3, "tubular_residential_70": 1, "mortise_latch": 2, "mortise_euro": 1})
        if op in ("push_plate", "pull_d"):
            latch = "none"
        lock = B.pick("co:lock", {"none": 6, "keyed_cylinder": 1, "electric_strike": 1, "card_reader": 1, "mag_lock": 1, "keypad_mechanical": 1, "privacy_button": 1})
        if op == "lever_card_reader":
            lock = "card_reader"
        closer = B.pick("co:closer", {"norton_1600": 3, "lcn_4040": 3, "none": 3, "concealed_overhead": 1})
        if op in ("push_plate", "pull_d") and closer == "none":
            closer = "norton_1600"
        hinge = B.pick("co:hinge", {"butt_45_plain": 3, "butt_45_bb": 3, "continuous_geared": 1, "butt_45_bb_4": 1})
        hinge_count = 4 if hinge == "butt_45_bb_4" or Hh > 2.3 else (1 if hinge == "continuous_geared" else 3)
        stop = B.pick("co:stop", {"wall_bumper": 3, "overhead_90": 2, "overhead_105": 1, "floor_dome": 1, "hinge_pin": 1, "corridor_wall_120": 1})
        seal = B.pick("co:seal", {"none": 4, "brush_pile": 2, "silicone_bulb": 1, "automatic_drop_seal": 1})
        cond = B.pick("co:cond", {"new": 4, "normal": 4, "worn": 2, "well_oiled": 1, "old_dry": 1})
        use = B.pick("co:use", ["private office door", "conference room door", "corridor door", "restroom door (public)", "server room door", "break room door", "reception door", "storage room door", "classroom door"])
        for e, p in (("kick_plate", 0.4), ("push_pull_sign", 0.2), ("rex_button", 0.15), ("keypad_reader_wall", 0.15), ("door_stop_wall", 0.2), ("bumper_rail", 0.1), ("coat_hook", 0.1)):
            if rng.random() < p:
                extras.append(e)
        if lock in ("mag_lock", "electric_strike", "card_reader") and "keypad_reader_wall" not in extras:
            extras.append("keypad_reader_wall")
        if lock == "mag_lock":
            extras.append("rex_button")
        frame = {"kind": "hollow_metal_frame", "material": "steel_painted", "casing": False, "stop_depth": 0.016, "jamb_depth": 0.146}
        handle_h = B.pick("co:hh", {1.016: 2, 1.040: 3, 0.965: 1})
    elif ctx == "fire_egress":
        slab = B.pick("fe:slab", {"hollow_metal_18ga": 4, "hollow_metal_16ga": 2, "mineral_core_90": 2, "mineral_core_45": 1, "mineral_core_20": 1, "hollow_metal_18ga_pu": 1})
        W = B.pick("fe:w", {0.914: 5, 0.965: 1, 1.067: 2, 1.219: 1})
        Hh = B.pick("fe:h", {2.134: 5, 2.032: 2, 2.438: 1})
        t = 0.044
        panel = B.pick("fe:panel", {"steel_flush": 5, "steel_vision": 2, "flush": 2, "steel_louvered": 1, "glass_vision": 1})
        op = B.pick("fe:op", {"panic_touchbar_rim": 4, "panic_touchbar_svr": 1, "panic_touchbar_mortise": 1, "panic_touchbar_stiff": 1, "panic_crossbar": 1, "panic_touchbar_rim_light": 1, "panic_touchbar_alarm": 1, "lever_straight": 2, "lever_return": 1})
        latch = {"panic_touchbar_rim": "rim_exit", "panic_touchbar_svr": "vertical_rods", "panic_touchbar_mortise": "mortise_latch", "panic_touchbar_stiff": "rim_exit", "panic_crossbar": "rim_exit", "panic_touchbar_rim_light": "rim_exit", "panic_touchbar_alarm": "rim_exit"}.get(op, "deadlatch_grade1")
        lock = B.pick("fe:lock", {"none": 7, "delayed_egress": 1, "mag_lock": 1, "keyed_cylinder": 1})
        if op == "panic_touchbar_alarm":
            lock = "delayed_egress"
        closer = B.pick("fe:closer", {"lcn_4040": 5, "norton_1600": 2, "magnetic_hold": 1, "lcn_4040_delayed": 1, "concealed_overhead": 1})
        hinge = B.pick("fe:hinge", {"butt_45_bb": 4, "butt_45_bb_4": 2, "continuous_geared": 1, "butt_5_bb_heavy": 1})
        hinge_count = 4 if hinge in ("butt_45_bb_4", "butt_5_bb_heavy") or Hh > 2.3 else (1 if hinge == "continuous_geared" else 3)
        stop = B.pick("fe:stop", {"overhead_90": 2, "overhead_105": 2, "wall_bumper": 2, "floor_dome": 1, "none": 1})
        seal = B.pick("fe:seal", {"smoke_seal_intumescent": 4, "silicone_bulb": 1, "none": 1, "automatic_drop_seal": 1})
        cond = B.pick("fe:cond", {"new": 3, "normal": 4, "worn": 2, "old_dry": 1, "rusty": 1, "damaged": 1})
        use = B.pick("fe:use", ["stairwell fire door", "exterior emergency exit", "corridor cross fire door", "loading dock egress door", "theater exit door", "parking garage stair door", "school egress door", "hospital fire door"])
        swing = "push"   # egress swings in direction of travel; robot may be on either side
        s["robot"]["is_push"] = not s["robot"]["robot_outside"]
        for e, p in (("exit_sign", 0.7), ("kick_plate", 0.5), ("push_pull_sign", 0.2), ("armor_plate", 0.15), ("warning_placard", 0.15), ("rex_button", 0.1), ("bumper_rail", 0.1)):
            if rng.random() < p:
                extras.append(e)
        if lock == "mag_lock":
            extras.append("rex_button")
        frame = {"kind": "hollow_metal_frame", "material": "steel_painted", "casing": False, "stop_depth": 0.016, "jamb_depth": 0.146}
        handle_h = B.pick("fe:hh", {1.016: 3, 1.040: 2, 0.965: 1})
    elif ctx == "institutional":
        sub = B.pick("in:sub", {"hospital": 3, "school": 2, "lab": 2, "hotel": 3, "radiology": 1, "psychiatric": 1})
        if sub == "hospital":
            slab, panel = "hospital_solid", B.pick("in:hp", {"flush": 3, "glass_vision": 2, "glass_half": 1})
            op = B.pick("in:hop", {"lever_return": 2, "paddle_push_pull": 2, "paddle_hospital_arm": 1, "push_plate": 1})
            latch = "roller_latch" if op == "push_plate" else ("mortise_latch" if op.startswith("paddle") else "deadlatch_grade1")
            lock = "none"
            closer = B.pick("in:hcl", {"lcn_4040": 2, "none": 2, "norton_1600": 1})
            hinge = B.pick("in:hhinge", {"lift_off": 2, "butt_45_bb": 2, "continuous_geared": 1})
            W = B.pick("in:hw", {1.067: 2, 1.219: 2, 0.914: 1})
            extras += ["kick_plate"] + (["bumper_rail"] if rng.random() < 0.5 else []) + (["armor_plate"] if rng.random() < 0.3 else [])
            use = B.pick("in:huse", ["patient room door", "operating room door (manual)", "ward corridor door", "clinic exam room door"])
            pal = "institutional"
        elif sub == "school":
            slab, panel = B.pick("in:ss", {"hollow_metal_18ga": 2, "solid_core_pb": 1}), B.pick("in:sp", {"steel_vision": 2, "glass_vision": 1, "steel_flush": 1, "flush": 1})
            op = B.pick("in:sop", {"lever_straight": 3, "lever_return": 1, "knob_round": 1})
            latch = "deadlatch_grade1"
            lock = B.pick("in:slock", {"keyed_cylinder": 2, "keypad_mechanical": 1, "none": 2, "electric_strike": 1})
            closer = B.pick("in:scl", {"lcn_4040": 2, "norton_1600": 2, "none": 1})
            hinge = B.pick("in:shg", {"butt_45_bb": 3, "butt_45_plain": 1})
            W = B.pick("in:sw", {0.914: 3, 1.067: 1})
            extras += ["kick_plate"] if rng.random() < 0.6 else []
            use = B.pick("in:suse", ["classroom door", "gymnasium door", "school office door", "library door"])
            pal = "institutional"
        elif sub == "lab":
            slab, panel = B.pick("in:ls", {"hollow_metal_18ga": 2, "stainless_hollow": 1, "solid_core_pb": 1}), B.pick("in:lp", {"glass_vision": 2, "steel_vision": 1, "steel_flush": 1})
            op = B.pick("in:lop", {"lever_straight": 2, "lever_card_reader": 1, "lever_return": 1})
            latch = "deadlatch_grade1"
            lock = B.pick("in:llock", {"card_reader": 2, "mag_lock": 1, "electric_strike": 1, "keypad_mechanical": 1})
            closer = B.pick("in:lcl", {"lcn_4040": 3, "norton_1600": 1})
            hinge = B.pick("in:lhg", {"butt_45_bb": 2, "continuous_geared": 1})
            W = B.pick("in:lw", {0.914: 2, 1.067: 1})
            extras += ["keypad_reader_wall", "warning_placard"] + (["rex_button"] if lock == "mag_lock" else [])
            use = B.pick("in:luse", ["biosafety lab door", "clean room anteroom door", "chemistry lab door", "data center door"])
            pal = "institutional"
        elif sub == "hotel":
            slab, panel = B.pick("in:hs", {"solid_core_pb": 2, "mineral_core_20": 1, "solid_wood_walnut": 1}), B.pick("in:hpn", {"flush": 3, "shaker_1": 1, "2_panel_arch": 1})
            op = "lever_card_reader"
            latch = "mortise_latch"
            lock = B.pick("in:hlock", {"card_reader": 3, "swing_bar_guard": 1, "chain": 1})
            closer = B.pick("in:hcl2", {"norton_1600": 2, "concealed_overhead": 1, "residential_light": 1, "lcn_4040": 1})
            hinge = B.pick("in:hhg", {"butt_45_bb": 3, "butt_45_plain": 1})
            W = B.pick("in:hw2", {0.914: 3, 0.864: 1})
            extras += ["peephole"] + (["kick_plate"] if rng.random() < 0.3 else []) + (["coat_hook"] if rng.random() < 0.3 else [])
            use = "hotel guest room door"
            pal = "commercial_office"
        elif sub == "radiology":
            slab, panel = "lead_lined", "flush"
            op = "lever_straight"
            latch = "deadlatch_grade1"
            lock = B.pick("in:rlock", {"none": 2, "keyed_cylinder": 1})
            closer = "lcn_4040"
            hinge = "butt_5_bb_heavy"
            W = B.pick("in:rw", {1.067: 2, 1.219: 1})
            extras += ["warning_placard"]
            use = "radiology (lead-lined) door"
            pal = "institutional"
        else:  # psychiatric / anti-ligature
            slab, panel = "hollow_metal_16ga", B.pick("in:pp", {"steel_vision": 2, "steel_flush": 1})
            op = B.pick("in:pop", {"lever_return": 2, "paddle_push_pull": 1, "push_plate": 1})
            latch = "deadlatch_grade1" if op != "push_plate" else "roller_latch"
            lock = B.pick("in:plock", {"keyed_cylinder": 2, "none": 1})
            closer = "lcn_4040"
            hinge = "continuous_geared"
            W = 0.914
            use = "behavioral health anti-ligature door"
            pal = "institutional"
        Hh = B.pick("in:h", {2.134: 4, 2.032: 2})
        t = 0.044
        hinge_count = 1 if hinge == "continuous_geared" else (4 if Hh > 2.3 or slab == "lead_lined" else 3)
        stop = B.pick("in:stop", {"wall_bumper": 2, "overhead_90": 2, "overhead_105": 1, "floor_dome": 1, "corridor_wall_120": 1})
        seal = B.pick("in:seal", {"none": 3, "brush_pile": 1, "silicone_bulb": 1, "smoke_seal_intumescent": 1, "automatic_drop_seal": 1})
        cond = B.pick("in:cond", {"new": 3, "normal": 4, "worn": 2, "well_oiled": 1})
        frame = {"kind": "hollow_metal_frame", "material": "steel_painted", "casing": False, "stop_depth": 0.016, "jamb_depth": 0.146}
        handle_h = B.pick("in:hh", {1.016: 2, 1.040: 3, 0.965: 1})
        ctx_pal = pal
    elif ctx == "industrial_utility":
        slab = B.pick("iu:slab", {"hollow_metal_18ga_pu": 3, "hollow_metal_16ga": 2, "hollow_metal_18ga": 2, "expanded_metal_gate": 1, "steel_plate_security": 1, "solid_core_pb": 1})
        W = B.pick("iu:w", {0.914: 4, 1.067: 2, 1.219: 1, 0.813: 1})
        Hh = B.pick("iu:h", {2.134: 4, 2.438: 1, 2.032: 1})
        t = M.SLABS[slab].typical_thickness[0]
        panel = B.pick("iu:panel", {"steel_flush": 4, "steel_louvered": 2, "steel_vision": 1, "mesh_panel": 1, "riveted_steel": 1})
        if slab == "expanded_metal_gate":
            panel = "mesh_panel"
        op = B.pick("iu:op", {"lever_straight": 3, "knob_round": 1, "panic_touchbar_rim": 1, "pull_d": 2, "push_plate": 1, "hasp_padlock": 1})
        latch = {"panic_touchbar_rim": "rim_exit", "pull_d": "none", "push_plate": "none", "hasp_padlock": "none"}.get(op, "deadlatch_grade1")
        lock = B.pick("iu:lock", {"none": 4, "padlock": 2, "keyed_cylinder": 2, "keypad_mechanical": 1, "slide_bolt": 1})
        if op == "hasp_padlock":
            lock = "padlock"
        closer = B.pick("iu:closer", {"lcn_4040": 3, "none": 3, "spring_hinge_single": 1, "norton_1600": 1})
        hinge = B.pick("iu:hinge", {"butt_45_plain": 3, "butt_rusty": 2, "continuous_geared": 1, "butt_45_bb": 1, "strap_pintle": 1})
        hinge_count = 1 if hinge == "continuous_geared" else 3
        stop = B.pick("iu:stop", {"none": 3, "wall_bumper": 1, "overhead_90": 1, "floor_dome": 1, "kick_down_holder": 1})
        seal = B.pick("iu:seal", {"none": 3, "silicone_bulb": 1, "brush_pile": 1, "kerf_foam": 1})
        cond = B.pick("iu:cond", {"rusty": 3, "worn": 3, "old_dry": 2, "damaged": 2, "normal": 2, "new": 1})
        use = B.pick("iu:use", ["mechanical room door", "electrical room door", "warehouse personnel door", "roof access door", "boiler room door", "workshop door", "pump house door", "substation door"])
        for e, p in (("warning_placard", 0.5), ("louver_vent", 0.3), ("kick_plate", 0.3), ("armor_plate", 0.15), ("push_pull_sign", 0.1)):
            if rng.random() < p:
                extras.append(e)
        frame = {"kind": "hollow_metal_frame", "material": "steel_painted", "casing": False, "stop_depth": 0.016, "jamb_depth": 0.146}
        handle_h = B.pick("iu:hh", {1.016: 2, 1.040: 2, 0.965: 1})
    elif ctx == "security_detention":
        slab = B.pick("sd:slab", {"hollow_metal_14ga": 3, "steel_plate_security": 2, "steel_bar_grille": 2, "hollow_metal_16ga": 1})
        W = B.pick("sd:w", {0.914: 3, 0.813: 1, 1.067: 1})
        Hh = B.pick("sd:h", {2.134: 3, 2.032: 1})
        t = M.SLABS[slab].typical_thickness[0]
        panel = "bar_grille" if slab == "steel_bar_grille" else B.pick("sd:panel", {"steel_flush": 2, "steel_vision": 2, "riveted_steel": 1})
        op = B.pick("sd:op", {"pull_d": 3, "lever_straight": 1, "knob_round": 1, "none": 1})
        latch = "none" if op in ("pull_d", "none") else "deadlatch_grade1"
        lock = B.pick("sd:lock", {"keyed_cylinder": 3, "electric_strike": 1, "mag_lock": 1, "slide_bolt": 1, "deadbolt_double": 1})
        closer = B.pick("sd:closer", {"none": 3, "lcn_4040": 1})
        hinge = B.pick("sd:hinge", {"butt_5_bb_heavy": 3, "continuous_geared": 1})
        hinge_count = 1 if hinge == "continuous_geared" else 3
        stop = B.pick("sd:stop", {"none": 2, "wall_bumper": 1, "overhead_90": 1})
        seal = "none"
        cond = B.pick("sd:cond", {"normal": 3, "worn": 2, "new": 1})
        use = B.pick("sd:use", ["detention cell door (swing)", "sally port door", "evidence room door", "safe room door", "armory door"])
        extras += (["keypad_reader_wall"] if lock in ("electric_strike", "mag_lock") else []) + (["warning_placard"] if rng.random() < 0.5 else [])
        frame = {"kind": "hollow_metal_frame", "material": "steel", "casing": False, "stop_depth": 0.019, "jamb_depth": 0.20}
        handle_h = 1.016
    elif ctx == "storefront_glass":
        slab = B.pick("sg:slab", {"storefront_alu": 3, "storefront_alu_igu": 2, "glass_frameless_12": 2, "glass_frameless_10": 1, "glass_frameless_19": 1})
        W = B.pick("sg:w", {0.914: 4, 1.067: 2, 0.813: 1})
        Hh = B.pick("sg:h", {2.134: 4, 2.438: 2, 2.743: 1})
        t = M.SLABS[slab].typical_thickness[0] if slab.startswith("glass_frameless") else 0.045
        panel = "glass_frameless" if slab.startswith("glass_frameless") else "glass_full"
        op = B.pick("sg:op", {"pull_bar_offset": 3, "pull_ladder_full": 2, "pull_d": 2, "push_plate": 1, "panic_touchbar_rim": 1, "paddle_push_pull": 1})
        latch = "rim_exit" if op == "panic_touchbar_rim" else ("mortise_latch" if op == "paddle_push_pull" else "none")
        lock = B.pick("sg:lock", {"none": 4, "thumbturn_only": 2, "keyed_cylinder": 1, "mag_lock": 1})
        closer = B.pick("sg:closer", {"floor_spring": 3, "lcn_4040": 2, "concealed_overhead": 2, "auto_low_energy": 1, "floor_spring_nohold": 1})
        hinge = B.pick("sg:hinge", {"pivot_offset": 3, "pivot_center": 2, "continuous_geared": 1, "butt_45_bb": 1})
        if slab.startswith("glass_frameless"):
            hinge = B.pick("sg:hinge_glass", {"pivot_center": 3, "pivot_offset": 1})
        hinge_count = 2 if hinge.startswith("pivot") else (1 if hinge == "continuous_geared" else 3)
        stop = B.pick("sg:stop", {"overhead_90": 2, "overhead_105": 1, "floor_dome": 1, "none": 2})
        seal = B.pick("sg:seal", {"brush_pile": 3, "none": 1, "door_sweep": 1})
        cond = B.pick("sg:cond", {"new": 4, "normal": 3, "worn": 1})
        use = B.pick("sg:use", ["retail storefront door", "office lobby glass door", "mall entrance door", "bank branch entry", "cafe glass door", "gym entrance"])
        for e, p in (("push_pull_sign", 0.5), ("kick_plate", 0.1), ("exit_sign", 0.2), ("keypad_reader_wall", 0.1)):
            if rng.random() < p:
                extras.append(e)
        frame = {"kind": "aluminum_storefront", "material": "aluminum" if rng.random() < 0.6 else "aluminum_dark", "casing": False, "stop_depth": 0.012, "jamb_depth": 0.114}
        handle_h = B.pick("sg:hh", {1.016: 2, 1.067: 2, 0.965: 1})
    else:  # heritage_rustic
        slab = B.pick("hr:slab", {"solid_wood_oak": 3, "solid_wood_pine": 2, "barn_plank": 2, "cedar_plank": 1, "solid_wood_fir": 1, "leather_padded": 1, "solid_wood_walnut": 1})
        W = B.pick("hr:w", {0.762: 2, 0.813: 2, 0.914: 2, 1.067: 1, 0.711: 1})
        Hh = B.pick("hr:h", {1.985: 2, 2.032: 3, 2.134: 1, 1.900: 1})
        t = B.pick("hr:t", {0.044: 3, 0.050: 1, 0.035: 1})
        panel = B.pick("hr:panel", {"plank_x_brace": 2, "arched_top": 2, "board_batten": 2, "plank_z_brace": 1, "carved_ornate": 2, "4_panel": 1, "padded_diamond": 1, "planks_diagonal": 1})
        if slab == "leather_padded":
            panel = "padded_diamond"
        op = B.pick("hr:op", {"knob_rim_lock": 2, "pull_ring": 2, "thumb_latch_suffolk": 3, "knob_glass_antique": 1, "knob_porcelain": 1, "lever_mortise_escutcheon": 1, "lever_loose": 1})
        latch = {"pull_ring": "gravity_bar", "thumb_latch_suffolk": "gravity_bar", "knob_rim_lock": "tubular_residential"}.get(op, "mortise_euro")
        lock = B.pick("hr:lock", {"none": 4, "slide_bolt": 2, "padlock": 1, "night_latch": 1, "jam_stuck": 1})
        closer = "none"
        hinge = B.pick("hr:hinge", {"strap_pintle": 3, "strap_heavy": 1, "butt_rusty": 2, "butt_35_worn": 1, "rising_butt": 1})
        hinge_count = 2 if hinge.startswith("strap") else 3
        stop = B.pick("hr:stop", {"none": 4, "wall_bumper": 1, "floor_dome": 1})
        seal = B.pick("hr:seal", {"none": 4, "brush_pile": 1})
        cond = B.pick("hr:cond", {"old_dry": 3, "worn": 3, "rusty": 1, "swollen": 2, "sagging": 1, "normal": 1})
        use = B.pick("hr:use", ["cottage front door", "castle chamber door", "pub cellar door", "church side door", "barn tack room door", "cabin door", "wine cellar door", "recording studio door"])
        for e, p in (("knocker", 0.3), ("pull_ring" if False else "house_number", 0.1), ("wreath", 0.1), ("peephole", 0.1)):
            if rng.random() < p:
                extras.append(e)
        frame = {"kind": "timber_frame_heavy", "material": "oak_white", "casing": False, "stop_depth": 0.020, "jamb_depth": 0.15}
        handle_h = B.pick("hr:hh", {0.914: 2, 0.965: 2, 1.050: 1, 1.10: 1})

    if ctx != "institutional":
        ctx_pal = ctx
    s["use_case"] = use
    finish = finish_for(slab, ctx_pal, B, rng)
    glass_mat = B.pick(f"glassmat:{ctx}", {"glass_clear": 4, "glass_frosted": 2, "glass_wired": 1 if ctx in ("fire_egress", "institutional", "industrial_utility") else 0.001, "glass_tinted": 1 if ctx in ("storefront_glass", "residential_exterior") else 0.001})
    glass_t = 0.006 if ctx not in ("storefront_glass",) else (0.012 if slab.startswith("glass_frameless") else 0.006)
    s["leaf"] = {"width": W, "height": Hh, "thickness": t, "slab": slab, "panel_style": panel, "finish": finish, "count": 1,
                 "glazing": glazing_for(panel, W, Hh, glass_mat, glass_t, rng)}
    if slab in ("screen_alu", "screen_wood"):
        s["leaf"]["glazing"] = None
    s["opening"] = {"width": _round(W + 0.006), "height": _round(Hh + 0.013), "wall_thickness": B.pick(f"wall:{ctx}", {0.115: 2, 0.145: 3, 0.20: 1, 0.30: 1}),
                    "frame": frame, "threshold": ("saddle" if "threshold_saddle" in extras else ("none" if ctx in ("residential_interior", "commercial_office") else B.pick(f"thr:{ctx}", {"none": 2, "saddle": 1, "ada_ramp": 1}))),
                    "sidelite": "sidelite" in extras, "transom": "transom_window" in extras}
    tilt = H.HINGES[hinge].axis_tilt_deg
    s["hinge"] = hinge_block(hinge, hinge_count, side, swing, tilt)
    max_open = min(H.STOPS[stop]["max_open_deg"], 178)
    if stop == "none":
        max_open = B.pick("maxopen:none", {110: 2, 130: 1, 160: 1, 178: 1})
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": max_open, "stop": stop}
    s["operator"] = {"model": op, "height": handle_h, "sides": "both"}
    if op.startswith("panic") or op in ("push_plate",):
        # egress side gets the device; far side gets a pull (or nothing) - robot may face either
        s["operator"]["sides"] = "push_side"
        s["operator"]["far_side"] = B.pick(f"farside:{ctx}", {"pull_d": 2, "none": 1, "pull_bar_offset": 1, "lever_straight": 1 if op.startswith("panic") and latch != "rim_exit" else 0.001})
    if op in ("pull_d", "pull_bar_offset", "pull_ladder_full"):
        s["operator"]["sides"] = "both"
    if op in ("push_plate", "pull_d", "pull_bar_offset", "pull_ladder_full", "pull_ring", "pull_barn_iron", "none") and lock in ("keyed_cylinder", "privacy_button", "keypad_code_4", "card_reader", "keypad_mechanical", "night_latch"):
        lock = "deadbolt_single" if op != "pull_ring" else "slide_bolt"
    if op == "pull_ring":
        latch = "none"
    if op == "hasp_padlock":
        op, lock, latch = "pull_d", "padlock", "none"
    s["latch"] = {"model": latch}
    s["lock"] = {"model": lock}
    s["closer"] = {"model": closer, "en_size": None, "spring_adjust": _round(_u(rng, 1.05, 1.3), 0.01)}
    s["seal"] = seal
    s["condition"] = cond
    s["extras"] = sorted(set(extras))
    if "pet_flap" in extras:
        s["leaf"]["pet_flap"] = {"width": B.pick("petw", {0.18: 2, 0.25: 2, 0.32: 1}), "height": B.pick("peth", {0.25: 2, 0.36: 2, 0.45: 1}), "slab": B.pick("petslab", {"pet_flap_pvc": 3, "pet_flap_acrylic": 1})}
    p_eng = {"residential_interior": 0.35, "residential_exterior": 0.55, "commercial_office": 0.5, "fire_egress": 0.6, "institutional": 0.55, "industrial_utility": 0.6, "security_detention": 0.8, "storefront_glass": 0.35, "heritage_rustic": 0.5}[ctx]
    choose_lock_engagement(s, B, rng, p_eng)
    s["tags"] = [ctx, "swing", "single_leaf", slab, panel, op, "lock:" + lock, "closer:" + closer, "hinge:" + hinge, cond]
    return s


def gen_swing_double(i, ctx, B, rng):
    s = base_spec(i, "swing_double", ctx, rng)
    swing = approach(s, B, rng)
    extras = []
    if ctx == "french":
        slab = B.pick("fd:slab", {"solid_wood_pine": 2, "solid_wood_fir": 1, "fiberglass_entry": 1, "solid_wood_mahogany": 1, "upvc_panel": 1, "mdf_solid": 1})
        W = B.pick("fd:w", {0.762: 2, 0.813: 2, 0.914: 1, 0.610: 1})
        Hh = B.pick("fd:h", {2.032: 3, 2.134: 1, 2.438: 1})
        panel = B.pick("fd:panel", {"glass_15_lite": 3, "glass_10_lite": 2, "glass_full": 1, "glass_6_lite": 1, "glass_1_lite_top": 1})
        op = B.pick("fd:op", {"lever_straight": 2, "knob_round": 1, "handleset_thumb": 1, "lever_euro_backplate": 1, "cremone_bolt": 1})
        latch = "tubular_residential_70" if op != "cremone_bolt" else "none"
        lock = B.pick("fd:lock", {"none": 3, "deadbolt_single": 2, "multipoint": 1, "thumbturn_only": 1})
        inactive_lock = "flush_bolts"
        closer, hinge, stop = "none", B.pick("fd:hinge", {"butt_35_plain": 2, "butt_45_plain": 1, "butt_45_bb": 1}), B.pick("fd:stop", {"wall_bumper": 2, "none": 1, "floor_dome": 1})
        seal, cond = B.pick("fd:seal", {"kerf_foam": 2, "none": 1, "brush_pile": 1}), B.pick("fd:cond", {"new": 2, "normal": 3, "worn": 1, "swollen": 1})
        use = B.pick("fd:use", ["french patio doors", "interior french doors (study)", "balcony french doors", "dining room french doors"])
        frame = {"kind": "wood_jamb_casing", "material": "pine", "casing": True, "stop_depth": 0.012, "jamb_depth": 0.125}
        astragal = B.pick("fd:astragal", {"T_astragal_on_inactive": 2, "meeting_stile_rabbet": 1, "none": 1})
        handle_h = 0.965
    elif ctx == "commercial_pair_panic":
        slab = B.pick("cp:slab", {"hollow_metal_18ga": 3, "hollow_metal_16ga": 1, "mineral_core_90": 1, "solid_core_pb": 1})
        W = B.pick("cp:w", {0.914: 4, 1.067: 1, 0.813: 1})
        Hh = B.pick("cp:h", {2.134: 3, 2.032: 1, 2.438: 1})
        panel = B.pick("cp:panel", {"steel_flush": 3, "steel_vision": 2, "flush": 1, "glass_vision": 1})
        op = B.pick("cp:op", {"panic_touchbar_svr": 2, "panic_touchbar_rim": 2, "panic_crossbar": 1, "panic_touchbar_mortise": 1})
        latch = {"panic_touchbar_svr": "vertical_rods", "panic_touchbar_rim": "rim_exit", "panic_crossbar": "rim_exit", "panic_touchbar_mortise": "mortise_latch"}[op]
        lock = B.pick("cp:lock", {"none": 4, "delayed_egress": 1, "mag_lock": 1})
        inactive_lock = "same_device"
        closer = B.pick("cp:closer", {"lcn_4040": 3, "norton_1600": 1, "magnetic_hold": 1})
        hinge = B.pick("cp:hinge", {"butt_45_bb": 3, "butt_45_bb_4": 1, "continuous_geared": 1})
        stop = B.pick("cp:stop", {"overhead_90": 2, "overhead_105": 1, "wall_bumper": 1, "none": 1})
        seal = B.pick("cp:seal", {"smoke_seal_intumescent": 2, "none": 1, "silicone_bulb": 1})
        cond = B.pick("cp:cond", {"new": 2, "normal": 3, "worn": 1, "old_dry": 1})
        use = B.pick("cp:use", ["auditorium exit pair", "gymnasium exit pair", "school corridor pair", "warehouse egress pair", "arena vomitory doors"])
        frame = {"kind": "hollow_metal_frame", "material": "steel_painted", "casing": False, "stop_depth": 0.016, "jamb_depth": 0.146}
        astragal = B.pick("cp:astragal", {"removable_mullion": 2, "none": 3, "overlapping_astragal": 1})
        s["robot"]["is_push"] = not s["robot"]["robot_outside"]
        extras += ["exit_sign"] + (["kick_plate"] if rng.random() < 0.6 else [])
        handle_h = 1.016
    elif ctx == "double_egress":
        slab = B.pick("de:slab", {"hollow_metal_18ga": 2, "hospital_solid": 2, "mineral_core_45": 1})
        W = B.pick("de:w", {0.914: 2, 1.067: 2, 1.219: 1})
        Hh = 2.134
        panel = B.pick("de:panel", {"steel_vision": 2, "glass_vision": 2, "steel_flush": 1})
        op = B.pick("de:op", {"push_plate": 2, "panic_touchbar_rim": 1, "paddle_push_pull": 1})
        latch = {"push_plate": "none", "panic_touchbar_rim": "rim_exit", "paddle_push_pull": "mortise_latch"}[op]
        lock = B.pick("de:lock", {"none": 3, "mag_lock": 1})
        inactive_lock = "opposite_swing"
        closer = B.pick("de:closer", {"lcn_4040": 3, "magnetic_hold": 1, "auto_low_energy": 1})
        hinge = B.pick("de:hinge", {"butt_45_bb": 2, "continuous_geared": 1, "lift_off": 1})
        stop = B.pick("de:stop", {"overhead_90": 2, "wall_bumper": 1})
        seal = B.pick("de:seal", {"smoke_seal_intumescent": 2, "none": 1})
        cond = B.pick("de:cond", {"new": 2, "normal": 2, "worn": 1})
        use = B.pick("de:use", ["hospital corridor cross-over (double egress)", "airport concourse double egress", "mall corridor smoke doors"])
        frame = {"kind": "hollow_metal_frame", "material": "steel_painted", "casing": False, "stop_depth": 0.016, "jamb_depth": 0.146}
        astragal = "none"
        extras += ["kick_plate", "bumper_rail"] if rng.random() < 0.6 else ["kick_plate"]
        handle_h = 1.016
    elif ctx == "storefront_pair":
        slab = B.pick("sp:slab", {"storefront_alu": 2, "storefront_alu_igu": 1, "glass_frameless_12": 1})
        W = B.pick("sp:w", {0.914: 3, 1.067: 1})
        Hh = B.pick("sp:h", {2.134: 2, 2.438: 1})
        panel = "glass_frameless" if slab.startswith("glass_frameless") else "glass_full"
        op = B.pick("sp:op", {"pull_bar_offset": 2, "pull_ladder_full": 1, "panic_touchbar_rim": 1, "pull_d": 1})
        latch = "rim_exit" if op == "panic_touchbar_rim" else "none"
        lock = B.pick("sp:lock", {"none": 3, "thumbturn_only": 1, "mag_lock": 1})
        inactive_lock = "same_device"
        closer = B.pick("sp:closer", {"floor_spring": 2, "concealed_overhead": 1, "lcn_4040": 1})
        hinge = B.pick("sp:hinge", {"pivot_offset": 2, "pivot_center": 1, "continuous_geared": 1})
        stop = B.pick("sp:stop", {"overhead_90": 2, "none": 1})
        seal = B.pick("sp:seal", {"brush_pile": 2, "none": 1})
        cond = B.pick("sp:cond", {"new": 3, "normal": 1})
        use = B.pick("sp:use", ["mall entrance pair", "hotel lobby doors", "office tower entry pair", "supermarket entrance pair"])
        frame = {"kind": "aluminum_storefront", "material": "aluminum", "casing": False, "stop_depth": 0.012, "jamb_depth": 0.114}
        astragal = "none"
        extras += ["push_pull_sign"] if rng.random() < 0.5 else []
        handle_h = 1.040
    else:  # barn_pair (hinged carriage / barn doors)
        slab = B.pick("bp:slab", {"barn_plank": 2, "solid_wood_pine": 1, "cedar_plank": 1, "reclaimed": 0.001})
        if slab == "reclaimed":
            slab = "barn_plank"
        W = B.pick("bp:w", {1.0: 2, 1.2: 1, 0.9: 1})
        Hh = B.pick("bp:h", {2.2: 2, 2.4: 1, 2.0: 1})
        panel = B.pick("bp:panel", {"plank_x_brace": 2, "plank_z_brace": 2, "board_batten": 1})
        op = B.pick("bp:op", {"pull_ring": 1, "pull_barn_iron": 2})
        latch = "none"
        lock = B.pick("bp:lock", {"slide_bolt": 2, "padlock": 1, "none": 2})
        inactive_lock = "cane_bolt"
        closer = "none"
        hinge = B.pick("bp:hinge", {"strap_pintle": 2, "strap_heavy": 2})
        stop = "none"
        seal = "none"
        cond = B.pick("bp:cond", {"old_dry": 2, "worn": 2, "rusty": 1, "sagging": 1})
        use = B.pick("bp:use", ["barn double doors", "carriage house doors", "farm shed doors", "stable doors"])
        frame = {"kind": "timber_frame_heavy", "material": "douglas_fir", "casing": False, "stop_depth": 0.0, "jamb_depth": 0.15}
        astragal = "none"
        handle_h = 1.05
    t = M.SLABS[slab].typical_thickness[0] if slab.startswith("glass_frameless") else (0.044 if not slab.startswith("storefront") else 0.045)
    both_active_ = ctx in ("commercial_pair_panic", "double_egress", "storefront_pair")
    if both_active_ and astragal != "removable_mullion" and latch not in ("none", "vertical_rods"):
        latch = "vertical_rods" if op.startswith("panic") else "none"
        if op.startswith("panic"):
            op = "panic_touchbar_svr"
    s["use_case"] = use
    finish = finish_for(slab, ctx if ctx in PAINT_PALETTES else "default", B, rng)
    glass_mat = B.pick(f"glassmat:{ctx}", {"glass_clear": 4, "glass_frosted": 1, "glass_wired": 1 if ctx in ("commercial_pair_panic", "double_egress") else 0.001})
    s["leaf"] = {"width": W, "height": Hh, "thickness": t, "slab": slab, "panel_style": panel, "finish": finish, "count": 2,
                 "glazing": glazing_for(panel, W, Hh, glass_mat, 0.006, rng), "astragal": astragal, "inactive_leaf": {"lock": inactive_lock, "active": ctx in ("commercial_pair_panic", "double_egress", "storefront_pair")}}
    s["opening"] = {"width": _round(2 * W + 0.009), "height": _round(Hh + 0.013), "wall_thickness": B.pick(f"wall:{ctx}", {0.145: 2, 0.20: 1, 0.30: 1}), "frame": frame, "threshold": B.pick(f"thr:{ctx}", {"none": 2, "saddle": 1}), "sidelite": False, "transom": rng.random() < 0.2}
    hc = 1 if hinge == "continuous_geared" else (2 if hinge.startswith(("pivot", "strap")) else (4 if Hh > 2.3 else 3))
    s["hinge"] = hinge_block(hinge, hc, "left", swing, H.HINGES[hinge].axis_tilt_deg)
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": min(H.STOPS[stop]["max_open_deg"], 178) if stop != "none" else 110, "stop": stop, "pair": True, "double_egress": ctx == "double_egress"}
    s["operator"] = {"model": op, "height": handle_h, "sides": "push_side" if op.startswith(("panic", "push_plate")) else "both", "far_side": B.pick(f"farside:{ctx}", {"pull_d": 2, "none": 1, "pull_bar_offset": 1}) if op.startswith(("panic", "push_plate")) else None}
    s["latch"] = {"model": latch}
    s["lock"] = {"model": lock}
    s["closer"] = {"model": closer, "en_size": None, "spring_adjust": _round(_u(rng, 1.05, 1.3), 0.01)}
    s["seal"], s["condition"], s["extras"] = seal, cond, sorted(set(extras))
    choose_lock_engagement(s, B, rng, 0.4)
    s["tags"] = [ctx, "swing", "double_leaf", slab, panel, op, "lock:" + lock, "closer:" + closer, cond]
    return s


def gen_dutch(i, ctx, B, rng):
    s = base_spec(i, "dutch", "residential", rng)
    swing = approach(s, B, rng)
    slab = B.pick("du:slab", {"solid_wood_pine": 2, "solid_wood_fir": 1, "fiberglass_entry": 1, "solid_wood_oak": 1, "mdf_solid": 1})
    W, Hh, t = B.pick("du:w", {0.813: 2, 0.914: 2, 0.762: 1}), B.pick("du:h", {2.032: 3, 2.134: 1}), 0.044
    panel = B.pick("du:panel", {"2_panel_arch": 1, "glass_9_lite": 2, "shaker_2": 1, "plank_x_brace": 1, "glass_6_lite": 1})
    op = B.pick("du:op", {"lever_straight": 2, "knob_round": 2, "handleset_thumb": 1, "lever_euro_backplate": 1})
    s["use_case"] = B.pick("du:use", ["kitchen dutch door", "nursery dutch door", "stable dutch door", "daycare dutch door"])
    s["leaf"] = {"width": W, "height": Hh, "thickness": t, "slab": slab, "panel_style": panel, "finish": finish_for(slab, "residential_exterior", B, rng), "count": 1,
                 "glazing": glazing_for(panel, W, Hh, "glass_clear", 0.006, rng), "dutch_split_height": _round(Hh * B.pick("du:split", {0.5: 2, 0.55: 1, 0.45: 1}))}
    s["opening"] = {"width": _round(W + 0.006), "height": _round(Hh + 0.013), "wall_thickness": 0.145, "frame": {"kind": "wood_jamb_casing", "material": "pine", "casing": True, "stop_depth": 0.012, "jamb_depth": 0.115}, "threshold": "none", "sidelite": False, "transom": False}
    hinge = B.pick("du:hinge", {"butt_45_plain": 2, "butt_35_plain": 1, "butt_45_bb": 1})
    s["hinge"] = hinge_block(hinge, 4, B.pick("du:side", ["left", "right"]), swing)
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": B.pick("du:mo", {110: 2, 90: 1, 160: 1}), "stop": "none", "dutch": True, "joining_bolt_engaged": rng.random() < 0.5}
    s["operator"] = {"model": op, "height": 0.965, "sides": "both"}
    s["latch"] = {"model": "tubular_residential_70"}
    s["lock"] = {"model": "none"}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.1}
    s["seal"], s["condition"], s["extras"] = B.pick("du:seal", {"kerf_foam": 1, "none": 2}), B.pick("du:cond", {"new": 1, "normal": 2, "worn": 1}), []
    choose_lock_engagement(s, B, rng, 0.4)
    s["tags"] = ["dutch", slab, panel, op]
    return s


def gen_saloon(i, ctx, B, rng):
    s = base_spec(i, "saloon", "hospitality", rng)
    approach(s, B, rng, allow_pull=False)
    slab = B.pick("sa:slab", {"solid_wood_pine": 2, "louver_wood": 2, "solid_wood_oak": 1, "hpl_partition": 1, "hospital_solid": 1, "stainless_hollow": 1})
    pair = B.pick("sa:pair", {True: 3, False: 1})
    W = B.pick("sa:w", {0.45: 2, 0.50: 1, 0.60: 1}) if pair else B.pick("sa:w1", {0.80: 1, 0.90: 1})
    Hh = B.pick("sa:h", {1.10: 2, 1.30: 1, 2.03: 1, 0.90: 1})
    # a hung leaf never rests on the floor: the lowest option is the hinged-door floor clearance, not 0 - five saloon
    # leaves used to sit on the floor (a zero-gap touch whose degenerate contact stalled the swing; one could not reach
    # 10 deg under the QA push); real double-acting pivots run >= 12-20 mm above the floor
    zb = _round(B.pick("sa:zb", {0.35: 2, 0.20: 1, 0.60: 1, 0.012: 1})) if Hh < 1.9 else 0.012
    zb = max(zb, 0.015)
    panel = "louver_full" if slab == "louver_wood" else B.pick("sa:panel", {"flush": 2, "shaker_1": 1, "glass_vision": 1, "hpl_flat": 1})
    s["use_case"] = B.pick("sa:use", ["saloon bar doors", "cafe kitchen pass doors", "restaurant kitchen swing door", "hospital utility double-acting door", "supermarket stockroom doors"])
    t_sal = B.pick("sa:t", {0.035: 2, 0.044: 1})   # double-acting pivots: hinge-edge gap >= t/2 + 6 mm so the corners clear the jamb
    s["leaf"] = {"width": W, "height": Hh, "thickness": t_sal, "slab": slab, "panel_style": panel, "finish": finish_for(slab, "default", B, rng), "count": 2 if pair else 1,
                 "glazing": glazing_for(panel, W, Hh, "glass_clear", 0.006, rng), "bottom_clearance": zb}
    s["opening"] = {"width": _round((2 * W if pair else W) + t_sal + 0.024), "height": _round(2.05 if Hh < 1.9 else Hh + zb + 0.013), "wall_thickness": 0.145, "frame": {"kind": "wood_jamb_casing", "material": "pine", "casing": True, "stop_depth": 0.0, "jamb_depth": 0.115}, "threshold": "none", "sidelite": False, "transom": False}
    s["hinge"] = hinge_block("spring_double", 2, "left", "push")
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": B.pick("sa:mo", {90: 2, 110: 1, 100: 1}), "stop": "none", "both_ways": True, "pair": pair}
    s["operator"] = {"model": B.pick("sa:op", {"none": 3, "push_plate": 1, "kick_plate_only": 0.001}), "height": 1.0, "sides": "both"}
    if s["operator"]["model"] == "kick_plate_only":
        s["operator"]["model"] = "none"
    s["latch"] = {"model": "none"}
    s["lock"] = {"model": "none"}
    s["closer"] = {"model": "spring_hinge_double", "en_size": None, "spring_adjust": 1.0, "spring_hinge_k": _round(_u(rng, 1.5, 3.5), 0.1)}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("sa:cond", {"new": 1, "normal": 2, "worn": 1, "old_dry": 1}), (["kick_plate"] if rng.random() < 0.5 else [])
    choose_lock_engagement(s, B, rng, 0.0)
    s["tags"] = ["saloon", "double_acting", slab]
    return s


def gen_pivot(i, ctx, B, rng):
    s = base_spec(i, "pivot", "architectural", rng)
    swing = approach(s, B, rng)
    slab = B.pick("pv:slab", {"solid_wood_walnut": 2, "solid_wood_oak": 1, "glass_frameless_12": 2, "glass_frameless_19": 1, "steel_plate_security": 1, "mdf_solid": 1, "stainless_hollow": 1, "solid_wood_teak": 1})
    W = B.pick("pv:w", {1.0: 2, 1.2: 2, 1.5: 1, 0.9: 1, 1.8: 1})
    Hh = B.pick("pv:h", {2.4: 2, 2.7: 2, 3.0: 1, 2.134: 1})
    t = M.SLABS[slab].typical_thickness[0] if slab.startswith(("glass", "steel_plate")) else B.pick("pv:t", {0.045: 1, 0.060: 1})
    panel = "glass_frameless" if slab.startswith("glass") else B.pick("pv:panel", {"flush": 3, "plank_vertical": 1, "glass_vision": 1})
    op = B.pick("pv:op", {"pull_ladder_full": 3, "pull_bar_offset": 1, "pull_d": 1, "lever_l_shape": 1, "none": 1})
    pivot = B.pick("pv:pivot", {"pivot_center": 2, "pivot_offset": 1, "pivot_center_heavy": 1})
    s["use_case"] = B.pick("pv:use", ["modern residence pivot entry door", "hotel lobby pivot door", "boutique glass pivot door", "museum gallery pivot door", "office reception pivot door"])
    s["leaf"] = {"width": W, "height": Hh, "thickness": t, "slab": slab, "panel_style": panel, "finish": finish_for(slab, "default", B, rng), "count": 1,
                 "glazing": glazing_for(panel, W, Hh, "glass_clear", t if slab.startswith("glass") else 0.008, rng)}
    s["opening"] = {"width": _round(W + 0.012), "height": _round(Hh + 0.012), "wall_thickness": B.pick("pv:wall", {0.20: 2, 0.30: 1, 0.145: 1}), "frame": {"kind": "minimal_reveal", "material": "aluminum_dark" if rng.random() < 0.5 else "steel_painted", "casing": False, "stop_depth": 0.0, "jamb_depth": 0.20}, "threshold": "none", "sidelite": rng.random() < 0.3, "transom": False}
    offset = _round(W * B.pick("pv:off", {0.15: 1, 0.2: 2, 0.25: 1, 0.33: 1})) if pivot != "pivot_offset" else 0.02
    s["hinge"] = hinge_block(pivot, 2, B.pick("pv:side", ["left", "right"]), swing)
    s["hinge"]["pivot_offset_m"] = offset
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": B.pick("pv:mo", {90: 3, 110: 1, 180: 1}), "stop": B.pick("pv:stop", {"floor_dome": 1, "none": 2, "overhead_90": 1})}
    s["operator"] = {"model": op, "height": 1.05, "sides": "both"}
    s["latch"] = {"model": B.pick("pv:latch", {"none": 3, "magnetic_catch": 1, "mortise_euro": 1 if op == "lever_l_shape" else 0.001})}
    s["lock"] = {"model": B.pick("pv:lock", {"none": 3, "thumbturn_only": 1, "mag_lock": 1, "electric_bolt": 0.001})}
    s["closer"] = {"model": B.pick("pv:closer", {"floor_spring": 2, "floor_spring_nohold": 1, "none": 2}), "en_size": None, "spring_adjust": _round(_u(rng, 1.0, 1.25), 0.01)}
    s["seal"], s["condition"], s["extras"] = B.pick("pv:seal", {"brush_pile": 1, "none": 2}), B.pick("pv:cond", {"new": 3, "normal": 2, "well_oiled": 1}), []
    choose_lock_engagement(s, B, rng, 0.3)
    s["tags"] = ["pivot", slab, op, pivot]
    return s


def gen_sliding_single(i, ctx, B, rng):
    s = base_spec(i, "sliding_single", ctx, rng)
    s["robot"]["robot_outside"] = rng.random() < 0.5
    s["robot"]["is_push"] = False
    extras = []
    if ctx == "pocket":
        slab = B.pick("pk:slab", {"hollow_core": 3, "solid_core_pb": 2, "mdf_solid": 1, "solid_wood_pine": 1, "glass_frameless_10": 1})
        W, Hh, t = B.pick("pk:w", {0.762: 2, 0.813: 2, 0.914: 1, 0.711: 1}), 2.032, 0.035
        panel = "glass_frameless" if slab.startswith("glass") else B.pick("pk:panel", {"flush": 3, "6_panel": 1, "shaker_1": 1, "glass_frosted_full": 0.001})
        op = B.pick("pk:op", {"pull_flush_recessed": 3, "pull_finger_cup": 1, "pull_d": 1})
        latch = "none"
        lock = B.pick("pk:lock", {"none": 3, "hook_lock": 2, "privacy_button": 0.001})
        roller = B.pick("pk:roller", {"ball_bearing_nylon": 2, "plain_nylon": 2, "plain_steel_worn": 1, "dirty_track": 1})
        track = "top_hung_pocket"
        use = B.pick("pk:use", ["bathroom pocket door", "pantry pocket door", "laundry pocket door", "office pocket door"])
        frame = {"kind": "wood_jamb_casing", "material": "pine", "casing": True, "stop_depth": 0.0, "jamb_depth": 0.115}
        travel_frac = 0.95
    elif ctx == "barn":
        slab = B.pick("bn:slab", {"barn_plank": 3, "solid_wood_pine": 2, "mdf_solid": 1, "glass_frameless_10": 1, "steel_plate_security": 0.5, "solid_wood_oak": 1})
        W = B.pick("bn:w", {0.914: 2, 1.067: 2, 1.219: 1, 0.813: 1})
        Hh = B.pick("bn:h", {2.134: 2, 2.032: 1, 2.4: 1})
        t = 0.010 if slab.startswith("glass") else (M.SLABS[slab].typical_thickness[0] if slab == "steel_plate_security" else 0.044)
        panel = "glass_frameless" if slab.startswith("glass") else B.pick("bn:panel", {"plank_z_brace": 2, "plank_x_brace": 2, "board_batten": 1, "shaker_1": 1, "5_panel_horizontal": 1})
        op = B.pick("bn:op", {"pull_barn_iron": 3, "barn_privacy_hook": 2, "pull_d": 1, "pull_ring": 1})
        latch = "teardrop" if op == "barn_privacy_hook" else "none"
        lock = B.pick("bn:lock", {"none": 4, "hook_lock": 1})
        roller = B.pick("bn:roller", {"barn_hanger": 3, "plain_steel_worn": 1, "dirty_track": 1})
        track = "surface_flat_track"
        use = B.pick("bn:use", ["barn door to bathroom", "barn door to office", "loft barn door", "restaurant barn door", "warehouse sliding door"])
        frame = {"kind": "wood_jamb_casing", "material": "pine", "casing": True, "stop_depth": 0.0, "jamb_depth": 0.115}
        travel_frac = 1.0
        extras += ["floor_guide"] if rng.random() < 0.7 else []
        extras += ["soft_close_damper"] if rng.random() < 0.3 else []
    elif ctx == "patio_glass":
        slab = B.pick("pg:slab", {"patio_slider_glass": 4, "storefront_alu_igu": 1, "glass_frameless_12": 1})
        W = B.pick("pg:w", {0.914: 2, 0.762: 1, 1.219: 1, 1.5: 1})
        Hh = B.pick("pg:h", {2.032: 2, 2.134: 1, 2.4: 1})
        t = 0.045
        panel = "glass_full"
        op = B.pick("pg:op", {"hook_lock_slider": 4, "pull_d": 1, "pull_flush_recessed": 1})
        latch = "hook_slider" if op == "hook_lock_slider" else "none"
        lock = B.pick("pg:lock", {"hook_lock": 3, "none": 1, "slide_bolt": 1})
        roller = B.pick("pg:roller", {"ball_bearing_steel": 2, "plain_steel_worn": 2, "dirty_track": 1, "ball_bearing_nylon": 1})
        track = "bottom_rolling"
        use = B.pick("pg:use", ["patio sliding glass door", "balcony slider", "sunroom slider", "hotel balcony door"])
        frame = {"kind": "vinyl_slider_frame", "material": "pvc" if rng.random() < 0.7 else "aluminum", "casing": False, "stop_depth": 0.0, "jamb_depth": 0.12}
        travel_frac = 0.92
        extras += ["threshold_saddle"]
    elif ctx == "shoji_fusuma":
        slab = B.pick("sj:slab", {"shoji": 3, "fusuma": 2})
        W = B.pick("sj:w", {0.90: 3, 0.86: 1, 0.95: 1})
        Hh = B.pick("sj:h", {1.80: 3, 1.76: 1, 2.0: 1})
        t = 0.030
        panel = "lattice_shoji" if slab == "shoji" else "hpl_flat"
        op = B.pick("sj:op", {"shoji_finger_pull": 4, "pull_flush_recessed": 1})
        latch = "none"
        lock = "none"
        roller = B.pick("sj:roller", {"wood_on_wood": 3, "wood_on_wood_waxed": 2, "glide_teflon": 1})
        track = "wood_groove_bottom"
        use = B.pick("sj:use", ["shoji screen (tatami room)", "fusuma closet door", "engawa shoji", "tea room shoji"])
        frame = {"kind": "kamoi_shikii", "material": "hinoki", "casing": False, "stop_depth": 0.0, "jamb_depth": 0.10}
        travel_frac = 1.0
    else:  # cell_industrial
        slab = B.pick("ci:slab", {"steel_bar_grille": 2, "hollow_metal_14ga": 2, "hollow_metal_16ga": 1, "elevator_landing": 1, "cold_storage_100": 1})
        W = B.pick("ci:w", {0.914: 2, 1.067: 1, 1.219: 1, 1.5: 1})
        Hh = B.pick("ci:h", {2.134: 3, 2.438: 1})
        t = M.SLABS[slab].typical_thickness[0]
        panel = "bar_grille" if slab == "steel_bar_grille" else B.pick("ci:panel", {"steel_flush": 2, "steel_vision": 1, "riveted_steel": 1})
        op = B.pick("ci:op", {"pull_d": 3, "pull_bar_offset": 1, "none": 1, "cold_storage_handle": 1})
        latch = B.pick("ci:latch", {"none": 2, "electric_bolt": 1, "slide_bolt_heavy": 1})
        lock = B.pick("ci:lock", {"keyed_cylinder": 2, "electric_strike": 1, "slide_bolt": 1, "none": 1, "padlock": 1})
        roller = B.pick("ci:roller", {"bottom_rolling_heavy": 2, "ball_bearing_steel": 1, "dirty_track": 1, "plain_steel_worn": 1})
        track = "top_hung_industrial"
        use = B.pick("ci:use", ["detention cell sliding door", "industrial sliding fire door", "cold room sliding door", "freight elevator manual gate"])
        frame = {"kind": "hollow_metal_frame", "material": "steel", "casing": False, "stop_depth": 0.0, "jamb_depth": 0.20}
        travel_frac = 1.0
        extras += ["warning_placard"] if rng.random() < 0.4 else []
    s["use_case"] = use
    finish = finish_for(slab, "default" if ctx != "cell_industrial" else "industrial_utility", B, rng)
    s["leaf"] = {"width": W, "height": Hh, "thickness": t, "slab": slab, "panel_style": panel, "finish": finish, "count": 1,
                 "glazing": glazing_for(panel, W, Hh, "glass_clear", 0.006 if ctx != "patio_glass" else 0.019, rng)}
    if slab == "shoji":
        s["leaf"]["glazing"] = {"material": "washi_paper", "thickness": 0.0002, "area_fraction": 0.8, "panel_style": panel, "count": 1}
    s["opening"] = {"width": _round(W * (1.0 if ctx not in ("patio_glass", "shoji_fusuma") else 2.0) + 0.02), "height": _round(Hh + (0.06 if ctx == "shoji_fusuma" else (0.045 if ctx == "patio_glass" else 0.02))), "wall_thickness": 0.145, "frame": frame, "threshold": "none" if ctx not in ("patio_glass",) else "saddle", "sidelite": False, "transom": False,
                    "fixed_panel": ctx in ("patio_glass", "shoji_fusuma")}
    s["hinge"] = hinge_block("none", 0, B.pick(f"slideside:{ctx}", ["left", "right"]), "slide")
    s["kinematics"] = {"type": "slide_horizontal", "travel_m": _round(W * travel_frac), "roller": roller, "track": track, "stop": "track_end", "opens_toward": s["hinge"]["side"]}
    s["operator"] = {"model": op, "height": B.pick("sl:hh", {1.0: 3, 0.95: 1, 1.05: 1}), "sides": "both"}
    s["latch"] = {"model": latch}
    s["lock"] = {"model": lock}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"] = B.pick(f"slseal:{ctx}", {"none": 3, "brush_pile": 2})
    s["condition"] = B.pick(f"slcond:{ctx}", {"new": 2, "normal": 3, "worn": 2, "old_dry": 1, "damaged": 1} if ctx != "shoji_fusuma" else {"new": 2, "normal": 2, "worn": 1})
    s["extras"] = sorted(set(extras))
    choose_lock_engagement(s, B, rng, 0.4)
    s["tags"] = [ctx, "sliding", slab, panel, op, roller, "lock:" + lock]
    return s


def gen_sliding_bypass(i, ctx, B, rng):
    s = base_spec(i, "sliding_bypass", "closet", rng)
    s["robot"]["is_push"] = False
    kind = B.pick("by:kind", {"closet_wood": 3, "mirror": 2, "shoji_pair": 1, "glass_frameless": 1})
    if kind == "closet_wood":
        slab = B.pick("by:slab", {"hollow_core": 3, "hollow_core_molded": 2, "mdf_solid": 1, "louver_wood": 1})
        panel = "louver_full" if slab == "louver_wood" else B.pick("by:panel", {"flush": 2, "6_panel": 1, "shaker_1": 1, "2_panel_arch": 1})
        op = B.pick("by:op", {"pull_finger_cup": 3, "pull_flush_recessed": 1, "bifold_knob": 1})
        roller = B.pick("by:roller", {"plain_nylon": 3, "ball_bearing_nylon": 1, "dirty_track": 1, "plain_steel_worn": 1})
    elif kind == "mirror":
        slab, panel = "mirror_bypass", "flush"
        op = B.pick("by:op_m", {"pull_finger_cup": 1, "pull_flush_recessed": 1, "none": 1})
        roller = B.pick("by:roller_m", {"ball_bearing_nylon": 2, "plain_nylon": 1, "plain_steel_worn": 1})
    elif kind == "shoji_pair":
        slab, panel = "shoji", "lattice_shoji"
        op = "shoji_finger_pull"
        roller = B.pick("by:roller_s", {"wood_on_wood": 2, "wood_on_wood_waxed": 1})
    else:
        slab, panel = "glass_frameless_10", "glass_frameless"
        op = B.pick("by:op_g", {"pull_d": 1, "pull_flush_recessed": 1})
        roller = "ball_bearing_steel"
    n = B.pick("by:n", {2: 4, 3: 1})
    W = B.pick("by:w", {0.61: 2, 0.762: 2, 0.914: 1, 1.219: 1})
    Hh = B.pick("by:h", {2.032: 3, 2.4: 1, 1.8: 1 if kind == "shoji_pair" else 0.001})
    t = 0.035 if kind == "closet_wood" else M.SLABS[slab].typical_thickness[0] if kind != "shoji_pair" else 0.03
    s["use_case"] = B.pick("by:use", ["bedroom closet bypass doors", "hallway linen closet doors", "mirrored wardrobe doors", "shoji closet (oshiire)"])
    s["leaf"] = {"width": W, "height": Hh, "thickness": t, "slab": slab, "panel_style": panel, "finish": finish_for(slab, "residential_interior", B, rng), "count": n, "glazing": glazing_for(panel, W, Hh, "glass_clear", 0.006, rng)}
    if slab == "shoji":
        s["leaf"]["glazing"] = {"material": "washi_paper", "thickness": 0.0002, "area_fraction": 0.8, "panel_style": panel, "count": 1}
    s["opening"] = {"width": _round(n * W - (n - 1) * 0.03 + 0.01), "height": _round(Hh + (0.06 if kind == "shoji_pair" else 0.02)), "wall_thickness": 0.145, "frame": {"kind": "wood_jamb_casing", "material": "pine", "casing": True, "stop_depth": 0.0, "jamb_depth": 0.115}, "threshold": "none", "sidelite": False, "transom": False}
    s["hinge"] = hinge_block("none", 0, "left", "slide")
    s["kinematics"] = {"type": "slide_horizontal", "travel_m": _round(W - 0.03), "roller": roller, "track": "top_hung_bypass" if kind != "shoji_pair" else "wood_groove_bottom", "stop": "track_end", "n_leaves": n}
    s["operator"] = {"model": op, "height": 1.0, "sides": "robot"}
    s["latch"] = {"model": "none"}
    s["lock"] = {"model": "none"}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("by:cond", {"new": 1, "normal": 3, "worn": 2, "damaged": 1}), (["floor_guide"] if rng.random() < 0.5 else [])
    choose_lock_engagement(s, B, rng, 0.0)
    s["tags"] = ["bypass", kind, slab, roller]
    return s


def gen_bifold(i, ctx, B, rng):
    s = base_spec(i, "bifold", "closet", rng)
    s["robot"]["is_push"] = False
    slab = B.pick("bf:slab", {"hollow_core": 3, "louver_wood": 3, "mdf_solid": 1, "solid_wood_pine": 1, "mirror_bypass": 1})
    panel = "louver_full" if slab == "louver_wood" else B.pick("bf:panel", {"flush": 2, "6_panel": 1, "louver_half": 1, "shaker_1": 1, "glass_frosted_full": 0.001})
    n = B.pick("bf:n", {2: 3, 4: 2})
    W_total = B.pick("bf:wt", {0.61: 1, 0.762: 2, 0.914: 2, 1.219: 1, 1.524: 1})
    W = _round(W_total / n)
    Hh = B.pick("bf:h", {2.032: 4, 2.4: 1})
    s["use_case"] = B.pick("bf:use", ["bedroom closet bifold", "laundry closet bifold", "pantry bifold", "utility closet bifold (louvered)"])
    t = 0.006 if slab == "mirror_bypass" else 0.035
    s["leaf"] = {"width": W, "height": Hh, "thickness": t, "slab": slab, "panel_style": panel, "finish": finish_for(slab, "residential_interior", B, rng), "count": n, "glazing": None}
    # opening sized around the panel set (doorbench/folding.py): pivot-jamb gap + the lead / meeting gap the face-hinged
    # zigzag needs while it folds; height = panels + floor gap + top track under the head jamb
    s["opening"] = {"width": _round(fold_opening_width(W, n, fold_groups(n, False), t)), "height": _round(fold_opening_height(Hh)), "wall_thickness": 0.145, "frame": {"kind": "wood_jamb_casing", "material": "pine", "casing": True, "stop_depth": 0.0, "jamb_depth": 0.115}, "threshold": "none", "sidelite": False, "transom": False}
    s["hinge"] = hinge_block("butt_35_plain", 2, "left", "fold")
    # max_open_deg = pivot panel travel; the panel pair folds to twice that (170 deg: knuckles keep the stack off flat)
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": FOLD_PIVOT_MAX_DEG, "stop": "track_end", "fold": True, "n_panels": n, "roller": B.pick("bf:roller", {"bifold_pivot_guide": 3, "plain_nylon": 1, "dirty_track": 1})}
    s["operator"] = {"model": B.pick("bf:op", {"bifold_knob": 4, "pull_d": 1}), "height": 1.0, "sides": "robot"}
    s["latch"] = {"model": B.pick("bf:latch", {"none": 3, "magnetic_catch": 1})}
    s["lock"] = {"model": "none"}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("bf:cond", {"new": 1, "normal": 2, "worn": 2, "damaged": 1}), []
    choose_lock_engagement(s, B, rng, 0.0)
    s["tags"] = ["bifold", slab, panel, f"{n}_panel"]
    return s


def gen_accordion(i, ctx, B, rng):
    s = base_spec(i, "accordion", "partition", rng)
    s["robot"]["is_push"] = False
    slab = B.pick("ac:slab", {"pvc": 0.001, "upvc_panel": 2, "hpl_partition": 1, "mdf_solid": 1, "canvas_tent": 1})
    if slab == "pvc":
        slab = "upvc_panel"
    n = B.pick("ac:n", {6: 2, 8: 2, 10: 1})
    W_total = B.pick("ac:wt", {0.914: 2, 1.219: 2, 1.829: 1})
    W = _round(W_total / n)
    Hh = B.pick("ac:h", {2.032: 3, 2.4: 1})
    s["use_case"] = B.pick("ac:use", ["accordion closet door", "room divider accordion", "laundry nook accordion", "office partition accordion"])
    t = 0.012 if slab != "mdf_solid" else 0.018
    s["leaf"] = {"width": W, "height": Hh, "thickness": t, "slab": slab, "panel_style": "hpl_flat", "finish": finish_for(slab, "residential_interior", B, rng), "count": n, "glazing": None}
    # opening sized around the panel set (doorbench/folding.py): pivot-jamb gap + the lead gap the face-hinged zigzag
    # needs while it folds (up to ~20 mm for ten 18 mm panels); height = panels + floor gap + top track under the head jamb
    s["opening"] = {"width": _round(fold_opening_width(W, n, fold_groups(n, True), t)), "height": _round(fold_opening_height(Hh)), "wall_thickness": 0.145, "frame": {"kind": "wood_jamb_casing", "material": "pine", "casing": True, "stop_depth": 0.0, "jamb_depth": 0.115}, "threshold": "none", "sidelite": False, "transom": False}
    s["hinge"] = hinge_block("piano", n - 1, "left", "fold")
    # max_open_deg = pivot panel travel (the primary joint); every panel pair folds to twice that, 170 deg at the stack
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": FOLD_PIVOT_MAX_DEG, "stop": "track_end", "fold": True, "accordion": True, "n_panels": n, "roller": "accordion_glides"}
    s["operator"] = {"model": B.pick("ac:op", {"pull_d": 2, "bifold_knob": 1, "pull_flush_recessed": 1}), "height": 1.0, "sides": "robot"}
    s["latch"] = {"model": B.pick("ac:latch", {"none": 2, "magnetic_catch": 2})}
    s["lock"] = {"model": "none"}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("ac:cond", {"new": 1, "normal": 2, "worn": 2}), []
    choose_lock_engagement(s, B, rng, 0.0)
    s["tags"] = ["accordion", slab, f"{n}_panel"]
    return s


def gen_revolving(i, ctx, B, rng):
    s = base_spec(i, "revolving", "commercial_entry", rng)
    s["robot"]["is_push"] = True
    wings = B.pick("rv:wings", {3: 2, 4: 3})
    D = B.pick("rv:d", {1.8: 1, 2.0: 2, 2.4: 1, 3.0: 1, 3.6: 1})
    Hh = B.pick("rv:h", {2.134: 2, 2.4: 2, 2.7: 1})
    s["use_case"] = B.pick("rv:use", ["office tower revolving door", "hotel revolving door", "department store revolving door", "airport revolving door", "hospital lobby revolving door"])
    s["leaf"] = {"width": _round(D / 2 - 0.03), "height": Hh, "thickness": 0.010, "slab": "revolving_wing", "panel_style": "glass_full", "finish": finish_for("revolving_wing", "default", B, rng), "count": wings, "glazing": {"material": "glass_clear", "thickness": 0.010, "area_fraction": 0.85, "panel_style": "glass_full", "count": 1}}
    s["opening"] = {"width": _round(D + 0.1), "height": _round(Hh + 0.3), "wall_thickness": 0.30, "frame": {"kind": "revolving_drum", "material": B.pick("rv:fm", {"aluminum": 2, "stainless": 1, "brass": 1, "aluminum_dark": 1}), "casing": False, "stop_depth": 0.0, "jamb_depth": 0.30}, "threshold": "none", "sidelite": False, "transom": False, "drum_diameter": D, "drum_opening_deg": B.pick("rv:open", {90: 1, 100: 2, 120: 1})}
    s["hinge"] = hinge_block("rotor_bearing", 1, "center", "rotate")
    s["kinematics"] = {"type": "rotor", "max_open_deg": None, "stop": "none", "wings": wings, "speed_governor_damping": _round(_u(rng, 20, 60), 1), "manual": B.pick("rv:manual", {True: 3, False: 1}), "breakout": True}
    s["operator"] = {"model": B.pick("rv:op", {"pull_d": 2, "push_plate": 1, "none": 2}), "height": 1.05, "sides": "both"}
    s["latch"] = {"model": "none"}
    s["lock"] = {"model": B.pick("rv:lock", {"none": 4, "electric_bolt": 1})}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "brush_pile", B.pick("rv:cond", {"new": 2, "normal": 2, "worn": 1}), ["push_pull_sign"]
    choose_lock_engagement(s, B, rng, 0.2)
    s["tags"] = ["revolving", f"{wings}_wing", f"D{D}"]
    return s


def gen_turnstile(i, ctx, B, rng, full_height=False):
    fam = "turnstile_fullheight" if full_height else "turnstile_tripod"
    s = base_spec(i, fam, "access_control", rng)
    s["robot"]["is_push"] = True
    if full_height:
        wings = B.pick("tf:wings", {3: 2, 4: 1})
        s["use_case"] = B.pick("tf:use", ["stadium full-height turnstile", "factory gate turnstile", "metro station turnstile", "parking garage pedestrian turnstile"])
        s["leaf"] = {"width": _round(B.pick("tf:r", {0.65: 2, 0.75: 1})), "height": 2.1, "thickness": 0.038, "slab": "turnstile_arm", "panel_style": "bar_grille", "finish": finish_for("turnstile_arm", "default", B, rng), "count": wings, "glazing": None, "arms_per_wing": 8}
        s["opening"] = {"width": 1.6, "height": 2.3, "wall_thickness": 0.1, "frame": {"kind": "turnstile_cage", "material": B.pick("tf:m", {"steel_galvanized": 2, "stainless": 1, "steel_painted": 1}), "casing": False, "stop_depth": 0, "jamb_depth": 0.1}, "threshold": "none", "sidelite": False, "transom": False}
        s["kinematics"] = {"type": "rotor", "max_open_deg": None, "stop": "none", "wings": wings, "ratchet_deg": 360 / wings, "one_way": B.pick("tf:oneway", {True: 2, False: 1}), "locked_until_credential": B.pick("tf:lock", {True: 2, False: 1})}
    else:
        s["use_case"] = B.pick("tt:use", ["subway tripod turnstile", "office lobby tripod turnstile", "gym entrance turnstile", "stadium tripod turnstile", "amusement park turnstile"])
        s["leaf"] = {"width": 0.50, "height": 1.0, "thickness": 0.038, "slab": "turnstile_arm", "panel_style": "bar_grille", "finish": finish_for("turnstile_arm", "default", B, rng), "count": 3, "glazing": None}
        s["opening"] = {"width": 0.55, "height": 1.05, "wall_thickness": 0.1, "frame": {"kind": "turnstile_cabinet", "material": B.pick("tt:m", {"stainless": 3, "steel_painted": 1}), "casing": False, "stop_depth": 0, "jamb_depth": 0.1}, "threshold": "none", "sidelite": False, "transom": False}
        s["kinematics"] = {"type": "rotor", "max_open_deg": None, "stop": "none", "wings": 3, "ratchet_deg": 120, "one_way": True, "locked_until_credential": B.pick("tt:lock", {True: 3, False: 1}), "axis_tilt_deg": 45, "drop_arm": rng.random() < 0.3}
    s["hinge"] = hinge_block("rotor_bearing", 1, "center", "rotate")
    s["operator"] = {"model": "turnstile_arm", "height": 0.95, "sides": "both"}
    s["latch"] = {"model": "none"}
    s["lock"] = {"model": "electric_strike" if s["kinematics"]["locked_until_credential"] else "none"}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("tt:cond", {"new": 2, "normal": 2, "worn": 1}), ["keypad_reader_wall"]
    s["lock"]["engaged"] = s["kinematics"]["locked_until_credential"]
    s["lock"]["robot_side_release"] = False
    s["tags"] = [fam, "ratchet"]
    return s


def gen_garage(i, ctx, B, rng, kind):
    s = base_spec(i, kind, "garage", rng)
    s["robot"]["is_push"] = False
    W = B.pick(f"{kind}:w", {2.44: 3, 2.74: 1, 3.05: 1, 4.88: 2, 5.49: 1} if kind != "rollup" else {2.44: 2, 3.05: 2, 3.66: 1, 1.2: 1})
    Hh = B.pick(f"{kind}:h", {2.13: 3, 2.44: 1, 2.0: 1} if kind != "rollup" else {2.44: 2, 3.05: 1, 2.13: 1, 3.66: 1})
    if kind == "garage_sectional":
        slab = B.pick("gs:slab", {"garage_steel_single": 3, "garage_steel_insulated": 3, "garage_wood_carriage": 1})
        panel = B.pick("gs:panel", {"sectional_raised_short": 3, "sectional_flush": 1, "sectional_long_windows": 2, "raised_carriage": 1})
        n_sections = 4 if Hh <= 2.2 else 5
        s["kinematics"] = {"type": "slide_vertical", "travel_m": _round(Hh - 0.05), "roller": B.pick("gs:roller", {"garage_nylon": 3, "garage_steel_dry": 2}), "track": "sectional_vertical_lift", "stop": "track_end", "n_sections": n_sections,
                           "counterbalance_fraction": _round(B.pick("gs:cb", {0.95: 3, 0.85: 1, 0.6: 1, 0.0: 1}), 0.01), "opener": B.pick("gs:opener", {"none_manual": 3, "chain_drive_disengaged": 1, "belt_drive_engaged": 1})}
        op = B.pick("gs:op", {"pull_lift_garage": 3, "pull_t_handle_garage": 2, "none": 1})
        use = B.pick("gs:use", ["residential single garage door", "residential double garage door", "detached garage door", "townhouse garage door"])
    elif kind == "garage_tiltup":
        slab = B.pick("gt:slab", {"garage_steel_single": 2, "garage_wood_carriage": 2, "plywood": 0.001})
        if slab == "plywood":
            slab = "garage_wood_carriage"
        panel = B.pick("gt:panel", {"sectional_flush": 1, "raised_carriage": 1, "plank_vertical": 1})
        s["kinematics"] = {"type": "hinge_horizontal", "max_open_deg": 88, "stop": "track_end", "mechanism": "retractable_top_roller_side_arm", "counterbalance_fraction": _round(B.pick("gt:cb", {0.9: 2, 0.7: 1}), 0.01)}
        op = B.pick("gt:op", {"pull_t_handle_garage": 2, "pull_lift_garage": 1})
        use = B.pick("gt:use", ["1960s tilt-up garage door", "carport tilt-up door"])
    else:  # rollup
        slab = B.pick("ru:slab", {"rollup_steel": 3, "rollup_alu_grille": 1})
        panel = "corrugated_slats" if slab == "rollup_steel" else "grille_rollup"
        s["kinematics"] = {"type": "slide_vertical", "travel_m": _round(Hh - 0.05), "roller": "rollup_curtain", "track": "coiling_guides", "stop": "track_end",
                           "counterbalance_fraction": _round(B.pick("ru:cb", {0.9: 3, 0.75: 1, 0.5: 1}), 0.01), "opener": B.pick("ru:opener", {"none_manual": 2, "chain_hoist": 2, "motor_disengaged": 1})}
        op = B.pick("ru:op", {"pull_lift_garage": 2, "pull_d": 1, "none": 1, "pull_ring": 1})
        use = B.pick("ru:use", ["self-storage unit roll-up door", "shop front security shutter", "warehouse coiling door", "loading dock roll-up door", "parking garage grille"])
    s["use_case"] = use
    s["leaf"] = {"width": W, "height": Hh, "thickness": M.SLABS[slab].typical_thickness[0], "slab": slab, "panel_style": panel, "finish": finish_for(slab, "default", B, rng), "count": 1,
                 "glazing": {"material": "glass_clear", "thickness": 0.003, "area_fraction": 0.06, "panel_style": panel, "count": 4} if panel == "sectional_long_windows" else None}
    s["opening"] = {"width": _round(W + 0.05), "height": _round(Hh + 0.05), "wall_thickness": 0.20, "frame": {"kind": "garage_jamb", "material": "pine" if kind != "rollup" else "steel_painted", "casing": False, "stop_depth": 0.0, "jamb_depth": 0.20}, "threshold": "none", "sidelite": False, "transom": False}
    s["hinge"] = hinge_block("none" if kind != "garage_tiltup" else "pivot_offset", 0 if kind != "garage_tiltup" else 2, "bottom", "lift")
    s["operator"] = {"model": op, "height": 0.9, "sides": "both"}
    s["latch"] = {"model": "none"}
    s["lock"] = {"model": B.pick(f"{kind}:lock", {"none": 3, "garage_slide_lock": 1, "padlock": 1, "keyed_cylinder": 1 if op == "pull_t_handle_garage" else 0.001})}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = B.pick(f"{kind}:seal", {"door_sweep": 2, "none": 1, "brush_pile": 1}), B.pick(f"{kind}:cond", {"new": 1, "normal": 2, "worn": 2, "rusty": 1, "damaged": 1}), []
    choose_lock_engagement(s, B, rng, 0.35)
    s["tags"] = [kind, slab, panel]
    return s


def gen_pet_door(i, ctx, B, rng):
    s = base_spec(i, "pet_door", "residential", rng)
    size = B.pick("pd:size", {"cat": 2, "small_dog": 2, "medium_dog": 2, "large_dog": 2, "xl_dog": 1})
    dims = {"cat": (0.15, 0.17), "small_dog": (0.18, 0.28), "medium_dog": (0.25, 0.38), "large_dog": (0.30, 0.50), "xl_dog": (0.38, 0.64)}[size]
    slab = B.pick("pd:slab", {"pet_flap_pvc": 3, "pet_flap_acrylic": 2})
    s["use_case"] = f"{size.replace('_', ' ')} pet door in {B.pick('pd:host', ['wall', 'exterior door', 'screen door', 'garage door'])}"
    s["leaf"] = {"width": dims[0], "height": dims[1], "thickness": M.SLABS[slab].typical_thickness[0], "slab": slab, "panel_style": "acrylic_flap", "finish": finish_for(slab, "default", B, rng), "count": 1, "glazing": None}
    s["opening"] = {"width": _round(dims[0] + 0.01), "height": _round(dims[1] + 0.01), "wall_thickness": B.pick("pd:wall", {0.044: 2, 0.145: 1, 0.25: 1}), "frame": {"kind": "pet_door_frame", "material": B.pick("pd:fm", {"pvc": 2, "aluminum": 1}), "casing": True, "stop_depth": 0, "jamb_depth": 0.05}, "threshold": "none", "sidelite": False, "transom": False, "host_panel": True}
    s["hinge"] = hinge_block("flap_pin", 1, "top", "both")
    s["kinematics"] = {"type": "hinge_horizontal", "max_open_deg": B.pick("pd:mo", {90: 2, 80: 1, 110: 1}), "stop": "none", "both_ways": True, "flap": True, "magnet_force_N": _round(B.pick("pd:mag", {3.0: 2, 5.0: 2, 0.0: 1}), 0.1)}
    s["operator"] = {"model": "none", "height": _round(dims[1] / 2 + 0.05), "sides": "both"}
    s["latch"] = {"model": "pet_flap_magnet" if s["kinematics"]["magnet_force_N"] > 0 else "none"}
    s["lock"] = {"model": B.pick("pd:lock", {"none": 3, "slide_bolt": 1})}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "brush_pile", B.pick("pd:cond", {"new": 2, "normal": 2, "worn": 1}), []
    choose_lock_engagement(s, B, rng, 0.3)
    s["tags"] = ["pet_door", size, slab]
    return s


def gen_hatch(i, ctx, B, rng, ceiling=False):
    fam = "hatch_ceiling" if ceiling else "hatch_floor"
    s = base_spec(i, fam, "utility", rng)
    if ceiling:
        slab = B.pick("hc:slab", {"attic_hatch": 3, "hollow_metal_18ga": 1, "steel_plate_security": 1})
        W, Hh = B.pick("hc:w", {0.56: 2, 0.76: 1, 0.9: 1}), B.pick("hc:h", {0.76: 2, 0.9: 1, 1.2: 1})
        use = B.pick("hc:use", ["attic access hatch", "roof scuttle hatch", "ceiling maintenance hatch"])
        op = B.pick("hc:op", {"none": 2, "pull_d": 1, "hatch_ring": 1})
        closer = B.pick("hc:closer", {"none": 2, "gas_strut": 2})
        max_open = B.pick("hc:mo", {80: 2, 90: 1, 100: 1})
    else:
        slab = B.pick("hf:slab", {"cellar_trapdoor": 2, "steel_plate_security": 2, "attic_hatch": 1, "aluminum": 0.001})
        if slab == "aluminum":
            slab = "steel_plate_security"
        W, Hh = B.pick("hf:w", {0.76: 2, 0.9: 2, 1.2: 1}), B.pick("hf:h", {0.76: 1, 0.9: 2, 1.2: 1, 1.5: 1})
        use = B.pick("hf:use", ["cellar trapdoor", "utility floor hatch", "ship deck hatch", "stage trapdoor", "storm shelter hatch"])
        op = B.pick("hf:op", {"hatch_ring": 3, "pull_d": 1, "pull_ring": 1})
        closer = B.pick("hf:closer", {"none": 3, "gas_strut": 2})
        max_open = B.pick("hf:mo", {90: 2, 100: 1, 110: 1, 75: 1})
    s["use_case"] = use
    s["leaf"] = {"width": W, "height": Hh, "thickness": M.SLABS[slab].typical_thickness[0], "slab": slab, "panel_style": B.pick(f"{fam}:panel", {"flush": 2, "plank_vertical": 1, "riveted_steel": 1}), "finish": finish_for(slab, "industrial_utility", B, rng), "count": 1, "glazing": None}
    s["opening"] = {"width": _round(W + 0.01), "height": _round(Hh + 0.01), "wall_thickness": 0.2, "frame": {"kind": "hatch_curb", "material": "steel_galvanized" if rng.random() < 0.5 else "pine", "casing": True, "stop_depth": 0.02, "jamb_depth": 0.2}, "threshold": "none", "sidelite": False, "transom": False, "horizontal": True, "elevation": 0.0 if not ceiling else 2.4}
    s["hinge"] = hinge_block("hatch_hinge", 2, "far", "up")
    s["kinematics"] = {"type": "hinge_horizontal", "max_open_deg": max_open, "stop": "prop_arm" if rng.random() < 0.4 else "none", "gravity_assisted_close": True, "ceiling": ceiling}
    s["operator"] = {"model": op, "height": 0.0, "sides": "robot"}
    s["lock"] = {"model": B.pick(f"{fam}:lock", {"none": 3, "padlock": 1, "slide_bolt": 1})}
    s["latch"] = {"model": "slide_bolt" if s["lock"]["model"] == "slide_bolt" else "none"}
    s["closer"] = {"model": closer, "en_size": None, "spring_adjust": 1.0, "gas_force_N": _round(_u(rng, 150, 400), 5)}
    s["seal"], s["condition"], s["extras"] = B.pick(f"{fam}:seal", {"none": 2, "gasket_rubber_heavy": 1}), B.pick(f"{fam}:cond", {"normal": 2, "worn": 2, "rusty": 1, "old_dry": 1}), []
    choose_lock_engagement(s, B, rng, 0.3)
    s["tags"] = [fam, slab]
    return s


def gen_ship(i, ctx, B, rng):
    s = base_spec(i, "ship_watertight", "marine", rng)
    swing = approach(s, B, rng)
    slab = B.pick("sw:slab", {"ship_watertight": 3, "submarine_hatch": 1})
    W, Hh = B.pick("sw:w", {0.65: 2, 0.7: 2, 0.8: 1}), B.pick("sw:h", {1.5: 1, 1.7: 2, 1.9: 1})
    op = B.pick("sw:op", {"dog_lever": 3, "wheel_ship_hatch": 2})
    n_dogs = B.pick("sw:dogs", {4: 1, 6: 2, 8: 1})
    s["use_case"] = B.pick("sw:use", ["ship bulkhead watertight door", "engine room WT door", "submarine bulkhead hatch", "offshore platform weathertight door"])
    s["leaf"] = {"width": W, "height": Hh, "thickness": M.SLABS[slab].typical_thickness[0], "slab": slab, "panel_style": B.pick("sw:panel", {"riveted_steel": 1, "porthole": 1, "dished_plate": 1, "steel_flush": 1}), "finish": finish_for(slab, "industrial_utility", B, rng), "count": 1, "glazing": {"material": "glass_laminated_security", "thickness": 0.012, "area_fraction": 0.03, "panel_style": "porthole", "count": 1} if rng.random() < 0.4 else None}
    s["opening"] = {"width": _round(W + 0.01), "height": _round(Hh + 0.01), "wall_thickness": 0.012, "frame": {"kind": "ship_coaming", "material": "steel_painted", "casing": True, "stop_depth": 0.03, "jamb_depth": 0.15}, "threshold": "coaming", "sidelite": False, "transom": False, "sill_height": _round(B.pick("sw:sill", {0.15: 1, 0.30: 2, 0.45: 1}))}
    s["hinge"] = hinge_block("ship_hinge", 2, B.pick("sw:side", ["left", "right"]), swing)
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": B.pick("sw:mo", {110: 1, 150: 1, 180: 1}), "stop": "hook_holdback", "dogs": n_dogs, "wheel_dogging": op == "wheel_ship_hatch"}   # quick-acting doors carry the same dogs; the wheel drives all of them at once
    s["operator"] = {"model": op, "height": _round(Hh * 0.5), "sides": "both"}
    s["latch"] = {"model": "dogs_6"}
    s["lock"] = {"model": "dogs"}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "watertight_rubber", B.pick("sw:cond", {"normal": 2, "worn": 1, "rusty": 2, "well_oiled": 1}), ["warning_placard"]
    choose_lock_engagement(s, B, rng, 1.0)
    s["lock"]["robot_side_release"] = True
    s["tags"] = ["watertight", slab, op, f"dogs{n_dogs}"]
    return s


def gen_vault(i, ctx, B, rng, blast=False):
    fam = "blast" if blast else "vault"
    s = base_spec(i, fam, "security", rng)
    swing = approach(s, B, rng)
    slab = "blast_steel" if blast else "vault_composite"
    t = B.pick(f"{fam}:t", {0.08: 1, 0.12: 2} if blast else {0.10: 2, 0.15: 2, 0.25: 1})
    W, Hh = B.pick(f"{fam}:w", {0.9: 2, 1.0: 2, 1.2: 1}), B.pick(f"{fam}:h", {2.0: 2, 2.1: 2, 1.9: 1})
    op = B.pick(f"{fam}:op", {"wheel_vault": 3, "dog_lever": 1, "lever_straight": 1}) if not blast else B.pick("bl:op", {"dog_lever": 2, "wheel_vault": 1, "lever_straight": 1})
    s["use_case"] = B.pick(f"{fam}:use", ["bank vault door", "safe room door", "data center vault door", "gun vault door"] if not blast else ["bunker blast door", "shelter blast door", "test cell blast door"])
    s["leaf"] = {"width": W, "height": Hh, "thickness": t, "slab": slab, "panel_style": B.pick(f"{fam}:panel", {"steel_flush": 2, "riveted_steel": 1}), "finish": finish_for(slab, "security_detention", B, rng), "count": 1, "glazing": None}
    s["opening"] = {"width": _round(W + 0.01), "height": _round(Hh + 0.07), "wall_thickness": _round(t + 0.2), "frame": {"kind": "vault_frame", "material": "steel", "casing": False, "stop_depth": 0.02, "jamb_depth": t + 0.2}, "threshold": "sill_step", "sidelite": False, "transom": False}
    s["hinge"] = hinge_block("vault_hinge", B.pick(f"{fam}:hn", {2: 2, 3: 1}), B.pick(f"{fam}:side", ["left", "right"]), swing)
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": B.pick(f"{fam}:mo", {100: 2, 120: 1, 170: 1}), "stop": "none", "bolts": B.pick(f"{fam}:bolts", {4: 2, 8: 1}) if op == "wheel_vault" else 0}
    s["operator"] = {"model": op, "height": _round(Hh * 0.5), "sides": "both" if op != "wheel_vault" else "robot"}
    s["latch"] = {"model": "multi_bolt_8" if s["kinematics"]["bolts"] == 8 else ("multi_bolt_4" if op == "wheel_vault" else "dogs_6")}
    s["lock"] = {"model": "vault_wheel" if op == "wheel_vault" else "dogs"}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = B.pick(f"{fam}:seal", {"gasket_rubber_heavy": 2, "none": 1}), B.pick(f"{fam}:cond", {"new": 1, "normal": 2, "well_oiled": 1, "old_dry": 1}), ["warning_placard"]
    choose_lock_engagement(s, B, rng, 1.0)
    s["lock"]["robot_side_release"] = True
    s["tags"] = [fam, slab, op]
    return s


def gen_gate_swing(i, ctx, B, rng):
    s = base_spec(i, "gate_swing", ctx, rng)
    swing = approach(s, B, rng)
    if ctx == "garden_picket":
        slab, panel = "cedar_plank", B.pick("gp:panel", {"pickets": 3, "board_batten": 1, "arched_top": 1, "plank_x_brace": 1})
        W, Hh = B.pick("gp:w", {0.9: 2, 1.0: 2, 1.2: 1}), B.pick("gp:h", {0.9: 1, 1.0: 2, 1.2: 1, 1.5: 1})
        op = B.pick("gp:op", {"thumb_latch_suffolk": 2, "gate_latch_fork": 1, "pull_ring": 1, "slide_bolt_barrel": 1})
        latch = {"thumb_latch_suffolk": "gravity_bar", "gate_latch_fork": "fork_gravity", "pull_ring": "none", "slide_bolt_barrel": "slide_bolt"}[op]
        hinge = B.pick("gp:hinge", {"garden_tee": 2, "strap_pintle": 1, "butt_rusty": 1})
        closer = B.pick("gp:closer", {"none": 2, "gate_spring": 2})
        lock = B.pick("gp:lock", {"none": 3, "padlock": 1, "slide_bolt": 1})
        use = B.pick("gp:use", ["garden picket gate", "cottage front gate", "side yard gate", "allotment gate"])
        fm = "cedar"
    elif ctx == "chain_link":
        slab, panel = "chain_link_gate", "mesh_panel"
        W, Hh = B.pick("cl:w", {0.9: 1, 1.0: 2, 1.2: 2}), B.pick("cl:h", {1.2: 2, 1.5: 1, 1.8: 2})
        op = B.pick("cl:op", {"gate_latch_fork": 3, "hasp_padlock": 1, "slide_bolt_heavy": 1})
        latch = {"gate_latch_fork": "fork_gravity", "hasp_padlock": "none", "slide_bolt_heavy": "slide_bolt_heavy"}[op]
        hinge = "chain_link_hinge"
        closer = B.pick("cl:closer", {"none": 2, "gate_spring": 1})
        lock = "padlock" if op == "hasp_padlock" else B.pick("cl:lock", {"none": 2, "padlock": 1})
        use = B.pick("cl:use", ["schoolyard chain-link gate", "tennis court gate", "utility yard gate", "dog run gate"])
        fm = "steel_galvanized"
    elif ctx == "wrought_iron":
        slab, panel = "wrought_iron_gate", "ornamental_scroll"
        W, Hh = B.pick("wi:w", {0.9: 1, 1.0: 2, 1.2: 1, 1.5: 1}), B.pick("wi:h", {1.5: 1, 1.8: 2, 2.1: 1, 2.4: 1})
        op = B.pick("wi:op", {"pull_ring": 1, "lever_euro_backplate": 2, "gate_latch_fork": 1, "knob_round": 1})
        latch = {"pull_ring": "none", "lever_euro_backplate": "mortise_euro", "gate_latch_fork": "fork_gravity", "knob_round": "mortise_euro"}[op]
        hinge = B.pick("wi:hinge", {"strap_pintle": 2, "strap_heavy": 1})
        closer = B.pick("wi:closer", {"none": 2, "gate_hydraulic": 1})
        lock = B.pick("wi:lock", {"none": 2, "deadbolt_double": 1, "electric_strike": 1, "padlock": 1})
        use = B.pick("wi:use", ["estate pedestrian gate", "courtyard iron gate", "cemetery gate", "apartment courtyard gate"])
        fm = "wrought_iron"
    elif ctx == "pool_safety":
        slab, panel = B.pick("ps:slab", {"baby_gate_steel": 0.001, "wrought_iron_gate": 1, "chain_link_gate": 1, "cedar_plank": 1}), "pickets"
        if slab == "baby_gate_steel":
            slab = "wrought_iron_gate"
        W, Hh = B.pick("ps:w", {0.9: 2, 1.0: 1}), B.pick("ps:h", {1.2: 2, 1.5: 1})
        op = "gate_latch_magnetic"
        latch = "magnalatch"
        hinge = B.pick("ps:hinge", {"strap_pintle": 1, "chain_link_hinge": 1})
        closer = "gate_hydraulic"
        lock = "none"
        use = "pool safety gate (self-closing, self-latching)"
        fm = "aluminum"
    else:  # ranch_tube
        slab, panel = "tube_gate", "bar_grille"
        W, Hh = B.pick("rt:w", {1.2: 1, 2.4: 1, 3.6: 2, 4.8: 1}), B.pick("rt:h", {1.2: 2, 1.5: 1})
        op = B.pick("rt:op", {"slide_bolt_heavy": 2, "hasp_padlock": 1, "gate_latch_fork": 1})
        latch = {"slide_bolt_heavy": "slide_bolt_heavy", "hasp_padlock": "none", "gate_latch_fork": "fork_gravity"}[op]
        hinge = B.pick("rt:hinge", {"strap_pintle": 2, "chain_link_hinge": 1})
        closer = "none"
        lock = "padlock" if op == "hasp_padlock" else B.pick("rt:lock", {"none": 2, "padlock": 1})
        use = B.pick("rt:use", ["ranch tube gate", "farm field gate", "cattle pen gate"])
        fm = "steel_galvanized"
    s["use_case"] = use
    s["leaf"] = {"width": W, "height": Hh, "thickness": M.SLABS[slab].typical_thickness[0], "slab": slab, "panel_style": panel, "finish": finish_for(slab, "default", B, rng), "count": 1, "glazing": None}
    s["opening"] = {"width": _round(W + 0.012), "height": _round(Hh + 0.02), "wall_thickness": 0.1, "frame": {"kind": "gate_posts", "material": fm, "casing": False, "stop_depth": 0.0, "jamb_depth": 0.1, "post_size": B.pick("gate:post", {0.1: 2, 0.15: 1, 0.06: 1})}, "threshold": "none", "sidelite": False, "transom": False, "outdoor": True, "ground_clearance": _round(B.pick("gate:gc", {0.05: 2, 0.1: 2, 0.15: 1}))}
    s["hinge"] = hinge_block(hinge, 2, B.pick("gate:side", ["left", "right"]), swing)
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": B.pick("gate:mo", {90: 2, 110: 1, 180: 2}), "stop": B.pick("gate:stop", {"none": 3, "wall_bumper": 1})}
    s["operator"] = {"model": op, "height": B.pick("gate:hh", {0.9: 2, 1.0: 2, 1.1: 1}) if op != "gate_latch_magnetic" else 1.5, "sides": "both"}
    s["latch"] = {"model": latch}
    s["lock"] = {"model": lock}
    s["closer"] = {"model": closer, "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("gate:cond", {"normal": 2, "worn": 2, "rusty": 2, "sagging": 1, "new": 1}), []
    choose_lock_engagement(s, B, rng, 0.35)
    s["tags"] = ["gate", ctx, slab, op]
    return s


def gen_gate_sliding(i, ctx, B, rng):
    s = base_spec(i, "gate_sliding", "outdoor", rng)
    s["robot"]["is_push"] = False
    slab = B.pick("gsl:slab", {"chain_link_gate": 2, "wrought_iron_gate": 1, "steel_bar_grille": 1, "expanded_metal_gate": 1})
    W, Hh = B.pick("gsl:w", {1.2: 2, 3.6: 2, 4.8: 1}), B.pick("gsl:h", {1.8: 2, 2.1: 1, 1.5: 1})
    s["use_case"] = B.pick("gsl:use", ["cantilever driveway gate (manual)", "pedestrian sliding gate", "warehouse yard sliding gate"])
    s["leaf"] = {"width": W, "height": Hh, "thickness": M.SLABS[slab].typical_thickness[0], "slab": slab, "panel_style": "mesh_panel" if slab != "wrought_iron_gate" else "pickets", "finish": finish_for(slab, "default", B, rng), "count": 1, "glazing": None}
    s["opening"] = {"width": _round(W + 0.05), "height": _round(Hh + 0.05), "wall_thickness": 0.1, "frame": {"kind": "gate_posts", "material": "steel_galvanized", "casing": False, "stop_depth": 0, "jamb_depth": 0.1, "post_size": 0.1}, "threshold": "none", "sidelite": False, "transom": False, "outdoor": True, "ground_clearance": 0.08}
    s["hinge"] = hinge_block("none", 0, "left", "slide")
    s["kinematics"] = {"type": "slide_horizontal", "travel_m": _round(W), "roller": B.pick("gsl:roller", {"cantilever_gate": 2, "bottom_rolling_heavy": 1, "dirty_track": 1}), "track": "cantilever" if W > 2 else "bottom_rail", "stop": "track_end"}
    op = B.pick("gsl:op", {"pull_d": 2, "slide_bolt_heavy": 1, "hasp_padlock": 1})
    s["operator"] = {"model": op, "height": 1.0, "sides": "both"}
    s["latch"] = {"model": "slide_bolt_heavy" if op == "slide_bolt_heavy" else "none"}
    s["lock"] = {"model": "padlock" if op == "hasp_padlock" else B.pick("gsl:lock", {"none": 2, "electric_bolt": 1})}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("gsl:cond", {"normal": 2, "worn": 1, "rusty": 2}), []
    choose_lock_engagement(s, B, rng, 0.3)
    s["tags"] = ["gate", "sliding", slab]
    return s


def gen_baby_gate(i, ctx, B, rng):
    s = base_spec(i, "baby_gate", "residential", rng)
    swing = approach(s, B, rng)
    W, Hh = B.pick("bg:w", {0.75: 2, 0.9: 2, 1.1: 1}), B.pick("bg:h", {0.75: 3, 0.9: 1, 1.0: 1})
    slab = B.pick("bg:slab", {"baby_gate_steel": 3, "cedar_plank": 1, "mdf_solid": 1})
    s["use_case"] = B.pick("bg:use", ["stair-top baby gate", "kitchen doorway baby gate", "pet gate in hallway"])
    s["leaf"] = {"width": W, "height": Hh, "thickness": 0.02, "slab": slab, "panel_style": "pickets" if slab != "mdf_solid" else "flush", "finish": finish_for(slab, "residential_interior", B, rng), "count": 1, "glazing": None}
    s["opening"] = {"width": _round(W + 0.012), "height": _round(Hh + 0.02), "wall_thickness": 0.115, "frame": {"kind": "pressure_frame", "material": "steel_painted", "casing": False, "stop_depth": 0.0, "jamb_depth": 0.05}, "threshold": "trip_bar" if rng.random() < 0.6 else "none", "sidelite": False, "transom": False, "ground_clearance": 0.03}
    s["hinge"] = hinge_block("baby_gate", 2, B.pick("bg:side", ["left", "right"]), swing)
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": B.pick("bg:mo", {90: 2, 100: 1, 180: 1}), "stop": "none", "both_ways": rng.random() < 0.5, "auto_close": rng.random() < 0.5}
    s["operator"] = {"model": "baby_gate_latch", "height": _round(Hh - 0.02), "sides": "both"}
    s["latch"] = {"model": "hook_slider"}
    s["lock"] = {"model": "none"}
    s["closer"] = {"model": "gate_spring" if s["kinematics"]["auto_close"] else "none", "en_size": None, "spring_adjust": 0.6}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("bg:cond", {"new": 2, "normal": 2, "worn": 1}), []
    choose_lock_engagement(s, B, rng, 0.0)
    s["tags"] = ["baby_gate", slab]
    return s


def gen_stall(i, ctx, B, rng):
    s = base_spec(i, "stall", "restroom", rng)
    swing = approach(s, B, rng)
    slab = B.pick("st:slab", {"hpl_partition": 2, "phenolic_partition": 2, "stainless_hollow": 1})
    W, Hh = B.pick("st:w", {0.61: 3, 0.66: 1, 0.86: 2}), B.pick("st:h", {1.47: 3, 1.83: 1, 2.0: 1})
    s["use_case"] = B.pick("st:use", ["public restroom stall door", "ADA accessible stall door (outswing)", "locker room changing stall", "school restroom stall"])
    s["leaf"] = {"width": W, "height": Hh, "thickness": M.SLABS[slab].typical_thickness[0], "slab": slab, "panel_style": "hpl_flat", "finish": finish_for(slab, "commercial_office", B, rng), "count": 1, "glazing": None, "bottom_clearance": _round(B.pick("st:bc", {0.30: 3, 0.15: 1, 0.05: 1}))}
    s["opening"] = {"width": _round(W + 0.012), "height": 2.1, "wall_thickness": 0.025, "frame": {"kind": "partition_pilasters", "material": "stainless" if slab == "stainless_hollow" else "hpl", "casing": False, "stop_depth": 0.0, "jamb_depth": 0.025}, "threshold": "none", "sidelite": False, "transom": False}
    s["hinge"] = hinge_block("gravity_pivot", 2, B.pick("st:side", ["left", "right"]), swing, 2.5)
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": B.pick("st:mo", {110: 2, 90: 1, 170: 1}), "stop": "none", "self_closing": True, "rest_angle_deg": B.pick("st:rest", {0: 1, 15: 2, 30: 1})}
    s["operator"] = {"model": B.pick("st:op", {"stall_slide_latch": 3, "pull_d": 1, "coat_hook_pull": 0.001}), "height": 1.0, "sides": "both"}
    if s["operator"]["model"] == "coat_hook_pull":
        s["operator"]["model"] = "pull_d"
    s["latch"] = {"model": "stall_slide"}
    s["lock"] = {"model": B.pick("st:lock", {"none": 2, "slide_bolt": 1})}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("st:cond", {"new": 1, "normal": 2, "worn": 2, "damaged": 1}), (["coat_hook"] if rng.random() < 0.6 else [])
    choose_lock_engagement(s, B, rng, 0.4)
    s["tags"] = ["stall", slab]
    return s


def gen_strip_curtain(i, ctx, B, rng):
    s = base_spec(i, "strip_curtain", "industrial", rng)
    s["robot"]["is_push"] = True
    W, Hh = B.pick("sc:w", {0.9: 2, 1.2: 2, 1.8: 1}), B.pick("sc:h", {2.1: 2, 2.4: 1, 3.0: 1})
    sw = B.pick("sc:sw", {0.2: 2, 0.3: 1, 0.4: 1})
    st = B.pick("sc:st", {0.002: 2, 0.003: 1, 0.004: 1})
    s["use_case"] = B.pick("sc:use", ["walk-in cooler strip curtain", "warehouse dock strip door", "food processing strip curtain", "loading bay PVC strips"])
    s["leaf"] = {"width": sw, "height": _round(Hh - 0.02), "thickness": st, "slab": "strip_curtain", "panel_style": "strips", "finish": finish_for("strip_curtain", "default", B, rng), "count": int(math.ceil(W / (sw * (1 - B.pick("sc:ov", {0.5: 2, 0.33: 1, 0.66: 1}))))), "glazing": None, "strip_width": sw, "overlap": 0.5}
    s["opening"] = {"width": _round(W + 0.02), "height": _round(Hh + 0.02), "wall_thickness": 0.2, "frame": {"kind": "strip_hanger_rail", "material": "stainless", "casing": False, "stop_depth": 0, "jamb_depth": 0.2}, "threshold": "none", "sidelite": False, "transom": False}
    s["hinge"] = hinge_block("piano", 1, "top", "both")
    # 85 deg, not 120: a strip is hinged on its own top edge just under the head, so past 90 deg its far end rises
    # ABOVE the opening and into the wall.  The model's strip hinges are built from this number, and the benchmark's
    # pass threshold is read from it - a curtain declaring 120 deg while its joints stopped at 71.6 deg was asking
    # for an opening its own strips could not make.
    s["kinematics"] = {"type": "hinge_horizontal", "max_open_deg": 85, "stop": "none", "both_ways": True, "strips": True}
    s["operator"] = {"model": "none", "height": 1.0, "sides": "both"}
    s["latch"] = {"model": "none"}
    s["lock"] = {"model": "none"}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("sc:cond", {"new": 1, "normal": 2, "worn": 1, "damaged": 1}), []
    choose_lock_engagement(s, B, rng, 0.0)
    s["tags"] = ["strip_curtain", f"strip{sw}"]
    return s


def gen_cold_storage(i, ctx, B, rng):
    s = base_spec(i, "cold_storage", "food_service", rng)
    swing = approach(s, B, rng)
    slab = B.pick("cs:slab", {"cold_storage_100": 3, "freezer_150": 2})
    W, Hh = B.pick("cs:w", {0.86: 2, 0.91: 2, 1.07: 1, 1.22: 1}), B.pick("cs:h", {1.98: 2, 2.13: 2, 2.44: 1})
    s["use_case"] = B.pick("cs:use", ["walk-in cooler door", "walk-in freezer door", "restaurant cold room door", "lab cold storage door", "florist cooler door"])
    s["leaf"] = {"width": W, "height": Hh, "thickness": M.SLABS[slab].typical_thickness[0], "slab": slab, "panel_style": B.pick("cs:panel", {"steel_flush": 3, "glass_vision": 1}), "finish": finish_for(slab, "default", B, rng), "count": 1, "glazing": glazing_for("glass_vision", W, Hh, "glass_clear", 0.025, rng) if rng.random() < 0.3 else None}
    s["opening"] = {"width": _round(W + 0.02), "height": _round(Hh + 0.02), "wall_thickness": M.SLABS[slab].typical_thickness[0], "frame": {"kind": "cold_room_frame", "material": "stainless", "casing": True, "stop_depth": 0.02, "jamb_depth": 0.12}, "threshold": B.pick("cs:thr", {"none": 2, "saddle": 1, "ada_ramp": 1}), "sidelite": False, "transom": False}
    s["hinge"] = hinge_block("cam_lift", B.pick("cs:hn", {2: 3, 3: 1}), B.pick("cs:side", ["left", "right"]), swing, 4.0)
    s["kinematics"] = {"type": "hinge_vertical", "max_open_deg": B.pick("cs:mo", {90: 1, 110: 2, 150: 1}), "stop": B.pick("cs:stop", {"none": 2, "wall_bumper": 1}), "self_closing": True, "gasket_compression_m": 0.012}
    s["operator"] = {"model": "cold_storage_handle", "height": 1.0, "sides": "both"}
    s["latch"] = {"model": B.pick("cs:latch", {"magnetic_gasket": 2, "roller_latch": 1})}
    s["lock"] = {"model": B.pick("cs:lock", {"none": 3, "padlock": 1})}
    s["closer"] = {"model": B.pick("cs:closer", {"none": 2, "lcn_4040": 1}), "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = B.pick("cs:seal", {"magnetic_gasket": 2, "gasket_rubber_heavy": 1}), B.pick("cs:cond", {"new": 1, "normal": 2, "worn": 1, "damaged": 1}), ["warning_placard", "kick_plate"]
    choose_lock_engagement(s, B, rng, 0.3)
    s["tags"] = ["cold_storage", slab]
    return s


def gen_auto_sliding(i, ctx, B, rng):
    s = base_spec(i, "automatic_sliding", "commercial_entry", rng)
    s["robot"]["is_push"] = False
    bi = B.pick("as:bi", {True: 3, False: 2})
    W, Hh = B.pick("as:w", {0.9: 2, 1.0: 2, 1.2: 1}), B.pick("as:h", {2.134: 3, 2.4: 1})
    slab = B.pick("as:slab", {"storefront_alu": 3, "storefront_alu_igu": 1, "glass_frameless_12": 1})
    s["use_case"] = B.pick("as:use", ["supermarket automatic door", "hospital entrance automatic door", "airport automatic door", "office lobby automatic slider", "pharmacy entrance"])
    s["leaf"] = {"width": W, "height": Hh, "thickness": 0.045 if not slab.startswith("glass") else 0.012, "slab": slab, "panel_style": "glass_full", "finish": finish_for(slab, "default", B, rng), "count": 2 if bi else 1, "glazing": {"material": "glass_clear", "thickness": 0.006, "area_fraction": 0.8, "panel_style": "glass_full", "count": 1}}
    s["opening"] = {"width": _round((2 * W if bi else W) * 2 + 0.04), "height": _round(Hh + 0.25), "wall_thickness": 0.2, "frame": {"kind": "auto_slider_header", "material": B.pick("as:fm", {"aluminum": 3, "aluminum_dark": 1, "stainless": 1}), "casing": False, "stop_depth": 0, "jamb_depth": 0.2}, "threshold": "saddle", "sidelite": True, "transom": False, "fixed_panel": True}
    s["hinge"] = hinge_block("none", 0, "center" if bi else "left", "slide")
    s["kinematics"] = {"type": "slide_horizontal", "travel_m": _round(W - 0.02), "roller": B.pick("as:roller", {"ball_bearing_nylon": 3, "plain_steel_worn": 1}), "track": "auto_header", "stop": "track_end", "bi_parting": bi,
                       "actuator": {"kind": B.pick("as:act", {"belt_drive": 3, "direct_drive": 1}), "max_force_N": 150, "open_speed_m_s": _round(_u(rng, 0.3, 0.6), 0.05), "close_speed_m_s": 0.25, "sensor": B.pick("as:sensor", {"microwave_motion": 2, "infrared_presence": 1, "wave_to_open": 1, "push_button": 1}), "sensor_range_m": _round(_u(rng, 1.0, 2.0), 0.1), "hold_open_s": _round(_u(rng, 1.5, 4.0), 0.5), "powered": B.pick("as:pow", {True: 4, False: 1})},
                       "breakout": True, "breakout_force_N": 220}
    s["operator"] = {"model": "none", "height": 1.0, "sides": "both"}
    s["latch"] = {"model": "none"}
    s["lock"] = {"model": B.pick("as:lock", {"none": 3, "electric_bolt": 1})}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "brush_pile", B.pick("as:cond", {"new": 2, "normal": 2, "worn": 1}), ["wave_sensor", "push_pull_sign"]
    choose_lock_engagement(s, B, rng, 0.2)
    s["tags"] = ["automatic", "sliding", "bi_parting" if bi else "single_slide", s["kinematics"]["actuator"]["sensor"]]
    return s


def gen_auto_swing(i, ctx, B, rng):
    s = gen_swing_single(i, B.pick("aw:ctx", {"commercial_office": 2, "institutional": 2, "storefront_glass": 1}), B, rng)
    s["family"] = "automatic_swing"
    s["id"] = f"db{i:04d}_automatic_swing"
    s["closer"] = {"model": B.pick("aw:op", {"auto_low_energy": 3, "auto_full_energy": 2}), "en_size": None, "spring_adjust": 1.1}
    s["kinematics"]["actuator"] = {"kind": "swing_operator", "max_torque_Nm": 60, "open_time_s": _round(_u(rng, 3, 5), 0.5), "hold_open_s": _round(_u(rng, 3, 6), 0.5), "sensor": B.pick("aw:sensor", {"push_button_wall": 2, "wave_to_open": 2, "motion": 1}), "powered": B.pick("aw:pow", {True: 3, False: 1}), "push_and_go": rng.random() < 0.5}
    s["extras"] = sorted(set(s["extras"] + ["wave_sensor"]))
    s["use_case"] = "automatic " + s["use_case"]
    s["tags"] = ["automatic", "swing"] + s["tags"]
    return s


def gen_elevator(i, ctx, B, rng):
    s = base_spec(i, "elevator", "vertical_transport", rng)
    s["robot"]["is_push"] = False
    center = B.pick("el:center", {True: 3, False: 2})
    W = B.pick("el:w", {0.9: 2, 1.07: 2, 1.2: 1})
    Hh = B.pick("el:h", {2.1: 3, 2.4: 1})
    s["use_case"] = B.pick("el:use", ["office elevator landing doors", "hospital elevator doors", "residential tower elevator", "freight elevator doors"])
    s["leaf"] = {"width": _round(W / 2 if center else W), "height": Hh, "thickness": 0.03, "slab": "elevator_landing", "panel_style": "steel_flush", "finish": finish_for("elevator_landing", "default", B, rng), "count": 2 if center else 1, "glazing": None}
    s["opening"] = {"width": _round(W + 0.02), "height": _round(Hh + 0.06), "wall_thickness": 0.25, "frame": {"kind": "elevator_entrance", "material": B.pick("el:fm", {"stainless": 3, "steel_painted": 1, "bronze": 1}), "casing": True, "stop_depth": 0, "jamb_depth": 0.25}, "threshold": "sill", "sidelite": False, "transom": False}
    s["hinge"] = hinge_block("none", 0, "center" if center else "left", "slide")
    s["kinematics"] = {"type": "slide_horizontal", "travel_m": _round((W / 2 if center else W) - 0.01), "roller": "elevator_hanger", "track": "elevator_hanger_track", "stop": "track_end", "center_opening": center, "interlocked": True,
                       "actuator": {"kind": "door_operator", "max_force_N": 135, "open_speed_m_s": 0.4, "close_speed_m_s": 0.3, "powered": True, "reopen_on_obstruction": True, "hold_open_s": _round(_u(rng, 3, 6), 0.5)}}
    s["operator"] = {"model": "elevator_none", "height": 1.0, "sides": "both"}
    s["latch"] = {"model": "elevator_interlock"}
    s["lock"] = {"model": "interlock"}
    s["closer"] = {"model": "none", "en_size": None, "spring_adjust": 1.0}
    s["seal"], s["condition"], s["extras"] = "none", B.pick("el:cond", {"new": 2, "normal": 2, "worn": 1}), ["call_button"]
    choose_lock_engagement(s, B, rng, 1.0)
    s["lock"]["robot_side_release"] = True  # call button
    s["tags"] = ["elevator", "center_opening" if center else "side_opening"]
    return s


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def generate_all(seed: int = 20260903) -> list[dict]:
    rng = random.Random(seed)
    B = Balanced(random.Random(seed + 1))
    specs = []
    i = 1

    def sub_ctx_list(d):
        out = []
        for k, n in d.items():
            out += [k] * n
        rng.shuffle(out)
        return out

    plan: list[tuple[str, str]] = []
    for fam, (quota, _, _) in T.FAMILIES.items():
        if fam == "swing_single":
            plan += [(fam, c) for c in sub_ctx_list(T.SWING_SINGLE_CONTEXTS)]
        elif fam == "swing_double":
            plan += [(fam, c) for c in sub_ctx_list(T.SWING_DOUBLE_CONTEXTS)]
        elif fam == "sliding_single":
            plan += [(fam, c) for c in sub_ctx_list(T.SLIDING_SINGLE_CONTEXTS)]
        elif fam == "gate_swing":
            plan += [(fam, c) for c in sub_ctx_list(T.GATE_SWING_CONTEXTS)]
        else:
            plan += [(fam, "")] * quota
    assert len(plan) == 1000
    # interleave families so ids are mixed (nice for browsing) but deterministic
    rng.shuffle(plan)
    for fam, ctx in plan:
        gen = {
            "swing_single": lambda: gen_swing_single(i, ctx, B, rng),
            "swing_double": lambda: gen_swing_double(i, ctx, B, rng),
            "dutch": lambda: gen_dutch(i, ctx, B, rng),
            "saloon": lambda: gen_saloon(i, ctx, B, rng),
            "pivot": lambda: gen_pivot(i, ctx, B, rng),
            "sliding_single": lambda: gen_sliding_single(i, ctx, B, rng),
            "sliding_bypass": lambda: gen_sliding_bypass(i, ctx, B, rng),
            "bifold": lambda: gen_bifold(i, ctx, B, rng),
            "accordion": lambda: gen_accordion(i, ctx, B, rng),
            "revolving": lambda: gen_revolving(i, ctx, B, rng),
            "turnstile_tripod": lambda: gen_turnstile(i, ctx, B, rng, False),
            "turnstile_fullheight": lambda: gen_turnstile(i, ctx, B, rng, True),
            "garage_sectional": lambda: gen_garage(i, ctx, B, rng, "garage_sectional"),
            "garage_tiltup": lambda: gen_garage(i, ctx, B, rng, "garage_tiltup"),
            "rollup": lambda: gen_garage(i, ctx, B, rng, "rollup"),
            "pet_door": lambda: gen_pet_door(i, ctx, B, rng),
            "hatch_floor": lambda: gen_hatch(i, ctx, B, rng, False),
            "hatch_ceiling": lambda: gen_hatch(i, ctx, B, rng, True),
            "ship_watertight": lambda: gen_ship(i, ctx, B, rng),
            "vault": lambda: gen_vault(i, ctx, B, rng, False),
            "blast": lambda: gen_vault(i, ctx, B, rng, True),
            "gate_swing": lambda: gen_gate_swing(i, ctx, B, rng),
            "gate_sliding": lambda: gen_gate_sliding(i, ctx, B, rng),
            "baby_gate": lambda: gen_baby_gate(i, ctx, B, rng),
            "stall": lambda: gen_stall(i, ctx, B, rng),
            "strip_curtain": lambda: gen_strip_curtain(i, ctx, B, rng),
            "cold_storage": lambda: gen_cold_storage(i, ctx, B, rng),
            "automatic_sliding": lambda: gen_auto_sliding(i, ctx, B, rng),
            "automatic_swing": lambda: gen_auto_swing(i, ctx, B, rng),
            "elevator": lambda: gen_elevator(i, ctx, B, rng),
        }[fam]
        s = gen()
        s["id"] = f"db{i:04d}_{s['family']}"
        s["index"] = i
        # thick hinged leaves need extra latch-edge clearance: the non-swing corner sweeps out by ~(t+7mm)^2/(2W)
        if s["kinematics"]["type"] == "hinge_vertical" and s["leaf"].get("count", 1) == 1 and s["family"] not in ("stall", "baby_gate", "gate_swing", "saloon", "pivot"):
            W_, t_ = s["leaf"]["width"], s["leaf"]["thickness"]
            pin_off = 0.065 if s["family"] in ("vault", "blast") else 0.007
            inset = 0.006 if (s["family"] in ("vault", "blast") or H.HINGES[s["hinge"]["model"]].kind.startswith("pivot")) else 0.003
            gap_needed = (t_ + pin_off) ** 2 / (2 * W_) + 0.003
            s["opening"]["width"] = _round(max(s["opening"]["width"], W_ + inset + max(0.003, gap_needed)))
        if s["kinematics"].get("type") == "hinge_vertical" and s["family"] not in ("saloon", "stall", "pet_door", "hatch_floor", "hatch_ceiling"):
            mo = s["kinematics"].get("max_open_deg") or 90
            cap = 140
            if s["opening"]["frame"].get("casing"):
                cap = 135                       # casing trim stops the leaf well short of flat
            if H.HINGES[s["hinge"]["model"]].kind == "strap":
                cap = min(cap, 120)             # strap hinges hit the post
            if H.CLOSERS[s["closer"]["model"]].kind in ("surface_overhead",):
                cap = min(cap, 100)             # surface closer arms/body reach the wall
            if s["family"] in ("vault", "blast", "dutch"):
                cap = min(cap, 100)
            if s["leaf"]["panel_style"] in ("plank_z_brace", "plank_x_brace", "board_batten"):
                cap = min(cap, 80)              # face braces reach the jamb beyond this
            if s["family"] == "gate_swing":
                cap = min(cap, 110)             # 100 mm posts
            if s["family"] == "stall":
                cap = min(cap, 110)             # adjacent pilaster
            if s["family"] == "baby_gate":
                cap = min(cap, 90)
            if H.HINGES[s["hinge"]["model"]].kind == "strap":
                cap = min(cap, 100)
            if s["leaf"]["panel_style"] == "glass_frameless":
                cap = min(cap, 100)             # corner patch fittings reach the jamb beyond this
            s["kinematics"]["max_open_deg"] = min(mo, cap)
        if s["family"] == "pet_door":
            s["kinematics"]["max_open_deg"] = min(s["kinematics"].get("max_open_deg") or 75, 75)
        if H.LOCKS[s["lock"]["model"]].kind == "keypad_code" and "keypad" not in H.OPERATORS[s["operator"]["model"]].style_params:
            # a keypad lock needs a keypad on the door: swap the operator for the matching keypad set
            s["operator"]["model"] = "knob_keypad_deadbolt" if H.OPERATORS[s["operator"]["model"]].kind == "knob" else "lever_keypad"
            s.setdefault("tags", []).append("keypad_operator_required")
        if s["kinematics"].get("both_ways") and s["family"] in ("baby_gate",):
            s["opening"]["width"] = _round(max(s["opening"]["width"], s["leaf"]["width"] + s["leaf"]["thickness"] + 0.024))
        if s["kinematics"].get("double_egress"):
            s["opening"]["wall_thickness"] = 0.10      # double-egress pairs sit in a thin partition (each leaf swings its own way)
            s["opening"]["frame"]["jamb_depth"] = 0.10
        if s['leaf']['slab'].startswith('glass_frameless_'):
            s['leaf']['thickness'] = M.SLABS[s['leaf']['slab']].typical_thickness[0]
        if (s["family"] == "sliding_bypass" or s["kinematics"].get("track") == "top_hung_pocket") and H.OPERATORS[s["operator"]["model"]].kind in ("pull", "ring_pull", "knob", "none"):
            s["operator"]["model"] = "pull_flush_recessed"    # leaves that pass each other / enter a pocket need flush pulls
            s.setdefault("tags", []).append("flush_pull_required")
        if s["kinematics"].get("track") in ("top_hung_pocket", "top_hung_bypass"):
            # Clear suspension space for the roller channel, axle and hanger.
            s["opening"]["height"] = _round(max(s["opening"]["height"], s["leaf"]["height"] + 0.095))
        if s["kinematics"].get("track") == "top_hung_bypass":
            depth = max(.145, (s["kinematics"].get("n_leaves", 2)-1)*(s["leaf"]["thickness"]+.05)+.052)
            s["opening"]["wall_thickness"] = depth
            s["opening"]["frame"]["jamb_depth"] = depth
        if s["kinematics"].get("track") == "top_hung_pocket":
            s["kinematics"]["travel_m"] = (s["leaf"]["width"] + s["opening"]["width"]) / 2
            s["kinematics"]["edge_pull"] = "press_to_deploy"
            if s["leaf"]["slab"].startswith("glass_frameless"):
                s["leaf"]["thickness"] = M.SLABS[s["leaf"]["slab"]].typical_thickness[0]
        if s["family"] == "sliding_single" and s["operator"]["model"] in ("cold_storage_handle", "none"):
            # These leaves have independent boltwork, not a cold-room cam
            # latch driven by a hinged-door handle. Give the leaf a fixed grip.
            s["operator"]["model"] = "pull_d"
        if s['family'] == 'hatch_ceiling' and s['operator']['model'] == 'none':
            s['operator']['model'] = 'hatch_ring'
        if s['id']=='db0264_swing_single':
            # The selected keypad-entry function has independent free inside
            # lever egress; a fixed far pull cannot perform that operation.
            s['operator']['sides']='both'
            s['operator']['far_side']=None
        if s['family'] in ('vault', 'blast') and H.OPERATORS[s['operator']['model']].kind != 'wheel':
            # Preserve all random draws and credential/condition decisions.
            # These are two independent sliding bolts, not six marine dogs.
            s['operator']['model'] = 'vault_lever'
            s['latch']['model'] = 'vault_bolts_2'
            s['lock']['model'] = 'vault_lever_boltwork'
            s.setdefault('tags', []).append('independent_vault_lever_bolts')
        if s['family'] == 'garage_sectional':
            kin = s['kinematics']
            fraction = kin['counterbalance_fraction']
            kin['counterbalance_state'] = 'failed' if fraction == 0 else 'weak' if fraction <= .6 else 'under_tensioned' if fraction < .95 else 'balanced'
            if fraction == 0:
                s['condition'] = 'damaged'
                s.setdefault('tags', []).append('failed_counterbalance')
            elif fraction <= .6:
                s.setdefault('tags', []).append('weak_counterbalance')
            if kin.get('opener') != 'belt_drive_engaged' and s['operator']['model'] == 'none':
                s['operator']['model'] = 'pull_lift_garage'
        if s["family"] in ("gate_swing", "gate_sliding") and s["leaf"]["slab"] == "chain_link_gate":
            s["leaf"]["infill_thickness"] = s["leaf"]["thickness"]
            s["leaf"]["thickness"] = 0.041275  # 1-5/8 inch structural frame, not mesh-equivalent sheet thickness
        if s["operator"]["model"] == "gate_latch_magnetic":
            s["opening"]["width"] = s["leaf"]["width"] + 0.022
        if s["operator"]["model"] == "gate_latch_fork":
            s["opening"]["width"] = s["leaf"]["width"] + 0.054
            s["operator"]["height"] = min(s["operator"]["height"], s["opening"].get("ground_clearance", .05) + s["leaf"]["height"] - .20)
        if s["family"] == "gate_swing" and s["operator"]["model"] == "thumb_latch_suffolk":
            s["operator"]["height"] = min(s["operator"]["height"], s["opening"].get("ground_clearance", .05) + s["leaf"]["height"] - .15)
        if s["kinematics"].get("stop") == "wall_bumper":
            # This environment has no return wall behind the open leaf. A
            # floor-mounted post provides the bumper's actual structural load path.
            s["kinematics"]["stop"] = "floor_post"
        if H.LOCKS[s['lock']['model']].kind == 'multipoint':
            # Lift-operated boltwork needs a bidirectional lever spindle.
            # A round knob, thumb press or cremone knob cannot perform that lift.
            s['operator']['model']='lever_euro_backplate'
            s['operator']['sides']='both'
            s['operator'].pop('far_side',None)
            s['latch']['model']='tubular_residential_70'
            s.setdefault('tags',[]).append('multipoint_lift_lever')
        if H.LOCKS[s['lock']['model']].kind in ('chain', 'swing_bar_guard'):
            # Interior security guards bridge an inward-opening leaf and
            # frame. Preserve the seeded inside/outside access choice, then
            # put the swing on that same physical side (Ives 481/482 layout).
            if s['robot']['is_push'] != s['robot']['robot_outside']:
                s['robot']['is_push'] = s['robot']['robot_outside']
                s.setdefault('tags', []).append('security_guard_inward_swing')
        if H.OPERATORS[s["operator"]["model"]].kind in ("panic_touchbar", "panic_crossbar"):
            # The panic bar defines the interior/push face. Side permissions
            # must follow the actual swing, not an independent random bit.
            s["robot"]["robot_outside"] = not s["robot"]["is_push"]
        if "keypad_reader_wall" in s.get("extras", []) and H.LOCKS[s["lock"]["model"]].kind not in ("keypad_code", "card_reader", "mag_lock", "electric_strike"):
            s["extras"].remove("keypad_reader_wall")
        if s["family"] == "saloon":
            s["kinematics"]["max_open_deg"] = min(s["kinematics"].get("max_open_deg") or 90, 90)
        if s['family']=='dutch' and H.OPERATORS[s['operator']['model']].kind=='handleset':
            # The former Dutch-only handleset had a fixed exterior grip but
            # no thumb latch. Use an actual through-spindle knob set with
            # both operating faces and its real latch coupling.
            s['operator']['model']='knob_round';s['operator']['sides']='both'
            s['operator'].pop('far_side',None)
            s.setdefault('tags',[]).append('dutch_supported_spindle_set')
        if s["family"] == "stall":
            s["kinematics"]["max_open_deg"] = min(s["kinematics"].get("max_open_deg") or 110, 110)
        if s["family"] == "stall" and s["lock"].get("engaged"):
            s["kinematics"]["rest_angle_deg"] = 0     # occupied stall: latched shut (rest angle applies only when vacant)
            s["lock"]["robot_side_release"] = not s["robot"]["is_push"]   # slide latch sits on the swing-side face
            s["robot"]["robot_outside"] = bool(s["robot"]["is_push"])
        if s['family'] == 'hatch_ceiling' and s['lock']['model'] == 'slide_bolt':
            s['lock']['robot_side_release'] = False  # loft-side bolt is inaccessible from below
        if s['family'] == 'strip_curtain':
            width=s['leaf']['width'];opening=s['opening']['width']
            count=min(s['leaf']['count'], math.floor((opening-.02-width)/(width/2)+1e-9)+1)
            s['leaf']['count']=count
            pitch=(opening-.02-width)/max(1,count-1)
            s['leaf']['overlap']=1-pitch/width
            s['leaf']['overlap_definition']='neighbor_fraction'
        if s["lock"]["model"] == "electric_strike" and s["latch"]["model"] == "none":
            s["lock"]["model"] = "mag_lock"           # a strike needs a latch bolt; pull-only doors get a maglock instead
        if s["family"] == "swing_double" and (s["leaf"]["panel_style"] == "glass_frameless" or s["leaf"]["slab"].startswith(("glass", "storefront"))) and H.LOCKS[s["lock"]["model"]].kind in ("deadbolt_single", "deadbolt_double", "thumbturn_only", "mortise_deadbolt", "night_latch", "multipoint", "keypad_code"):
            s["lock"] = {"model": "none", "engaged": False, "robot_side_release": True}
            s.setdefault("tags", []).append("frameless_glass_patch_lock_omitted")
        if s["operator"]["model"] in ("slide_bolt_barrel", "slide_bolt_heavy", "cane_bolt_drop") and s["lock"]["model"] == "none":
            s["lock"] = {"model": "slide_bolt", "engaged": True, "robot_side_release": True}
        assign_task(s, B)
        if H.LOCKS[s['lock']['model']].kind=='jam_stuck':
            # Preserve the seeded catalogue draw, then describe this as the
            # authored breakaway-friction condition. It is not a security lock.
            s['lock'].update(engaged=False,robot_side_release=True)
            s['kinematics']['breakaway_friction_model']='hinge_coulomb_proxy'
            s['kinematics']['extra_stick_torque_multiplier']=2.
            s.setdefault('tags',[]).append('high_breakaway_friction')
            s['task']='open_and_traverse'
        if s['family'] in ('sliding_bypass', 'bifold', 'hatch_floor', 'hatch_ceiling') and not (s['lock']['engaged'] and not s['lock'].get('robot_side_release', True)):
            # These authored closet openings expose storage. They need not
            # provide a human passage between the stacked panels and handles.
            s['task'] = 'open_only'
        if s['family'] in ('garage_tiltup','garage_sectional','rollup'):
            # These authored tracks/curtains store into +Y. The fixed -Y
            # approach is outside; an interior slide lock or rear hasp has
            # no modeled front key release. Normalize after all RNG draws.
            s['robot']['robot_outside']=True
            if s['lock']['model'] in ('garage_slide_lock','keyed_cylinder','padlock'):
                s['lock']['robot_side_release']=False
                if s['lock']['engaged']:s['task']='locked_recognize'
        if s['family']=='rollup' and s['kinematics'].get('opener')=='chain_hoist':
            # The hand chain is installed inside the garage. Operating it
            # through the outside wall is not an approach-side interaction.
            s['robot']['approach_side']='+y'
            s['robot']['robot_outside']=False
        if H.LOCKS[s['lock']['model']].kind not in ('mag_lock','delayed_egress','card_reader','electric_strike'):
            s['extras']=[e for e in s['extras'] if e!='rex_button']
        if H.LOCKS[s['lock']['model']].kind not in ('mag_lock','card_reader','electric_strike'):
            s['extras']=[e for e in s['extras'] if e!='keypad_reader_wall']
        if s['family'] in ('turnstile_tripod','turnstile_fullheight'):
            # Normalize only after all source RNG draws and availability/task
            # decisions. Every rotor has an actual fail-secure index bolt,
            # including initially powered cases; no M62 maglock is installed.
            s['lock']['model']='turnstile_index_bolt'
        specs.append(s)
        i += 1
    assert len(specs) == 1000
    return specs
