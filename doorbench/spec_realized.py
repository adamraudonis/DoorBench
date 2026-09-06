"""Deterministic SPEC-REALIZED gate: the spec is a contract and the geometry must satisfy it.

Every door carries a spec that promises hardware by name - an operator on both faces, a latch with six
dogs, a closer, an opening stop of a named kind, three hinges, a louvre vent - and a model that is
supposed to contain that hardware.  Nothing checked that the two agreed.  The vision review
(docs/VISION_REVIEW.md) found 156 declared extras with no geometry on 153 doors, 35 doors whose named
hold-open stop was not modelled, 129 doors whose operator was declared on both faces and drawn on one,
and 12 whose latch model named more dogs than the builder made - and every one of those doors passed
every other gate, because no other gate ever reads the spec.

This module is that gate.  It walks the spec's declared hardware and asserts that the model contains
geometry with the matching semantic and name for each, in the right place, with the right multiplicity:

  operator_missing     ``spec["operator"]["model"]`` is a real operator and no geom carries the
                       ``operator`` semantic.  The parts may well be drawn - as ``latch`` or ``lock`` or
                       ``decor`` - but then the benchmark's grip sites, the viewer's handle camera and the
                       review's hardware close-up all miss them, and a robot asked to find the handle
                       finds nothing.
  operator_faces       ``spec["operator"]["sides"] == "both"`` and the operator geometry is on ONE face of
                       the leaf.  A robot approaching from the far side finds a blank slab.
  latch_missing        a declared latch with no ``latch``-semantic geometry.
  latch_multiplicity   the latch/lock model names N bolts or dogs (``dogs_6``, ``multi_bolt_8``) or the
                       kinematics declare ``dogs: N``, and the builder makes a different number.
  lock_missing         a declared lock with no ``lock``-semantic geometry.
  closer_missing       a declared closer with no ``closer``-semantic geometry.
  stop_missing         ``spec["kinematics"]["stop"]`` names a stop and the geometry that realizes it is
                       absent (a hatch standing 90 deg open on a ``prop_arm`` that was never drawn).
  stop_wrong_kind      the stop is drawn, but not as the kind the spec names: a floor riser under a
                       caption that says wall bumper.
  hinge_count          ``spec["hinge"]["count"]`` hinge stations are declared and a different number is
                       drawn.
  extra_missing        a declared entry of ``spec["extras"]`` with no geometry anywhere on the door.

Exceptions.  Anything deliberately not drawn is listed in one of the exception tables below - WITH ITS
REASON - and is counted and reported (``metrics["spec_realized_exceptions"]``) rather than silently
skipped.  A door may additionally carry ``model.meta["spec_realized_allow"]`` entries, each of which
must carry a written justification::

    ["<rule>", "<declared-item>", "reason"]

Enforced vs reported.  ``ENFORCED_RULES`` are the rules this gate signs off on: they are zero over all
1000 doors and ``checks["spec_realized"]`` fails the moment one of them comes back.  ``REPORTED_RULES``
are the same walk applied to declarations whose realization has not been built yet; they are counted
into ``metrics["spec_realized_open"]`` on every door, by rule and by item, so the size of the remaining
work is a number in the dataset rather than a thing nobody measured.  Moving a rule from REPORTED to
ENFORCED is the whole of the work it names.  Today's open counts, measured over all 1000 doors:

    lock_missing   127 doors  - a declared lock with no lock geometry: 22 privacy buttons, 21 maglocks
                                on sliding / rollup / turnstile leaves, 15 slide bolts, 11 padlocks,
                                11 electric strikes, 9 key cylinders, 8 hook locks, 8 child-lock covers,
                                6 electric bolts, 5 card readers, and 11 `jam_stuck` doors that
                                correctly have no lock hardware at all.

    latch_missing   56 doors  - 17 watertight/vault doors whose dogs and bolts carry the `lock` semantic
                                rather than `latch`, 14 magnetic catches, 8 elevator interlocks and 17
                                other bolts drawn under another semantic or not at all.

Both of those are the same class this gate exists for - a declaration with nothing behind it - and both
are the next tranche of the same work.

Every absence not covered by an exception is a defect and must be fixed in the generator.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re

import numpy as np

from .ir import quat_to_mat

# ---------------------------------------------------------------------------
# Contract tables
# ---------------------------------------------------------------------------

# Operator models that really are "no operator": there is nothing to draw and nothing to grip.
NO_OPERATOR_MODELS = {
    "none": "no operator at all (push face, free-swinging leaf, powered door with no manual trim)",
    "elevator_none": "an elevator landing door has no landing-side operator - it is opened by the car",
}

# Closer models whose closer IS another part of the door, so no `closer`-semantic geometry is expected.
CLOSER_IN_ANOTHER_PART = {
    "spring_hinge_single": "the closer is the hinge: a spring butt hinge, drawn as the hinge geometry",
    "spring_hinge_double": "double-acting spring hinges, drawn as the hinge geometry",
    "gate_spring": "a gate spring wound round the hinge pintle, drawn with the gate hinge",
    "gate_hydraulic": "a hydraulic gate closer built into the hinge post, drawn with the gate hinge",
}

FACE_EPS = 0.0005   # m off the leaf's mid-plane before a part counts as being ON one of its faces

# Face-sign rules do not apply where a 'face' is not a thing: a rotor wing, a curtain of strips, a flap.
NO_FACE_RULE_FAMILIES = {"revolving", "turnstile_tripod", "turnstile_fullheight", "strip_curtain"}

# What realizes each opening stop.  (semantic, name globs, mount) - `mount` is checked against
# meta["stops"] where the builder records what it actually built.
STOP_CONTRACT = {
    "wall_bumper":      {"globs": ("door_stop_bumper*",), "mount": "wall"},
    "floor_bumper":     {"globs": ("door_stop_bumper*",), "mount": "floor"},
    "floor_dome":       {"globs": ("floor_stop_dome*",),  "mount": None},
    "wall_180":         {"globs": ("door_stop_bumper*",), "mount": "wall"},
    "corridor_wall_120": {"globs": ("corridor_return_wall*",), "mount": None},
    "prop_arm":         {"globs": ("prop_arm*",),         "mount": None},
    "hook_holdback":    {"globs": ("holdback_hook*", "holdback_eye*"), "mount": None},
    "kick_down_holder": {"globs": ("kickdown_*",),        "mount": None},
}

# Stops that are deliberately not modelled as separate hardware, with the reason.  Each is reported.
STOP_EXCEPTIONS = {
    "none": "no stop is declared: the frame and hinge geometry limit the travel",
    "track_end": "a sliding leaf is stopped by the end of its own rail; the rail is modelled over the "
                 "full travel and the joint range ends where it ends.  Bolted end bumpers are drawn "
                 "only where a roller runs on the rail (barn hangers).",
    "hinge_pin": "a hinge-pin stop is a set-screw collar inside the hinge knuckle: it is not visible "
                 "hardware and is not drawn; the joint limit carries it",
    "overhead_90": "a concealed overhead stop lives inside the door's top rail and the frame soffit; "
                   "the track and slide arm are not modelled and the joint limit carries the travel",
    "overhead_105": "a concealed overhead stop at 105 deg: as overhead_90, inside the rail and the soffit",
    "overhead_110_hold": "a concealed overhead stop and holder: as overhead_90, inside the rail and the soffit",
    "wedge_jammed": "a wedge kicked under the leaf; modelled as the joint range, not as loose furniture",
}

# What realizes each extra.  Every entry of taxonomy.EXTRAS must appear here or in EXTRA_EXCEPTIONS.
EXTRA_CONTRACT = {
    "kick_plate":        ("*kick_plate*",),
    "armor_plate":       ("armor_plate*",),
    "push_plate":        ("*push_plate*", "*pushplate*"),
    "peephole":          ("peephole*",),
    "mail_slot":         ("mail_slot*",),
    "knocker":           ("knocker*",),
    "house_number":      ("house_numbers*",),
    "pet_flap":          ("*pet_flap*", "pet_frame*"),
    "chain_lock":        ("*chain*",),
    "swing_bar_guard":   ("*swing_bar*",),
    "exit_sign":         ("exit_sign*",),
    "push_pull_sign":    ("sign_push*", "sign_pull*"),
    "vision_lite_grille": ("vision_grille*", "*vision_lite*"),
    "door_stop_floor":   ("floor_stop_dome*", "door_stop_bumper*"),
    "door_stop_wall":    ("wall_stop_*",),
    "hold_open_kickdown": ("kickdown_*",),
    "wreath":            ("wreath*",),
    "keypad_reader_wall": ("wall_reader*",),
    "rex_button":        ("rex_*",),
    "wave_sensor":       ("wave_sensor*",),
    "call_button":       ("call_button*", "call_plate*"),
    "threshold_saddle":  ("threshold*",),
    "weather_drip_cap":  ("weather_drip_cap*",),
    "door_viewer_camera": ("peephole*", "door_viewer*"),
    "coat_hook":         ("coat_hook*",),
    "bumper_rail":       ("bumper_rail*",),
    "louver_vent":       ("louver_vent*",),
    "transom_window":    ("transom*",),
    "sidelite":          ("sidelite*",),
    "warning_placard":   ("warning_placard*",),
    "floor_guide":       ("*floor_guide*",),
    "soft_close_damper": ("*soft_close*",),
}


# Families whose leaves are not hung on a set of numbered hinge stations at all.
HINGE_NOT_STATIONED_FAMILIES = {
    "bifold": "a bi-fold panel pair hangs on a top pivot, a guide and the knuckles between the panels",
    "accordion": "an accordion's panels hang on a carrier and hinge to each other, not on N butt hinges",
    "strip_curtain": "each strip hangs on its own hanger off the header",
    "revolving": "a rotor turns on a floor pivot and a top bearing",
    "turnstile_tripod": "a rotor turns in its cabinet bearing",
    "turnstile_fullheight": "a rotor turns on a floor bearing and a roof bearing",
    "pet_door": "a flap hangs on a pin line through its frame",
}

# Hinge models whose realization is not a set of numbered butt-hinge stations, so ``hinge["count"]``
# cannot be counted off the geometry.  Each is excused with the reason.
HINGE_NOT_STATIONED = {
    "pivot_center": "a pivot door turns on one top and one bottom pivot, not on N leaves",
    "pivot_center_heavy": "a heavy centre-pivot set is still one top and one bottom fitting",
    "pivot_offset": "an offset pivot set is a top and a bottom fitting",
    "gravity_pivot": "a gravity hinge is a top and a bottom pivot with a cam",
    "spring_double": "a double-acting spring pivot set is top and bottom",
    "rotor_bearing": "a rotor turns on a floor pivot and a top bearing",
    "piano": "a continuous piano hinge is one member, however long",
    "continuous_geared": "a continuous geared hinge is one member",
    "flap_pin": "a flap hangs on a pin line, not on separate hinges",
    "hatch_hinge": "surface hatch hinges are drawn as an interleaved knuckle line",
    "cam_lift": "a cam-lift hinge set is drawn as its cam and its knuckles",
    "ship_hinge": "a watertight door's hinges are drawn with their lugs and jamb lugs",
    "vault_hinge": "a crane hinge is drawn as its barrel, arm and frame brackets",
    "concealed_soss": "a concealed hinge is invisible when the door is shut",
    "bifold_pivot": "a bi-fold panel turns on a top pivot and a bottom pivot",
    "baby_gate": "a pressure-mounted child gate hangs on a top and a bottom pivot in its frame",
}

# The rules ``checks["spec_realized"]`` signs off on.  Every one of them is zero over all 1000 doors.
ENFORCED_RULES = ("operator_missing", "operator_faces", "latch_multiplicity", "closer_missing",
                  "stop_missing", "stop_wrong_kind", "extra_missing", "hinge_count")
# The same walk, applied to declarations whose realization is not built yet.  Counted, never silent.
REPORTED_RULES = ("latch_missing", "lock_missing")


def _load(door_dir: str):
    with open(os.path.join(door_dir, "spec.json")) as f:
        spec = json.load(f)
    with open(os.path.join(door_dir, "model.json")) as f:
        model = json.load(f)
    return spec, model


def _tier_ok(o, tier):
    t = o.get("tiers")
    return True if not t else (tier in t)


class SpecRealized:
    """Walk the spec's declared hardware against the built geometry."""

    def __init__(self, door_dir: str, tier: str = "full"):
        spec, model = _load(door_dir)
        self._init(spec, model, tier)

    def _init(self, spec: dict, model: dict, tier: str = "full"):
        self.spec, self.model = spec, model
        self.tier = tier
        self.bodies = {b["name"]: b for b in self.model["bodies"]}
        self.meta = self.model.get("meta", {}) or {}
        self.findings = []
        self.exceptions = []
        self.allow = []
        for e in (self.meta.get("spec_realized_allow") or []):
            if len(e) == 3:
                self.allow.append((e[0], e[1], e[2]))
        self.geoms = [(b, g) for b in self.model["bodies"] for g in b["geoms"] if _tier_ok(g, tier)]

    # ---- helpers ---------------------------------------------------------
    def has(self, globs, semantic=None):
        for b, g in self.geoms:
            if semantic is not None and g.get("semantic") != semantic:
                continue
            n = g["name"]
            if any(fnmatch.fnmatch(n, p) for p in globs):
                return g["name"]
        return None

    def any_name(self, globs):
        for b in self.model["bodies"]:
            if any(fnmatch.fnmatch(b["name"], p) for p in globs):
                return b["name"]
        return self.has(globs)

    def sem(self, semantic):
        return [(b, g) for b, g in self.geoms if g.get("semantic") == semantic]

    def _excused(self, rule, item):
        for r, it, why in self.allow:
            if fnmatch.fnmatch(rule, r) and fnmatch.fnmatch(item, it):
                return why
        return None

    def fail(self, rule, item, detail, **extra):
        why = self._excused(rule, item)
        if why:
            self.exceptions.append({"rule": rule, "item": item, "reason": why, "scope": "door"})
            return
        self.findings.append({"rule": rule, "item": item, "detail": detail, **extra})

    def excuse(self, rule, item, reason):
        self.exceptions.append({"rule": rule, "item": item, "reason": reason, "scope": "global"})

    # ---- leaf frames -----------------------------------------------------
    def owner_leaf(self, body, geom=None):
        """The nearest ancestor (or self) that is a leaf body, and the geom's position in ITS frame.

        The transform has to be composed properly, rotations included: ``Model.bake_initial`` writes a
        door that ships part-open (a stall door at its 15 deg rest angle) with its whole leaf authored
        in the rotated pose, so a raw local y is not a face offset."""
        p = np.zeros(3) if geom is None else np.asarray(geom["pos"], dtype=float)
        b = body
        seen = 0
        while b is not None and seen < 12:
            if b.get("semantic") == "leaf" and b.get("joint"):
                return b, p
            if not b.get("parent"):
                return None, p
            p = np.asarray(b["pos"], dtype=float) + quat_to_mat(np.asarray(b["quat"], dtype=float)) @ p
            b = self.bodies.get(b["parent"])
            seen += 1
        return None, p

    @staticmethod
    def leaf_face_plane(leaf):
        """(point, unit normal) of the leaf's own mid-plane, read off its slab.

        The slab is the largest face-parallel box the leaf carries; its thinnest axis is the leaf's
        thickness, so that axis of its frame is the face normal.  Working relative to this plane rather
        than to the body's y axis is what makes the face test hold for a leaf that is authored rotated
        (a stall door at rest) or offset from its body origin (a roll-up curtain)."""
        boxes = [g for g in leaf["geoms"] if g.get("type") == "box" and g.get("semantic") in ("leaf", "glass")]
        if not boxes:
            return None
        slab = max(boxes, key=lambda g: float(g["size"][0]) * float(g["size"][1]) * float(g["size"][2]))
        size = [float(x) for x in slab["size"][:3]]
        ax = int(np.argmin(size))
        n = quat_to_mat(np.asarray(slab["quat"], dtype=float))[:, ax]
        return np.asarray(slab["pos"], dtype=float), n / max(float(np.linalg.norm(n)), 1e-12)

    # ---- rules -----------------------------------------------------------
    def check_operator(self):
        op = self.spec["operator"]["model"]
        if op in NO_OPERATOR_MODELS:
            self.excuse("operator_missing", op, NO_OPERATOR_MODELS[op])
            return
        ops = self.sem("operator")
        if not ops:
            self.fail("operator_missing", op,
                      f"spec declares operator '{op}' and no geom carries the 'operator' semantic")
            return
        if self.spec["operator"].get("sides") != "both":
            return
        if self.spec["family"] in NO_FACE_RULE_FAMILIES:
            self.excuse("operator_faces", op, f"{self.spec['family']}: a wing/strip has no near and far face in the leaf frame")
            return
        signs, seen_leaf = set(), False
        for b, g in ops:
            leaf, p = self.owner_leaf(b, g)
            if leaf is None:
                continue
            plane = self.leaf_face_plane(leaf)
            if plane is None:
                continue
            seen_leaf = True
            d = float(np.dot(p - plane[0], plane[1]))
            if abs(d) > FACE_EPS:
                signs.add(1 if d > 0 else -1)
        if not seen_leaf:
            self.excuse("operator_faces", op, "the operator is not carried by a leaf with a face-parallel slab")
            return
        if len(signs) < 2:
            self.fail("operator_faces", op,
                      f"operator declared on BOTH faces; operator geometry only on face {sorted(signs) or 'none'}")

    def check_latch_lock_closer(self):
        lat = self.spec["latch"]["model"]
        if lat != "none" and not self.sem("latch"):
            self.fail("latch_missing", lat, f"spec declares latch '{lat}' and no geom carries the 'latch' semantic")
        lk = self.spec["lock"]["model"]
        if lk != "none" and not self.sem("lock"):
            self.fail("lock_missing", lk, f"spec declares lock '{lk}' and no geom carries the 'lock' semantic")
        cl = self.spec["closer"]["model"]
        if cl != "none":
            if cl in CLOSER_IN_ANOTHER_PART:
                if not self.sem("hinge"):
                    self.fail("closer_missing", cl, f"'{cl}' is a spring hinge and no hinge geometry is drawn")
                else:
                    self.excuse("closer_missing", cl, CLOSER_IN_ANOTHER_PART[cl])
            elif not self.sem("closer"):
                self.fail("closer_missing", cl, f"spec declares closer '{cl}' and no geom carries the 'closer' semantic")

    def declared_bolt_count(self):
        """N bolts / dogs the spec's own names promise, and where the promise is written."""
        for key in ("latch", "lock"):
            mid = self.spec[key]["model"] or ""
            m = re.match(r"(?:dogs|multi_bolt)_(\d+)$", mid)
            if m:
                return int(m.group(1)), f"{key}.model={mid}"
        kin = self.spec.get("kinematics", {})
        for key in ("dogs", "bolts"):
            if kin.get(key):
                return int(kin[key]), f"kinematics.{key}={kin[key]}"
        return None, None

    def check_multiplicity(self):
        want, where = self.declared_bolt_count()
        if not want:
            return
        # A lever-bolt door carries a `dog_k` lever AND the `bolt_k` it throws, so count the distinct STATIONS
        # rather than the joints: a handwheel door has bolts only, a watertight door dogs only, a lever-bolt door
        # one of each per station.
        idx = set()
        for b in self.model["bodies"]:
            if not b.get("joint"):
                continue
            m = re.match(r"(?:.*_)?(?:dog|bolt)_(\d+)_(?:hinge|slide)$", b["joint"]["name"])
            if m:
                idx.add(int(m.group(1)))
        built = len(idx)
        if built != want:
            self.fail("latch_multiplicity", where,
                      f"{where} promises {want} dogs/bolts; the model builds {built}",
                      declared=want, built=built)

    def check_stop(self):
        stop = (self.spec.get("kinematics") or {}).get("stop")
        if not stop:
            return
        if stop in STOP_EXCEPTIONS:
            self.excuse("stop_missing", stop, STOP_EXCEPTIONS[stop])
            return
        c = STOP_CONTRACT.get(stop)
        if c is None:
            self.fail("stop_missing", stop, f"stop '{stop}' has no realization contract and no documented exception")
            return
        hit = self.any_name(c["globs"])
        if hit is None:
            self.fail("stop_missing", stop, f"spec declares stop '{stop}' and no geometry realizes it (expected {c['globs']})")
            return
        if c["mount"]:
            mounts = {s.get("mount") for s in (self.meta.get("stops") or [])}
            if mounts and c["mount"] not in mounts:
                self.fail("stop_wrong_kind", stop,
                          f"spec declares stop '{stop}' ({c['mount']} mount); the model builds {sorted(mounts)}")

    def hinge_stations(self):
        """Distinct hinge stations, counted off the station index the builders put in the geom names
        (``hinge_0``, ``hinge_1_jamb``, ``hinge_2_lug`` ...).  A dutch door numbers each half from 0, so
        the union of the indices is the number of stations per leaf, which is what the spec counts."""
        idx = set()
        for b, g in self.geoms:
            if g.get("semantic") != "hinge":
                continue
            m = re.search(r"_(\d+)(?:_|$)", g["name"])   # hinge_2, hinge_1_jamb, hinge_pintle_0, hinge_strap_eye_1
            if m:
                idx.add(int(m.group(1)))
        return len(idx)

    def check_hinges(self):
        h = self.spec.get("hinge") or {}
        want = h.get("count")
        if not want:
            return
        model_id = h.get("model") or ""
        fam = self.spec["family"]
        if fam in HINGE_NOT_STATIONED_FAMILIES:
            self.excuse("hinge_count", model_id, HINGE_NOT_STATIONED_FAMILIES[fam])
            return
        if model_id in HINGE_NOT_STATIONED:
            self.excuse("hinge_count", model_id, HINGE_NOT_STATIONED[model_id])
            return
        if not self.sem("hinge"):
            self.fail("hinge_count", model_id, f"spec declares {want} x '{model_id}' and no hinge geometry is drawn",
                      declared=want, built=0)
            return
        built = self.hinge_stations()
        if built != want:
            self.fail("hinge_count", model_id,
                      f"spec declares {want} hinge stations of '{model_id}'; {built} are drawn",
                      declared=want, built=built)

    def check_extras(self):
        for e in self.spec.get("extras", []):
            globs = EXTRA_CONTRACT.get(e)
            if globs is None:
                self.fail("extra_missing", e, f"extra '{e}' has no realization contract")
                continue
            if self.any_name(globs) is None:
                self.fail("extra_missing", e, f"spec declares extra '{e}' and no geometry realizes it (expected {globs})")

    def run(self):
        self.check_operator()
        self.check_latch_lock_closer()
        self.check_multiplicity()
        self.check_stop()
        self.check_hinges()
        self.check_extras()
        enforced = [f for f in self.findings if f["rule"] in ENFORCED_RULES]
        reported = [f for f in self.findings if f["rule"] not in ENFORCED_RULES]
        by_rule, open_by_rule = {}, {}
        for f in enforced:
            by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
        for f in reported:
            open_by_rule[f["rule"]] = open_by_rule.get(f["rule"], 0) + 1
        return {"ok": not enforced, "n_findings": len(enforced), "by_rule": by_rule, "findings": enforced,
                "n_open": len(reported), "open_by_rule": open_by_rule, "open": reported,
                "n_exceptions": len(self.exceptions), "exceptions": self.exceptions}


def run_spec_realized_objects(spec: dict, model: dict, tier: str = "full") -> dict:
    """The same gate, run against an in-memory spec and IR dict (``Model.to_dict()``).

    Used by the tests, which build a door from its spec rather than reading ``assets/`` - so the contract
    holds whether or not the shipped dataset has been regenerated since the generator changed."""
    try:
        sr = SpecRealized.__new__(SpecRealized)
        sr._init(spec, model, tier)
        return sr.run()
    except Exception as e:
        return {"ok": False, "n_findings": 1, "by_rule": {"error": 1},
                "findings": [{"rule": "error", "item": "-", "detail": f"{type(e).__name__}: {e}"}],
                "n_open": 0, "open_by_rule": {}, "open": [], "n_exceptions": 0, "exceptions": []}


def run_spec_realized(door_dir: str, tier: str = "full") -> dict:
    """The spec-realized gate for one door.  A gate that cannot run is a failure, not a pass."""
    try:
        return SpecRealized(door_dir, tier).run()
    except Exception as e:
        return {"ok": False, "n_findings": 1, "by_rule": {"error": 1},
                "findings": [{"rule": "error", "item": "-", "detail": f"{type(e).__name__}: {e}"}],
                "n_open": 0, "open_by_rule": {}, "open": [], "n_exceptions": 0, "exceptions": []}
