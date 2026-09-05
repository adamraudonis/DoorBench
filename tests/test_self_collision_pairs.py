"""Intra-articulation (moving-vs-moving) contact pairs, now that ``physxArticulation:enabledSelfCollisions`` is on.

Background.  Before the 2026-09 export fix PhysX had self-collision disabled for the whole articulation, so no two
moving links of a door ever touched in Isaac Sim.  The fix turns it on and authors every pair MuJoCo suppresses
(same weld body, weld parent/child, ``<contact><exclude>``) as ``PhysxFilteredPairsAPI``, so the two engines filter
the same set.  That is right - but it also makes every *unfiltered* moving pair live in PhysX for the first time,
and PhysX is not MuJoCo: MuJoCo at margin 0 resolves an authored overlap with a soft constraint, PhysX resolves it
rigidly inside a 5 mm ``contactOffset`` and pushes the parts apart at up to ``maxDepenetrationVelocity``.

So an overlap that a MuJoCo-only dataset could carry harmlessly is now a hazard, and the two geometric gates do not
see it: ``run_running`` measures MOVING vs STATIC only, and the interpenetration gate allows the by-design
pass-throughs on ``clearance.DEFAULT_ALLOW`` (a thumbturn spindle bored through its lock case, a pin in its
housing).  This module closes that gap for the at-rest configuration, which is frame 0 of every episode.

The measured defect this pins: ``leaf_deadbolt_box`` (on the bolt body) and ``leaf_deadbolt_thumbturn_mesh`` (on
the thumbturn body) overlapped 2-5 mm at rest on 6 doors, unfiltered - two links coupled by the very joint equality
that makes the deadbolt work, with a rigid PhysX contact fighting it.  ``add_deadbolt`` now excludes the pair.

Run:  pytest -q tests/test_self_collision_pairs.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.export.usd import mujoco_filtered_pairs
from doorbench.spec import generate_all

# doors whose deadbolt / thumbturn pair was measured overlapping at rest before the exclude was added
DEADBOLT_OVERLAP_BEFORE = ["db0016_swing_single", "db0418_swing_single", "db0011_automatic_swing",
                           "db0161_pivot", "db0817_pivot", "db0912_swing_single"]
MARGIN = 0.025      # m; the distance query's cutoff, well past PhysX's 5 mm contact offset


@pytest.fixture(scope="module")
def specs():
    return {s["id"]: s for s in generate_all()}


def live_pairs_at_rest(door_dir: str):
    """(geom_a, geom_b, signed distance) for every moving-vs-moving collider pair PhysX will simulate at q0.

    PhysX with self-collisions enabled skips exactly two things: geoms on the same link, and links joined by a
    joint (parent/child).  Everything else is live unless it was authored into ``physxFilteredPairs``, which is
    ``mujoco_filtered_pairs`` - recomputed here from the same model.json the exporter reads, so the test does not
    depend on a USD file having been written.
    """
    import mujoco

    from doorbench.ir import Model

    model_json = json.load(open(os.path.join(door_dir, "model.json")))
    m = mujoco.MjModel.from_xml_path(os.path.join(door_dir, "door.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
    bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    excl = {tuple(sorted(x)) for x in model_json.get("contact_excludes", [])}
    bw, bp = m.body_weldid, m.body_parentid
    ids = [i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and bw[m.geom_bodyid[i]] != 0]
    out = []
    pos = np.asarray(d.geom_xpos)
    for a, gi in enumerate(ids):
        for gj in ids[a + 1:]:
            w1, w2 = int(bw[m.geom_bodyid[gi]]), int(bw[m.geom_bodyid[gj]])
            if w1 == w2 or int(bw[bp[w1]]) == w2 or int(bw[bp[w2]]) == w1:
                continue                                    # one link, or joint-adjacent: PhysX skips it too
            if tuple(sorted((bname[w1], bname[w2]))) in excl:
                continue                                    # authored as PhysxFilteredPairsAPI
            if np.linalg.norm(pos[gi] - pos[gj]) - m.geom_rbound[gi] - m.geom_rbound[gj] > MARGIN:
                continue
            dist = float(mujoco.mj_geomDistance(m, d, int(gi), int(gj), MARGIN, None))
            if dist < MARGIN:
                out.append((gname[gi], gname[gj], dist))
    return out


def test_a_moving_pair_that_overlaps_at_rest_is_detected(tmp_path):
    """The detector itself: two unfiltered moving links authored 3 mm into each other must be reported."""
    import mujoco
    xml = """<mujoco model="fixture">
      <worldbody>
        <body name="world_env"><geom name="floor" type="box" size="2 2 0.05" pos="0 0 -0.05"/></body>
        <body name="leaf" pos="0 0 1">
          <joint name="leaf_hinge" type="hinge" axis="0 0 1"/>
          <geom name="leaf_slab" type="box" size="0.4 0.02 0.9" pos="0.4 0 0"/>
          <body name="bolt" pos="0.8 0 0">
            <joint name="bolt_slide" type="slide" axis="1 0 0" limited="true" range="0 0.03"/>
            <geom name="bolt_box" type="box" size="0.03 0.01 0.01"/>
          </body>
          <body name="turn" pos="0.75 0 0">
            <joint name="turn_hinge" type="hinge" axis="0 1 0" limited="true" range="0 1.57"/>
            <geom name="turn_mesh" type="box" size="0.011 0.01 0.01"/>
          </body>
        </body>
      </worldbody>
    </mujoco>"""
    # bolt_box spans x 0.77..0.83 and turn_mesh 0.739..0.761 (9 mm clear); at x = 0.762 the turn's face is 3 mm
    # inside the box and 20 mm inside it in y and z, so the minimum translation distance is exactly the 3 mm in x
    d = str(tmp_path)
    with open(os.path.join(d, "door.xml"), "w") as f:
        f.write(xml.replace('name="turn" pos="0.75 0 0"', 'name="turn" pos="0.762 0 0"'))
    with open(os.path.join(d, "model.json"), "w") as f:
        json.dump({"meta": {}, "bodies": [], "contact_excludes": []}, f)
    hits = [h for h in live_pairs_at_rest(d) if h[2] < 0]
    assert hits, "an unfiltered 3 mm overlap between two moving links must be reported"
    assert {hits[0][0], hits[0][1]} == {"bolt_box", "turn_mesh"}
    assert hits[0][2] == pytest.approx(-0.003, abs=1e-4)

    # ... and excluding the pair silences it, exactly as PhysxFilteredPairsAPI will
    with open(os.path.join(d, "model.json"), "w") as f:
        json.dump({"meta": {}, "bodies": [], "contact_excludes": [["bolt", "turn"]]}, f)
    assert not [h for h in live_pairs_at_rest(d) if h[2] < 0]


@pytest.mark.parametrize("door_id", DEADBOLT_OVERLAP_BEFORE)
def test_thumbturn_no_longer_overlaps_its_lock_case(door_id, specs, tmp_path):
    """The thumbturn spindle is bored through the lock case: the pair must be excluded, not resolved."""
    export_door(specs[door_id], os.path.join(str(tmp_path), "doors"), os.path.join(str(tmp_path), "hardware"),
                formats=("mjcf", "json"))
    door_dir = os.path.join(str(tmp_path), "doors", door_id)
    model_json = json.load(open(os.path.join(door_dir, "model.json")))
    excl = {tuple(sorted(x)) for x in model_json.get("contact_excludes", [])}
    pair = next((p for p in excl if p[0].endswith("_deadbolt") and p[1].endswith("_deadbolt_thumbturn")), None)
    assert pair is not None, f"{door_id}: the deadbolt / thumbturn pair must be in contact_excludes, got {sorted(excl)}"
    assert not [h for h in live_pairs_at_rest(door_dir) if h[2] < 0]


def test_the_exclude_reaches_the_usd_filtered_pairs(specs, tmp_path):
    """``mujoco_filtered_pairs`` (what the exporter authors) must contain the excluded pair, so both engines agree."""
    from doorbench.build import build_model
    s = specs["db0016_swing_single"]
    model = build_model(s)
    pairs = mujoco_filtered_pairs(model, model.bodies, "full")
    assert ("leaf_deadbolt", "leaf_deadbolt_thumbturn") in [tuple(p) for p in pairs]


def test_no_door_family_overlaps_itself_at_rest(specs, tmp_path):
    """One door per family: nothing PhysX will simulate as a self-collision may start the episode interpenetrating.

    Sub-millimetre laps of a soft meeting-stile astragal against the other leaf are the documented exception (a real
    astragal IS compressed against the passive leaf); anything deeper is a hard contact on frame 0.
    """
    ASTRAGAL_LAP = 0.001        # m; a compressible meeting-stile seal may lap the other leaf by up to 1 mm
    seen, bad = set(), []
    for s in specs.values():
        if s["family"] in seen:
            continue
        seen.add(s["family"])
        export_door(s, os.path.join(str(tmp_path), "doors"), os.path.join(str(tmp_path), "hardware"),
                    formats=("mjcf", "json"))
        for ga, gb, dist in live_pairs_at_rest(os.path.join(str(tmp_path), "doors", s["id"])):
            soft = "astragal" in ga or "astragal" in gb or "seal" in ga or "seal" in gb or "gasket" in ga or "gasket" in gb
            if dist < (-ASTRAGAL_LAP if soft else 0.0):
                bad.append((s["id"], ga, gb, round(dist, 5)))
    assert not bad, bad
