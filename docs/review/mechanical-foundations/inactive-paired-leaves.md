# Inactive paired-leaf holding hardware

This repair replaces the artificial `leaf_b_hinge` range of 0–0.001 rad on
28 paired doors. Leaf B now retains its source opening range and is held by
physical steel bolts entering prepared fixed receivers. These are original
generic mechanisms, not OEM CAD or certified hardware.

| Installed hardware | Door IDs |
|---|---|
| Two edge flush bolts (20 pairs) | DB0149, DB0222, DB0334, DB0341, DB0413, DB0454, DB0534, DB0577, DB0604, DB0682, DB0702, DB0707, DB0714, DB0733, DB0792, DB0832, DB0846, DB0929, DB0944, DB0963 |
| One inside cane bolt (8 pairs) | DB0127, DB0144, DB0279, DB0395, DB0467, DB0704, DB0788, DB0918 |

The flush-bolt installation envelope follows the dimensions in the
[Ives FB358 installation template](https://allegion.ca/content/dam/allegion-us-2/web-files/ives/installation-documents/Ives_FB358_Manual_Flush_Bolt_Installation_Template_101738.pdf):
171 × 25 × 35 mm housing and a 19 mm edge backset. The original direct finger
slider uses a 12 mm shaft, 35 mm stroke, open finger window, bored guides, and
a shaft collar between physical end stops. The housing and complete shaft
stroke are cut from the actual stile; the steel is not embedded in uncut stock.
Separate upper/lower knob surfaces support finger presses in opposite
directions. This does not assume a pinch grasp fits inside the 16 mm window.

The cane-bolt dimensions are informed by the
[National N165-902 drawing](https://nationalhardwarestorage.blob.core.windows.net/documents/nh_td_836_n165-902.pdf):
a 12.3 mm shaft with a 317 mm vertical section and 77.5 mm bent grip. The
original model adds two supported open guides and finite travel collars for an
80 mm lift. Guide bases meet the actual plank or brace surface. The inside
face is derived from both the source approach direction and whether the robot
is outside. Floor and head receivers have 0.75 mm radial clearance, real metal
walls and a back cap, with matching cavities in the fixed substrate.

Both faces of B receive a fixed service pull with supported mounting pads and
a clear grip. Bolts, guides, receivers, pulls, and the functional astragal remain
present in full, simple, and minimal exports. New moving parts retain their
geometry-derived steel masses; exact routed stock is deducted from B's material
budget. The prior source contained no separate inactive-bolt allowance.

The implementation uses ideal prismatic bearings and frictional position
retention. It does not model screw threads, fastener pullout, steel yielding,
or certify durability, reach, grasp feasibility, or one-handed usability. The
native contact model requires a timestep no larger than 0.5 ms. Other engines
must provide corresponding contact resolution; successful loading alone is
not equivalent to this native proof.

## Runtime contract and access

`model.meta.paired_leaf_holds` contains one row per installed rod. Each row
names its leaf and primary joints; moving body; rod and grip geometry; guide,
keeper and stop geometries; withdrawal `site`; and reverse `engage_site`.
`travel_m`, `withdrawn_threshold_m`, `nominal_joint_range_m`, and the wider
`joint_safety_range_m` distinguish usable motion from a safety bound. Actual
collar contacts stop motion before the joint limit. `force_cap_N` is 20 N.

Flush controls are on the meeting edge. Their
`requires_primary_open_rad = 0.20` is a dynamic acquisition precondition;
their potential input permission must not grant contact through a closed seam.
Cane controls retain their actual inside-face permission.
`meta.inactive_leaf_pulls` names each fixed pull and its face. These are handles
for moving B after its rods are clear, not alternative release inputs.

The source itself can represent an inaccessible upper or lower control.
Approach-side permission is not a whole-body reachability certificate. The
repair does not alter an active leaf's credentials or imply that every task
must open the inactive leaf.

The scripted-hand adapter uses actual current joint state. It first operates A
through its existing source inputs, presses the now-exposed flush controls,
then pulls B. Closing reverses that sequence: B seats, each rod re-enters its
receiver, then A closes. An inaccessible cane bolt keeps the policy on A only.
Both direct-joint and site-force entry points revoke flush permission whenever
A hides the edge again. Finger site forces are projected inward and capped at
20 N; B pull forces are capped at 50 N. The contact controller runs at the
authored native timestep, with no pose or range changes.

Native runtime regressions exercise DB0222 (flush bolts) and DB0467 (inside
cane bolt) in `open_then_close` and `close_only`, plus closed-seam input
rejection and DB0127 outside-cane rejection. Those representative runtime
checks are distinct from the all-28 mechanical fixture below.

The final local runs passed 15 dedicated mechanical tests and 11 runtime and
related regression tests. All 84 native tier fixtures completed both cycles;
their worst penetration was 0.255 mm. Runtime `open_then_close` completed in
15.259 s on DB0222 and 13.736 s on DB0467, with respective worst native contact
penetrations of 0.456 and 0.118 mm. Both `close_only` cases also passed. These
are source-bound spot checks, not a rerun of the published leaderboard.

## Verification

The reproducible regression entry point is:

```sh
python -m pytest tests/test_paired_leaf_holds.py -q \
  --basetemp=out/mechanical-foundations/paired-holds/pytest-local
```

The service fixture prescribes A at 0.8 rad once, resolves its authored closer
loop at setup, and then holds it with at most 100 Nm. It leaves the original
model unchanged and retains authored passive closer fields. This is explicitly
an already-open active-leaf fixture, not a recorded primary unlock or benchmark
result. There are no configuration writes during the two repeated cycles.

Each cycle loads B toward opening through its actual pull, verifies each
receiver carries load, unloads the leaf, withdraws each bolt with at most 20 N,
opens and closes B through a fixed pull with at most 50 N, and reinserts the
bolts from their real grip surfaces. Hinge-friction compensation stays inside
those caps. Each rod must hit its physical stroke stops, while bolt joint-limit
force remains at most 0.01 N. All native contacts are measured against a 1 mm
penetration limit; warning callback strings and native counters both fail the
fixture. Ten-hertz and phase-end native configurations are retained for
independent geometry inspection.
Each installed rod and collar must show a positive measured normal reaction
above 0.05 N on its exact receiver/stop pairs. A candidate contact with zero
force does not prove load transfer. Flush finger forces are inward-only at the
chosen face; the opposite face is used for re-engagement.

Separate checks cover every complete nominal bolt stroke and the full B leaf
range with A open, including visual geometry and parent-filtered pairs. Negative
fixtures recompile modified XML for removed receivers, a filled guide, removed
stroke stops, and the old artificial hinge range. Off-surface grips, incomplete
top/bottom metadata, and callback-only warnings also fail closed. Per-tier
native receipts bind the generated spec, model, XML and tier XML hashes.

The final local test log and generated per-door receipts are under
`out/mechanical-foundations/paired-holds/pytest-pressbound/`. Generated assets are
not committed. Original installation documents and hashes are recorded in
`out/mechanical-foundations/paired-dutch/sources/receipt.json`; no OEM drawing
or CAD is incorporated into the geometry exports.
