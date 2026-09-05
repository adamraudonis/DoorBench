# Closer mechanisms as real, tunable linkages (task G5)

Read `handoffs/README.md` first. Branch to resume: `worktree-agent-a919977c25b0ad6a2` (1 WIP commit
"Closers as real mechanisms (WIP): pinion-spring two-bar arm closers, telescoping struts, ..." plus a snapshot
commit; new files `doorbench/closers.py` and `doorbench/geometry/closers.py`). Start with
`git diff master...HEAD --stat` and read both new modules; decide what to keep.

## Why (owner's words)

"https://adamraudonis.github.io/DoorBench/#/door/db0012_swing_single has clearly broken parts where things are
floating and the hinges for this self closing mechanism are not actuating. This is absolutely unacceptable. Also
this self-closing mechanism and power and mechanics needs to be perfectly modeled with tunable physics parameters."

## Current state

* db0012 has a `norton_1600` surface closer (EN size 5, pull side, regular arm). `add_closer` in
  `doorbench/geometry/common.py` (~line 873) builds `closer_arm_main` (child of `leaf`, hinge `closer_pinion`) ->
  `closer_arm_fore` (child of the main arm, hinge `closer_elbow`) and a `connect` equality `closer_arm_connect`
  pinning the forearm tip to a world bracket (the shoe). FULL tier only.
* The closing torque is NOT transmitted through that linkage: `doorbench/physics.py::closer_params` puts a spring
  (`spring_preload_Nm`, `spring_stiffness_Nm_per_rad`) and asymmetric damping (`damping_closing/opening`,
  `backcheck_*`, `latch_boost`, `hold_open_rad`) directly on the door hinge joint. The arms are decoration.
* The web viewer already solves the closed loop kinematically (merged: `viewer/src/kinematics.ts`), so the arms
  no longer float on the site. The viewer reads `model.json["linkages"]` when present (schema below) and otherwise
  derives the loop from `bodies` + `equalities`.
* The viewer found 5 rising-hinge cold-storage doors whose planar loop cannot close (see
  `door-cold-storage-rising-hinge-closer.md`).

## Goal

Every self-closing / power-operating mechanism in `doorbench/hardware.py::CLOSERS` is a physically faithful
mechanism in the MuJoCo full tier, with tunable parameters, and the reduced tiers/exports stay calibrated to it:

1. **Arm closers** (surface overhead regular arm pull side / parallel arm push side, hold-open arms, automatic
   swing operators): the spring (preload + rate) and the hydraulic damping act on the **pinion** joint on the
   closer body; sweep / latch (last ~10-15 deg) / backcheck (from ~70 deg) / delayed-action are angle windows;
   the door torque emerges through the two-bar linkage's varying mechanical advantage. Real geometry from
   installation templates (LCN 4040XP / Norton 1600 / Dorma TS 83: main arm ~0.28-0.32 m, forearm 0.20-0.30 m,
   pinion 60-110 mm from the hinge edge). Automatic operators: motor (actuator) on the pinion with the spec'd
   opening torque/speed, spring-close on power loss.
2. **Telescoping** (gas struts on garage / tilt-up / hatches, pneumatic and spring-tube gate closers, screen-door
   closers): cylinder hinged at the frame bracket, rod on a slide joint with the force ALONG the axis (gas force with
   progression; air-cushion damping with a latch valve), tip pinned to the door bracket by `connect`.
3. **Floor springs / concealed closers / spring hinges**: torque at the pivot/hinge, concealed body geometry right.
4. **Sliding operators**: belt/carriage drive with the spec'd force.

Constraints: `connect` violation < 1 mm over the whole sweep; arms pass the clearance gate; tunable parameters in
`spec.json["physics"]["closer"]` with units + a `mechanism` string, derived from EN 1154 as now and overridable
from `spec.json["closer"]` (spring_adjust, valve settings, hold_open, delayed_action, arm lengths, pinion/shoe
positions, gas force/stroke/progression); document each in the physics doc.

Reduced models: simple/minimal tiers and URDF keep a joint-level spring/damper on the door joint **calibrated** to
the full mechanism (fit torque-vs-angle and damping curves measured in MuJoCo, < 10 % error; store the fitted
values in physics). USD (`doorbench/export/usd.py`, default prim `/<door_id>` with `Env` + `Articulation`): export
the loop-closure joints (PhysX supports loops) and the pinion drive; `door_rl.usda` (canonical 8-link
articulation) keeps the reduced model; `scripts/isaaclab/validate_usd_static.py` must stay 1000/1000.

`model.json` must carry a `linkages` block (the viewer consumes it):

```json
"linkages": [
 {"name": "closer", "type": "two_bar",
  "pinion": {"body": "<main arm body>", "joint": "<pinion hinge joint>", "parent": "<leaf body>"},
  "elbow":  {"body": "<forearm body>", "joint": "<elbow hinge joint>"},
  "anchor": {"body": "world or <frame body>", "pos": [0, 0, 0]},
  "equality": "<connect equality name>", "axis": [0, 0, 1], "L1": 0.30, "L2": 0.25, "elbow_sign": 1},
 {"name": "gas_strut", "type": "telescoping",
  "base":  {"body": "<cylinder body>", "joint": "<hinge at bracket>", "parent": "world or <frame body>", "pos": [0, 0, 0]},
  "slide": {"body": "<rod body>", "joint": "<slide joint>", "axis_local": [1, 0, 0], "offset": 0.0},
  "anchor": {"body": "<leaf body>", "pos": [0, 0, 0]}, "equality": "<connect equality name>"}
]
```

## QA to add (`doorbench/qa.py`)

`closer_linkage` (connect violation < 1 mm over the sweep; arm joints move when the door moves),
`closer_closes` (from 90 deg and from 15 deg within the EN 1154 / product closing-time window; latch engages;
no slam: final angular speed bound; backcheck limits peak speed when flung open; hold-open holds; delayed action
delays), automatic operators open to the spec'd angle in the spec'd time and close on power loss.
Add `tests/test_closers.py` covering every closer kind.

## Files

`doorbench/geometry/closers.py` (new, most code), `doorbench/geometry/common.py` (call sites only),
`doorbench/physics.py` (closer params + reduced-model fit), `doorbench/ir.py` / `doorbench/build.py`
(linkages block), `doorbench/export/{mjcf,usd,urdf}.py`, `doorbench/qa.py`, docs (physics doc),
`tests/test_closers.py`.

## Done when

1000 signed off, clearance 1000/1000, static USD validation 1000/1000, tests green;
`docs/media/closer_db0012_{0,45,90}.png` (MuJoCo render) show the linkage following the door;
`docs/media/closer_curves.png` shows full vs reduced torque-vs-angle with the fit error.
