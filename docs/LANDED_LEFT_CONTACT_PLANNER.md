# Left contact from an actual landed stance

`scripts/dexterous/plan_landed_left.py` adapts the left-palm target to a measured
H1/Shadow stance. It preserves the actual root, torso, legs, right arm, fingers
and door coordinates. Its separate MuJoCo scene only computes geometry; it
never steps or writes to the active plant. This is a privileged planning tool,
outside the sensor actor boundary.

The first continuous walking/opening attempt landed roughly 13° from the old
reference heading. Reusing the original fixed left-palm position was infeasible.
The planner instead keeps the palm's panel-local Y plane and normal, while
allowing up to 15 cm in panel-local X/Z. Seven left arm/wrist joints are fitted
inside their authored limits with a 25 mrad planning margin. The objective is:

```text
100 × panel-Y error
 10 × palm-normal vector error
 .15 × displacement from the clipped original terminal arm posture
100 × any tangential displacement beyond the declared 15 cm bound
```

The endpoint must converge, preserve Y within 1 mm and the normal within 0.5°,
and stay inside the tangential bound. The original 61-row nominal path is
adapted by cubic smoothstep endpoint corrections. Every target's position and
normal are recomputed in the actual leaf frame; no saved root is installed.

## Run

Use the pinned asset Python with MuJoCo and SciPy. A fresh output directory is
required. This reproduces the actual 45.502 s state of the failed first run:

```bash
PYTHONPATH=. "$ASSET_PYTHON" scripts/dexterous/plan_landed_left.py \
  --robot "$ROBOT_V2_XML" --door "$ACTUAL_DOOR_XML" \
  --targets "$WALKING_RUN_001/static-plan.json" \
  --trial "$WALKING_RUN_001" --time 45.502 \
  --output out/landed-left/new-plan
```

The archive loader verifies the robot/door input XML hashes and selects an
exact timestamp from `trace.json`, matching its row to `trajectory.npz`. It
does not infer a frame rate or round to a nearby state. The run's failed task
status is recorded, never inherited as a successful prefix.

For a newly attained state, replace `--trial … --time …` with `--state state.json`.
The JSON contains `pose_time_s`, `root` (world XYZ and unit WXYZ quaternion),
all 69 actual `joints` by unprefixed name, and every actual `door_positions`
coordinate by native joint name. Missing/extra coordinates are rejected. These
values must come from the active simulator; the caller remains responsible for
their measurement and timestamp provenance.

The output includes `target-config.json`, `frozen-state.json`, a complete
report and frozen planner/auditor/CLI sources. To independently audit an existing
candidate, add `--audit-targets existing-target-config.json`; it does not rewrite
or promote that file.

## Independent checks and limits

The independent auditor uses fresh planning data and does not trust a plan's
`passed` flag. It verifies the initial actual arm, unchanged torso, leaf-angle
metadata, and every stored FK position/normal. It then checks two 601-sample
paths (ten subdivisions per original segment): nominal joint interpolation,
and Cartesian position/normal interpolation reconstructed with bounded IK.
The latter follows the same target representation used by `LeftPalmContact`.

All native collision pairs are considered, including receiving-only colliders.
Left-hand environment contact is prohibited before path coordinate 0.98, then
allowed only on the leaf body. All nonfoot penetration remains bounded by 3 mm;
individual limits and passive Shadow loopbacks retain their 20 mrad physical
ceilings. Positive-gap collision-buffer entries are not declared touch.

These are sampled, frozen-state geometry checks. They do not certify gaps
between samples, moving-root tracking, the later force-control contact offset,
force feasibility, balance, valid right-hand finger pads, or continuous opening.
Runtime physical checks remain required. No changes are made to masses,
collision shapes, joints, tendons or motor caps.

## Measured reproduction

The reusable planner exactly reproduced all 61 arm target rows of the original
`landed-left-plan-001` candidate. Its contact shifted X by −8.472 mm and Z by
−12.474 mm. Panel-Y error was 0.253 micrometres; normal error was 0.000208°.
Both independent 601-sample paths passed. Cartesian IK's maximum position error
was 0.424 micrometres; all 1,202 poses had no left-hand environment touch, at
most 0.136 mm nonfoot penetration and 0.450 mrad loopback excess. These are
geometric measurements, not measured contact loads.

The audit found stale old `leaf_rad=0.0888702084` metadata in the original
candidate. The new file records the actual `0.0850058831` rad angle. That
correction does not change the 61 arm targets.

It also rejected an older Door55 XML (`d3b367b62eed…`) against the actual source
run's revised door (`5b12a0a19c8b…`). The two have different handle/strike/latch
geometry and mechanics. Earlier continuation009/010 on the older door cannot
qualify passage on this door. The root's separate walking/opening002 run reached
1.200824 rad with the adapted arm path but failed its right-pad contact gate;
its 28/29 result remains a failed full-task result.

Compact receipts are in `results/dexterous/2026-09-08/landed-left/`. Initial
planner001 (enum comparison),002 (correct model mismatch rejection), and003
(NumPy boolean JSON serialization) remain recorded. No physics ran in any of
these planner attempts.

Source-design identities include authored XML and referenced asset bytes.
Compiled geometry is bound separately to the runtime. On a different OS/runtime,
regenerate this complete static plan and audit; do not bypass an older compiled
identity check or reuse a stale runtime rescreen receipt.
