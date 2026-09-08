# Right-hand release from the actual walked grasp

**Development only; no release fix is qualified.** The original
`AxialRightRelease` remains the default. These tests keep the v2 robot, original
motor caps, exact Door55 scene, and every anatomy/mechanical acceptance gate.

The continuous native `walking-opening-native-002` baseline acquires the handle
and loads the left palm correctly, but its right hand follows the springing
lever while sliding sideways with curled fingers. The resulting middle and
proximal finger contacts fail the declared pad contract. A geometric screen
that freezes a pressed lever does not certify release of a springing mechanism.
Close-ups from four exact archived pre-states, viewed from two angles, confirm
the contact path. They use the actual source door XML `5b12a0a1…`; an initial
render attempt using the older deep-stance door was rejected by the body-frame
consistency check.

## Isolated continuous comparisons

`probe_walking_release.py` unpacks the baseline's hash-verified source archive
and exact configuration, including the measured landed stance and left-arm
plan. It changes the release primitive in an isolated process. Every completed
raw chunk before the intervention must match the baseline byte-for-byte, or the
run stops. Equal chunks are SHA-verified and hardlinked to conserve disk space.
Both complete comparisons match **all raw evidence through 53.500 s**; the
release intervention starts at 53.504 s. There are no runtime pose resets.

| Variant | Observed result | Decision |
|---|---|---|
| Pressed handle frame bound to measured moving leaf; original 0.4 s preload decay | 18 invalid contacts across 10 intervals, including a 61.5 N thumb-middle impact at preload removal; original joint/motor/collision gates pass | Failed; later palm support is lost and aperture is reached by coasting |
| Same frame; retain original grip preload until measured 20 mm clearance, then decay over 0.4 s | 262 thumb-distal end/side contacts, 157 joint-stop violations, maximum 45.66 mrad | Failed; do not escalate preload |

The first variant reduces the baseline's roughly 1,200 invalid contacts but
does not solve the release. Neither trial passes the sustained palm, right
release completion, or full-opening requirements. The second is also physically
invalid under the unchanged joint-stop tolerance. Final aperture alone is not
success. [Frozen results and archive receipts](../results/dexterous/2026-09-08/walked-right-release-development.json)

## Development API and reproduction

`PressedLeafFrameRightRelease` is an explicit opt-in subclass. Before each
`begin` and `update`, call `observe_leaf(t, leaf_pose)` with the current measured
xyz/wxyz leaf pose on the same local teacher clock. It binds the attained
pressed handle frame once, then follows actual leaf motion rather than copying
the springing operator angle. It changes only analytic teacher targets; the
original force controller still emits the original capped motor commands.

Its optional `retain_grip_until_clear=True` keeps the original digit preload
until the caller's measured clearance freezes the palm goal. The test above
rejects that variant; it is retained solely for reproduction. A stale leaf
measurement is an error. The wrapper records its precise intervention and
source hashes separately from the baseline manifest.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/dexterous/probe_walking_release.py \
  --source-run /path/to/walking-opening-native-002 \
  --output out/walked-pressed-release-new --seconds 66
# Reproduce the rejected grip-retention comparison by adding:
# --retain-grip-until-clear
```

## Controlled return was rejected by the measured workspace

An explicit `ControlledLeverReturn` development component preserves the actual
opposed grip and moves the operator target smoothly to rest over four seconds,
using the measured operator/leaf joint geometry. The left palm retains the
attained aperture. Reproduce it with `--release-mode controlled-return` in the
same isolated wrapper; the original release remains unchanged.

The 60 s continuous trial again reproduces the baseline raw prefix through
53.500 s. It maintains 0.08566 rad aperture and 3.727 N final left-palm support;
all original mechanical gates pass. However, the lever stalls at 0.2174 rad,
right-palm tracking error grows to 27.83 mm, and 2,106 intervals have invalid
middle-link contact patches. This is **not a qualified controlled return**.

Unstepped inverse-kinematics screens use the exact archived pre-release state
and original joint limits. With a frozen waist, the resting-lever target has
14.64 mm position and 4.16° rotation error and reaches the right elbow and wrist
limits. Allowing the waist and both arms reduces error to 0.43 mm / 1.64°, but
right wrist limits still bind. Allowing the left palm to rotate in its contact
plane or lowering its target does not remove that right-hand restriction.
These are geometric diagnostics, not physical trials. More grip force will not
resolve the required wrist pose.

The first attempt ended before release because the disk filled; its incomplete
prefix is retained. The recorder now compares compressed chunks in memory
before writing, so an equal walking prefix needs only its verified hardlink.
The successful recording and the interrupted attempt are archived separately
from the earlier release failures. [Trial, workspace and archive receipt](../results/dexterous/2026-09-08/controlled-lever-return-development.json)

The next bounded hypothesis is to open the fingers while withdrawing the palm
along a screened curve that respects the returning lever. The earlier measured
release came from an **unpressed canonical grasp**; its timing and 160 mm retreat
are proposals, not evidence that the actual walked, pressed grasp can release.

## A small physical body shift allows the lever to return

The next experiment succeeds at the **controlled lever-return component**.
It is still not a completed right-hand release or full opening.

A whole-body geometric path keeps the two attained foot frames and both palm
poses while returning the lever. Its final pelvis displacement is
`[+8.47, -14.55, +15.11]` mm, with a small orientation change. The 401-state
interpolation audit has at most 13.8 µm palm error, 0.35 µm foot error, 2.336°
torso tilt, and no forbidden collision. The robot COM remains inside the hull
of the measured initial foot contacts. This geometric evidence does not impose
physical foot constraints.

`WholeBodyLeverReturn` supplies the existing stance QP with those pelvis and
leg targets and the existing arm teacher with the screened torso target. The
physical trial retains all original motors, limits, contact geometry and
attained foot references. Its raw walking/acquisition prefix again matches the
original run through 53.500 s, before intervention at 53.504 s.

The 60 s native trial physically returns the lever to rest while retaining the
opposed grasp and left-palm support. An independent audit reads all 30,000 raw
transitions, checks motor caps and state continuity, and reconstructs the actual
body/contact frames for all 3,250 intervals after the intervention. Frame error
is exactly zero and no invalid loaded distal patch occurs. During the final
half second the operator stays within 0.630 mrad of rest and all five pads stay
loaded. Final left-palm load is 13.02 N and aperture is 0.09692 rad. Every
original mechanical, anatomy and stance check passes; the original full task
report correctly fails **usable aperture** and **right release completion**.
[Measured result and canonical archive receipts](../results/dexterous/2026-09-08/whole-body-lever-return-development.json)

```sh
# Regenerate the unstepped geometry path from the exact recorded state.
python scripts/dexterous/screen_whole_body_return.py \
  --source-run /path/to/walking-opening-native-002 \
  --state-run /path/to/walked-controlled-return-002 \
  --source-package /path/to/walked-controlled-return-002-source \
  --at 53.504 --output out/whole-body-return-screen

# Apply only motor targets in a fresh continuous physical comparison.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/dexterous/probe_walking_release.py \
  --source-run /path/to/walking-opening-native-002 \
  --whole-body-path /path/to/controlled-return-whole-body-path-001/report.json \
  --release-mode whole-body-return --seconds 60 \
  --output out/whole-body-return-repeat
```

These targets are tied to the actual walked state and its robot/scene inputs.
The adapter rejects a different attained root/joint configuration. Re-screen a
different robot, scene, or initial stance; this is a privileged teacher component,
not a vision/tactile actor.

## Reject disconnected release configurations

A shorter diagonal ungrip initially appeared feasible when each hand frame was
allowed to choose its own collision-free lever angle. Dense checking revealed
that these feasible sets were disconnected: near 53% progress only a pressed
lever fit, while at 54% only a resting lever fit. No continuous motion connected
them. `release_connectivity.monotone_release_path` now rejects this specific
failure mode; a returned sampled path still requires swept-edge refinement and
physical execution. Seventeen reachable fixed-timing curves and four searches
with independent hand, finger, thumb and operator progress produced no qualified
release. Those failures and the diagnostic plot are retained in the archive.
The unrun curved controller prototype was not installed as a release option.

The next trial must withdraw from the **actually attained resting-lever state**.
The measured canonical ungrip still exceeds the current fixed-waist arm reach,
so another bounded body movement or a newly screened hand path is required.

## Resting-grasp adjustment and measured withdrawal remain developmental

A new geometric route first adjusts the actual attained resting grasp to the
measured canonical grasp (about 12 mm palm depth and 2° orientation), then uses
the source release's **absolute finger angles** and handle-relative palm path.
Adding source finger deltas to the attained grasp was rejected: its finger
angles differ by as much as 0.16 rad. Decimating the source also failed dense
collision checks between samples. Retaining all 330 recorded release frames
passes 2,360 interpolated geometry samples, including a continuous prelude from
the attained state: maximum palm error 5.31 µm, foot error 1.18 µm, and torso
tilt 4.45°. This is geometric evidence, not contact-force qualification.

`WholeBodyMeasuredUngrip` is an explicit development option. It retains the
qualified return commands until episode 60 s, verifies the exact measured
root/joints/door configuration, bridges desired commands smoothly, and follows
the screened body and finger route. The caller supplies both
`observe_operation(...)` and `observe_state(t, root, joints)` on the same clock.
`body_goal(...)` feeds only the existing landed-foot stance targets. Its
`ready_for_panel` remains false until the entire route finishes, so an early
20 mm clearance measurement cannot interrupt the planned motion. The core
still requires actual clearance and all original physical gates.

Both continuous trials reproduce all 120 raw prefix chunks through 60 s
(310,752,869 bytes). Neither qualifies:

| Ungrip goal frame | Result |
|---|---|
| Current measured handle | Removed right-hand leaf stiffness during adjustment; the door coasted to 1.2016 rad at 64.198 s with no palm support, before ungrip completed. Rejected. |
| Attained resting-handle world frame | Completed the withdrawal and measured-clearance gate, with original motor/joint/stance limits intact. Ten brief RF/MF distal-tip contacts at 68.470–68.582 s fail the unchanged volar-normal gate. Left-palm support also unloads; final 1.2001 rad aperture has zero palm load. Rejected. |

The independent second-trial audit checks all 37,415 state transitions and
original motor caps, and reconstructs all 10,665 post-transfer body/contact
frames with exactly zero error. Its six invalid contact intervals agree with
the primary audit; late valid instantaneous samples do not erase them.
[Sources, measured outcomes and canonical archive receipt](../results/dexterous/2026-09-08/whole-body-ungrip-development.json)

```sh
python scripts/dexterous/screen_whole_body_ungrip.py \
  --source-run /path/to/walked-whole-body-return-001 \
  --measured-release /path/to/measured-release-portable.json \
  --release-trajectory /path/to/release-v2-001/trajectory.npz \
  --output out/ungrip-screen
python scripts/dexterous/audit_whole_body_ungrip_screen.py \
  --source-run /path/to/walked-whole-body-return-001 \
  --screen out/ungrip-screen/report.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/dexterous/probe_walking_release.py \
  --source-run /path/to/walking-opening-native-002 \
  --verified-prefix-run /path/to/walked-whole-body-return-001 \
  --whole-body-path /path/to/controlled-return-whole-body-path-001/report.json \
  --ungrip-path out/ungrip-screen/target-plan.json \
  --release-mode whole-body-ungrip --seconds 82 --output out/ungrip-repeat
python scripts/dexterous/audit_walking_release.py --run out/ungrip-repeat
```

The fixed-frame option is the default only within this new development helper;
`AxialRightRelease` remains unchanged. Use `--ungrip-goal-frame measured-handle`
to reproduce the rejected moving-frame option. These exact-state native paths
must not be admitted to another robot, attained state, or Isaac import without
a new geometric screen and physical trial. No sensor-only actor or traversal
success is claimed.

## Full palm orientation fixes the gross support mismatch

The next single-change comparison retains the same right-hand/body route and
original total-load feedback. It binds the **full attained left-palm rotation**
at episode 60s. The previous arm IK constrained position and palm normal, leaving
a free twist about that normal; actual FK found roughly 10° twist during the body
adjustment, although the normal error stayed below 0.53°. The geometry screen
had preserved the full rotation, so that was a controller/screen mismatch.

The optional `palm_orientation_error` retains the original normal residual when
no full rotation is supplied. `bind_attained_palm_orientation` reads current
measured state in the unstepped FK model and binds the rotation relative to the
leaf. `--hold-full-left-orientation` applies this one change in the isolated
probe; it does not change the shared default, contact-force target, or motors.

Actual 2 ms load evidence over 61–67s:

| Left orientation target | Mean palm load | Minimum | Zero-load samples | Samples below 2 N |
|---|---:|---:|---:|---:|
| Position + normal (002) |0.788N|0N|2,319/3,000|2,511/3,000|
| Full pose (003) |3.529N|1.766N|0/3,000|38/3,000|

The remaining 38 low-load samples still fail the frozen gate. Total hand load
averages 4.053 N while palm load averages 3.529 N, so total-load control does not
match the scored palm-load objective exactly. No threshold is relaxed.

Trial 003 also remains a failed release/opening trial: six distal-tip contacts
occur at 68.576–68.674s. Later, while the long withdrawal delays panel continuation,
the left arm reaches its workspace limit; 76 joint-stop samples at 70.758–70.926s
reach 59.79 mrad. Right clearance is already above 20 mm around 69 s, but this tested
route waits until 71.584 s to finish. A new shorter route must be densely screened
and end with the fingers open and all right-hand collision shapes at least 40 mm
from scene geometry before a fresh physical test. This does not reclassify 003.
[Comparison and archive receipt](../results/dexterous/2026-09-08/left-palm-orientation-development.json)

## Short withdrawal and panel-force diagnosis remain separate failures

The next declared geometry ends at 69.8 s, with the fingers fully open and a
40 mm minimum clearance requirement across **all** right-hand collision shapes
and scene colliders. The `clearance-lift-v4` screen follows the measured source,
adds an early 2 mm lift, then finishes with 6 mm away from the strike, 16 mm back
and 40 mm up. Dense FK gives 44.00 mm minimum clearance; the actual attained
endpoint gives 49.55 mm and remains balanced. `ready_for_panel` stays false until
the entire route finishes, and actual 20 mm release evidence remains required.
These distances are geometric release checks, not an anatomical qualification.

Physical trial 004 still fails: **nine middle-finger wrong-pad contacts** occur
at 68.576–68.590 s. Their small forces do not waive the unchanged 0.5 volar-normal
criterion. All earlier failures and the separate cumulative anatomy gate remain.
The first 134 raw chunks, through 67 s, exactly match trial 003.

The later plain-v1 panel continuation fails for a different reason. The actual
500 Hz archive contains a **137.35 N palm pulse at 70.658 s**, which the 50 Hz
summary missed. Total kinetic energy grows from about 0.2 J at 70.1 s to 15.5 J
by 70.9 s; left-arm motors perform +9.60 J net work during 70.8–70.9 s. The cup
motor uses only roughly 0.004–0.08 Nm of its original 1 Nm cap while contact
torque reaches −0.8 Nm during this growth. In this shorter trial its minimum
angle is −10.36 mrad, within the frozen 20 mrad tolerance. The largest late
violation is instead **right hip yaw, 39.27 mrad at 71.33 s**, after balance loss.
In the earlier long trial 003, the 59.79 mrad violation was **left LFJ5**, not the
main arm; its wrist excursion was only 8.54 mrad.

`audit_panel_energy.py` uses archived actual `mj_step` forces and pre-state
Jacobians. It never solves replacement contact forces. Motor transmission and
body-frame reconstruction agree exactly with the archive. Signed work uses
force × pre-state velocity × 2 ms; it is a diagnostic quadrature, not a complete
energy-conservation test. The implementation follows MuJoCo's documented
[contact-frame wrench API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce)
and [actuator transmission](https://mujoco.readthedocs.io/en/stable/computation/index.html#transmission).

The existing `hybrid-surface-v2` profile already removes opposing normal servo
acceleration, uses the actual palm support point, blends control over one second,
doubles damping, and strengthens cup feedback within its original 1 Nm cap.
The next panel comparison should use that explicitly declared profile after
the release error is corrected. The attained body target may still need a new
panel workspace plan; the evidence does not prove hybrid control will pass.

```sh
python scripts/dexterous/audit_panel_energy.py \
  --run /path/to/walked-whole-body-ungrip-004 \
  --output out/panel-energy-audit
```

[Frozen failed report, interval mechanics and verified archive](../results/dexterous/2026-09-08/short-release-panel-development.json)

For the next fresh comparison, `--early-lift-m .006` increases only the early
geometric lift within `clearance-lift-v4`; omission preserves 2 mm. The parameter
is limited to 0–10 mm and recorded in the geometry receipt. It must pass the
same dense anatomy/collision/foot/endpoint screen before execution. The replay
probe also accepts `--panel-profile hybrid-surface-v2`, which selects the existing
frozen controller implementation and records the change from baseline. Omission
preserves the baseline profile. These are experiment controls, not claimed fixes.

The fresh 6 mm trial 005 removes the nine MF contacts but retains three LF/FF
wrong-pad contacts across two intervals, so the withdrawal is still rejected.
Its hybrid panel attempt also fails. Source review identified a separate target
ownership bug: the finished ungrip helper rewrites torso and RH-finger targets
at 500 Hz, while panel IK supplies its new targets at 100 Hz. The archived torso
force exhibits the corresponding 10 ms spikes and later reaches its original
200 Nm cap. Merely choosing hybrid contact control cannot correct this conflict.

`handoff_posture_targets=True` (probe `--handoff-posture-targets`) now relinquishes
those posture targets after the complete route has actually been frozen by the
clearance gate. The release still preserves its cleared palm target and maintains
zero grip preload; panel IK owns subsequent waist/cup targets. A regression runs
interleaved 500/100 Hz updates and verifies the target persists between panel
updates. Omission retains the failed historical behavior for reproduction.
This source correction requires a new physical comparison and does not promote
trial 005 or change any original motor, joint, anatomy, or contact gate.

**Controlled null result:** trial 006 with that ownership option is bit-identical
to 005 across all 143 raw chunks through 71.410 s. `AcquisitionTeacher` caches its
consumed motor targets at the same 100 Hz phase as panel IK, so the intermediate
500 Hz path writes did not change this fixture's forces. The ownership cleanup
is not an explanation or physical fix for the observed spikes.

The next explicit comparison, `--hybrid-include-waist`, projects the actual
eight-motor waist + left-arm command after ordinary force assembly. Panel IK
changes all eight joints, whereas the previous normal projection included only
the seven arm motors. The new projection uses the complete 8×8 analytic mass
block and eight-entry support-point Jacobian. The original 200 Nm waist cap and
all seven arm caps remain active. A runtime assertion verifies that the later
walking leg-force override leaves these eight commands exactly unchanged.

`--record-panel-targets` records the consumed motor targets, nominal IK steps,
actual joint positions/velocities, task-Jacobian singular values and assembled
forces at every 2 ms interval. The eight-joint option enables this record
automatically. These are privileged diagnostic records, never actor inputs.
The experiment must still distinguish infeasible/sudden targets from contact
control effects and retain the independent LF/FF release failure.

The paired physical experiment is complete and **both trials remain failed**.
Instrumented seven-joint trial 007 reproduces 005/006 exactly; eight-joint 008
preserves the physical prefix until panel intervention but also loses balance.
Independent alignment verifies all 803/808 recorded panel intervals against
actual archived motor forces with exactly zero force or clock disagreement.

The consumed seven-joint targets reach 0.120 rad waist and 0.242 rad shoulder-yaw
changes in one 10 ms update: about 12 and 24 rad/s, with shoulder target
acceleration above 1,500 rad/s². The eight-joint version still demands up to
20.5 rad/s. At the onset the weighted task Jacobian condition is only about
12–16; very large condition numbers appear later as the robot falls. Therefore
the large late condition number alone does not explain the first impulses.
The old angle-driven 15 cm palm-lowering path reaches a rapidly changing IK
region in the attained stance. A new destination-state whole-body path with
explicit target velocity/acceleration bounds is needed before another physical
opening claim. Neither projection nor filtering can establish that feasibility.

```sh
python scripts/dexterous/summarize_panel_targets.py \
  --run /path/to/walked-whole-body-ungrip-008 \
  --output out/panel-target-audit.json
```

[Controlled results, exact command audit and verified archive](../results/dexterous/2026-09-08/panel-chain-comparison.json)
