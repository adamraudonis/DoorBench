# Bimanual contact-transfer development

The continuous native teacher now passes **20 physical gates** from a contact-free
deep-stance start through loaded **1.201 rad** opening. The extracted controller
repeated the result with identical states and controls. Walking into that start,
traversal and Isaac opening remain separate, unqualified steps. Earlier static
candidates and physical failures are retained below; the current result and API
are in the final section.

A later timing audit found that the earlier native driver combined derived
force-evaluation body poses with just-integrated joint positions. Its actual
opening/control evidence is preserved, but synchronized ground-truth claims
require the separate coherent-measurement FullOpeningTeacher trial below.

## Measured-release candidate: screen-014

The fix reuses the actual 0.9–7.5 s motion from the separately qualified native
`release-v2-001` experiment. It transports measured palm motion through the
initial palm/handle frame and retains measured finger-angle changes. This
preserves the new grasp's actual axial offset. It does not replace the hand
with an arbitrary zero-angle pose.

All **445 sampled waypoints** pass the screen. Maximum adjacent arm motion is
0.0834 rad; maximum static arm demand is 40.76% of the original motor cap.
The released right hand clears the lever by **45.69 mm**. The nearest left
geometry is the actual palm, approximately 0.472 mm from the slab. This gap is
geometric proximity, not measured contact load. A physical controller still
needs to establish left-palm loading and maintain it through opening.

Run the command below with the additional argument:

```bash
--measured-release /path/to/qualified/release-v2-001
```

The source model hash and physical release receipt must match. The output also
contains `measured-release-portable.json`: source times, finger deltas and palm
poses in the handle frame, with source hashes. `relocate_release()` applies the
same relative motion to an actual reached grasp pose. Small sampled finger
target corrections restore original joint bounds and legal loopback ordering;
the maximum correction is 16.23 mrad and is recorded in the report. Those targets need new
physical qualification.

See [the measured-release screen report](../results/dexterous/2026-09-08/bimanual-v2-measured-release-screen.json).
Close views of the grasp and release were inspected. The next gate is continuous
unchanged-motor execution, starting from the actual achieved partial-opening
state. Contact loading, motor tracking, dynamic balance and actual aperture
remain unverified for this combined candidate.

The archive `DoorBench-runs/2026-09-08-robust-opening/bimanual-measured-release-001.tar.gz`
retains screens 012–014, measured inputs, close views and source. SHA-256:
`2f2a3ec8a82104dd6ac4fd4f17af69fcad48b65f3d3cff87db7eef8cb89e2a54`.

## Earlier rejected candidate: screen-011

The sequence places the left palm on the closed leaf, uses the qualified right
grasp to rotate the operator and open the leaf to 0.08 rad, releases the right
hand, then tracks the leaf with the left palm to 1.2 rad. The left contact point
is 0.18 m from the hinge, 0.85 m above the floor. Robot geometry, mass, motor
limits, free-base pose and documented v2 loopbacks are preserved in the screen.

| Sampled phase | Frames | Collision failures |
|---|---:|---:|
| Left hand approaches | 41 | 0 |
| Right hand rotates operator | 9 | 0 |
| Both hands begin opening | 9 | 0 |
| Right hand releases | 41 | **4** |
| Left palm continues to 1.2 rad | 56 | 0 |

The rejected release penetrates the handle by up to **11.97 mm** while the
fingers unfold. Close hand views were inspected and agree with this failure.
The other sampled stages pass pose accuracy, hand-to-panel proximity, the
original arm-force reserve, all eight loopback inequalities, and waypoint
continuity. The maximum calculated arm demand is 40.76% of its original cap.
The independent finite-foot support problem remains solvable, but this is
not a dynamic balance or force-transfer result.

Full evidence is summarized in
[bimanual-v2-static-screen.json](../results/dexterous/2026-09-08/bimanual-v2-static-screen.json).
The planner never advances a physical robot: it sets hypothetical qpos only
inside its separate FK/inverse-dynamics analysis instance. A physical teacher
must command original bounded motors and earn the motion through contacts.

## Reproduce the sampled candidate

Use the final versioned v2 robot plus the **grasp-only** reference extracted from
the qualified grasp endpoint:

```bash
PYTHONPATH=. python scripts/dexterous/plan_bimanual_panel.py \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --door assets/doors/db0055_swing_single \
  --reference /path/to/canonical-grasp.json \
  --radius .18 --height .85 --transfer-angle .08 \
  --reach-pitch-bump .6 --reach-roll-bump .15 \
  --output out/bimanual-new
```

Each new run saves its source, input reference, robot XML/audit, hashes, full
candidate qpos and per-waypoint diagnostics. Use a new output directory for
every attempt. The robot XML may refer to meshes outside the XML directory;
retain the original robot package for reproduction.

Early screens 001–006 incorrectly used an acquisition container's open-hand
`initial_joints` instead of its qualified final grasp. Those results are
retained as invalid-initialization development evidence. The planner now
rejects acquisition containers. Screens 007–011 use the exact legally projected
grasp-only reference; they are still failed route candidates. The successful
physical acquisition's realized final state should replace this ideal endpoint
before an executed sequence is attempted.

The local archive contains all eleven attempts, endpoint searches, inspected
close views, inputs, and source snapshots:

`DoorBench-runs/2026-09-08-robust-opening/bimanual-development-001.tar.gz`

SHA-256: `f7d1c1421f6e64bd627b0c3495925b97ae41283ae69a7c9170d11e0471e20a71`.
Assets and rendered frames remain outside Git.

## Remaining gates

1. Find a collision-free release under the documented passive finger coupling.
2. Replay from the actual reached acquisition state through unchanged motor
   dynamics. Require loaded left contact before right release and sustained
   contact-driven travel; an already-open panel brushed after coasting fails.
3. Preserve every-2-ms joint, loopback, motor, nonfoot contact, no-assistance,
   upright and fingertip-pad checks. Measure actual final aperture.
4. Repeat in Isaac only after native execution passes. Static FK, a solvable
   stance problem and a plausible rendered pose are insufficient evidence.

## Continuous native left contact: 2026-09-08

`contact-006` passed all 18 gates over **34 uninterrupted seconds** with the
corrected Shadow v2 mechanics. Starting with the existing contact-free deep
acquisition, it physically pressed the lever, opened the leaf to 0.08953 rad,
and maintained approximately 4 N of left-palm panel load while the right-hand
five-distal-pad grasp remained valid. The every-2-ms audit checks all original
motor caps, individual joint and unilateral loopback limits, upright stance,
non-foot penetration and absence of external assistance. No invalid right pad
patch occurred during operation or contact transfer (five brief digit-unload
samples occurred earlier during lever operation). This is a contact stage,
not a usable-aperture opening or traversal result.

The original perpendicular palm near the hinge required a waist turn. Physical
trial 004 reached the left contact but lost right distal pads during that turn.
Trial 005 added fixed-pad force feedback and also failed an individual joint
limit. Both failures are archived. The successful hypothesis keeps the waist
under the unchanged right-hand controller, reaches 0.22 m from the hinge at
1.0 m height, and permits a roughly 24-degree left-palm tilt. I inspected close
views of both actual simulated hands and the complete body pose.

The portable target configuration is
[`bimanual-left-contact-v2.json`](../configs/dexterous/bimanual-left-contact-v2.json).
The measured-state interface is in
[`bimanual_transfer.py`](../doorbench/dexterous/bimanual_transfer.py):

```python
names, targets = load_screen_targets(config, robot_xml, door_dir)
left = LeftPalmContact(acquisition_teacher, motors, (names, targets), fixed_waist=True)
# Begin only after actual strict RH pad hold and partial opening are qualified.
left.begin(t, root_state, joint_positions, leaf_pose, handle_pose)
# At every 2 ms tick, before the acquisition/operation teacher motor call:
left.update_targets(t, root_state, joint_positions, leaf_pose,
                    measured_left_panel_load_N, handle_pose)
forces, info = operation_teacher.force(...)  # consumes current measured state
forces = left.apply_forces(forces, joint_positions, joint_velocities)
```

`root_state` has 13 entries: world position, **wxyz** orientation, world linear
velocity, world angular velocity. Body poses have seven entries in the same
position/wxyz convention. Named joint dictionaries use the imported motor
contract's complete joint names, radians and rad/s. Left normal load is the
actual force of hand-to-panel contact projected onto the panel normal. The
reported loaded duration requires at least 2 N; the target tracks about 4 N.
Only original left-arm/wrist motor forces are overridden; the analytic FK
model is never stepped. The adapter is privileged teacher control, not actor
observations. Isaac transfer, right release and loaded substantial opening
remain unverified.

Reproduce the native stage with `probe_bimanual_contact.py`, supplying the v2
robot, imported motor JSON, qualified acquisition reference, Door55 directory,
`--plan configs/dexterous/bimanual-left-contact-v2.json`, and a fresh `--output`.
The numerical receipt and complete evidence hashes are in
[`native-bimanual-contact-v2.json`](../results/dexterous/2026-09-08/native-bimanual-contact-v2.json).
The source archive includes the endpoint scans, failed trials, exact executed
sources, 50 Hz full-state trajectories, 500 Hz audit rows, and close-view images.
The third trial's final serialization failed before its full evidence was saved;
its remaining source and progress logs are preserved without a full-gate claim.

### Robot identity when moving to another cluster

New target receipts also carry `doorbench.compiled-robot-identity.v1`, generated
by `robot_identity.robot_file_identity`. The reader verifies that identity even
when the XML SHA is unchanged, so an altered mesh behind the same filename does
not silently pass. Older target files retain their exact XML SHA requirement.
The identity includes 327 compiled physical arrays for this H1: collision mesh
vertices/faces, body mass/inertia, joint and motor properties, transmissions,
passive tendons, named bindings, and the native solver settings. It pins the
MuJoCo version, normalizes floating arrays to 12 decimal places and integers to
64-bit little-endian encoding, and excludes asset paths, memory layout and
render-only assets. An identity match is not an Isaac runtime parity claim.

Only the version-pinned built-in `mujoco.sensor.touch_grid` plugin is accepted,
with its compiled sensor arrays and expanded XML configuration included. It
must be attached exclusively to sensors. Other plugins fail closed pending an
explicit identity contract. MuJoCo documents this plugin as a contact-force
sensor, not a physical actuator:
[official plugin source and documentation](https://github.com/google-deepmind/mujoco/blob/main/plugin/sensor/README.md).

The local relocation check recompiled the real H1 after rewriting every asset
location to a different symlink path and obtained the same identity. Tests
reject changes to geometry, masses, motor caps, passive tendon limits, solver
settings, joint names and sensor configuration. Cross-operating-system
compilation still needs a live check with the pinned MuJoCo version; a mismatch
must be investigated rather than bypassed or assigned a fresh expected hash.

The identity compile explicitly clears MuJoCo's compiler asset cache first.
A regression test exposed that two edits to one mesh path within the filesystem
timestamp granularity could otherwise reuse cached geometry. Clearing the cache
changes no existing model or simulation state, and the test now rejects a mesh
change while the XML bytes remain identical.

## Axial release and rejected opening trials (2026-09-08)

The new native `axial-release-001` completes the right-hand release without
invalid distal pad contact patches, joint-limit violations, extra forces or
collision violations. It relaxes the active grip preload over 0.4 seconds,
retains the attained legal finger targets, and slides 140 mm toward the lever's
free end over six seconds. The palm follows the measured handle frame as its
spring returns. A measured 20 mm all-right-hand collision-geometry clearance
freezes the retreat goal. This avoids the elbow limit encountered by the
previous straight-back withdrawal. These are privileged teacher measurements.

The full 42-second episode still **fails** loaded substantial opening: the
left palm unloads after about 0.178 rad. Its final 0.882 rad angle includes
coasting and is not an opening success. The receipt is
[`native-bimanual-axial-release-v2.json`](../results/dexterous/2026-09-08/native-bimanual-axial-release-v2.json).
It points to the complete byte-verified local archive and records the unchanged
full-episode failure alongside the useful physical release primitive.

`doorbench.dexterous.right_hand_release.AxialRightRelease` exposes the isolated
release controller. Call `begin(t, named_joints, root13, handle_pose7)` only after
qualifying another loaded physical contact. Set its measured `handle_pose`
each tick, then call `update(t)` before the acquisition/operation teacher's
`force(...)`. The initializer and updates touch only the teacher's analytic
model and target arrays; they never step it or write the plant. Its caller owns
the actual clearance check and may set `frozen=(world_palm_position,
world_palm_rotation)` once every right-hand collider clears the lever by 20 mm.
The compact screen in `configs/dexterous/bimanual-axial-release-v2.json` is for
the qualified Mac runtime only; it does not waive a different runtime's
geometry check. The native run used the same numerical release formula; the
extracted module adds finite-input and rotation validation. Six focused tests
check rotated handle frames, smooth endpoints and invalid inputs. The extracted
module itself still needs an uninterrupted integration repeat before an Isaac
transfer claim.

`native_hand_surface_loads` separately records actual **palm-only** force and
other hand link forces. Two physical tests check both geometry orders. The old
`left_panel_load_N` sum cannot establish a palm contact on its own.

The first coordinated full-palm continuation reached 0.7 rad with 12.5 N of
actual palm load, but its full 500 Hz trace exposes earlier joint-limit
violations and intermittent unloading. It remains rejected. The follow-on
trial retains the independently successful release behavior until measured
clearance before beginning the torso turn; this is development, not a claimed
complete opening, traversal, natural human reference or sensor-only policy.

Cross-OS identity verification also caught a real portability limit: the same
v2 assets and MuJoCo 3.12 compiled to Mac identity `18191db3…` and Linux
`98c0fdc7…`. Mesh topology/BVH and related derived arrays differ. Exact compiled
identity rejection remains enabled. A path-independent source-design identity
and runtime-specific geometry rescreen are being added separately; changing a
hash without those checks is not a valid port.

## Destination-runtime contact rescreen

The Linux MuJoCo 3.12 rescreen passed on the owned node. All 1,700 recorded poses
from contact006 were recompiled and checked against the actual destination
robot and Door55 geometry: maximum joint violation 10.13 mrad, loopback
violation 0.644 mrad, and nonfoot penetration 0.410 mm. No simulation steps were
executed. This is 50 Hz sampled geometry compatibility, with no interpolation
or destination dynamics claim. The complete source-design receipt matches
across OS while the distinct compiled identities remain recorded.

`load_screen_targets(..., runtime_screen=receipt_path)` now accepts a different
compiled runtime only if its authored source design matches and a passed
rescreen binds the exact target-config bytes, original measured trajectory,
source compiled identity, destination compiled identity, MuJoCo version and
door XML. Linux testing confirmed rejection without the receipt and acceptance
with the exact receipt. Eleven tests reject changed inputs and incomplete or
failed evidence; the original six target-loader tests also pass.

Reproduce on a prepared destination, with the original archived contact006
trajectory and expected source-design JSON available:

```bash
python scripts/dexterous/rescreen_bimanual_contact.py \
  --robot "$ROBOT_XML" --door "$DOOR_DIRECTORY" \
  --trajectory "$CONTACT_006_TRAJECTORY" \
  --target-config configs/dexterous/bimanual-left-contact-v2.json \
  --expected-design "$EXPECTED_SOURCE_DESIGN_JSON" \
  --output out/contact-runtime-rescreen.json
```

The expected source-design JSON is also embedded as `source_design_identity`
in the frozen target configuration. The measured trajectory hash is pinned
there too. A new robot, changed mechanics or different door cannot reuse this
receipt. After static compatibility, run the uninterrupted physical contact
primitive on the destination engine; that remains a separate gate.

## Earlier loaded-aperture result: full-push005 and portable repeat

The original runs below mixed just-integrated joint coordinates with cached
body poses and contacts. Their reported gates and actual trajectory are retained,
but they do **not** establish a synchronized sensor/controller qualification.
The unified teacher therefore receives explicit current measured palm poses,
and its new native driver refreshes derived measurements after every step.
MuJoCo documents that `mj_step` updates the state last, leaving derived quantities
from the preceding dynamics calculation; `mj_forward` refreshes those quantities
without integrating time. See the [official simulation pipeline](https://github.com/google-deepmind/mujoco/blob/main/doc/computation/index.rst#simulation-pipeline).

The complete native episode now starts with a closed leaf, resting lever,
contact-free hands and the qualified deep stance. Without further pose resets,
it acquires the lever, depresses it, opens the leaf about 0.08 rad, places the
left hand on the panel, slides the unloaded right grip off the lever, and turns
the waist while pushing with the left palm. The episode ends at the **first
crossing of a declared 1.2 rad aperture**, with all prior physical samples
included. It does not continue past that threshold to select a convenient
later result. This primitive does not hold the door indefinitely at 1.2 rad.

Both prototype005 and the extracted-module repeat pass **20/20** gates. They
produce bitwise-identical recorded qpos, qvel and motor commands; every 2 ms
physical/contact measurement also matches. The repeat includes a newly
explicit `grasp_profile=distal-pad-v1` metadata label with unchanged strict
contact tests. The full record contains 18,369 samples over 36.736 seconds.

| Measurement | Result |
| --- | ---: |
| Final actual leaf aperture | 1.200999 rad |
| Actual final palm load | 3.373 N |
| Minimum palm load in final 0.5 s | 2.027 N |
| Leaf angle across that loaded interval | 0.9525 → 1.2010 rad |
| Maximum torso tilt | 1.861° |
| Maximum joint soft-limit violation | 11.353 mrad |
| Maximum unilateral loopback violation | 1.903 mrad |
| Maximum nonfoot penetration | 0.453 mm |
| Invalid right-hand pad patches | 0 |
| Runtime robot pose writes / helper forces | 0 / 0 |

The successful change allows the released right arm to reorient while the
waist turns. A clearance term keeps its colliders away from the lever, and its
wrist targets retain extra distance from the joint stops. Freezing that arm's
joint posture caused recontacts and wrist overshoot; freezing its full world
pose consumed the workspace needed by the pushing left arm. Both failures are
preserved. The left hand retains its full panel-relative orientation while its
contact location moves from roughly 1.0 m high / 0.22 m from the hinge toward
0.85 m / 0.18 m. Bounded load feedback maintains palm pressure using only the
original robot motors.

Personally inspected close views show the acquired four-finger grip and the
open left palm against the panel at the final aperture; body views show the
upright torso and deep starting stance. These views and the full source, state,
500 Hz audits and failed alternatives are in the byte-verified archives listed
in [`native-loaded-bimanual-opening-v2.json`](../results/dexterous/2026-09-08/native-loaded-bimanual-opening-v2.json).
No generated images or model assets are committed.

### Measured-state integration

`LeftPalmContact` remains active alongside the acquisition/operation teacher.
After its actual load and the retained right grasp qualify, instantiate
`AxialRightRelease(teacher, axial_screen_path)` and call `begin(...)`, as above.
Each physics tick supplies the actual current handle pose to `release`, updates
its targets, calls the base teacher's bounded `force(...)`, and applies the
left-arm motor override. Determine clearance from **all actual right-hand
collision shapes**, not a fingertip centroid or the commanded pose.

Once all right-hand shapes clear the lever by 20 mm, freeze the release's
measured world-palm goal and start:

```python
push = CoordinatedPanelPush(left)
push.begin(t, root13, named_joints, leaf_pose7, actual_leaf_angle)
# Every tick, before base teacher.force(...) and left.apply_forces(...):
push.update(t, root13, named_joints, leaf_pose7,
            actual_palm_normal_load_N, actual_leaf_angle, right_clear=True)
```

`CoordinatedPanelPush` lives in `doorbench.dexterous.panel_continuation`. The
13-vector is measured world position, wxyz quaternion, world linear velocity,
and world angular velocity; the leaf pose is xyz+wxyz. It operates on the
teacher's analytic model only, solving at 100 Hz while original motor commands
and physical audits run at 500 Hz. Robot state, exact door frames, contact
identity and geometry clearance are **privileged teacher/evaluator data**.
They must not enter actor observations or sensor-only runtime routing.

`scripts/dexterous/probe_bimanual_full_push.py` reproduces the complete native
qualification and captures the actual controller tree and local module
sources. Its `--target-aperture` defaults to 1.2. It preserves the original
joint, loopback, force, upright and collision bounds. The measured-source
archive contains all required reference and import contracts.

The validated start is the original deep stance, not the different root/yaw
attained by the separate walking approach. That attained state needs a fresh
workspace and physical check. Isaac transfer, left release, rising, walking
through the aperture, robustness across starts and sensor-only imitation remain
unqualified. Do not score any of those from this result.

## Unified measured-state teacher: development qualification

`doorbench.dexterous.full_opening_teacher.FullOpeningTeacher` now owns the
continuous acquisition, lever operation, qualified partial opening, left contact,
axial right release and coordinated panel continuation. It accepts numeric
measurements and returns exactly 61 forces clipped to the original motor caps.
Its owned model is an unstepped calculator. No active simulator is accepted.

```python
teacher = FullOpeningTeacher(
    robot_xml, motors, reference, joint_geometry,
    door_xml=door_directory, left_targets=left_target_config,
    release_screen=axial_release_config,
    runtime_screen=destination_geometry_receipt,  # when required
    target_aperture=1.2,
)
forces, info = teacher.force(
    t, root13, named_joints, named_velocities, handle_pose7, leaf_pose7,
    dict(operator=handle_angle, leaf=leaf_angle, latch=latch_displacement),
    hand_forces_world, evidence=measured_evidence,
    right_palm_pose=measured_right_palm_xyz_wxyz, pose_time_s=t,
    contact_interval_s=(max(0., t - physics_dt), t),
)
```

`joint_geometry` contains the compiled local `operator_origin`, `operator_axis`,
`leaf_origin`, and `leaf_axis`. `root13` and body poses use the conventions above.
The exact evidence dictionary is `grasp_qualified`, `physics_qualified`,
`right_pad_patches_valid`, `hand_contact_count`, `left_panel_load_N`,
`left_palm_load_N`, and `right_lever_clearance_m`. The first three values are
actual booleans; loads are measured newtons and clearance is the actual signed
minimum across every right-hand collider. These are privileged teacher inputs.

The consumer supplies current poses/joints and explicitly dated forces from the
last completed physics interval. See [native timing](NATIVE_TRANSITION_AUDIT.md)
for the actual-transition recorder; an extra `mj_forward` force solution cannot
substitute for the preceding physical interval. Repeated timestamps,
stale poses, opaque objects and unexpected evidence fields reject. Missing
samples cancel the sustained-contact window. A sample count cannot replace the
required half-second interval. `info` timestamps the current mechanism/evidence
measurements and separately identifies the left controller's last 100 Hz update.
The following force acts on the subsequent physics interval; it must not be
reported as though its input measurements were taken after that interval.

Native reproduction uses `scripts/dexterous/probe_full_opening_teacher.py` with
the same `--robot`, `--door`, `--reference`, `--motors`, `--plan`,
`--release-path`, optional `--runtime-screen`, and a fresh `--output`. It records
every 2 ms contact/physics sample, 50 Hz states, exact inputs and local sources.
`--target-aperture` is independent of the controller's gains; 1.5 rad or another
target requires a new physical run and must not inherit the 1.2 rad result.

The synchronized controller is still under qualification. Coherent001 reached
1.20114 rad but violated the left LFJ5 joint bound and lost continuous palm load.
Coherent002 corrected the joint bound but lost final contact and acquired an
invalid right-hand patch during axial withdrawal. Coherent003 retained correct
right anatomy and finished with 6.97 N palm load, but failed joint and continuous
load gates. The original 20 gates remain unchanged. Exact failures and sources
are preserved in the byte-verified archives referenced by
[`full-opening-teacher-development.json`](../results/dexterous/2026-09-08/full-opening-teacher-development.json).
Do not score a completed opening from these development runs.

The operation controller also exposes explicit engine-transfer options:
`open_on_latch_clear=True` permits the leaf trajectory once the **actual** handle
has reached 0.8 rad and the latch has retracted 11 mm, without waiting for the
nominal press schedule. `operator_compliance_gain=.5` integrates measured handle
tracking error into a palm-reference rotation, clipped to 0.15 rad and frozen at
that measured release. Both options reproduce the separately qualified Isaac
operation controller's settings. They remain **off by default** in the native
full-opening reproduction, and do not change any door state, joint limit or
motor cap. The CLI equivalents are `--open-on-latch-clear` and
`--operator-compliance-gain .5`. Porting these options into the full-opening
controller does not establish an Isaac full-opening pass.


The latest development continuation replaces the left arm servo's normal
acceleration component using its actual analytic mass matrix and Jacobian, then
commands a bounded 0–12 N equivalent normal effort through the original motors.
Tangential/rotational tracking and all original caps remain active. A 20 ms load
filter is controller-only; gates use every raw physical load sample. Its flat
palm target includes actual collision-mesh support-plane compensation and a
30 mrad extra wrist margin. This is an unqualified controller revision until a
complete actual-transition run passes; it must not inherit legacy trial scores.

The controlled comparison is explicit: `panel_profile="plain-v1"` selects the
original 3 N target, 12 mm offset bound, 2 s left cup and original left-arm
damping, without the later hybrid/flat-palm controller or LFJ5 impedance change.
`palm_load_target=8` changes only its desired physical load (accepted range above
2 N through 10 N); the qualification floor remains 2 N at every physics tick.
CLI options are `--panel-profile plain-v1 --palm-load-target 8`. The development
`hybrid-surface-v2` profile adds a one-second control-mode blend and the actual
palm collision support point in its normal Jacobian. Neither profile is qualified.

The plain actual007 comparison passes all mechanical/anatomical gates and reaches
1.200269 rad with 6.23 N final palm force, but 92/251 final hold samples unload
below 2 N. Increasing only the desired load to 8 N in actual008 reduces that count
to 18/251 (final force 13.67 N), while original joint/collision/motor/anatomy gates
remain clean. These are recorded failures, not completed opening references.
`scripts/dexterous/summarize_native_opening.py RUN` computes these summaries from
the preserved every-step evidence without modifying the original pass/fail report.

The 10 N comparison actual009 worsens unloading to 49/251 samples; pressure
escalation stopped. An exact-state/control convergence study of actual008 finds
only 17.55 µm maximum palm separation and 2 ms unsupported intervals, while
every rolling 10 ms window still carries at least 5.85 N average palm load.
Finer 1/0.5 ms integration preserves the integrated load and aperture. The
[contact investigation and proposed support protocol](DEXTEROUS_PALM_SUPPORT_PROTOCOL.md)
records the evidence and validation needed before any new acceptance profile.
Actual008 remains failed under its frozen 20/22 score.

### Attained-state panel workspace and bounded target path (development)

The frozen seven-joint and eight-joint force-projection comparisons both failed.
Their actual consumed targets demanded shoulder speeds up to24rad/s. A fresh
whole-body geometric route therefore replaces the old15cm downward palm motion.
The new route starts from native005's exact attained state at69.804s, holds the
cleared right hand and both attained foot frames, keeps left-palm height, and
moves its contact point4cm toward the hinge as the leaf opens to1.2rad.

`scripts/dexterous/screen_whole_body_panel.py` owns an unstepped model. The
101-node screen permits at most30mm per root translation coordinate and0.03rad
per rotation-vector coordinate. These are planning bounds, not a demonstrated
physical capability. The source's9.65microradian right-elbow solver excursion
is admitted only at the exact initial state; the screen then regains its original
joint margin. It never changes the plant's joint limits. Screen004's first
cubic interpolant overshot a wrist limit by72microradians and remains a failed
geometric candidate. Screen005 confines the temporary initial admission to
joints that actually need it and passes the independent dense screen.

`audit_whole_body_panel_screen.py` checks2,001 interpolated states against the
original collision geometry and compares all308 RH shapes with all64 scene
shapes (conservative enclosing-sphere pruning, followed by exact shape distance).
Screen005's maximum palm/foot errors are4.57/4.43micrometres; all RH shapes
remain at least40mm clear. Torso tilt stays below2.64degrees, root translation
below23.2mm, root rotation below0.0422rad, and COMxy displacement below5.47mm.
A20s quintic/C2 interpolation has maximum joint speed0.08895rad/s and
acceleration0.3145rad/s². This is **geometry and target-rate evidence only**.
The source has zero measured palm load and0.1453rad/s leaf velocity at this
instant, and the three previous wrong-pad contacts remain disqualifying.

The optional `screened_panel_teacher.py` candidate uses the existing original
motors and landed-foot stance controller. It admits only the exact screened
root, all69 joint positions, and aperture; it accepts no active plant handle.
A measured-aperture reference phase retains the actual initial leaf velocity,
limits reference speed to0.149rad/s and acceleration to0.08rad/s², and brakes
before its endpoint. The geometric path then supplies both body and arm targets
at the same current physics clock. An independent phase-envelope calculation
bounds joint speed/acceleration at0.1563rad/s and1.817rad/s², root speed at
16.82mm/s and root rotation-vector speed at0.02356rad/s.

The declared experimental force profile is `screened-position-v1`: existing
left-arm position control, doubled damping and original cup-joint feedback,
with the original3.5N Jacobian feedforward; no hybrid normal projection, no
new actuators, and no changes to caps. The existing cup angle is retained so
that the controller does not silently alter the screened collision geometry.
This remains physically unqualified until a new actual-force archive completes.
The optional probe argument is `--whole-body-panel-plan PATH`; historical
controller defaults and every actual anatomy/mechanics gate remain unchanged.

The first continuous physical comparison, `walked-whole-body-panel-001`, ran
95 seconds from the original walking reset and passed 26/29 frozen checks.
It remained upright and satisfied the original joint, loopback, collision-depth,
motor-limit and landed-stance checks throughout. It finished at 0.640815 rad
with 0.792383 N palm load, so the aperture and sustained-palm checks failed.
The original three RH wrong-pad contacts remain disqualifying. Independent
reconstruction checked all 47,500 transitions and 20,750 post-return contact/body
frames: maximum frame disagreement and motor-cap excess were both exactly zero.
The 139 byte-identical raw chunks preserve the old prefix through 69.5 seconds.
During the new panel stage, measured target joint speed and acceleration stayed
below 0.11390 rad/s and 1.35764 rad/s²; torso tilt stayed below 2.694 degrees.
This resolves the previous balance collapse for this declared experiment but
qualifies neither the hand release nor a complete opening.

`inspect_palm_support.py` reconstructs one exact archived pre-integration pose
and its actual interval wrenches, then renders the original collision meshes.
Six close-ups at 94.998 s were personally inspected. The palm normal is tilted
23.290 degrees from the panel normal. Its loaded corner is at palm-local
`[35.93, 45.43, -21.64]` mm; the little-finger middle contacts lie around
`[37.8, -94, -3.3]` mm. All support therefore lies along the same outer edge
of the hand. The actual loads are palm 0.792 N, little-finger knuckle 0.601 N,
and little-finger middle 1.131 N. The pressure at the palm centre cannot be
mistaken for broad face contact. The next declared geometric candidate rotates
the palm face toward the panel while preserving the support plane of every
original palm collision vertex. It keeps the same rate, force-cap, and
palm-specific load gates; it does not increase motor strength or relabel the
inherited RH failures.

Immutable local evidence lives in
`DoorBench-runs/2026-09-08-robust-opening/native-screened-panel-development-001`:
615 files; manifest SHA256
`b7fc9b955a85a18479435908893e95507e2733dff9686693bd75aaf35616839e`.
The preceding 31-file geometric archive is `native-attained-panel-workspace-001`,
manifest SHA256
`3ac423b7ccfe92cae7b6c133e387fb87658f6023bbc0f929e0cce8d76f2d14af`.
Working-tree `out/` links preserve these canonical files without duplicate raw
archives. Generated evidence is not committed to the source repository.

The flattened-palm candidate `attained-panel-screen-009` passes the unchanged
geometric and target-rate checks. It rotates the palm over the first 0.35 rad
of opening while preserving its original collision support plane; all finger
joint targets stay as before. Its 201 nodes and 2,001 dense samples keep RH
clearance >=40 mm, palm tracking error <=25.45 micrometres, foot error
<=2.08 micrometres, body translation <=28.55 mm and body rotation <=0.03446 rad.
Exact cubic-segment derivative extrema bound the measured-aperture reference at
0.1735 rad/s joint speed, 1.8865 rad/s² joint acceleration, 17.28 mm/s body
translation and 0.01924 rad/s body rotation-vector speed. The aperture phase
limits remain 0.149 rad/s and 0.08 rad/s². Intermediate screens006–008 retain
their original bounds or interpolation failures; no tolerance was enlarged.
The candidate is ready for a fresh physical comparison with the same original
force profile. The close-up tool and original actual-contact reports do not
synthesize forces from an unstepped geometric calculation.

The fresh physical comparison `walked-whole-body-panel-002` completed 95 s
and passed 27/29 frozen checks. Its sole geometric change was the screened
flattened-palm route; all force profiles and limits were held constant. It
maintained the required palm-specific support, stayed upright and passed every
mechanics check, but stopped at 0.542711 rad. The three inherited RH wrong-pad
contacts still disqualify the release. Independent reconstruction checked all
47,500 transitions and 20,750 post-return frames with zero disagreement and
zero motor-cap excess. Actual final palm load was 3.776147 N; the palm normal
was 2.981 degrees from the panel normal, and only the palm carried panel load.
This qualifies the support correction as a component result, not a full opening.

The hinge-moment audit uses the archived physical contact wrenches, their
recorded body side and contact frames, and the actual hinge axis. At 94.998 s,
normal palm load contributes +0.474277 Nm opening moment; tangential contact
reduces the net moment to +0.447349 Nm. The unchanged hinge frictionloss limit
is 0.457638 Nm, and leaf speed is approximately 1.13e-8 rad/s. The reference
has also settled exactly 5 mrad ahead of the door with zero reference speed.
These measurements support an insufficient-torque/tracking equilibrium; they
do not establish that reference phase alone caused the stall. The frictionloss
value is the original model limit, not a measured friction-constraint multiplier.
`audit_leaf_contact_moment.py` makes the wrench-side convention explicit and
has tests for tangential moments, side reversal, and global-frame invariance.

The next declared geometric comparison reduces the planned inward radial
slide of the palm from 40 mm to 20 mm. It keeps the same palm orientation,
force profile, target-rate bounds and all physical gates. This should increase
the opening moment arm without more force; the physical outcome remains
unqualified. Keeping the entire original radius was rejected by screens010–012
because their final RH orientation errors exceeded the unchanged 1 mrad screen.
Their failures are retained. No target path is admitted to a physical run until
the complete dense collision, joint, body and derivative audit passes.

Flat-palm002 and its actual contact close-ups, torque audit, and screens006–009
are preserved in `native-flat-palm-development-001`: 644 files, manifest SHA256
`c94d1068b510cc872242df3aab721ab77dad5ba4c55ff42520fdd3397394a5ca`.

Retained-radius screens013–015 passed their geometry checks but failed the
unchanged global target-rate envelope. Screen016 uses a smaller 10 mm increase
in contact radius (30 mm inward slide rather than 40 mm) and passes all 2,001
samples. Its exact worst-case envelope is 0.16472 rad/s joint speed,
2.61619 rad/s² joint acceleration, 18.402 mm/s root translation and
0.01736 rad/s root rotation-vector speed. The declared physical comparison
`walked-whole-body-panel-003` uses this admitted route with the original
3.5 N feedforward and every original cap and gate. It makes no correction to
the inherited right-hand withdrawal and cannot erase its three wrong-pad
contacts. No outcome is claimed before the actual-force trial and independent
archive audit complete.

Panel003 completed the declared 95 s and passed 27/29 frozen checks. All
mechanics and sustained palm support passed; the final leaf angle was
0.549978 rad with 3.718344 N palm load. The original three right-pad errors
remain, and usable aperture failed. The actual moment-arm correction increased
final normal moment to +0.498577 Nm and net moment to +0.457343 Nm, still below
the unchanged 0.457638 Nm hinge frictionloss limit. This confirms that the
realized moment increased but does not qualify a complete opening. The
independent audit verified all 47,500 transitions and 20,750 post-return frames
with zero force-cap excess or body-frame disagreement. The identical prefix
through 69.5 s occupies 139 byte-verified hardlinked raw chunks.

Panel003, torque series for baseline002, and all retained-radius screens010–016
are immutable in `native-panel-moment-arm-development-001`: 673 files, manifest
SHA256 `35d5f355a7901fdbd50fba2ddf10eaa7d4222bdc8896fa0a4b5d3cfb94c522af`.

The next controlled comparison, `walked-whole-body-panel-004`, returns to the
original qualified flat-palm002 path (screen009) and changes only normal
Jacobian feedforward from 3.5 N to 4 N. This is a declared 0.5 N target-force
increment motivated by the actual hinge-moment deficit; it changes no motor
cap or plant parameter. The probe records `--screened-panel-feedforward-n 4`
and the controller reports the consumed feedforward explicitly. Actual contact
force and hinge moment still require measurement; the configured 4 N value is
not a measured contact load. The default remains 3.5 N. All anatomy, support,
clearance, balance, derivative and collision gates remain unchanged, including
the cumulative inherited right-hand failure.
