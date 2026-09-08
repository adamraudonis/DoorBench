# Bimanual contact-transfer development

The latest candidate passes **11 sampled FK/static planning gates**, including
right-hand release. It is not an executed teacher or a successful door-opening
reference. The earlier failed candidate is retained below.

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
