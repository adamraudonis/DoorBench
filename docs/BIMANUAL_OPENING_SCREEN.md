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
