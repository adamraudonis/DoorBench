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
