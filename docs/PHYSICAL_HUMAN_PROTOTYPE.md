# One-door physical human reference

The stick figure grasps a lever with **four fingers on one side and the thumb
underneath on the other**, opens the door, releases its hand, walks through and
settles beyond the frame. Its body floats freely; joint motors and native
contacts produce the entire **28.58-second** sequence.

[Watch the full phone video](https://github.com/adamraudonis/DoorBench/releases/download/physical-traversal-20260907/doorbench-full-sequence.mp4)
· [Local replay](http://127.0.0.1:5184)
· [Download the native evidence](https://github.com/adamraudonis/DoorBench/releases/tag/physical-traversal-20260907)
· [Recorded checks](review/physical-human/prototype-checks.json)

This is an **engineered synthetic reference for one simple door**, separate from
the catalogue-wide benchmark. The gait is cautious, with bent knees and a partly
sideways passage. It is not captured human ground truth, a validated human-style
motion, or a robot-retargeting result.

| Opposing thumb during the press | Four fingers during the hold |
|---|---|
| ![Cyan thumb below the lever](review/physical-human/thumb-side.png) | ![Four fingers on the other side](review/physical-human/finger-side.png) |

The enlarged skeleton views hide the compliant contact envelopes. Those
surfaces remain active in physics. The video shows the whole body alongside a
hand close-up, followed by an overhead foot view. Adjacent wall wings are hidden
for inspection; their collisions were active throughout the recorded run.

## Measured result

Native **MuJoCo 3.12.0**, CPU; full run **September 7, 2026,
06:54:41–06:55:23 UTC**. The two causal controls use the same source, rig and
controller parameters, stopping after the opening phase.

| Measurement | Result |
|---|---:|
| Maximum door opening | **45.06°** |
| Opening with hand contact disabled | **0.00°** |
| Opening with latch mechanically blocked | **0.36°** of strike clearance |
| All five opposing digit contacts: press / pull / hold | **100% / 100% / 100%** |
| Maximum backward chest lean, entire sequence | **1.03°** (rounded up) |
| Maximum chest tilt while walking | **0.77°** (rounded up) |
| Maximum planted-foot slip | **6.95 mm** |
| Minimum horizontal clearance at the doorway plane | **64.35 mm** |
| Final whole-body clearance beyond the far frame | **0.683 m** |
| Body contact with door/frame/walls; hand contact during walking | **None** |
| Native acceptance checks / regression tests | **18/18 / 12/12 passed** |

The torso briefly inclines forward during the reach: maximum total tilt is
**7.59°**, with **2.69° RMS** over the sequence. Hand/environment penetration is
at most **0.472 mm**, skeleton self-penetration **0.359 mm**, and summed hand
contact **27.41 N**. The latter is a sum of contact-force magnitudes, not net pull
force. No native simulator warnings occur.

The door is passive after release and settles near **35.8°**; the figure passes
through the remaining opening without pushing against it. “Held open” is checked
during the actual holding phase, not after the hand releases.

These checks run at **1 kHz**, including measured torso angles, qualifying
contacts, joint limits, release/arm speeds, stance slip, and exact body-geometry
intersections with the doorway plane. The replay saves native states at **50 Hz**.
The old `max_foot_drift_m` report field measures displacement from the initial
stance, so it includes intentional walking; use
`traversal.max_planted_foot_slip_m` to assess sliding.

## How it works

1. **Grasp and opening.** A MyoHand-derived articulated hand follows the observed
   handle pose through regularized arm IK and bounded joint torques. Eight small
   finger corrections come from the earlier 64-rollout cross-entropy search.
   They are unchanged in this version. Each qualifying finger contact must oppose
   a loaded thumb pad by at least 120° around the usable lever, with at least
   0.05 N normal load. Palm, thumb-base, stem and rose contacts cannot substitute.
2. **Upright posture.** The opening controller's backward spine target is removed.
   The search reward now includes `-25 mean(tilt²) - 50 mean(backward_lean²)` in
   radians. Walking also directly penalizes achieved chest tilt in its control
   objective. Acceptance requires total tilt below 8° and backward lean below 5°.
   The old backward-leaning pose is retained as a rejection regression. This update
   adds a reward and physical posture control; it does not claim a new RL training run.
3. **Release.** The lever returns, the fingers open, and the wrist lifts clear.
   A deterministic collision-checked arm path lowers the hand through smooth
   quintic segments. The executed native contacts are audited independently.
4. **Foot placement and balance.** A footstep schedule is checked against the
   observed door and frame using native geometry-distance queries. Unsafe
   footprints and swings receive small smooth corrections. A center-of-mass plan
   keeps the planned support point inside the feet's support region.
5. **Physical traversal.** A 200 Hz inverse-dynamics quadratic program calculates
   bounded joint torques while satisfying floating-body dynamics, friction and
   foot-pressure limits. MuJoCo executes those torques at 1 kHz. No root forces,
   foot anchors, hand welds, door motors, mocap bodies or runtime mechanism pose
   writes are used.

The balance formulation draws on standard
[quadratic-program whole-body control](https://arxiv.org/abs/1311.1839), implemented
here with [OSQP](https://osqp.org/docs/interfaces/python.html). No external
controller implementation was copied.
[Real door-opening footage](https://www.pexels.com/video/person-opening-and-closing-the-door-2108274/)
guided the opposing grasp; no 3D motion or force was inferred from it.
[Finger policy](../scripts/physical_human/grip_policy.json) ·
[Earlier search record](review/physical-human/grip-search.json).

## Anatomy, inspection and limits

The custom door is **0.81 × 2.07 m, 19 kg**, with a spring lever, mechanically
coupled retracting latch, and passive hinge. The hand uses five metacarpals,
distinct finger proportions, two thumb-base axes, **20 digit DoF per hand**, two
wrist axes and separate forearm rotation. Its 21 landmarks follow COCO-WholeBody
hand ordering; Sapiens inference and motion capture are not used.

Hand parameters come from [MyoHub/myo_sim](https://github.com/MyoHub/myo_sim), commit
`eb327acbae0fad12279495040607f5235d962328`.
[Source, modifications and Apache-2.0 license](../scripts/physical_human/anatomy/README.md).
Masses and servos are approximations. Skin deformation, muscles and calibrated
human inertias are not modeled. Other doors, perturbation robustness and robot
retargeting remain outside this recording's validation.

The earlier **v2 grasp was rejected** because its thumb did not touch the lever.
This sequence preserves the corrected v3 grasp and adds full-body checks.
[Visual review](review/physical-human/visual-review.json) records the enlarged
native images and browser views actually inspected. Passing physical checks does
not establish natural human style; continuous playback style has not received
independent human validation.

## Reproduce on CPU

```sh
python -m pip install -e '.[reference]'
python scripts/physical_human/prototype.py --sequence --out out/physical-human/normal
python scripts/physical_human/prototype.py --sequence --duration 6.3 \
  --no-touch --out out/physical-human/no-touch
python scripts/physical_human/prototype.py --sequence --duration 6.3 \
  --latch-blocked --out out/physical-human/blocked
python -m pytest tests/test_physical_human_prototype.py -q
```

The sequence stops automatically after traversal and settling. Omitting
`--sequence` preserves the shorter opening-only experiment. `--anchors` is a
labeled debugging option and is rejected for full sequences. Loading the XML
alone does not run the controller.

```sh
python scripts/physical_human/review.py out/physical-human/normal
python -m scripts.physical_human.sequence_video out/physical-human/normal \
  --out out/physical-human/doorbench-full-sequence.mp4
cd viewer
bun install
cd ..
python scripts/physical_human/export.py out/physical-human/normal \
  --web out/physical-human-demo --three viewer/node_modules/three
python -m http.server 5184 --bind 127.0.0.1 --directory out/physical-human-demo
```

The exporters check the scene/trace hashes and refuse failed runs. The local
viewer provides whole-body and enlarged hand cameras, phase shortcuts, contact
points, live torso/joint angles, per-digit forces, slow motion and downloads.
It interpolates saved native states rather than resimulating physics.

`trajectory.npz` contains timestamps, qpos, qvel, motor targets/forces, touch
sensors and `hand_keypoints[frame, left/right, 21, world_XYZ_metres]`.
`walking-plan.npz` and `release-plan.json` preserve the planned paths;
`report.json` includes full-rate achieved checks, source hashes and exact run
times. Generated scenes, trajectories and videos are release attachments, not
committed catalogue assets.
