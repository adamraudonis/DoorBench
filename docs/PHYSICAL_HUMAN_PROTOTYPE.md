# One physical human, one simple door

A small, inspectable experiment: an original stick figure reaches for a lever, closes its fingers and thumb, depresses the lever, pulls a passive door open, and releases it. The hand moves the door through MuJoCo contact forces. There is no door motor, animated door joint, hand-to-handle weld, mocap body, root force or foot anchor in the demonstrated run.

**Local demo:** <http://127.0.0.1:5184>. Whole-body and isolated-hand views, an orbitable camera, slow motion, a scrubber, a transparent-door option, actual contact points, and separate finger/thumb contact readings are available. This is playback of recorded simulation states, not browser physics. Appearance is simplified; dimensions and motion come from the native scene.

## Scope

- One **custom prototype door**, not `db0002` or a validation of the catalogue: approximately 0.81 × 2.07 m, 19 kg, with an offset hinge, strike pocket, sliding spring latch and lever. A joint equality models the spindle-to-latch transmission. Blocking the latch physically blocks opening.
- Original approximate adult skeleton, floating pelvis, articulated limbs, four three-joint fingers and a two-joint opposing thumb per hand: **28 finger/thumb joints**. Self-collision is enabled; MuJoCo's usual parent-child exclusion applies. Both hands have sensors; the left hand performs the task.
- Torque-limited joint servos track a smooth authored arm trajectory with inverse-kinematics targets and joint bias-force compensation. Lower-body servos hold a standing posture; ground contact supports the figure. No external force stabilizes the pelvis.
- This is **standing opening**, not traversal, trained control, motion capture, calibrated human biomechanics, a safety certificate or demonstrated robustness across doors. Joint torque limits and inertias are engineering approximations. The grasp is scheduled, not learned or touch-conditioned. Touch is simulated mechanical contact, not a skin/deformation model.

## Observed result

Run on **September 6, 2026 at 23:37 UTC**, MuJoCo 3.12.0, locally on CPU:

| Case | Maximum opening |
|---|---:|
| Physical hand | **49.52°** |
| Hand contact disabled | **0.00°** |
| Latch blocked | **0.36°** (strike clearance) |

Maximum foot drift was **0.23 mm**; measured hand/door penetration remained below **0.80 mm** and self-contact penetration below **0.31 mm**. No non-hand body contact moved the door, and no simulator warnings were reported. Peak total hand contact was **115 N**; this is not a validated human force profile. [Exact run timestamps, hashes and measurements](review/physical-human/prototype-checks.json).

## Reproduce

From the repository root, using Python with the project's reference dependencies:

```sh
python -m pip install -e '.[reference]'
python scripts/physical_human/prototype.py --out out/physical-human/final
python scripts/physical_human/prototype.py --out out/physical-human/no-touch --no-touch
python scripts/physical_human/prototype.py --out out/physical-human/blocked --latch-blocked
python -m pytest tests/test_physical_human_prototype.py -q
```

MuJoCo 3.12.0 was used. No GPU or cloud resource is required. `--anchors` exists only as an explicitly labeled debugging mode; it was **off** in the demonstration and tests. The blocked-latch case is a diagnostic failure, not a successful reference.

Install the viewer's existing dependencies if needed, then package and serve the recorded run:

```sh
cd viewer
bun install
cd ..
python scripts/physical_human/export.py out/physical-human/final \
  --web out/physical-human-demo --three viewer/node_modules/three
python -m http.server 5184 --bind 127.0.0.1 --directory out/physical-human-demo
```

The packaged demo includes Three.js locally and does not need a CDN. Open <http://127.0.0.1:5184>. The supplied scene has an `initial` keyframe; reset to that keyframe and use the controller to reproduce the rollout. Loading XML alone without the controller does not reproduce the opening.

Optional native-renderer video (Pillow, imageio and imageio-ffmpeg required):

```sh
python scripts/physical_human/render.py out/physical-human/final --video
```

## Evidence and tests

`report.json` records the simulator version, UTC timestamps, source/rig/scene hashes, phase, opening angle, lever angle, bolt travel, contact measurements, foot drift, penetration and warnings. `trajectory.npz` contains 50 Hz post-step timestamps, positions, velocities, servo targets, actuator forces and native touch sensors. Simulation and full-run diagnostic extrema use a **1 ms** timestep. Saved contact rows are also 50 Hz, so a narrow peak in the full-run extrema can exceed every saved row.

The UI sums the magnitudes of **hand–door contact-pair forces**, grouped by the participating finger/palm/thumb. This is not the resultant pull force or a measured human grip force. Raw MuJoCo touch sensors additionally include self-contact; summing all sensors can double-count a contact, so that sum is not used as the UI hand force.

The native regression tests require:

| Check | Acceptance |
|---|---:|
| Normal opening | 40–65° |
| Same controller, hand contact disabled | < 0.05° |
| Latch mechanically blocked | < 1° of strike clearance |
| Pelvis height | > 0.90 m |
| Foot horizontal drift | < 5 mm |
| Hand/door and self-contact penetration | < 2 mm |
| Non-hand body contact impulse on door | Zero |
| Simulator warnings | Zero |

The exact observed measurements and checks are downloadable in the local demo. No unsupported all-door reference claims are made. Before using this as human ground truth, the controller needs broader robustness tests, biomechanical validation, natural-motion review and a locomotion/traversal extension.

Native contact and constraint semantics: [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/index.html) and [MJCF equality constraints](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality).
