# Door55 standing grasp and initialized regrasp diagnostics

The standing grasp is feasible. A six-second initialized native hold passed the physical and five-digit opposition gates. Acquisition and opening are **not** established by this result.

| Experiment | Result | Limitation |
|---|---|---|
| Quiet walking stance, 1.046 m pelvis, canonical hand IK | 4 nm position error; 2.43 cm root XY adjustment; no screened collisions | Geometric relocation from a measured plane walking stop |
| Initialized standing hold | Max torso tilt 1.162°; palm error 2.321 mm; FF/MF/RF/LF/thumb loads 4.20/4.17/3.66/2.61/9.50 N | Starts already grasping; physical gates sampled every 20 ms in this historical probe |
| Actual Door55 approach-005 stop, fixed root and legs | Canonical grasp IK again exact, no screened collisions | Wrist yaw at its lower limit; geometric fit only |
| Standing workspace sweep | Handle rotation through 0.773 rad exact and collision-clean | At full 0.87 rad press, wrist limits leave almost no opening workspace; leaf >0.03 rad begins error |
| Old 438-input/6-action tactile PPO from failed acquisition-013 | Both filtered and unfiltered trials release fingers and let the lever return; thumb gap 11.37 → about 1.5 mm | No opposed grasp; filter occasionally infeasible in filtered trial |
| Thumb-first targets after PPO release | Stable thumb contact 4.46 N; canonical finger targets reached | Fingers remain unloaded |
| Thumb-first plus 6 N finger motor feedforward | MF/RF load 4.61/3.92 N | FF/LF unload, thumb lifts 2.82 mm; closure pauses at 94.7% |

The last four trials record safety and opposition every **2 ms**. All passed the physical joint, motor, penetration, and upright checks; none passed sustained five-digit opposition. `ppo-summary.json` preserves exact results. The two thumb-first trials also passed every hand-limit-filter feasibility check.

The thumb gate requires ≥0.2 N continuously for 0.1 s before finger closure. Closure advances only while thumb remains loaded. In the force trial the last loaded-thumb sample is 1.662 s, then persistent loss begins 1.682 s at 94.7% closure. The lever is still at rest (−0.000189 rad), so this differs from the earlier failure caused by fingers depressing the lever before thumb seating.

## Scope and retained evidence

All trials initialize once from explicitly identified states. No runtime root/joint pose writes, added body wrenches, direct door commands, joint-limit changes, or stronger actuators occur. Body control uses a privileged landed-foot inverse-dynamics motor QP. These are not continuous approach/acquisition, Isaac parity, opening, traversal, or sensor-only whole-robot results.

The scripts here are immutable **historical diagnostic snapshots**, retaining the workstation paths used for the run. They are not the portable production runner. `source-index.json` records original paths and SHA-256 hashes. Native Python used `/tmp/doorbench-dexterous-humanoid/out/dexterous/venv/bin/python`; set `PYTHONPATH=/tmp/doorbench-isaac-integration`. Robot assets were `out/dexterous/robot/h1-shadow.xml` and door `out/dexterous/assets/doors/db0055_swing_single` in the dexterous-humanoid worktree.

Local full evidence, including uncommitted generated trajectories and zoomed visual inspection:

- `/tmp/doorbench-standing-grasp/README.md`
- `/tmp/doorbench-standing-grasp/fit-001/reference.json`
- `/tmp/doorbench-standing-grasp/fit-approach-005-v2/reference.json`
- `/tmp/doorbench-standing-grasp/initialized-hold-001/initialized-hold-audit.json`
- `/tmp/doorbench-ppo-transfer/README.md`
- `/tmp/doorbench-ppo-transfer/{transfer-001,transfer-002,thumb-first-001,thumb-first-002}/`

Every transfer/regrasp directory includes source/checkpoint hashes, raw states, actions and observations, a trace, and per-2-ms physical gates. The initialized standing hold has an additional source/dependency override manifest, explicit scope audit, upright-body image and three-angle zoomed hand montage. Its generic acquisition report intentionally remains false because it starts in contact. All hand montages were personally inspected; no render was used as a replacement for force and geometry evidence.

Exact last staged invocation, from a fresh output path:

```sh
PYTHONPATH=/tmp/doorbench-isaac-integration \
  /tmp/doorbench-dexterous-humanoid/out/dexterous/venv/bin/python \
  /tmp/doorbench-ppo-transfer/probe_thumb_first_force.py \
  --source /tmp/doorbench-ppo-transfer/transfer-002 \
  --output /tmp/doorbench-ppo-transfer/thumb-first-new \
  --duration 5 --limit-filter --grip-force 6
```

Controller work returned to the root agent after these bounded trials. Further work should preserve the distinction between contact-force positivity and an anatomically valid opposed grasp, and should evaluate every native physics step.
