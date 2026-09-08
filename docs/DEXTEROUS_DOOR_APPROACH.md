# H1 physical approach to Door55

The full H1 with two Shadow hands can walk from a separated start to Door55's standing-grasp body waypoint and stop quietly. **9/9 nearby-start development trials passed** in the complete native MuJoCo door scene. No handle acquisition, door opening or traversal is performed by this primitive.

## Measured result

[Protocol](../configs/dexterous/h1-door55-approach-development.json) and [all results/failures](../results/dexterous/2026-09-08/h1-door55-approach.json), run September 8, 2026 UTC. Each 25 s trial varies initial leg angles uniformly within ±0.005 rad.

| Start behind target | Lateral offset | Initial heading error | Complete checks |
|---|---:|---:|---:|
| 0.5 m | −5 cm | −5° | 3/3 |
| 0.7 m | 0 cm | 0° | 3/3 |
| 0.9 m | +5 cm | +5° | 3/3 |

The worst final position error was **20.34 mm**, heading error **0.932°**, and final-second speed **0.00452 m/s**. Final-second root excursion stayed below 2.02 mm, and each foot carried at least 187 N. The maximum torso tilt over all runs was 2.61°. Every native 2 ms step was checked for motor/joint limits, upright posture, finite state, robot-ground/self/scene contacts, and external applied robot forces. Original physical-parameter arrays are compared after each rollout. There were no robot contacts with the door or wall, no excess motor torque, and no nonfoot-ground/self penetration. Maximum joint-stop penetration was 0.003186 rad.

A representative uncut rollout starts 0.7 m away and finishes 8.74 mm / 0.902° from the target after two physical braking attempts. Its exact native state is available in the generated trajectory; it is not a relocated standing snapshot. The arms remain in a neutral walking posture throughout.

These nine trials establish only a small neighborhood of this waypoint. They are not a broad obstacle-navigation test, a door coverage score, an Isaac result, or proof that the resulting arm trajectory will successfully grasp the handle.

## Controller and supervision

The [pinned official H1 actor](DEXTEROUS_LOCOMOTION.md) retains its proprioceptive 41-value observation and original motor limits. A separate privileged waypoint teacher reads world pelvis pose and filtered velocity, then supplies body-frame forward/lateral/yaw commands. Bounds are −0.15 to +0.30 m/s forward, ±0.12 m/s lateral, and ±0.4 rad/s yaw.

Near the target, the teacher predicts the stopping location from filtered velocity and brakes at a repeatable point in the 0.8 s gait cycle. The phase observation amplitude fades to zero over one second. After four seconds of settling, a stop outside the target tolerance triggers another physical correction attempt. The command waypoint receives 80% of the observed residual error, limited to ±0.12 m per world axis; evaluation always uses the original fixed target. At most four stop attempts are permitted. This bounded online bias correction compensates the actor's repeatable landing offset without moving the simulated root or feet.

The official source [zeros sampled translation commands with norm at most 0.2 m/s](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py#L306), explaining why small proportional commands are a weak precision-control regime. This controller is an empirical adaptation, not new RL training. Earlier approaches stopped 4–9 cm away or continued stepping; all remain in the report. The first prototype also misidentified Door55's box-shaped floor as a scene obstacle; that diagnostic error is explicitly retained and corrected by identifying the actual floor geometry. No floor shape or friction was changed.

The robot begins at least 0.5 m from the target. Initialization alone places its original collision soles on the floor and sets a neutral posture. There are no runtime root/foot pose writes, root actuators, external supports, welds or foot anchors. Only robot motor controls are written. DoorEnv continues its ordinary native mechanical/passive logic; this controller sends no door command.

## Reproduce

Use the [robot and actor setup](DEXTEROUS_LOCOMOTION.md#reproduce) and generated full Door55 assets. The [portable waypoint](../configs/dexterous/h1-door55-approach-target.json) contains the target and source hash; no local-only grasp file is needed.

```bash
python scripts/dexterous/probe_locomotion_approach.py \
  --robot out/dexterous/robot/h1-shadow.xml \
  --door out/dexterous/assets/doors/db0055_swing_single \
  --reference configs/dexterous/h1-door55-approach-target.json \
  --checkpoint out/h1-walking-upstream/deploy/pre_train/h1/motion.pt \
  --output out/h1-door55-approach-001
python scripts/dexterous/render_locomotion_approach.py \
  --trial out/h1-door55-approach-001
```

For the frozen nine-trial matrix, replace `probe_locomotion_approach.py` with `evaluate_locomotion_approach.py` and choose a fresh output directory. Every invocation stores exact source/dependency provenance, arguments, target, logs, report and trajectory. The renderer uses those physics timestamps at 25 fps; changes to diagnostic colors and light affect only the replay.

For a continuous downstream sequence, keep the existing plant alive after success and hand its actual `qpos`, `qvel`, contacts and actor/controller memory to the next controller. Do not reset to the reference pose. The saved full-scene trajectory uses the native DoorEnv joint ordering, with robot joint names prefixed `robot/`. The waypoint teacher remains privileged; vision/tactile distillation and Isaac validation are separate tasks.
