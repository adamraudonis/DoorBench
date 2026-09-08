# Continuous H1 body transition

This development primitive physically walks, stops, lowers the pelvis to 0.87 m, holds, rises, walks again, and stops. It uses the full H1 with two Shadow hands on an empty MuJoCo plane. It does not include reaching, door contact, a vision/tactile policy, or doorway traversal. The earlier held-raised-arm stop failures remain in the [walking report](DEXTEROUS_LOCOMOTION.md).

## Mechanism

The official H1 walking actor reaches a quiet stop when the two gait-phase observation channels fade to zero over one second. H1 has no ankle-roll joint; an ordinary learned landing leaves the feet narrow and rolled. Holding those foot frames while deeply bending the legs caused slipping and unreliable stance transitions.

During the final approach steps, a smooth motor/proprioception offset of +0.07 rad at the left hip roll and −0.07 rad at the right hip roll widens the actual landing. This changes the command and proprioceptive reference, not the physical model. The feet land approximately 0.368 m apart in the demonstrated trial, versus 0.276 m without the offset. The offset is removed smoothly after walking restarts.

The privileged inverse-dynamics teacher then captures the independently landed foot frames and current root position. It drives the body height with a minimum-jerk reference while producing bounded commands for the original ten leg motors. Its support bounds use the full native sole geometry. These are planning assumptions; no support constraint, root actuator, external wrench, foot anchor, or runtime pose write is inserted into the simulator. Original native masses, inertias, joint limits, damping, armature, friction, motor parameters, control limits, and force limits stay unchanged.

The walking actor is paused during the stance teacher and its previous-action memory is reset when walking resumes. The upper-body motors maintain the declared initial arm and hand posture. A real door sequence must explicitly acquire/release the handle and retract the arm; this test does not pretend that those actions occurred.

| Time | Motor-controlled behavior |
|---|---|
| 0–1 s | Initial standing gait phase |
| 1–9 s | Walk forward at requested 0.4 m/s; widen hip reference over 5–8 s |
| 9–14 s | Zero command, phase fade and physical settling |
| 14–18 s | Lower pelvis to 0.87 m |
| 18–22 s | Hold manipulation height; score quietness over 20–22 s |
| 22–26 s | Rise to the captured standing height |
| 26–27 s | Hold before actor handoff |
| 27–32 s | Resume walking at requested 0.3 m/s; remove hip offset over 27–30 s |
| 32–37 s | Phase-adapted stop again |

## Evidence and limits

The [frozen development matrix](../configs/dexterous/h1-locomotion-transition-development.json) covers three seeds with ±0.005 rad initial leg noise, at headings 0, 1.6785, and −1.2 rad. This is a narrow development check, not environmental robustness or a door benchmark. See the [machine-readable report](../results/dexterous/2026-09-08/h1-locomotion-transition.json) for every trial and retained failures.

| Native development trial | Complete checks | Worst torso tilt | Quiet low-foot excursion | QP failures |
|---|---:|---:|---:|---:|
| Ordinary narrow landing | 2/9 | Includes one fall | Up to 14.5 mm | Present |
| Widened landing, 2 ms safety audit | **9/9** | **4.07°** | **3.58 mm** | **0** |

The widened matrix held the pelvis between 0.86965 and 0.87000 m, with maximum horizontal low-stance speed 0.00173 m/s and at least 244 N on each foot. The final stop stayed below 0.0151 m/s. There were zero measured self/nonfoot-ground penetrations, zero excess motor torque, and maximum joint-stop penetration 0.002585 rad. The native safety audit runs at every 2 ms step, while these quiet-state quantities use the declared 20 ms trace.

The successful widened-landing sequence still has approximately 2.2 cm of physical foot motion during the whole lowering/rising interval. Quiet feet during the measured low hold move less than 3.6 mm. This distinction matters: passing the hold check does not establish perfectly planted feet throughout the transition. A future stepping or contact-aware controller should reduce that transient motion.

Safety checks run at every 2 ms native step: torque and joint limits, upright posture, nonfoot ground contact, self contact, finite state, no external applied forces, and unchanged physical parameters. Root/foot quietness and contact loads are sampled at 20 ms. All QP failures are counted and fail the run, even when the robot remains standing. The limited contact planning model remains subject to the measured physical rollout.

Retained failures include the direct crouch-offset controller (fell), immediate flat-foot QP (infeasible), pose-IK handoff (fell), the lower-iteration landed-foot QP (12 solver failures), the ordinary narrow-landing matrix (2/9), and continually refreshing foot references (fell while rising). The ordinary phase-braking protocol remains 30/40, including all ten raised-arm failures and the separate long-hold fall.

## Reproduce and integrate

Use the unchanged robot and pinned actor from [walking setup](DEXTEROUS_LOCOMOTION.md#reproduce), with MuJoCo, Torch, NumPy, SciPy, and OSQP:

```bash
python scripts/dexterous/probe_locomotion_transition.py \
  --robot out/dexterous/robot/h1-shadow.xml \
  --checkpoint out/h1-walking-upstream/deploy/pre_train/h1/motion.pt \
  --output out/h1-body-transition-001
python scripts/dexterous/evaluate_locomotion_transition.py \
  --robot out/dexterous/robot/h1-shadow.xml \
  --checkpoint out/h1-walking-upstream/deploy/pre_train/h1/motion.pt \
  --output out/h1-body-transition-matrix-001
python scripts/dexterous/render_locomotion.py \
  --robot out/dexterous/robot/h1-shadow.xml \
  --trial out/h1-body-transition-001
```

Every run needs a fresh output directory and records exact source, dependency manifest, arguments, logs, report, and trajectory. The MP4 is a labeled replay of those recorded physics states at their original timestamps. Generated checkpoints, meshes, trajectories and video remain outside git.

`LandedFootStanceController(sim)` needs the same simulation adapter as `StanceController`, including named leg joints/motors, root indices, pelvis body, and the full plant state. Construct it only after the robot physically stops. Change `target_root[2]` smoothly, call `command()` at 100 Hz, and write only the returned bounded native leg motor controls. It relies on privileged state and inverse dynamics; it is a teacher that can support later distillation, not a sensor-only deployed policy. The standalone runner demonstrates the exact 50 Hz actor / 100 Hz stance / 500 Hz physics handoffs.
