# First actual sensor actor: failed before acquisition

The first sensor-only actor trial fell after 0.522 seconds; it never acquired a
valid grasp. It crossed the 12-degree upright criterion at 0.268 seconds and
reached 47.41 degrees at the final recorded step. Its original motor limits,
passive hand mechanics and torque delivery passed. The original failed report,
all physics samples and the actual pre-step actor decision remain archived.

Its actual initial root and joint state exactly match the qualified teacher002
reset. All seven initial sensor streams were invalid, their timestamps were
-1, and previous action was zero. The initial knee forces had the wrong sign:

| Motor | Actor command Nm | Teacher command Nm | Error Nm |
|---|---:|---:|---:|
| Right knee | +13.774 | -53.533 | +67.307 |
| Left knee | +13.471 | -51.424 | +64.895 |

That packet was absent from the first training data. Fixed 32-step burn-in also
masked the first 64 milliseconds of a true start window, and random interior
windows rarely reached the beginning. Merely making initial sensors valid does
not solve the recurrent initialization problem: with a valid recorded packet
and zero hidden state, the original actor still predicts left/right knee forces
of +1.87/-15.46 Nm instead of -52.52/-54.63 Nm. Even under unchanged teacher
states, body force RMSE remains about 2.17 Nm at 80–100 ms. These are gaps in
start supervision and physical feedback robustness, not a simulator force-cap
failure.

Same-time teacher commands **after** the first physical tick are not corrective
labels for the diverged actor state. They are used only to characterize drift.
At 0.1 s the actor already tilts 2.68 degrees versus the teacher's 0.103 degrees,
with a 0.299-radian norm of joint-state differences. The [independent receipt](evidence/sensor-actor-initial-failure-001.json)
preserves exact initial errors, progression, hashes and the original failure.

## Bounded second training experiment, frozen before execution

Keep the existing actor architecture, motor/observation interface and original
physical gates. Add the actual all-invalid reset observation from actor001,
paired only with teacher002's actual first action. Admission requires identical
robot/door reset, physical joint order, mechanics and sensor calibration. The
failed actor's action is explicitly discarded. The rest of the qualified
acquisition prefix and its causal cutoff remain unchanged.

Train a fresh seed-zero model for 200 CPU AdamW steps. Half the batches start at
the true episode boundary with zero hidden state and no masked prefix; half
retain random interior windows and 32-step burn-in. Keep batch two, supervised
length 32 and learning rate 0.0001. The augmented prefix has 6,457 labels,
including the previously missing t=0 decision. Record early force error and
full-prefix persistence comparison before another physical trial. This
experiment has not run at this protocol commit, and no success is assumed.

For future DAgger corrections, the teacher must be queried on the student's
actual states, with its own continuously advanced controller history. The
current runner preserves 500 Hz root and joint position/door state, but only
50 Hz body-resolved hand loads. A faithful future query stream should separately
record pre-decision clean root13, named joint positions and velocities, handle
pose7 and body-resolved world hand forces at t=0 and every 2 ms. Those privileged
values and returned counterfactual teacher forces belong in evaluator/training
label files, never the actor's numeric sensor packet. Missing impulses must not
be filled by interpolating the sparse contact trace.

## Second training result

The frozen experiment ran at **2026-09-08 12:44:18 UTC** and completed 200 CPU
steps in 72.22 seconds. Its initial knee commands now have the correct sign:
left -48.83 Nm and right -42.50 Nm, versus teacher -51.42/-53.53 Nm. Under
recorded teacher states with the actor's own command history, body-force RMSE
falls from 5.057 to 1.269 Nm over the first 100 ms, and from 2.737 to 1.605 Nm
over the first 500 ms. This is an offline prediction improvement, not evidence
that the robot remains standing.

The tradeoff is visible: full-prefix normalized MSE increases to 0.000234584,
versus persistence 0.0000335349. No physical student trial is included in this
training receipt. The checkpoint passed all four runtime integration checks,
including 64 consecutive commands from the actual cold-start observation. Its
SHA-256 is `1314a277d81b2a93eeaccaff6416fd54272d15c227434398106270ef78e4cf01`.
The [result receipt](evidence/sensor-imitation-acquisition-002.json) retains
source commit, exact episode provenance, inference checks and both early-time
and full-prefix comparisons.

Reproduce the training change by adding these options to the first-run command:

```sh
--reset-observation-run "$ACTOR001" --episode-start-probability .5
```

Use a new output directory. Add `--reset-observation-run "$ACTOR001"` when
running the prediction and runtime checkers on that augmented prefix. To compare
both checkpoints' early commands:

```sh
python scripts/dexterous/evaluate_sensor_start.py \
  --episode "$RUN002" --legacy-teacher-receipt "$CURRICULUM/provenance.json" \
  --reset-observation-run "$ACTOR001" \
  --checkpoint "$TRAINING001/actor.pt" --checkpoint "$TRAINING002/actor.pt" \
  --output "$TRAINING002/early-force-comparison.json"
```
