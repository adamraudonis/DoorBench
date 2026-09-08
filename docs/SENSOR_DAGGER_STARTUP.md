# Student-state startup corrections

The second actual Isaac sensor actor still failed: it crossed 12 degrees of
torso tilt at 0.314 s and 45 degrees at 0.760 s, ending at 0.762 s with no hand
contact. The cold-start training change improved the initial commands and
delayed the fall from 0.522 s, but did not learn stable feedback. Original
actuator caps, delivered torques, joint stops, passive hand mechanics and
collision checks remained valid. Both failures remain failures.

Actor002 now records separate evaluator-only measurements before its t=0
decision and every 2 ms. The 381 rows include actual root13, named joints and
velocities, handle/leaf poses and body-resolved normal hand loads. Root angular
velocity and loads use world coordinates. Normal forces omit friction and
moments, matching the existing acquisition teacher approximation. There was no
hand contact in this failed prefix. The actor never receives these privileged
arrays and no expert ran during the student trial.

## Offline label contract

Run one continuously maintained `AcquisitionTeacher` over the actual measured
sequence. Do not reset its filter/integrator state between queries. Native
physics stepping is forbidden; its model is an analytic calculator. Keep all
queries, errors and infeasible labels. Admit only the contiguous initial prefix
whose actual state remains within the original upright/height/mechanics bounds,
whose QP status is `solved`, and whose proposed forces are finite and within the
unchanged 61 motor caps. This is candidate correction supervision, not a
physically validated recovery trajectory or a successful teacher episode.

The first diagnostic has 157 eligible labels at 0–0.312 s. Its next query at
0.314 s is rejected despite a solved QP because actual tilt is 12.058 degrees.
The remaining 224 queries are retained and excluded. The failure is already
substantial there: the largest expert-versus-student force difference is
111.84 Nm. A fresh query will be made after this source/protocol commit.

`CorrectionDemonstration` is a separate admission path. It verifies original
archive hashes, pose/force conventions, reset identity, robot/reference hashes,
motor order, joint order and exact pre-action clocks. Decision zero uses the
real initial invalid-sensor packet; decision i uses actual sensor row i−1. Its
target is the same-time counterfactual teacher force, never the failed student's
force. Recorded student previous actions remain legitimate input history.
No world state, phase, teacher state or absolute clock enters the actor tensor.

## Frozen next experiment

Train a fresh seed-zero model for 200 CPU steps, with unchanged architecture,
61 force outputs, batch two, learning rate 0.0001, 32 supervised steps and
32-step interior burn-in. Sample uniformly between the original qualified
teacher002 acquisition prefix (including its audited cold-start observation)
and the separately identified 157-label correction prefix. Half the batches
start at the actual episode boundary with zero hidden state and no masked
prefix. This deliberately oversamples the observed early feedback failure.

Evaluate correction prediction, the original full teacher prefix against
previous-command persistence, early teacher-state commands, and the runtime
checkpoint contract before a fresh physical actor trial. None of these offline
scores establish standing, acquisition, opening or generalization. Do not use
the later fallen states as correction labels and do not relax the physical
evaluation criteria if the next actor also falls.

```sh
python scripts/dexterous/query_acquisition_teacher.py \
  --run "$ACTOR002" --robot "$ROBOT" --reference "$ACTUAL_REFERENCE" \
  --output "$CORRECTIONS"
python scripts/dexterous/train_sensor_imitation.py \
  --episode "$TEACHER002" --qualification acquisition-report.json \
  --legacy-teacher-receipt "$CURRICULUM/provenance.json" \
  --reset-observation-run "$ACTOR001" --episode-start-probability .5 \
  --correction-dataset "$CORRECTIONS" --iterations 200 --device cpu \
  --sequence-length 32 --burn-in 32 --batch-size 2 --learning-rate .0001 \
  --seed 0 --output "$TRAINING"
python scripts/dexterous/evaluate_sensor_predictions.py \
  --correction-dataset "$CORRECTIONS" --checkpoint "$TRAINING/actor.pt" \
  --output "$TRAINING/correction-prediction.json"
```

Use new output directories and the actual actor002 reset reference. Canonical
archives are under `DoorBench-runs/2026-09-08-shadow-loopback`. This protocol
commit precedes the fresh label query and training experiment.

## First correction fit result

The frozen query reproduced 157 admissible labels. Model003 completed 200 CPU
steps in 70.02 seconds, starting **2026-09-08 13:17:15 UTC**. Correction MSE on
the actual visited-state prefix falls from 0.044499 for model002 to 0.005537,
versus recorded-command persistence 0.044533. That is an eightfold prediction
improvement on the newly supplied training states.

Nominal behavior regresses: original teacher-prefix MSE rises to 0.001859,
versus persistence 0.000033535. With actor-owned previous-command history on
teacher sensor states, first-100-ms body force RMSE increases from 1.269 to
4.263 Nm, and first-500-ms error from 1.605 to 4.469 Nm. The reset knee commands
remain similar to model002: left −48.63 and right −42.33 Nm, versus teacher
−51.42/−53.53 Nm. All four runtime checks pass, but this tradeoff needs an actual
physical trial. The [complete receipt](evidence/sensor-imitation-acquisition-003.json)
preserves the checkpoint hash, source, labels and both favorable and unfavorable
scores. A live actor003 trial is being evaluated separately by the parent agent;
its result is not assumed here.

## Bounded longer fit, frozen before execution

Keep the identical seed, dataset admission, balanced sampling, recurrent
windows, architecture, learning rate and physical evaluation. Train a fresh
model for **1,000 CPU optimizer steps**, changing only the iteration count from
the first correction fit. The hypothesis is that 200 steps underfit the two
distinct feedback regimes; additional optimization can reduce their conflict.
Do not add failed-state labels, alter gates or change the motor interface.
Preserve model003 regardless of the result. Evaluate both the correction prefix
and the original teacher trajectory with the same scripts, including cold-start
commands and runtime integration. This protocol update precedes the longer run.
