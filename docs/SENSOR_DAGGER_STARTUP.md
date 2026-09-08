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

## Longer fit and actual model003 failure

The unchanged longer fit completed at 1,000 steps in 336.06 CPU seconds,
starting **2026-09-08 13:21:10 UTC**. Its offline results are:

| Model | Correction-prefix MSE | Nominal-prefix MSE | Nominal first-100-ms body RMSE |
|---|---:|---:|---:|
| 002: cold-start BC | 0.044499 | 0.000234584 | 1.269 Nm |
| 003: 200-step correction fit | 0.005537 | 0.001859410 | 4.263 Nm |
| 004: 1,000-step correction fit | 0.000781 | 0.000355112 | 0.971 Nm |

The MSEs use normalized forces. The last column uses actor-owned command
history on recorded nominal sensor states. Nominal first-500-ms RMSE for
model004 is 2.127 Nm, still worse than model002's 1.605 Nm; its nominal-prefix
MSE remains 10.59 times command persistence. Model004's reset left/right knee
forces are −51.04/−52.32 Nm versus teacher −51.42/−53.53 Nm. The four runtime
integration checks pass, with the original mechanics and observation contract.
The [model004 receipt](evidence/sensor-imitation-acquisition-004.json) retains all
results and checkpoint SHA
`afce5233becb9eb32fedfe79a5619c0efe1f55979b9dbe6353549057e09a9508`.

In a separate actual Isaac trial, **model003 failed at 0.522 s** with no hand
contact. At 100 ms its root angular velocity about world X was −0.509 rad/s,
versus model002's +0.226 rad/s: the correction fit drove an opposite drift.
Its initial left-hip-yaw force was −9.903 Nm versus teacher +0.046 Nm. Original
joint, tendon, collision, force-cap and delivered-torque checks still passed.
This is a learned-feedback failure, not physical task success hidden by a
score. The [failure receipt](evidence/sensor-actor-initial-failure-003.json)
preserves the original report and its hashes.

The continuous expert query on model003's 261 visited states found 146
individually eligible labels, but only a 20-label contiguous valid prefix:
the QP returned `solved inaccurate` at 40 ms, while tilt was still 0.077 degrees.
The stricter numerical admission criterion remains unchanged. This short
prefix and all later query evidence are retained; they were not used in
model004, whose source data was already frozen.

An independent frame audit reconstructs the torso-mounted gyro from the real
model002 query states using analytic FK/Jacobians. Across the 156 valid
post-reset samples, it agrees with actual actor IMU to maximum
1.13×10⁻⁶ rad/s and RMSE 2.48×10⁻⁷ rad/s. The exact model, source and frame
hashes accompany the receipt. This confirms sensor-frame consistency, not
closed-loop controllability. Reproduce it with:

```sh
python scripts/dexterous/audit_correction_imu.py \
  --corrections "$CORRECTIONS" --robot "$ROBOT" --output "$IMU_AUDIT"
```

## Bounded numerical query retry

Model003's 40-ms QP used the original 4,000-iteration budget. Before making a
fresh query, freeze a numerical-only retry with `max_iter=100000` and initial
`rho=0.001`. The QP objective, support/contact assumptions, constraints, motor
caps and both 1e-4 residual tolerances remain unchanged. Only `max_iter` and
`rho` may be configured through the new optional API; defaults reproduce the
previous controller. Record settings, solver iterations and primal/dual
residuals. Keep the earlier rejected query and its admission result intact.
No physical recovery or new training follows merely from a solved query.

```sh
python scripts/dexterous/query_acquisition_teacher.py \
  --run "$ACTOR003" --robot "$ROBOT" --reference "$ACTUAL_REFERENCE" \
  --solver-max-iterations 100000 --solver-rho .001 \
  --output "$NUMERICAL_RETRY"
```

The penalty/iteration retry admits 151 contiguous candidate labels through
0.300 s. A more controlled follow-up changes **only** the iteration budget to
100,000 and achieves the same admission: the formerly inaccurate 40-ms solve
finishes at 5,200 iterations, with primal/dual residuals 7.70×10⁻⁵/9.98×10⁻⁶.
This iterations-only result is the selected candidate for possible future
training. All later states remain excluded once actual tilt exceeds 12 degrees
at 0.302 s. No new training is implied by this label admission.

A default-settings replay reproduces all 261 original force vectors and label
validity bits **exactly**, including the rejected 40-ms query. The selected
candidate differs from the original forces by at most 0.958 Nm over the first
151 labels. The [numerical receipt](evidence/sensor-actor003-query-numerics.json)
retains every trial, setting, hash and residual. The 1e-4 absolute and relative
stopping tolerances are unchanged; raw residuals can exceed 1e-4 under the
relative rule. A solver status is still not a physical recovery qualification.

The selected query command uses `--solver-max-iterations 100000` and omits
`--solver-rho`. Its canonical directory is
`teacher-corrections-actor003-004-iterations-only`.

## Actual model004 and accumulated correction experiment

Model004 still failed in Isaac, ending at **1.042 s**. Its first 12-degree
upright violation moves to 0.412 s; it later returns near upright while losing
height, then ends at 0.419 m pelvis height and 28.65 degrees of tilt. Initial
force error improves: its largest error is 1.367 Nm at the right hip yaw,
followed by 1.208 Nm at the right knee. It never acquires a qualified handle
grip. Incidental hand/door contacts must not be omitted: they begin at 0.390 s,
reach 5.305 N within the upright prefix and later reach 1,576.94 N on the middle
finger knuckle at 0.482 s, after the first physical-bound failure.

The iterations-only continuous expert query admits 206 initial labels through
0.410 s. It preserves the actual measured normal contact vectors and the
acquisition teacher's body-origin approximation; friction/moments remain
outside that declared query contract. Later states that return inside the
upright bound are still excluded by the contiguous-prefix rule. These labels
remain counterfactual corrections, not a validated recovery.

On the admitted actual model004 observations, full recurrent replay matches
recorded motor commands to body RMSE 0.000684 Nm and maximum difference
0.002392 Nm. There is a measurable training-history discrepancy: over seven
matching 32-step windows, resetting and warming up for 32 steps gives body
RMSE 0.6174 Nm versus full history; warming up for 64 steps reduces it to
0.2297 Nm. This is a history approximation error, not an inference frame bug.
The audit requires the exact checkpoint that executed the recorded trial.

Freeze one next candidate before execution: a fresh seed-zero model trained
for **1,000 CPU steps**, with uniform sampling across the qualified nominal
prefix and the 157/151/206 eligible correction prefixes from actor002/003/004.
Use **64-step burn-in** and the existing 32 supervised steps, batch two,
learning rate 0.0001, and 50% true-start batches without a masked prefix.
The actor architecture, sensors, force outputs, physical limits and task gates
remain unchanged. No other parameter sweep is part of this experiment.
Evaluate all four sources, original startup commands and runtime integration
before another physical attempt. Use these additional training arguments:

```sh
--burn-in 64 --iterations 1000 \
--correction-dataset "$ACTOR002_CORRECTIONS" \
--correction-dataset "$ACTOR003_ITERATIONS_ONLY_CORRECTIONS" \
--correction-dataset "$ACTOR004_CORRECTIONS"
```

Reproduce the history audit with:

```sh
python scripts/dexterous/audit_sensor_recurrent_history.py \
  --corrections "$ACTOR004_CORRECTIONS" --checkpoint "$MODEL004/actor.pt" \
  --output "$HISTORY_AUDIT"
```

The [model005 result](evidence/sensor-imitation-acquisition-005.json) is a
regression on the nominal trajectory despite fitting two new correction
prefixes more closely. Its actual Isaac actor falls at **0.482 s**, with no
qualified grip or handle operation. The
[physical failure receipt](evidence/sensor-actor-initial-failure-005.json)
preserves that outcome; the original motor, coupling, collision and delivery
checks pass. Model004's physical failure and recurrent-history diagnostic are
also [retained](evidence/sensor-actor-initial-failure-004.json).

| Full-history prediction source | Model004 MSE | Model005 MSE |
|---|---:|---:|
| Nominal acquisition prefix | 0.0003551 | 0.0026507 |
| Actor002 correction prefix | 0.0007813 | 0.0023188 |
| Actor003 correction prefix | 0.0795481 | 0.0037923 |
| Actor004 correction prefix | 0.0821776 | 0.0025981 |

These are normalized-force prediction errors on recorded observations, not
success metrics. With the actor's own previous-command history on nominal
recorded sensor states, its first-100-ms body force RMSE worsens from 0.971 to
2.999 Nm, and first-500-ms RMSE from 2.127 to 5.924 Nm. The physical regression
is consistent with this lost nominal fit. More data alone has not solved it.

## Frozen longer fit on another machine

Freeze the [5000-step convergence experiment](../configs/dexterous/sensor-imitation-convergence-006.json)
before execution. It uses exactly model005's four admitted datasets, architecture,
sampling, burn-in and seed. Only training length and compute device change.
Do not add actor005 states to this experiment. Every 500 steps, preserve atomic
actor weights, latest optimizer/RNG state, full-history four-source errors and
actual-start command errors. Stop at 5000 steps or 45 minutes. The command
does not resume an optimizer state automatically.

All six predeclared thresholds must improve on the best previous value for
that source or startup interval before independent runtime checks and any new
physical attempt. Report every checkpoint's scores, including regressions;
none of these fitting checks is a closed-loop success claim. A single fitting
run cannot establish generalization.

Build the portable training bundle from the immutable evidence directory:

```sh
python scripts/dexterous/build_sensor_training_bundle.py \
  --base "$RUN_ARCHIVE" \
  --experiment configs/dexterous/sensor-imitation-convergence-006.json \
  --output "$NEW_BUNDLE"
```

The bundle contains source, frozen experiment settings, motor/sensor calibration,
original evidence and receipts, plus explicit relative paths. The loader checks
every file hash, repeats teacher/correction admission, and verifies that all
four dataset identities match model005. Original report paths and bytes remain
unchanged. The launch file provides an argument array and environment for an
existing CUDA PyTorch environment; it neither provisions a GPU nor launches
Isaac. Simulator meshes are unnecessary for this fitting-only bundle.

After extraction, run `launch.json`'s argument array from the bundle root with
`PYTHONPATH=source`, `OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. The trainer
captures actual dependencies and source hashes before training. `progress.json`
shows optimizer progress; `latest-fit.json` shows all six comparison checks;
`fit-step-*.json` and `actor-step-*.pt` preserve the complete checkpoint series.
If the wall limit interrupts the requested optimization count, the report
explicitly marks `completed: false` and retains the last measured checkpoint.

The [portable preflight receipt](evidence/sensor-training-convergence-006-preflight.json)
records the final 62.6-MB package hash, independent relocated loading,
70 focused passing tests, and a one-step CPU run whose loss exactly matches
model005's first step. Periodic evaluation reproduces model005's prior scores
and startup errors. Its evaluation leaves weights, training mode and Torch RNG
unchanged. A 1e-6 relative comparison margin prevents summation roundoff from
counting as an improvement.

Cold-start inputs are identical across all four sources. The correction
targets agree with one another and differ from the nominal target by at most
0.0546 Nm, far below model005's multi-Nm initial error. This small mismatch
does not explain the observed failure by itself; all original labels remain
unchanged for the longer-fit experiment.
