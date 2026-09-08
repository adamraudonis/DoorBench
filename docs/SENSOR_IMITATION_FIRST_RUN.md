# First sensor imitation run

The first real job trained a 557,949-parameter stereo/touch recurrent actor on
CPU for 200 optimizer steps in 82.57 seconds, starting **2026-09-08 12:27:57 UTC**.
It used a qualified acquisition-only prefix of Isaac operation002. It has **not
been evaluated in closed-loop physics and has no door-opening success result**.
All failed operation reports remain failed.

On all 6,456 causal next-action examples from the training prefix, its mean
squared normalized force error is 5.62 times worse than repeating the previous
command. Persistence is a strong baseline at the 2 ms timestep. The idle left
hand exposes unnecessary predicted-force noise; the result is useful as a first
training/integration baseline, not evidence of reliable control.

| Motor group | Learned actor MSE | Repeat previous command MSE | Zero command MSE |
|---|---:|---:|---:|
| All 61 motors | 0.000181862 | 0.0000323611 | 0.00865848 |
| Body | 0.000244735 | 0.0000658911 | 0.0205828 |
| Right hand | 0.000194813 | 0.0000295083 | 0.00461415 |
| Left hand | 0.000102895 | 0.0000000074353 | 0.000182265 |

The actual checkpoint passed four runtime integration checks: 64 consecutive
finite inferences, original motor caps, explicit reset reproducibility, and
rejection of changed valid motor caps. These checks used recorded observations
and the actor's own previous commands, without stepping a simulator. The
prediction evaluation used teacher action history and carried recurrent state
through the complete prefix. Neither check establishes generalization or
student task success. The [compact receipt](evidence/sensor-imitation-acquisition-001.json)
preserves results, UTC start, source and checkpoint hashes, qualification scope,
and durable artifact locations.

## Qualified acquisition data

The original episode remains intact. The separately audited prefix contains
6,457 sensor samples from 0.002 to 12.914 seconds and 6,456 next-action labels.
Its final distal-pad grasp is continuously valid for all 251 physics samples
from 12.414 to 12.914 seconds. The reviewed runner switches to operation at
clock 12.914; that controller's first resulting action is recorded at 12.916
seconds and is excluded. Failed opening actions never enter this curriculum.

That legacy recorder predates `control_source`, so admission requires an
explicit [compatibility receipt](evidence/legacy-acquisition002-teacher-receipt.json)
bound to its exact reviewed immutable source package. The loader recomputes
source/manifest/actual remote copy hashes, configuration, original reports,
physical/pad/sensor hashes, complete clocks and the qualified grip interval.
Unknown source packages, actor captures, altered receipts and failed
acquisitions are rejected. Legacy command-versus-sent motor delivery was
recorded at 50 Hz; bounded forces, physical state and pad checks were recorded
at 500 Hz. This limitation remains explicit.

Fresh demonstrations require `control_source=privileged_teacher`, a passing
selected task and mechanics report, complete captures, and matching numeric
hashes when declared. Actor rollouts cannot enter teacher imitation by selecting
an acquisition report. Qualification metadata never enters the actor packet.

## Reproduce

Use the original durable `isaac-operation-v2-002` archive as `RUN002`, a separate
new `CURRICULUM` receipt directory, and a new `TRAINING` output directory. Source
commit `69df6bc34` froze this first job before training.

```sh
python scripts/dexterous/make_legacy_teacher_receipt.py \
  --run "$RUN002" --output "$CURRICULUM/provenance.json"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/dexterous/train_sensor_imitation.py \
  --episode "$RUN002" --qualification acquisition-report.json \
  --legacy-teacher-receipt "$CURRICULUM/provenance.json" \
  --output "$TRAINING" --device cpu --iterations 200 \
  --batch-size 2 --sequence-length 32 --burn-in 32 --seed 0
python scripts/dexterous/evaluate_sensor_predictions.py \
  --episode "$RUN002" --qualification acquisition-report.json \
  --legacy-teacher-receipt "$CURRICULUM/provenance.json" \
  --checkpoint "$TRAINING/actor.pt" --output "$TRAINING/prediction-evaluation.json"
python scripts/dexterous/check_trained_sensor_runtime.py \
  --episode "$RUN002" --qualification acquisition-report.json \
  --legacy-teacher-receipt "$CURRICULUM/provenance.json" \
  --checkpoint "$TRAINING/actor.pt" --output "$TRAINING/runtime-inference.json"
```

The causal input at tick i is post-step stereo RGB, local touch, encoders, IMU,
sensor ages/validity and previous motor command. Its label is the command
applied on tick i+1. Future images, door geometry, teacher phase, world pose and
absolute episode time are excluded. The model uses a shared stereo CNN, touch
encoder and GRU, with 32-step burn-in plus 32 supervised steps per window,
batch two, AdamW at 0.0001 and seed zero. Reusing one episode is not a validation
split.

Checkpoint schema `doorbench.sensor-actor.v2` binds the full canonical JSON motor
contract, robot XML, calibration, action order and timestep. Changed caps,
transmission or passive mechanics require new qualification, even if an XML
label is retained. Seventy data, provenance and inference tests pass, including
causal cutoff, future-frame rejection, source identity and motor-contract
mismatch cases. A digest still cannot inspect a live imported physical plant;
engine validation remains separate.
