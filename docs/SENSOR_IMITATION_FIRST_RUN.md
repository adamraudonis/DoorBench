# First sensor imitation run

The first CPU job is prepared but has **not run**: operation-v2-002, 003 and 004
are failed opening experiments and are not admitted as qualified opening data.
A fresh trial must pass its declared contact profile and physical task gates.
Historical results are not relabeled to make them training data.

The causal input at physical tick *i* is its post-step stereo RGB, local touch,
encoders, IMU, sensor ages/validity, and the force applied on tick *i*. The target
is the force applied on tick *i+1*. Future frames, door geometry, teacher phases,
world poses and the absolute episode clock are excluded. Windows do not cross
episode boundaries. The 557,949-parameter actor uses a shared stereo CNN, touch
encoder and GRU. A trained checkpoint would still require independent physical
student rollouts before any claim that it opens a door.

Run from a source checkout containing the frozen sensor interface, with a new
output directory and the eventual qualified episode path:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/dexterous/train_sensor_imitation.py \
  --episode /path/to/fresh-qualified-isaac-run \
  --qualification operation-report.json \
  --output /path/to/sensor-imitation-001 \
  --device cpu --iterations 200 --batch-size 2 \
  --sequence-length 32 --burn-in 32 --seed 0

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/dexterous/evaluate_sensor_predictions.py \
  --episode /path/to/fresh-qualified-isaac-run \
  --checkpoint /path/to/sensor-imitation-001/actor.pt \
  --output /path/to/sensor-imitation-001/prediction-evaluation.json
```

Report next-force MSE for the actor, a zero-command predictor, and a predictor
that repeats the previous force. At 2 ms, persistence is a strong baseline;
small supervised loss alone is insufficient. The evaluator carries recurrent
state through the complete recording, uses teacher sensor/action history, and
labels training versus separate episodes. Reusing one recording is not a
generalization test. Synthetic CPU timing was approximately 0.295 seconds for a
batch of two 32-step forward/backward windows; that is a throughput check only.

Checkpoint schema `doorbench.sensor-actor.v2` binds the full canonical JSON motor
contract as well as robot XML hash, sensor calibration, action order and timestep.
Changing valid force caps, tendon transmission or passive parameters while
retaining the source XML label is rejected by runtime inference. JSON dictionary
ordering and file location do not affect this identity. Any static metadata
change requires a deliberate new checkpoint qualification. The actual imported
plant must still be validated against that contract; a digest cannot inspect a
running simulator. Older schema checkpoints without this identity are rejected.

Each demonstration also preserves hashes of its selected qualification report,
mechanical audit, passive-tendon audit, motor contract, sensor report and sensor
arrays. The sensor report must explicitly identify `control_source` as
`privileged_teacher`; actor rollouts cannot enter teacher imitation by selecting
an acquisition report instead of an operation report. Missing source identity
is rejected. Qualification metadata never enters the actor packet. The controller and
demonstration boundary pass 56 tests, including future-frame rejection, explicit
episode-local memory, original force caps, changed-mechanics rejection and
preservation of causal next-step targets.

## Frozen acquisition curriculum before the first optimizer step

Operation002 also contains a physically qualified acquisition. A separate,
explicit legacy receipt admits only its 0.002–12.914-second sensor prefix:
6,457 samples and 6,456 causal next-action labels. The uninterrupted distal-pad
hold covers every one of the 251 physics samples from 12.414 to 12.914 s. The
reviewed runner starts the operation controller at clock 12.914; that controller's
first resulting action is recorded at 12.916 s and is excluded. No failed opening
labels enter this curriculum. This is acquisition-only training, and does not
upgrade operation002's failed mechanism outcome.

Because that old recorder predates explicit source identity, compatibility is
limited to its exact reviewed immutable source-package digest. A receipt binds
the source tar, manifest, actual remote source copies, configuration, original
reports, complete physical/pad/sensor archives, and the cutoff. Admission
recomputes those facts and the full half-second grip. Unknown sources, actor
captures, incomplete clocks, failed acquisition or modified receipts fail
closed. Every original report and NPZ remains unchanged. Legacy motor delivery
was independently recorded at 50 Hz, while bounded forces, physical state and
pad checks were recorded at 500 Hz; the receipt retains this limitation.

The first actual job is frozen at 200 AdamW steps, CPU, one thread, seed 0,
batch 2, sequence length 32, burn-in 32 and learning rate 0.0001. It remains
unrun at this protocol commit. The result must include the previous-force
persistence baseline and a tested runtime checkpoint load, while retaining
`closed_loop_evaluated=False`.

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
```
