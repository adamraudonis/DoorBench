# Learning balance from actual policy mistakes

More imitation updates on one successful trajectory have not solved feedback
away from that trajectory. We now collect bounded recovery examples from states
the actor actually reaches. This is MuJoCo development, not an Isaac policy result.

| Experiment | Training | Unassisted result | Evaluation completed (UTC) |
|---|---|---|---|
| CPU full-history student002 recovery | 5 complete updates | Falls at 0.304 s | September 9, 2026, 05:35 |
| GPU window student001 | 1,000 updates, 221 s | Falls at 0.428 s | September 9, 2026, 05:55:41 |
| GPU full-history fine-tune001 | 3 full-history updates, fresh Adam, 335 s | Falls at 0.392 s | September 9, 2026, 06:11:15 |
| GPU recovery curriculum001 | 30 updates on three recovery traces, 178 s | Falls at 0.580 s | September 9, 2026, 06:30:52 |
| GPU motor-target curriculum001 | 100 updates, original motor feedback | Falls at 1.306 s | September 9, 2026, 06:42:44 |

All use the same original reset and capped force interface. None opens the door.
The latter rollout passes 15 sensor join checks; this verifies recorded data,
not policy competence. The first command is close to the teacher, but force
errors grow sharply as the body drifts.

## Physical recovery experiment

The frozen actor controls the first 100, 200 or 300 ms, then the privileged teacher
recovers for the remainder of a two-second physical trial. The teacher observes
actual state throughout. Its labels never replace actual preceding motor commands
in sensor packets. No runtime pose writes or external root forces are introduced.

| Actor prefix | Physical duration | Peak torso tilt | Final torso tilt |
|---|---|---|---|
| 100 ms | 2.000 s | 2.337° | 1.006° |
| 200 ms | 2.000 s | 9.080° | 1.091° |
| 300 ms | 2.000 s | 22.968° | 2.546° |

These teacher-assisted recoveries are **not policy successes**. A teacher-only
control reproduces all 500 original transitions exactly over one second. Each
actor prefix also exactly matches the failed actor rollout: positions, velocities
and forces. The intervention, rather than a different reset, changes the outcome.

All three recoveries pass 15 sensor join checks and 21 correction checks, including
independently reconstructed teacher labels, actual teacher forces, uninterrupted
states, original caps, warning counters, reset observations and all 51 camera
frames. Label error is zero. The auditor never steps physics. Gyro/tactile
reconstruction is sampled; independent accelerometer reconstruction remains absent.
The geometric stability audit uses root tilt/height; torso tilt above comes from
the runtime diagnostic. Correction data retains these limitations explicitly.

## Reproduce on another cluster

Use the project environment with MuJoCo 3.12 and pinned H1/Shadow v2 inputs.
`TEACHER` is qualified native sensor capture003, `CAMERA` its audited camera variant,
and `ACTOR` fine-tune001. Preserve model, motor, camera and source hashes. Record
relocations separately; never rewrite an archived receipt to fit another machine.

```bash
python scripts/dexterous/probe_native_approach_recovery.py \
  --teacher-run "$TEACHER" --checkpoint "$ACTOR" \
  --actor-seconds .1 --seconds 2 --output out/recovery-100ms
python scripts/dexterous/audit_native_sensor_capture.py --run out/recovery-100ms
python scripts/dexterous/audit_native_approach_recovery.py --run out/recovery-100ms
```

Repeat with `.2` and `.3` into new directories. The strict correction adapter
verifies bound evidence and separates teacher targets from actual actor inputs.

```bash
python scripts/dexterous/train_native_continuous.py \
  --run "$TEACHER" --camera-variant "$CAMERA" \
  --initialize-actor "$ACTOR" --device cuda --correction-only \
  --correction-run out/recovery-100ms --correction-run out/recovery-200ms \
  --correction-run out/recovery-300ms \
  --iterations 30 --learning-rate .0001 --max-wall-seconds 900 \
  --output out/recovery-training
python scripts/dexterous/evaluate_native_sensor_policy.py \
  --teacher-run "$TEACHER" --checkpoint out/recovery-training/actor.pt \
  --seconds 130 --output out/recovery-student-rollout
```

`--correction-only` explicitly trains the short approach curriculum. The complete
source verifies the embodiment/camera contract but contributes no examples to
these updates. Full-task retention must be tested and reintroduced later. Weights
update only after all three sources; recurrent state resets between sources.
Without correction flags, training remains on the complete original episode.

Both new GPU curricula and their unassisted evaluations have finished. The motor
target actor has its own checkpoint schema and realizes targets through the
original gains, transmissions, target bounds and force caps, using current joint
encoders. Its constant first-target baseline also falls, at 0.942 s. Target
training uses recorded actual force history only; predicted targets never enter
the previous-force sensor channel. All 3,000 correction labels are realizable,
but the complete task is not: 13 motors have later force commands that exceed
the equivalent original target bounds, with a maximum 23.946 Nm discrepancy.
The strict adapter rejects those labels. This mode is an early-balance experiment,
not a replacement for the complete force-controlled teacher.

[Pinned sensor locomotion](SENSOR_LOCOMOTION_BASELINE.md) provides a stronger
hierarchical foundation. Isaac transfer, a complete sensor-only door task, varied
starts and broad coverage remain unfinished.

Evidence lives in `DoorBench-runs/2026-09-09/` with collector or local SHA receipts.
Run Center lists the correction and motor-target experiments. Closed historical
`opening-010`, `opening-015`, `opening-018`, `opening-020` and `opening-027` contact
JSON files were losslessly gzip-packed to recover disk space.
Each includes `packed-contacts.json` and `restore-contacts.py`; restore before
older tools that require the plain path.
