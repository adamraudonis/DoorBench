# Continuous native handle operation, 2026-09-08

One uninterrupted native simulation now acquires the Door55 handle from a
contact-free start, turns it to retract the latch, and holds the door slightly
open with the right hand. The second run ends at **0.086204 rad (4.94°)** of actual
leaf opening. The right grasp remains active for a future left-palm transfer.
This is partial opening, not traversal, broad robot capability, Isaac parity or
a vision/tactile-only policy.

The controller is the shared `AcquisitionTeacher` plus the phase driver
[probe_acquisition_operation.py](../scripts/dexterous/probe_acquisition_operation.py).
The only physical pose writes are the initial reset. The driver subsequently
changes Cartesian reference goals; only the original capped motor forces reach
the free-root robot. It never commands door efforts, changes model strength,
steps the teacher's analytic model, welds a hand or adds external supports.

## Sequence and actual measurements

The driver first executes the [qualified pre-curl acquisition](DOOR55_PRE_CURL_ACQUISITION.md).
At 10.6 s it requires the preceding 0.5 s of all-five-pad qualification before
binding the **actual reached** palm position and orientation in the measured
handle frame. That binding preserves the acquired grip's axial safety margin.

A five-second smooth lever goal runs to 0.87 rad. The opening goal starts only
after actual lever travel reaches 0.80 rad and actual latch retraction reaches
11 mm. A three-second leaf goal then runs to 0.08 rad and remains there. The
right-hand grip feedforward remains active. Actual leaf/handle frames continually
reproject the palm goal; neither the physical door nor physical hand is posed.

| Measurement | Continuous run 002 |
|---|---:|
| Duration / physics interval | 22 s / 2 ms |
| Maximum actual lever travel | 0.842468 rad |
| Maximum actual bolt retraction | 12.212 mm |
| Final actual bolt retraction | 12.091 mm |
| Final actual leaf angle | 0.086204 rad |
| Maximum robot joint-limit excursion | 7.568 mrad |
| Maximum Shadow loopback excursion | 0.949 mrad |
| Maximum non-foot penetration | 0.454 mm |
| Maximum torso tilt | 0.296° |
| Final FF / MF / RF / LF / TH loads | 1.667 / 1.513 / 1.735 / 2.580 / 10.831 N |

All physical gates cover reset and every 2 ms step. The final 0.5 s retains all
five qualified volar pads, and the leaf stays in the declared 0.075–0.10 rad
transfer interval. During lever motion, four isolated 2 ms samples briefly unload
a digit below the required contact-force threshold. There are **no invalid pad
patches** during operation. The audit does not claim uninterrupted five-digit
loading at every instant; these transient unloads remain disclosed.

Run 001 also passed the declared end-to-end gates, but had thirteen isolated
digit-unload samples and a 10.265 mrad maximum robot joint excursion. Its phase
transition changed the leaf goal from zero to its small measured deflection,
creating a wrist-target step. Run 002 continues from the previous **commanded**
leaf goal and uses a five-second rather than four-second press. Both runs and
their exact executed sources are retained.

## Reproduce and integrate

```bash
python scripts/dexterous/probe_acquisition_operation.py \
  --robot out/dexterous/robot/h1-shadow-loopback-v2.xml \
  --door out/dexterous/assets/doors/db0055_swing_single \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --motors out/dexterous/import-v2/h1-import.motors.json \
  --output out/dexterous/continuous-operation-001 \
  --press-seconds 5 --seconds 22
```

The motor contract must name the versioned V2 XML and corrected mechanics.
Both recorded operation runs used `AcquisitionTeacher` source SHA-256
`537dbf70788230857e224a59fb13ff3405de59bdf12bb0171490bc7f43d77b01`.
Later controller changes require requalification. Each artifact includes the
actual imported source tree, dependencies and model hashes plus the executed
phase-driver override. This avoids attributing a scratch driver to an unrelated
repository source snapshot.

For continuous left-hand integration, keep calling `OperationGoals.update()` and
the same `AcquisitionTeacher.force()` while approaching the panel. The right hand
must remain loaded until actual left-palm contact supports the door. Do not load
a saved pose to claim a continuous sequence, and do not start the right release
from an idealized canonical grasp. The collaborator's current left-palm target
is leaf-local `[0.18, -0.048, 0.85]`; that transfer has separate static evidence
and still requires physical integration.

Full evidence is under `/tmp/doorbench-shadow-loopback/continuous-operation-001/`
and `continuous-operation-002/`, with durable copies in
`~/Desktop/Projects/DoorBench-runs/2026-09-08-shadow-loopback/`. The compact
[measurement receipt](evidence/door55-continuous-operation-2026-09-08.json)
includes both runs. Original recordings contain 50 Hz physical states and full
2 ms gates; their final state sample is at 21.982 s, not exactly 22 s. The committed
driver additionally saves an exact terminal state and the measured handle-frame
binding on future runs; these fields are not retroactively claimed for the old
artifacts.

## Portable operation controller

[`DoorOperationTeacher`](../doorbench/dexterous/operation_teacher.py) wraps the
same acquisition motor law using measured poses and mechanism travel. It has no
simulator dependency and returns only the wrapped teacher's native-capped motor
forces. The wrapped teacher's analytic robot FK supplies the measured palm at
the handoff; the physical robot or door is never reset.

```python
operation = DoorOperationTeacher(
    acquisition, joint_geometry,
    qualified_hold_seconds=0.5, min_acquisition_seconds=0.0,
)
force, info = operation.force(
    time_s, root_state, joint_positions, joint_velocities,
    measured_handle_pose, measured_leaf_pose,
    {"operator": lever_rad, "leaf": leaf_rad, "latch": bolt_m},
    measured_hand_body_forces, grasp_qualified=actual_pad_audit_passed,
)
```

Poses contain world `[x, y, z, qw, qx, qy, qz]`. `joint_geometry` supplies
`operator_origin`, `operator_axis`, `leaf_origin`, and `leaf_axis`, all in their
respective measured child-body frames; the axes must be unit vectors. Use the
unchanged asset's joint frames. Door55's operator and leaf targets are 0.87 and
0.08 rad, with a five-second lever press and three-second partial opening.

The event trigger requires a completed acquisition route and 0.5 s of actual
qualified contact. An unload or more than 50 ms without observations resets
that qualification interval. `min_acquisition_seconds` is an optional additional
delay, default zero, so a valid earlier grasp need not wait for a particular
recording boundary. Opening still waits for actual lever and bolt travel.
The caller must continue the full physical audit every simulation step and
report transient digit unloads during operation.

The portable controller passed a separate 22 s native test with all sixteen
declared gates clean: final leaf **0.086459 rad**, maximum lever **0.840455 rad**,
and maximum bolt travel **12.183 mm**. Seven isolated 2 ms samples during
operation briefly unload a digit; no pad patch is invalid, and the final 0.5 s
is fully qualified. Its exact terminal state and handle-frame binding are saved.
See the [portable measurement receipt](evidence/door55-portable-operation-2026-09-08.json).
This establishes the portable native implementation, not an Isaac result.

Add `--portable-wrapper --min-acquisition-seconds 10.6` to the command above to
reproduce its fixed-time native comparison. Use zero for event-triggered engine
integration. The wrapper defaults do not change acquisition forces. Unit tests cover qualification interruptions, measured release gates,
continuous leaf goals, coordinate-frame invariance and invalid measurements.

The default event trigger was subsequently executed in a separate native run:
actual qualified acquisition started lever operation at **7.780 s**, followed by
opening at **12.780 s**. All sixteen gates pass over 22 s; final leaf travel is
**0.087022 rad**, maximum lever travel 0.841613 rad and maximum latch travel
12.200 mm. There are eight isolated digit-unload samples during operation and
zero invalid contact patches. Maximum joint excursion is 8.399 mrad and maximum
loopback excursion 0.948 mrad. The [event-trigger receipt](evidence/door55-event-operation-2026-09-08.json)
retains the exact timings and source hashes. Close hand views at three phases
and final body views were inspected: thumb and fingers oppose the handle and
the torso remains upright. The artifact directory is
`/tmp/doorbench-shadow-loopback/continuous-operation-portable-002/`, also copied
to the durable run archive above.
