# Sensor balance with explicit arm goals

`SensorArmBalanceController` adds opt-in numeric arm and wrist goals to the
[qualified stationary controller](SENSOR_BALANCE.md). The stationary source and
its default behavior are unchanged. Native trial004 passed all 22 checks over
six seconds while both arms moved out and back. This is **sensor feedback plus
scripted high-level joint goals**, not a learned or vision-based door policy.

## Input contract

```python
from doorbench.dexterous.sensor_arm_balance import SensorArmBalanceController
controller = SensorArmBalanceController(
    robot_xml, motor_contract, sensor_layout, calibration['desired_posture'])
controller.reset_episode()
goals = {n: calibration['desired_posture'][n] for n in controller.goal_names}
forces, diagnostics = controller.force(packet, now_s=0., joint_goals=goals)
```

`packet` remains the exact numeric sensor.v2 packet. `joint_goals` is a separate
high-level command: exactly twelve named H1 shoulder/elbow and Shadow wrist
angles. The first command must match calibration. Later commands must stay
within original joint/control bounds and change by no more than 0.5 rad/s at
the 2 ms cadence. A slower high-level generator must interpolate its command
stream to this cadence. Omitting goals holds the last accepted target; at reset
that target is the original stationary posture. Rejected input makes the episode
terminal until `reset_episode()`.

Torso, leg and finger commands are forbidden. A native transmission check
requires these twelve arm motors to be independent of all other joint columns.
Only those twelve motor target entries can change. The same calibrated torso
and leg targets, encoder/IMU/touch estimator, gravity terms and finite-support
QP remain in use. The estimator reads actual arm encoders, so arm motion changes
its estimated robot mass distribution and resulting leg feedback. No arm-goal
coordinate is interpreted as a body target, world pose or door command.

The unchanged feedback returns all 61 original capped motor forces. Its last
submitted force remains the sole previous-action authority. There is no second
history buffer, force override, hidden active-plant reference or balance oracle.
Finger goals are deliberately excluded; adding them later requires the original
coupled joint/tendon anatomy contract and a separate physical qualification.

## Native component protocol

The reusable evaluator has no simulator dependencies:

```python
from doorbench.dexterous.arm_balance_schedule import validate_schedule, scripted_goals
validate_schedule(schedule, controller.goal_names, duration=6.0)
goals = scripted_goals(local_time_s, calibration['desired_posture'],
                       controller.goal_names, schedule)
```

The frozen schedule is
[`sensor-arm-balance-v1.json`](../configs/dexterous/sensor-arm-balance-v1.json):
1 s initial balance, 1.5 s smooth outward motion, .5 s hold, 1.5 s return, then
1.5 s final balance. Nine joints move 0.08–0.20 rad; three arm joints remain at
their fixed goals. Quintic easing gives zero commanded endpoint velocity.

```bash
python scripts/dexterous/probe_sensor_arm_balance.py \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --motors /path/to/h1-import.motors.json \
  --door /path/to/db0055_swing_single \
  --reference /path/to/actor006/frozen-source/reference.json \
  --output /fresh/path/arm-balance-001
```

The evaluator starts at the same hands-away actor006 reset. A separate unstepped
full-scene model screens 501 joint poses before physical execution, including
all self/environment contacts and original joint limits. At 101 poses it also
checks actual hand-to-environment shapes using complete compatible pair masks,
including receiving-only colliders. This component declares a 5 mm static
clearance requirement. Its accepted path has at least 29.883 mm clearance.
The screen fixes the initial body pose; it is not a guarantee of dynamic safety.

Every subsequent 2 ms physical interval is audited independently under the same
19 stationary gates, plus the static screen, actual arm motion and tracking.
All original force limits, passive loopbacks, collisions and assistance checks
remain. No root, foot or joint pose is written after reset. Goal tracking must
stay within .04 rad, and every requested moving joint must physically achieve
at least 70% of its declared excursion. The final quiet/support checks cover the
last second. True simulator state is used only for evaluation and the disclosed
offline geometric screen, never for balance feedback.

## Results and retained rejections

Evidence lives under `/tmp/doorbench-continuous/out/continuous/`.

| Trial | Outcome | Reason/result |
|---|---|---|
| sensor-arm-balance-001 | Static rejection; zero physics steps | Initial script brushed the slab with RH distal fingers, maximum 2.334 mm penetration. |
| sensor-arm-balance-002 | Static rejection; zero physics steps | Revised script avoided contact but minimum clearance was 2.014 mm, below 5 mm. |
| sensor-arm-balance-003 | 22/22; 6 s / 3,000 steps | Both arm directions corrected using measured geometric clearance. |
| sensor-arm-balance-004 | 22/22; 6 s / 3,000 steps | Same physical trajectory, source/dependency provenance. |
| sensor-arm-balance-005 | 22/22; 6 s / 3,000 steps | Reusable schedule module and explicitly bound static-screen receipt. |

Trial004 measured maximum torso tilt **0.3331°**, arm tracking error
**0.002760 rad**, original joint-stop penetration **0.696 mrad**, and passive
loopback difference **0.224 mrad**. There were zero hand contacts, nonfoot
penetrations, native warnings or failed QPs. Pelvis height stayed
.870684–.872467 m. Final-second horizontal speed was at most 0.0000646 m/s,
with at least 252.17 N actual normal floor support on each foot. These results
do not establish contact manipulation, arbitrary arm trajectories, sensor
noise robustness, Isaac success or door opening.

The trial004 provenance SHA256 is
`65be1b1a8720cb29c70a03d163b811499c8b779c8f49ceb02458114544060fd4`.
The probe freezes imported project sources, calibration, scripted schedule,
repository revision, reset, motor/sensor inputs, actual contact intervals and
endpoint trajectory. Both rejected scripts and their original screens remain
intact. No generated evidence is committed.

Run `pytest tests/test_sensor_arm_balance.py tests/test_sensor_balance.py`:
44 tests pass, including exact stationary force equivalence, denied body/finger
commands, authored limits, slew continuity, hold/reset behavior and sensor-only
boundary checks. A separate review found no arm-goal path into torso/leg targets
or divergence in previous-action history.

To inspect recorded physics:

```bash
python scripts/dexterous/render_sensor_arm_balance.py --trial /path/to/arm-balance-004
```

Rendering replays saved physical states only and is explicitly labeled with the
scripted high-level input. It is not a new simulation or a contact-force audit.

## Static screen binding

`static-screen.json` schema `doorbench.sensor-arm-static-screen.v1` now includes
`binding`: exact SHA256 hashes of schedule, robot XML, door XML, reset JSON,
reference, motors JSON, fixed calibration, sensor layout, and the screen,
schedule and controller source files. **These are file-byte hashes**; the motors
hash here differs from the canonical contract fingerprint inside calibration.
The reset JSON contains the exact full native reset qpos/qvel, root WXYZ pose
and all69 named joint angles. Its world coordinates are evaluator evidence,
not balance-controller inputs. A different reset/model/schedule requires a new
screen; a source screen does not qualify a different imported collision model.

Trial005's provenance hash is
`ed62e22cd65e68014894f92dfec1f1ce2932126fd5545a58704c2813d6925973`.
Its six-second physical trajectory exactly matches004; it independently repeats
all22 gates with the reusable module. The video of004 was personally inspected
at beginning, maximum arm excursion and end: the torso remains upright and both
hands stay clear during this modest out-and-back motion.
