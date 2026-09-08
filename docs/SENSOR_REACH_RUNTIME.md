# Portable contact-free reach

The eleven-second component commands a fixed joint route while balancing from encoders, local IMU, foot touch and its own previous action. It controls 30 torso/right-arm/right-hand joints through 26 motor coordinates. It is scripted, uses no vision, and does not establish grasping or opening.

`SensorReachBalanceRuntime(robot_xml, motors, layout, calibration_json, reach_protocol_json, joint_route_json)` requires `reset_episode()` before `force(packet, now_s)`. The public command is 61 original capped motor forces. `previous_action` is that command normalized by the original force caps. Runtime rejects root/door observations, malformed packets, changed gains or protocol, borrowed previous commands and invalid clocks; failures require reset.

Project the source outside runtime with:

```sh
python scripts/dexterous/prepare_sensor_reach_runtime.py \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --schedule configs/dexterous/sensor-reach-balance-feedforward-v3.json \
  --motors /path/to/h1-import.motors.json --output out/reach-runtime-inputs
```

The output route retains only the 30 named joint columns. The original root, door and other joint trajectories are excluded. The protocol binds the exact route bytes, original reference hash and source schedule hash. Pass file paths unchanged to runtime. Preserve the inputs with every run. The frozen protocol follows the first 45% of the route over eight seconds, after one second of settling, then holds for two seconds. The finger controller uses the explicitly declared moving-target damping profile; plant gains, motor caps and passive tendon mechanics stay unchanged.

## Independent qualification

`evaluate_sensor_reach_balance` consumes actual post-step root/joint/contact records and each preceding controller command. Those observations are evaluation privilege only. It loads a separate, unstepped robot model for named transmission mapping and palm FK; it never loads a door or steps physics.

All 5,500 intervals must be present at 2 ms. Existing physical checks must pass, including motor delivery/caps, joint stops, both hands' unilateral loopbacks and collisions. Additional checks require no hand contact, height 0.82–0.92 m, tilt at most 12°, all QPs solved, a cold first packet, and a quiet final second supported by both feet. Each active commanded motor must deliver more than 70% of its requested excursion. Continuous motor-coordinate error must stay below 0.04 rad, and the actual final palm must end within 20 mm of the static initial-body target.

Coupled finger J1/J2 tracking is measured in its actuated sum coordinate. The passive split error is reported separately; individual joint stops and `J1 <= J2` are still independent hard gates. A deliberately reversed split with an identical motor sum fails the verifier. A legal passive split is accepted and disclosed.

## Native evidence

The adapter replayed every packet of the qualified `sensor-reach-balance-006` capture without stepping a plant: all 5,500 commands match to 6.09e-10 Nm. Independent actual-state qualification passed 20/20 grouped checks and retained the source's 23/23 checks. Maximum motor-coordinate error was 0.0171654 rad, nominal individual-joint error 0.0688235 rad, and palm endpoint error 0.00330169 m. The larger nominal finger error is a passive split, not an independently controllable motor error.

Reproduce the two checks with `check_sensor_reach_runtime.py` and `evaluate_native_sensor_reach.py`; each accepts `--run`, `--robot`, `--calibration`, `--protocol`, `--joint-route` and an unused `--output` path. They refuse to overwrite evidence. The committed compact receipt binds all original inputs and recorded arrays. These native results do not establish Isaac success; a fresh actual Isaac run must pass the same evaluator.

The 84 focused tests cover this adapter and the unchanged five-second stationary and six-second arm protocols. Reach tests include strict input projection/hash binding, previous-action ownership, missing/shifted evidence, actual movement, root/FK endpoint error, contact/support and illegal coupled-finger geometry.
