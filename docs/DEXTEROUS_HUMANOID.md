# Simulated dexterous humanoid: execution record

Owner-approved objective, September 7, 2026: a full simulated humanoid with two high-DOF hands that opens at least 95% of independently reviewed human-openable DoorBench doors, with vision/touch/proprioception as the final policy inputs. Opening and traversal are separate metrics. No real hardware is required. This is a new experimental track; existing baseline scores are unchanged.

## Current status

The H1/dual-Shadow robot compiles and attaches to a freshly generated `db0055_swing_single` lever door. The source model has 61 actuators (20 for each Shadow Hand), 69 articulated joints plus a free base, 53.239896 kg mass, and 448 three-axis tactile cells. Native state is separated from copied student observations. No door-opening policy has been achieved yet.

A borrowed two-hand reaching policy maintained balance in a short stationary/reaching probe, but crouched substantially and fell on a longer approach. Those are integration results and failures, not a door-opening score. The next step is adapting the body-reaching skill under the full articulation before proceeding to hand contact and the complete task.

See [cluster reproduction and Isaac Sim migration](DEXTEROUS_REPRODUCTION.md) for portable configurations, artifact requirements, and backend limitations. The [approved full plan](DEXTEROUS_PLAN.md) remains the project scope.

## Reproduce the model and native probe

Use a dedicated environment with MuJoCo 3.12.0, PyTorch, Pillow, and the DoorBench package. The optional PPO training stage uses stable-baselines3 2.7.0 and gymnasium 1.2.2. The upstream clone and generated assets go under ignored `out/`, never into source commits.

```bash
python scripts/dexterous/setup_robot.py
python scripts/generate_dataset.py --out out/dexterous/assets --ids db0055_swing_single --workers 1 --formats mjcf,json --no-thumbs
python scripts/dexterous/probe.py --upstream out/dexterous/upstream/humanoid-bench
python -m pytest -q tests/test_dexterous_humanoid.py
```

`setup_robot.py` checks the upstream revision and rejects modified robot/checkpoint files. The audit records joints, actuator ranges, mass, and absence of mocap or gravity compensation. The upstream model includes authored adjacent-part contact exclusions and tendon transmissions, which still require task-specific review. Generated robot XML currently contains resolved external mesh paths and must be prepared on each host.

## Training and observation contracts

`DexterousDoorEnv.observe()` exposes only joint position/velocity, body gravity direction, gyro/accelerometer, tactile arrays, previous actuator command, and optionally two rendered head-camera RGB arrays. Tactile values are local normal/tangential forces clipped to +/-100 N per component; they are idealized simulated force cells, not calibrated pressure sensors. No noise or latency model has yet been added. Head images default to 128x128.

The evaluator retains `DoorEnv`, exact mechanism state, and diagnostics. Policy code receives copied arrays and returns 61 normalized robot-actuator commands. The action adapter cannot command door actuators. This is an API boundary, not a security sandbox against malicious policy code.

The first PPO stage is explicitly **privileged body reaching**, with 19 learned body actuators and the hands held open. It is not the final dexterous or sensor-only policy. It initializes from the upstream reaching checkpoint and adds upright posture and reaching objectives. Read its diagnostics before increasing the training scope.

```bash
python scripts/dexterous/train_reach.py \
  --upstream out/dexterous/upstream/humanoid-bench \
  --robot out/dexterous/robot/h1-shadow.xml \
  --door out/dexterous/assets/doors/db0055_swing_single \
  --output out/dexterous/training/reach-001 --envs 8 --steps 500000 --device cuda
```

## Resource isolation

`scripts/dexterous/pod.py` uses its own allocation journal, never the legacy shared pod record. Creation arms a detached local deadline guard tied to the exact allocation ID; it also refuses to overwrite an active allocation. Copy evidence/checkpoints before termination. The local guard requires this host and its network connection. The current allocation also has an independently running remote deadline guard. Keep its private configuration outside the repository and never include it in artifacts.

`scripts/dexterous/bootstrap.sh` installs a lean native-MuJoCo/PyTorch training runtime. Isaac Lab porting and GPU physics are separate compatibility tasks, not implied by using a GPU for neural-network updates. Runtime dependencies are recorded after installation.

## Sources

Robot and body-skill source: [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench), revision `cb1189039151c8aadaaa987b442da54383c87fab`. The external clone retains its complete LICENSE and third-party model attributions. The reaching adapter follows the documented network/observation convention; imported state dictionaries are loaded with `weights_only=True`.
