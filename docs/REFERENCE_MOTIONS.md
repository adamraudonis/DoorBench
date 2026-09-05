# Reference humanoid motions

Every door has a deterministic recording of its **primary core scenario**, using seed 0 and the full MuJoCo model. Open a door in the [catalogue](https://adamraudonis.github.io/DoorBench/) and choose **Play reference**. Scrub the timeline, change speed, or enable **Mechanism contrast** to inspect brown doors with gold hardware.

The door moves under the existing `scripted_hand` policy in native MuJoCo. An original articulated figure follows a procedural approach, hardware operation and traversal reference. Its limbs keep fixed metric lengths; hand targets beyond reach are measured and shown in orange. It is useful for reviewing task sequences and starting retargeting work.

**This is a kinematic motion reference, not a dynamically controlled humanoid baseline.** Generalized joint forces actuate the door. The figure does not generate those forces through physical hand contact; balance, collision avoidance, grip and actuator feasibility are not certified. Its display path differs from the benchmark's synthetic base. Hand release in the reference does not imply the oracle's generalized forces have ended. Failed attempts and damage remain in the dataset. Small pet openings and some overhead hardware are unsuitable for the fixed-size figure.

## Download and reproduce

The [Hugging Face release](https://huggingface.co/datasets/adamraudonis/DoorBench) contains the full recordings. See [download instructions](DATASET_RELEASE.md) for component archives and checksum verification.

```bash
python -m doorbench.reference.record --assets assets --doors all \
  --fps 20 --workers 6 --out out/reference-motions
```

The command runs the normal benchmark runner with a read-only recording observer. It preserves the original policy, timestep, strength limits, scenario criteria and episode termination. A regression test compares observed and unobserved episode outcomes and event timelines.

The initial `lead_in_s` seconds of the **actor timeline** depict an approach (at least 2 seconds; longer for distant gate starts) while the door remains at its initial pose. Native **physics time** starts at zero after that presentation lead-in. Native states are sampled at approximately 20 Hz, with the exact terminal state included. Sampled controls/forces are observations, not a complete control-rate action log; replay the policy to reproduce dynamics.

## Files and coordinates

`index.json` lists every door, source SHA256 values (`spec.json`, `model.json`, `door.xml`), generator hashes, file hashes, outcome and reach diagnostics. The format is `doorbench.reference-motion.v1`. All positions use metres, angles radians, time seconds, and Z up.

| File | Content |
|---|---|
| `clips/<door_id>.json` | Browser recording, native joint names/addresses, scalar joint coordinates, 16-joint avatar positions, target positions, phases, outcome and limitations |
| `clips/<door_id>.json.gz` | Byte-equivalent gzip recording for efficient browser delivery |
| `trajectories/<door_id>.npz` | Compressed numeric arrays, readable with NumPy and `allow_pickle=False` |

Native arrays: `time`, `qpos`, `qvel`, `ctrl`, `tau`, `body_pos`, `body_quat`, `base`, `target`. Body quaternions use MuJoCo's WXYZ order. `tau` contains strength-clamped policy generalized forces at each sample. `base` is the synthetic benchmark base, not the articulated figure's root.

Actor arrays: `actor_time`, `actor_joints` (frames × 16 × XYZ), `actor_root`, `hand_target_error`, `foot_contact` (left/right). Foot contact is the procedural stance schedule, not a force-sensor label. The canonical joint order and bone graph are also in each JSON clip. Arms use 0.30 m and 0.28 m segments; legs use 0.43 m segments. The fixed-length IK clamps unreachable targets instead of stretching the limbs.

```python
import json
import numpy as np

clip = json.load(open("out/reference-motions/clips/db0002_swing_single.json"))
with np.load("out/reference-motions/trajectories/db0002_swing_single.npz",
             allow_pickle=False) as motion:
    print(clip["outcome"]["outcome"], motion["qpos"].shape)
    actor_pose = motion["actor_joints"][40]  # actor timeline, about 2 seconds at 20 Hz
```

The viewer interpolates recorded scalar coordinates and applies all recorded joints directly, including coupled/linkage joints. It does not run the kinematic loop solver over recorded physics a second time. Linear interpolation between sampled skeleton positions is a display approximation; use the canonical samples for analysis.

## Scope and results

This release records one primary scenario per door. It does not replace the benchmark's multi-scenario, multi-seed scores, and the optional human-interaction suite is separate. Outcome totals are in the released index. A successful door task does **not** certify humanoid contact feasibility. Source geometry issues are tracked in [the takeover review](review/takeover/REVIEW.md) and [Blender screening](review/blender/REVIEW.md).

The avatar geometry and motion code are original DoorBench code under the repository MIT license; no third-party humanoid mesh or trained robot policy is redistributed in this component.

## Animated Blender scenes

Create an optional packed `.blend` with source door geometry, the recorded native body motion, and an original capsule figure. Prepare the appearance job in the project Python environment, then run the exporter in Blender (tested with Blender 5.2.1):

```bash
python - <<'PY'
import json
from pathlib import Path
from doorbench.appearance.pipeline import prepare_job
out = Path("out/blender-reference")
out.mkdir(parents=True, exist_ok=True)
job = prepare_job("assets", "db0079_sliding_single", out,
                  quality="photo", width=960, height=960, view="iso")
(out / "job.json").write_text(json.dumps(job))
PY
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_reference_motion.py -- \
  --job out/blender-reference/job.json \
  --clip out/reference-motions/clips/db0079_sliding_single.json \
  --trajectory out/reference-motions/trajectories/db0079_sliding_single.npz \
  --out out/blender-reference/db0079.blend \
  --render-time 3 --image out/blender-reference/db0079.png
```

On macOS, replace `blender` with `/Applications/Blender.app/Contents/MacOS/Blender`. To use downloaded CC0 maps, pass `texture_library=load_texture_library("out/appearance-textures/manifest.json")` to `prepare_job`, importing the function from `doorbench.appearance.textures`.

Frame 1 represents actor time zero; native door playback begins after the clip's `lead_in_s`. The default camera covers the whole trajectory and stays fixed. An explicit snapshot camera retains its calibration and may crop the figure. The scene packs textures, animation keyframes, and provenance; its adjacent JSON records input and output checksums. Native sample poses are preserved, with linear position/quaternion-component interpolation between samples. The figure remains a kinematic visual reference: overlaps, reach failures, balance, and physical hand contact are not repaired or certified by the exporter. Animated scenes are generated on demand and are not included in the release archive.
