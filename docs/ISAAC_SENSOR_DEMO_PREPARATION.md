# Prepare the sensor-feedback reach or grasp

On an **existing ready `shadow-loopback-v2` Isaac host**, this single command
prepares the nineteen-second grasp experiment:

```bash
/workspace/asset-venv/bin/python scripts/isaac/prepare_sensor_demo.py \
  --receipt out/isaac-ready/shadow-loopback-v2/ready.json \
  --task grasp --joint-passive-profile backend-dry-v2 \
  --isaac-python /workspace/venv/bin/python \
  --output out/sensor-grasp-preparation --check-only
```

Use `--task reach` for the separate eleven-second contact-free reach. Replace
`--check-only` with `--dry-run` to inspect the plan without creating output,
running native physics, launching Isaac or downloading anything. An explicit
ready-receipt path supports a checkout/cache outside the default directory.
`DOORBENCH_READY_DIR` remains supported. This command does not provision a GPU
or alter its teardown timer. See [v2 environment setup](ISAAC_V2_READY.md) first.

The command runs five CPU phases, keeping each log and failure:

The asset environment supplies MuJoCo, SciPy, standalone USD and OSQP. Fresh
bootstrap installs OSQP 1.1.3 there. For a host prepared before that addition,
run `uv pip install --python /workspace/asset-venv/bin/python osqp==1.1.3` once.
The separate Isaac environment keeps its pinned simulator dependencies.
Interpreter paths retain their virtual-environment symlinks.

1. Project the frozen reference into the strictly joint-only runtime route.
2. Run the existing native component, including its geometric screen and all
   500-Hz physical, motor, joint, loopback and contact checks.
3. Independently evaluate the actual recorded trajectory and raw evidence.
4. Replay all 5,500 reach or 9,500 grasp sensor packets through the portable
   runtime, preserving its previous-action ownership and original tolerance.
5. Generate a fresh closed-door, resting-operator, contact-free reset preflight
   bound to this host's exact native and USD assets.

Every phase must pass before `isaac-launch.json` and `isaac-command.txt` exist.
Each CPU phase has a fifteen-minute wall cap. `progress.json` reports the current
phase; a zero process exit with a failed or incomplete qualification still stops
preparation. The emitted launch command revalidates source, assets, generated
inputs and qualification receipts immediately before launching Isaac:

```bash
sh out/sensor-grasp-preparation/isaac-command.txt
```

The preparer itself never executes that GPU command. After a future actual run,
execute the `isaac_audit_argv` in `isaac-launch.json`, and inspect both the original
`isaac-trial/balance-report.json` and the independent report. A successful native
preparation does not establish an Isaac grasp. The original distal-pad surface,
opposed five-digit half-second hold and all physical gates remain unchanged.

## Moving to another host

The historical constant-posture calibration contains the original native XML
SHA and motor-contract fingerprint. Generated XML currently includes mesh paths,
so relocating otherwise identical assets can change that identity. The wrapper
creates a **candidate calibration** with only those two identity fields changed;
the same 69 joint angles, gains, timing and provenance remain. It always requires
the fresh native phases above. `calibration-binding.json` records the change and
states that no historical qualification was reused. The committed calibration
and old reports are never edited.

Top-level XML/USD hashes alone are insufficient. Preparation records a canonical
native source-design identity with every referenced asset's bytes, and resolves
the USD dependency closure, including binary hardware layers. Missing geometry,
texture or layer dependencies fail. The imported `OmniPBR.mdl` rendering shader
is an explicit installed-SDK dependency; its bytes are not part of this asset
receipt. The pinned Isaac runtime owns that shader. Source-design equality is
not compiled geometry or dynamic equivalence: the fresh native qualification,
existing live v2 readiness and subsequent actual Isaac checks remain necessary.

Source files and all input hashes are frozen before preparation. Candidate
calibration and generated route/evidence hashes are bound immediately after their
producing phase and verified before and after each dependent phase. The copied
source tree is evidence; commands execute the checked checkout. Launch-time
verification catches changes made after preparation.

## Current scope and validation

This is the existing H1 with corrected Shadow hands, one fixed Door55 reset and
one scripted route. Balance uses encoders, IMU and tactile packets; RGB is
recorded but unused. It does not retarget another robot, learn a vision policy,
walk up to a door, open it or traverse it. Those require separate protocols.
The passive-profile default remains `legacy-tanh-v1`; the command above explicitly
selects the separately tested `backend-dry-v2` adapter.

Both final-source CLI dry runs passed on the actual ready v2 host. They correctly
detected a different XML/calibration identity and required fresh native phases.
They hashed 731 robot asset references, 18 native door asset references and the
USD dependency closure without starting any physics/GPU process or creating a
preparation output directory. Fourteen tests cover failure retention, short
evidence, generated-input edits, launch-time validation and referenced USD assets.
The complete new `--check-only` orchestration has not yet been executed on that
host; the individual native probes/evaluators/replays have their earlier
independent qualification. See [the smoke receipt](evidence/sensor-demo-preparation-smoke.json).
