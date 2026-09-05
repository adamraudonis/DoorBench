# Isaac Lab integration — verification status

Written on an Apple-silicon Mac **without an NVIDIA GPU**: Isaac Sim / Isaac Lab cannot run here. This file lists
exactly what was verified locally and what awaits the GPU run (task board I4).

## Verified on this machine

| what | result |
|---|---|
| USD export rewrite (`doorbench/export/usd.py`): default prim = door root, `Env` (static) + `Articulation` (fixed `base` link, single tree), PhysX applied schemas (`PhysxArticulationAPI`, `PhysxRigidBodyAPI`, `PhysxJointAPI`, `PhysxJointAxisAPI:rotX/transX` friction efforts, `PhysxCollisionAPI`, `PhysxConvexHullCollisionAPI`, `PhysxMimicJointAPI` with `rel referenceJoint`), `/PhysicsScene` outside the default prim, per-joint `doorbench:*` metadata | regenerated 1000 doors in 37 s (`scripts/generate_dataset.py --formats mjcf,urdf,usd,json`); MJCF / URDF / thumbnails byte-identical; `n_signed_off = 1000` (QA now also requires `usd_rl_opens`) |
| New canonical `door_rl.usda` (8 links / 7 joints for every door, `doorbench:rl` metadata) | 1000/1000 written; slot histogram in `assets/usd_validation.json` |
| `scripts/isaaclab/validate_usd_static.py` over all 1000 doors, both files | **1000/1000 pass** — full: 3 618 joints, 4 650 rigid bodies, 21 719 colliders, 5 347 mesh references resolved, 0 warnings; rl: 7 000 joints, 8 000 rigid bodies, 20 600 colliders, 57 warnings (doors that cannot open by spec: jammed / interlocked / child-locked, and 35 doors without a handle site → leaf point used). Checks: stage metadata, single articulation root, fixed base, tree connectivity, mass/inertia > 0, joint frames consistent through body0/body1 (anchor & axis), limits, drives (gains, spring targets vs `model.json` within 1e-3), friction efforts == IR Coulomb values, collision APIs + physics materials, mesh references, JSON attributes, RL slot consistency |
| `pytest -q tests/test_doorbench.py` | pass (6 tests) |
| `pytest -q tests/test_mujoco_import.py` on the regenerated dataset | pass (261 tests: MJCF/URDF still byte-identical, QA sign-off intact) |
| `pytest -q tests/test_isaaclab_ext.py` (new) | pass: static validation of one door per family + 20 random, RL structure / meta, doors index & easy-100 curation, hand USD, `py_compile` of the extension + scripts, offline API-name checklist |
| `python -m py_compile` of every new file | pass |
| `scripts/isaaclab/check_api_names.py` | 130 Isaac Lab / rsl-rl symbol references in 22 files, all present in the Isaac Lab **v2.3.2** reference list; `--source <IsaacLab v2.3.2 clone> --source <rsl_rl v3.1.2 clone>` additionally resolves every symbol in the real tree and checks the keyword arguments of 128 config-class / function calls against the fields defined there (2026-09-05, all pass) |
| Isaac Lab v2.3.2 API audit (after the train stage crashed on `dump_pickle`) | `isaaclab.utils.io` lost `dump_pickle`/`load_pickle` (isaaclab 0.47.0, security) -> local replacement in `scripts/isaaclab/_common.py`; `RslRlVecEnvWrapper.get_observations()` returns a TensorDict (no `(obs, extras)` tuple); the policy normalizer is `policy.actor_obs_normalizer` (play.py export fixed); `empirical_normalization` is deprecated in rsl-rl 3.1 (dropped from the PPO cfgs); the runner is picked by `agent_cfg.class_name`; `env_cfg.log_dir` is set.  Everything else we use (`ArticulationCfg.articulation_root_prim_path`, `MultiUsdFileCfg`, actuator / schema / sensor cfg fields, `mdp.*`, `set_camera_view`, `scene.stage` / `env_prim_paths`, `sim._app_control_on_stop_handle`) exists in v2.3.2 |
| `isaaclab/cloud/run_all.sh` | validation-first: the validate stage runs the Isaac parity runner over ALL doors when the repo has one (auto-detected `scripts/isaaclab/*parity*.py`, override `PARITY_RUNNER`), else `validate_usd_isaacsim.py --all`; train / hero / eval are OFF unless `TRAIN=1` (skipped stages write `STAGE_<name>_EXIT=skipped`).  Stage output is teed to `logs/stage_<name>.log` and a Python traceback counts as a failure: the 2026-09-05 pod run printed `STAGE_train_EXIT=0` right after the `dump_pickle` ImportError because the simulator shutdown masked the exit status; `_env.sh`'s `set -e` is switched off so a failed stage no longer aborts the pipeline.  Dry-run with a fake launcher (4 scenarios) on 2026-09-05 |
| `isaaclab/doorbench_isaaclab/data/gantry_hand.usda` | generated + validated (6 DoF, fixed base, drives) |

## Not executed (needs the GPU box; run `bash isaaclab/cloud/validate.sh` first)

* `isaaclab/cloud/setup.sh` now wraps `scripts/pod_bootstrap.sh`, which was executed end to end on a RunPod L40S on 2026-09-04/05 (Isaac Sim 5.1.0 wheels via uv, Isaac Lab **v2.3.2**, `ISAACLAB_IMPORT_OK`); see `docs/RUNPOD.md`. `Dockerfile`: still untested.
* `scripts/isaaclab/validate_usd_isaacsim.py`: Isaac Sim import of all doors (spawn, settle, actuate). This is the
  first thing to run; it tells whether PhysX parses the articulations as intended (fixed base, joint frames,
  friction efforts, mimic joints).
* `doorbench_isaaclab` environments (`DoorBench-Open-Hand-v0`, `DoorBench-Open-G1-v0`): never instantiated. Risk
  points, in order of likelihood: (1) `articulation_root_prim_path="/Articulation"` with `MultiUsdFileCfg` (fallback:
  remove the argument), (2) the 0-dim `DoorMechanismAction` (fallback: interval event), (3) contact-sensor filter
  regexes for the G1, (4) G1 body-name candidates for the hands (`.*_palm_link`), (5) rsl-rl config fields if the
  installed Isaac Lab is not v2.3.x.
* Training quality: reward weights follow the benchmark events + Isaac Lab's G1 regularisers but were never tuned.
* `record_hero.py` (viewport render via `env.render()` with `--enable_cameras`), `eval_all_doors.py`.
* PhysX semantics that only a run can confirm: mimic-joint gearing units; whether `PhysxJointAxisAPI` friction
  efforts are honoured on articulation joints in Isaac Sim 5.1 (if not, hinge friction is absent: the legacy
  coefficient is small by design).

## How to report back

After `bash isaaclab/cloud/validate.sh` and one `train.sh` run, paste `assets/usd_validation_isaacsim.json`'s summary
and the first 20 lines of the training log into the task board (I4); fixes will be small and local.
