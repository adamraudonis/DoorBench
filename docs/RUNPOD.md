# Running DoorBench in Isaac Lab on a rented GPU (RunPod)

This is the exact, replicable path we used. Total cost for validation + a short training run + the hero shot +
a full evaluation is on the order of **$10** (an L40S is $1.09/h on RunPod Secure Cloud; an A40 at $0.49/h also
works). Nothing here needs a browser after the API key exists.

## 0. Requirements (why these choices)

| | |
|---|---|
| GPU | Must have **RT cores** (Isaac Sim renders with RTX): L40S, RTX 6000 Ada, A40, RTX 4090, A6000, RTX PRO 6000. **A100 / H100 are not supported by Isaac Sim.** 24 GB VRAM minimum, 48 GB comfortable for 1000+ envs. |
| Driver | CUDA 12.8-capable host driver (≥ 570). The pod request asks for `allowedCudaVersions: ["12.8","12.9","13.0"]` so RunPod only places you on such hosts. Our pod had driver 580.126. |
| OS / image | Ubuntu 22.04 (glibc 2.35+). Image `runpod/pytorch:1.2.0-rc.162-cu1281-torch271-ubuntu2204` (CUDA 12.8, sshd, `/workspace` volume). |
| Python | **3.11**, in a **uv venv** (`/workspace/venv`). Not conda: conda's ICU library needs a newer libstdc++ than Ubuntu 22.04 ships and Isaac Sim loads the system one first (`CXXABI_1.3.15 not found`). |
| Isaac Sim | `isaacsim[all,extscache]==5.1.0` pip wheels from `https://pypi.nvidia.com` (~10 GB). |
| Isaac Lab | **v2.3.2** (the release paired with Isaac Sim 5.1). `main` now targets Isaac Sim 6 / Python 3.12 and refuses to install on 3.11. |
| Disk | 150 GB volume: Isaac Sim + extension cache ≈ 22 GB after install, plus Isaac Lab, DoorBench (250 MB), checkpoints and videos. |

## 1. RunPod account (once, ~5 minutes, in the browser)

1. runpod.io → sign up → **Billing** → add credit ($30 is plenty).
2. **Settings → API Keys → Create** ("Read & Write"; pod creation needs write). Keep it out of git:
   ```bash
   export RUNPOD_API_KEY=rpa_...          # or: runpodctl doctor  (stores it in ~/.runpod/config.toml, mode 600)
   ```
3. Optional CLI: `brew install runpod/runpodctl/runpodctl` (macOS) or `curl -sSL https://cli.runpod.net | bash`.
   Useful: `runpodctl user` (balance), `runpodctl gpu list` (stock + $/h per data center), `runpodctl pod list`.
4. Register an SSH key on the account (the pod images add it to `authorized_keys`):
   ```bash
   ssh-keygen -t ed25519 -N "" -C doorbench-runpod -f ~/.ssh/runpod_doorbench
   runpodctl ssh add-key --key-file ~/.ssh/runpod_doorbench.pub
   ```
   (`scripts/runpod_pod.py create` also passes the public key as the pod's `PUBLIC_KEY` env var, so step 4 is a
   belt-and-braces measure.)

## 2. Create the pod (one command)

```bash
python scripts/runpod_pod.py create          # POST https://rest.runpod.io/v1/pods with the spec below
python scripts/runpod_pod.py wait            # prints:  ssh -i ~/.ssh/runpod_doorbench -p <port> root@<ip>
python scripts/runpod_pod.py status          # GPU, $/h, hours, spend estimate
```

Pod spec used (`scripts/runpod_pod.py`, `DEFAULT_POD`): Secure Cloud, GPU priority `L40S → RTX 6000 Ada → A40 →
RTX 4090` (first available), 1 GPU, `allowedCudaVersions 12.8/12.9/13.0`, ≥ 8 vCPU and ≥ 32 GB RAM per GPU,
150 GB volume at `/workspace`, 60 GB container disk, port `22/tcp` exposed (direct SSH with a public IP, so
`scp`/`rsync` work), env `PUBLIC_KEY`, `OMNI_KIT_ACCEPT_EULA=YES`, `ACCEPT_EULA=Y`, `PRIVACY_CONSENT=Y`.
The pod we used came up in about 90 s: L40S 46 GB, 16 vCPU, 188 GB RAM, `$1.09/h`.

## 3. Install everything (one command, ~25 minutes, unattended)

Verified end to end on 2026-09-04/05: a brand-new Secure Cloud L40S pod (driver 580.159) went from `create` to
`BOOTSTRAP DONE` with only the commands below; Isaac Sim wheels via uv took ~7 min at 24 MB/s, Isaac Lab v2.3.2 +
torch cu128 ~6 min, DoorBench + extension ~1 min, first headless Kit start ~5 min.

```bash
python scripts/runpod_pod.py bootstrap       # scp scripts/pod_bootstrap.sh + run it in tmux; log: /workspace/bootstrap.log
python scripts/runpod_pod.py ssh tail -f /workspace/bootstrap.log
```

`scripts/pod_bootstrap.sh` (also reachable as `bash isaaclab/cloud/setup.sh` from a checkout on any Ubuntu 22.04 GPU box) does, idempotently: apt libs for Kit (GLU, X11, Vulkan), uv + Python 3.11 venv,
`pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`, Isaac Lab `v2.3.2` with
`./isaaclab.sh --install rsl_rl` (installs torch 2.10 cu128), `pip install -e DoorBench` (+ the Isaac Lab
extension in `isaaclab/` when present), and a first headless `SimulationApp` start that pulls the extension
registry (that first start is the slow part). Expected markers in the log: `ISAACLAB_IMPORT_OK`, `ISAACSIM_OK` and `BOOTSTRAP DONE`.

Known noise you can ignore: on first start Isaac Sim logs errors from `omni.kit.test` / `omni.graph.*.tests`
extensions; they are test-only extensions and the app still reports `app ready`.

## 4. Smoke tests we ran (copy-paste)

```bash
python scripts/runpod_pod.py ssh
source /workspace/DoorBench/isaaclab/cloud/env.sh     # activates the venv, sets ISAACLAB_DIR / DOORBENCH_ASSETS, accepts the EULA
cd /workspace/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless --num_envs 512 --max_iterations 5
```
Then the DoorBench pieces, from the DoorBench checkout (`isaaclab/cloud/README.md` has the details):

```bash
cd /workspace/DoorBench
bash isaaclab/cloud/validate.sh --limit 40        # headless Isaac Sim import + settle + actuate of door.usda and door_rl.usda
bash isaaclab/cloud/train.sh --task DoorBench-Open-Hand-v0 --num_envs 1024 --max_iterations 300
bash isaaclab/cloud/hero.sh                        # docs/media/isaaclab_hero.{png,mp4}
bash isaaclab/cloud/eval.sh logs/rsl_rl/doorbench_hand/<run>/model_300.pt
```

## 5. Tear down (stops billing)

```bash
python scripts/runpod_pod.py terminate       # DELETE /v1/pods/<id>; the volume is deleted with the pod
```
Copy results out first (`scp -i ~/.ssh/runpod_doorbench -P <port> -r root@<ip>:/workspace/DoorBench/results .`).
A *stopped* pod (`python scripts/runpod_pod.py stop`) pauses GPU billing and keeps `/workspace`, but it is tied to
its host: `start` fails with `There are not enough free GPUs on the host machine` whenever that host is busy
(it happened to us after a 3-hour pause). Treat stop/start as a short pause only; for anything longer, copy your
results out and *terminate*, then re-create + bootstrap (~25 min, fully scripted). Revoke the API key in Settings when done.

## Troubleshooting log (what went wrong for us, so it does not for you)

| Symptom | Cause | Fix |
|---|---|---|
| `libstdc++.so.6: version CXXABI_1.3.15 not found (required by .../libicui18n.so.78)` on Isaac Sim start | conda Python's ICU vs Ubuntu 22.04 system libstdc++ | use the uv venv (Python builds from python-build-standalone); or `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib` if you must use conda |
| `Package 'isaaclab' requires a different Python: 3.11.x not in '>=3.12'` | Isaac Lab `main` moved to Isaac Sim 6 / Python 3.12 | `git checkout v2.3.2` before `./isaaclab.sh --install` |
| `ModuleNotFoundError: No module named 'isaaclab'` although `./isaaclab.sh --install` exited 0 | the installer loop skipped the core `source/isaaclab` package (all other sub-packages were installed) | `pip install -e /workspace/IsaacLab/source/isaaclab`; the bootstrap now does this and checks the import (`ISAACLAB_IMPORT_OK`) |
| `pip install -e source/isaaclab` fails: `Failed to build 'flatdict'` / `No module named 'pkg_resources'` | flatdict's setup.py imports pkg_resources; setuptools ≥ 81 in pip's isolated build env dropped it | `pip install "setuptools<81" wheel && pip install --no-build-isolation flatdict==4.0.1` (the exact pin from `source/isaaclab/setup.py`; a newer flatdict does not satisfy it), then install isaaclab (the bootstrap does this) |
| `python -c "import isaaclab"` → `No module named 'pxr'` | `isaaclab` imports USD (`pxr`) at import time, which only exists inside a running Kit app | not an error: import Isaac Lab after `AppLauncher` (as every script does); to check an install use `importlib.util.find_spec('isaaclab')` |
| `pip install isaacsim…` sits for 30+ min with the venv not growing | pip downloads one wheel at a time; on one RunPod host that was 0.4 MB/s to pypi.nvidia.com (10 GB → 7 h) | the bootstrap now uses `uv pip install` (concurrent downloads: 24 MB/s on the same host, ~7 min) for Isaac Sim and torch |
| `wheel 0.48.0 requires packaging>=24.0` | isaacsim wheels pin an old `packaging` | harmless; `pip install "packaging>=24"` |
| pod has no public IP / port | pod still starting | `runpod_pod.py wait` polls `GET /v1/pods/<id>` until `portMappings["22"]` exists (~90 s) |
| `GET /v1/gputypes` → 400 | wrong endpoint | use `runpodctl gpu list` or `GET https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE` |
