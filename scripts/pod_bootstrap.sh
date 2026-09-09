#!/usr/bin/env bash
# DoorBench Isaac Lab GPU box bootstrap (tested 2026-09-04 on a RunPod Secure Cloud L40S, driver 580,
# image runpod/pytorch:1.2.0-rc.162-cu1281-torch271-ubuntu2204, Ubuntu 22.04).  Idempotent; re-run freely.
#
#   Isaac Sim 5.1.0 (pip wheels, Python 3.11)  +  Isaac Lab v2.3.2 (the release paired with Isaac Sim 5.1)
#   in a uv-managed venv at /workspace/venv (NOT conda: conda's ICU needs a newer libstdc++ than Ubuntu 22.04
#   ships and Isaac Sim loads the system one first).  Isaac Lab `main` needs Python 3.12 / Isaac Sim 6 - do not
#   use it with the 5.1 wheels.
#
# Usage on the box:  bash scripts/pod_bootstrap.sh 2>&1 | tee /workspace/bootstrap.log   (root or sudo; DOORBENCH_WORK=/workspace)
# isaaclab/cloud/setup.sh is a thin wrapper around this script.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# Isaac Lab calls tput even without a TTY; some cloud shells inherit an unsupported TERM.
export TERM=xterm-256color
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
# Resolve the invoking checkout before changing directories: BASH_SOURCE may be relative.
HERE_DB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
W="${DOORBENCH_WORK:-/workspace}"; mkdir -p "$W"; cd "$W"
SUDO=""; [ "$(id -u)" = "0" ] || SUDO="sudo -n"
# when this script is run from inside a DoorBench checkout, use that checkout instead of cloning a new one
if [ -f "$HERE_DB/pyproject.toml" ] && [ -d "$HERE_DB/doorbench" ]; then DB="$HERE_DB"; else DB="$W/DoorBench"; fi
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"
ISAACLAB_TAG="${ISAACLAB_TAG:-v2.3.2}"
DOORBENCH_REPO="${DOORBENCH_REPO:-https://github.com/adamraudonis/DoorBench.git}"

phase() {
  echo "$1"
  python3 "$DB/scripts/isaac/status.py" "$DB/out/launch/pipeline.json" "$1"
}
phase "== [1/6] system packages"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq --no-install-recommends git git-lfs rsync tmux htop ffmpeg curl ca-certificates \
  libglu1-mesa libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 libx11-xcb1 libxcb1 libgl1 libglib2.0-0 \
  libvulkan1 vulkan-tools libegl1 libsm6 libice6 libfontconfig1 libfreetype6 >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

phase "== [2/6] uv + Python 3.11 venv"
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1)
export PATH=$HOME/.local/bin:$PATH
[ -x $W/venv/bin/python ] || uv venv --python 3.11.15 $W/venv
source $W/venv/bin/activate
uv pip install -q pip setuptools wheel "packaging>=24"
python -V

phase "== [3/6] Isaac Sim $ISAACSIM_VERSION (pip wheels, ~10 GB)"
# uv downloads the ~10 GB of wheels concurrently (24 MB/s measured) where pip crawled at 0.4 MB/s on one RunPod host.
export UV_LINK_MODE=copy UV_CACHE_DIR="$W/.uv-cache"
# Install the actual CUDA build first. Plain ==2.7.0 also accepts a PyPI build,
# which made the later Isaac Lab upgrade download a second large Torch wheel.
if [ "$ISAACSIM_VERSION" = "5.1.0" ] && [ "$ISAACLAB_TAG" = "v2.3.2" ]; then
  uv pip install "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" "torchaudio==2.7.0+cu128" --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import isaacsim" 2>/dev/null || {
  uv pip install "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com || exit 1
}

phase "== [4/6] Isaac Lab $ISAACLAB_TAG (installs torch cu128 + rsl_rl)"
if [ ! -d $W/IsaacLab ]; then git clone -q --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git $W/IsaacLab; fi
# pre-fetch torch cu128 with uv (fast, concurrent) so isaaclab.sh's pip step finds it installed
uv pip install "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" --index-url https://download.pytorch.org/whl/cu128
# Stream installation progress into the launch log. A trailing `tail` buffered
# every package message until completion and made a slow FUSE install look hung.
cd $W/IsaacLab && git fetch -q --depth 1 origin "$ISAACLAB_TAG" && git checkout -q FETCH_HEAD && ./isaaclab.sh --install rsl_rl
# isaaclab.sh can skip the core package and still exit 0 (seen on 2026-09-04: every sub-package installed, `isaaclab` missing);
# install it explicitly and fail loudly if the import does not work.
# isaaclab's dependency `flatdict` builds from source with a setup.py that imports pkg_resources, which setuptools >= 81
# (what pip's isolated build env picks) no longer ships -> build it once without isolation against setuptools < 81.
FLATDICT_PIN=$(grep -o "flatdict[=<>~!]*[0-9.]*" $W/IsaacLab/source/isaaclab/setup.py | head -1)   # e.g. flatdict==4.0.1
pip install -q "setuptools<81" wheel && pip install -q --no-build-isolation "${FLATDICT_PIN:-flatdict}" 2>&1 | tail -1
pip install -e $W/IsaacLab/source/isaaclab
# NOTE: `import isaaclab` only works inside a running Kit app (it needs pxr), so check the packages resolve instead.
python -c "import importlib.util as u; assert all(u.find_spec(m) for m in ('isaaclab', 'isaaclab_tasks', 'rsl_rl')); print('ISAACLAB_IMPORT_OK')" || { echo "ISAACLAB_IMPORT_FAILED"; exit 1; }

phase "== [5/6] DoorBench"
if [ ! -d "$DB" ]; then git clone -q "$DOORBENCH_REPO" "$DB"; fi
cd "$DB" && pip install -q -e . 2>&1 | tail -1
[ -d "$DB/isaaclab" ] && pip install -q -e "$DB/isaaclab" 2>&1 | tail -1 || true
# Asset generation has its own environment: importing pip's usd-core into Kit's
# Python process risks mixing incompatible USD libraries. MuJoCo QA is required
# even when the consuming simulator is Isaac.
[ -x "$W/asset-venv/bin/python" ] || uv venv --python 3.12.13 "$W/asset-venv"
uv pip install --python "$W/asset-venv/bin/python" -e "$DB" 'mujoco==3.12.0' 'usd-core==26.8' 'osqp==1.1.3' || exit 1
ASSET_CHECK=("$W/asset-venv/bin/python" "$DB/scripts/isaac/check_assets.py" "$DB/assets")
if [ -n "${DOORBENCH_GENERATE_IDS:-}" ]; then ASSET_CHECK+=(--ids "$DOORBENCH_GENERATE_IDS"); fi
if ! "${ASSET_CHECK[@]}" >/dev/null 2>&1; then
  if [ -n "${DOORBENCH_GENERATE_IDS:-}" ]; then
    "$W/asset-venv/bin/python" scripts/generate_dataset.py --out assets --ids "$DOORBENCH_GENERATE_IDS" --workers 1 --no-thumbs
  else
    "$W/asset-venv/bin/python" scripts/generate_dataset.py --out assets --workers 8 --no-thumbs
  fi
fi
"${ASSET_CHECK[@]}"
# Isaac Lab's torch upgrade can replace Isaac Sim's pinned utility dependencies.
# Restore the shared 5.1 / 2.3.2 contract after all editable packages are installed.
if [ "$ISAACSIM_VERSION" = "5.1.0" ] && [ "$ISAACLAB_TAG" = "v2.3.2" ]; then
  uv pip install "filelock==3.13.1" "fsspec==2024.6.1" "markupsafe==2.1.3" \
    "networkx==3.3" "packaging==23.0" "sympy==1.13.3" "numpy==1.26.0" \
    "typing-extensions==4.12.2" "psutil==5.9.8" "fastapi==0.121.0" "wheel==0.45.1" "ipython==8.37.0" "onnx==1.21.0" "osqp==0.6.7.post3" || exit 1
  uv pip install "torchaudio==2.7.0" --index-url https://download.pytorch.org/whl/cu128 || exit 1
  python "$DB/scripts/isaaclab/check_g1_runtime.py" || exit 1
fi
# Native kinematics support for the optional closed-loop teacher; no pip USD in Kit.
uv pip install 'mujoco==3.12.0' || exit 1
# environment file used by isaaclab/cloud/*.sh (validate / train / hero / eval)
{
  echo "# generated by scripts/pod_bootstrap.sh - source me:  source isaaclab/cloud/env.sh"
  echo "source \"$W/venv/bin/activate\""
  echo "export ISAACLAB_DIR=\"$W/IsaacLab\" DOORBENCH_DIR=\"$DB\" DOORBENCH_ASSETS=\"$DB/assets\""
  echo "export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y TERM=xterm-256color"
} > "$DB/isaaclab/cloud/env.sh"

phase "== [6/6] first headless Isaac Sim start (pulls the extension registry, up to ~10 min)"
cd $W/IsaacLab
STARTUP_LOG="$W/isaacsim-first-start.log"
if timeout 1500 python -u -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('ISAACSIM_OK', flush=True); app.close()" 2>&1 | tee "$STARTUP_LOG"; then
  STARTUP_RC=0
else
  STARTUP_RC=$?
fi
if [ "$STARTUP_RC" != 0 ] || ! grep -q '^ISAACSIM_OK$' "$STARTUP_LOG" || grep -q 'Traceback (most recent call last)' "$STARTUP_LOG"; then
  echo "ISAACSIM_STARTUP_FAILED (exit=$STARTUP_RC; see $STARTUP_LOG)"
  tail -30 "$STARTUP_LOG"
  exit 1
fi
echo "ISAACSIM_OK"
echo "== BOOTSTRAP DONE  (next: source $DB/isaaclab/cloud/env.sh; bash isaaclab/cloud/validate.sh | train.sh | hero.sh | eval.sh)"
