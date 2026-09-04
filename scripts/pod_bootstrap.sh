#!/usr/bin/env bash
# DoorBench Isaac Lab GPU box bootstrap (tested 2026-09-04 on a RunPod Secure Cloud L40S, driver 580,
# image runpod/pytorch:1.2.0-rc.162-cu1281-torch271-ubuntu2204, Ubuntu 22.04).  Idempotent; re-run freely.
#
#   Isaac Sim 5.1.0 (pip wheels, Python 3.11)  +  Isaac Lab v2.3.2 (the release paired with Isaac Sim 5.1)
#   in a uv-managed venv at /workspace/venv (NOT conda: conda's ICU needs a newer libstdc++ than Ubuntu 22.04
#   ships and Isaac Sim loads the system one first).  Isaac Lab `main` needs Python 3.12 / Isaac Sim 6 - do not
#   use it with the 5.1 wheels.
#
# Usage on the box (as root):  bash pod_bootstrap.sh 2>&1 | tee /workspace/bootstrap.log
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
W=/workspace; mkdir -p $W; cd $W
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"
ISAACLAB_TAG="${ISAACLAB_TAG:-v2.3.2}"
DOORBENCH_REPO="${DOORBENCH_REPO:-https://github.com/adamraudonis/DoorBench.git}"

echo "== [1/6] system packages"
apt-get update -qq
apt-get install -y -qq --no-install-recommends git git-lfs rsync tmux htop ffmpeg curl ca-certificates \
  libglu1-mesa libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 libx11-xcb1 libxcb1 libgl1 libglib2.0-0 \
  libvulkan1 vulkan-tools libegl1 libsm6 libice6 libfontconfig1 libfreetype6 >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== [2/6] uv + Python 3.11 venv"
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1)
export PATH=$HOME/.local/bin:$PATH
[ -x $W/venv/bin/python ] || uv venv --python 3.11 $W/venv
source $W/venv/bin/activate
uv pip install -q pip setuptools wheel "packaging>=24"
python -V

echo "== [3/6] Isaac Sim $ISAACSIM_VERSION (pip wheels, ~10 GB)"
python -c "import isaacsim" 2>/dev/null || pip install "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com 2>&1 | tail -1

echo "== [4/6] Isaac Lab $ISAACLAB_TAG (installs torch cu128 + rsl_rl)"
if [ ! -d $W/IsaacLab ]; then git clone -q https://github.com/isaac-sim/IsaacLab.git $W/IsaacLab; fi
cd $W/IsaacLab && git fetch -q --tags && git checkout -q "$ISAACLAB_TAG" && ./isaaclab.sh --install rsl_rl 2>&1 | tail -3
# isaaclab.sh can skip the core package and still exit 0 (seen on 2026-09-04: every sub-package installed, `isaaclab` missing);
# install it explicitly and fail loudly if the import does not work.
pip install -q -e $W/IsaacLab/source/isaaclab 2>&1 | tail -1
python -c "import isaaclab, isaaclab_tasks, rsl_rl; print('ISAACLAB_IMPORT_OK', isaaclab.__version__)" || { echo "ISAACLAB_IMPORT_FAILED"; exit 1; }

echo "== [5/6] DoorBench"
if [ ! -d $W/DoorBench ]; then git clone -q "$DOORBENCH_REPO" $W/DoorBench; fi
cd $W/DoorBench && git pull -q && pip install -q -e . 2>&1 | tail -1
[ -d $W/DoorBench/isaaclab ] && pip install -q -e $W/DoorBench/isaaclab 2>&1 | tail -1 || true

echo "== [6/6] first headless Isaac Sim start (pulls the extension registry, up to ~10 min)"
cd $W/IsaacLab
timeout 1500 python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('ISAACSIM_OK'); app.close()" 2>&1 | grep -E "ISAACSIM_OK|Traceback" | tail -2
echo "== BOOTSTRAP DONE  (activate with: source /workspace/venv/bin/activate)"
