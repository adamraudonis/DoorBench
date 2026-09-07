#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git libegl1 libgl1 libglfw3 tmux
python3 -m venv --system-site-packages /workspace/dex-venv
source /workspace/dex-venv/bin/activate
python -m pip install 'mujoco==3.12.0' 'stable-baselines3==2.7.0' 'gymnasium==1.2.2' pillow scipy pytest imageio imageio-ffmpeg
if [ ! -d /workspace/DoorBenchDex/.git ]; then
  git clone --depth 1 --filter=blob:none --sparse --branch codex/dexterous-humanoid https://github.com/adamraudonis/DoorBench.git /workspace/DoorBenchDex
  git -C /workspace/DoorBenchDex sparse-checkout set doorbench scripts tests docs
fi
cd /workspace/DoorBenchDex
python -m pip install -e .
export PYTHONPATH=/workspace/DoorBenchDex
export MUJOCO_GL=egl
python scripts/dexterous/setup_robot.py
python scripts/generate_dataset.py --out out/dexterous/assets --ids db0055_swing_single --workers 1 --formats mjcf,json --no-thumbs
python -m pip freeze > out/dexterous/runtime-requirements.txt
python - <<'PY'
import torch,mujoco,json
print(json.dumps({'torch':torch.__version__,'mujoco':mujoco.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None}))
assert torch.cuda.is_available()
PY
echo DEXTEROUS_BOOTSTRAP_READY
