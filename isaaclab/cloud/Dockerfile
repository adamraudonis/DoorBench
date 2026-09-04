# DoorBench x Isaac Lab container (alternative to setup.sh).
#
#   docker login nvcr.io                     # free NGC account + API key (https://ngc.nvidia.com/setup/api-key)
#   docker build -t doorbench-isaaclab -f isaaclab/cloud/Dockerfile .
#   docker run --gpus all --rm -it --network host \
#       -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
#       -v $PWD/logs:/workspace/DoorBench/logs -v $PWD/docs/media:/workspace/DoorBench/docs/media \
#       doorbench-isaaclab bash isaaclab/cloud/train.sh --task DoorBench-Open-Hand-v0 --num_envs 1024 --max_iterations 300
#
# Base: NVIDIA's Isaac Sim 5.1.0 container (runs as a non-root user since 5.1; python at /isaac-sim/python.sh).
# Isaac Lab v2.3.0 is installed on top with its own installer (uses /isaac-sim as the _isaac_sim symlink).
# NOT BUILT/EXECUTED ON THE AUTHOR'S MACHINE (no NVIDIA GPU).
FROM nvcr.io/nvidia/isaac-sim:5.1.0

USER root
ENV DEBIAN_FRONTEND=noninteractive OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
    ISAACLAB_DIR=/workspace/IsaacLab DOORBENCH_DIR=/workspace/DoorBench DOORBENCH_ASSETS=/workspace/DoorBench/assets \
    ISAACLAB_PATH=/workspace/IsaacLab
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs build-essential cmake ffmpeg libgl1 libglu1-mesa xvfb && rm -rf /var/lib/apt/lists/*

# Isaac Lab v2.3.0 (+ rsl_rl)
RUN git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_DIR} && \
    ln -s /isaac-sim ${ISAACLAB_DIR}/_isaac_sim && \
    cd ${ISAACLAB_DIR} && ./isaaclab.sh --install rsl_rl

# DoorBench + the extension
COPY . ${DOORBENCH_DIR}
WORKDIR ${DOORBENCH_DIR}
RUN ${ISAACLAB_DIR}/isaaclab.sh -p -m pip install -e ".[usd,mujoco]" imageio imageio-ffmpeg && \
    ${ISAACLAB_DIR}/isaaclab.sh -p -m pip install -e isaaclab && \
    ( [ -f assets/manifest.json ] || ${ISAACLAB_DIR}/isaaclab.sh -p scripts/generate_dataset.py --out assets --workers 8 --no-thumbs ) && \
    ${ISAACLAB_DIR}/isaaclab.sh -p scripts/isaaclab/validate_usd_static.py --workers 8 --quiet --out assets/usd_validation.json

ENTRYPOINT []
CMD ["bash"]
