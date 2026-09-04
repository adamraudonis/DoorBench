# Third-party assets used by the humanoid-in-the-loop demo

Nothing under `robot_demo/third_party/` is committed (see `.gitignore`); `robot_demo/setup.sh` re-clones the exact
commits below. All of it is royalty-free, BSD-3-Clause licensed, and used unmodified.

| Component | Source | Commit | Licence | What we use |
|---|---|---|---|---|
| MuJoCo Menagerie – `unitree_g1` | https://github.com/google-deepmind/mujoco_menagerie | `e4049d0a3bfd58d2a3081614e6777d4007e3f86a` (2026-09-01) | BSD-3-Clause (`unitree_g1/LICENSE`, copyright HangZhou YuShu Technology / Unitree Robotics; Menagerie packaging by Google DeepMind, Apache-2.0 for the repo tooling) | `g1.xml` (29-dof G1, rev 1.0) + `assets/*.STL` meshes |
| MuJoCo Menagerie – `unitree_h1` | same repo / commit | same | BSD-3-Clause | cloned for reference only; not used in the recorded runs |
| unitree_rl_gym | https://github.com/unitreerobotics/unitree_rl_gym | `276801e46c5d433564f24658bac64f254b7d2d4b` (2025-07-25) | BSD-3-Clause (`LICENSE`, copyright HangZhou YuShu Technology / Unitree Robotics) | `deploy/pre_train/g1/motion.pt` (pretrained sim2sim locomotion policy, TorchScript), `deploy/deploy_mujoco/configs/g1.yaml` (PD gains, scales), `deploy/deploy_mujoco/deploy_mujoco.py` (observation/action construction, ported into `run_g1_door.py`), `resources/robots/g1_description/g1_12dof.xml` (fallback robot model, `--robot rlgym`) |

Python dependencies added to the venv for this demo: `torch` (BSD-3-Clause, CPU wheel), `pyyaml` (MIT), `imageio` +
`imageio-ffmpeg` (BSD-2 / bundled FFmpeg binary, LGPL/GPL – used only as an external encoder process).

The BSD-3-Clause licence requires that redistributions reproduce the copyright notice and disclaimer; the third-party
files are not redistributed by this repository (they are downloaded by `setup.sh`), and the videos in `docs/media/`
are renders produced with these assets. DoorBench's own code and door assets remain MIT.
