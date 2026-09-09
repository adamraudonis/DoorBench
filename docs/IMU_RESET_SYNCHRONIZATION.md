# Reset synchronization for the ideal interval gyro

The first actual `pose-delta-angle-v1` grasp experiment retained a stale reset
body transform. Its initial tensor quaternion was near identity, while the first
completed interval reflected the newly assigned root and torso joints. The
resulting first gyro interval contained about0.761rad of artificial rotation.
The existing estimator skips its first gyro sample, so this defect can be
separated from later executed control, but the original producer reset does not
qualify. Keep the original experiment and its failed independent check.

NVIDIA documents that GPU articulation link transforms require an explicit
kinematic update after assigning joint positions. The installed tensor method
delegates to the backend without advancing simulation. Isaac Lab's `forward`
uses this same operation before refreshing Fabric. We use the narrower tensor
operation once after reset assignments and before the producer's reset read.
There is no extra physics step, render, synthetic native-FK measurement or new
actor field. [NVIDIA tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html),
[Isaac Lab implementation](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/sim/simulation_context.html).

Pinned installed evidence on the owned Isaac5.1 host:

| File | SHA256 | Relevant lines |
|---|---|---|
| `omni.physics.tensors-107.3.26+107.3.3/omni/physics/tensors/impl/api.py` | `5dd16f8a37eccc94ac82338d6c1127e785cccf761cae1f6f18ef03d55b0f325f` | 546–560: documented `update_articulations_kinematic`, backend delegation |
| `IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py` | `d95f2eaa138b4fff845ac72439caa2eca5deab4f0d5609e06f163d20eef4df93` | 533–539: forward uses the tensor update |
| `IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py` | `fb179aacdd848906367d01105020e1d36412bd0f1da7187be09f0781ff589309` | 399–446,561–609: reset state assignment/buffer invalidation |

`scripts/dexterous/isaac_reset_kinematic_fixture.py` is a separate two-link
actual-backend test. It assigns two distinct root/joint reset poses, preserves
the stale reads, calls the documented update, checks the measured child pose
against closed-form geometry and verifies that the simulation time and step
index did not advance. The producer then reads the synchronized body directly.
It does not run the humanoid or qualify a grasp. Each launch must preserve its
source hash, process cap and resulting report before a corrected grasp trial.

The actual L40S retry `reset-sync-002` passes **7/7 checks** on September 9,
2026 at **02:59 UTC**. Both reset cases preserve the simulation clock and joint
state, reproduce the synchronized producer read exactly, and match closed-form
positions within 1.77e-7 m and rotations within 3.21e-7 rad. The stale-read
positive control detects both deliberately changed poses. No physics step is
taken during these reset checks. [Report and source identity](evidence/imu-reset-synchronization-002.json).

The independent collector verified the complete archive before this result was
recorded. The first fixture attempt failed in its final Python assertion because
it applied `abs` to a list; that attempt remains retained. The correction uses
NumPy's elementwise absolute value and changes no physical tolerance. This
fixture qualifies the reset synchronization operation, not the humanoid grasp;
fresh native preparation and the subsequent actual Isaac trial remain required.
