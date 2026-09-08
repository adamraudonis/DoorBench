# PhysX pose/velocity consistency experiment

The isolated L40S experiment reproduced a numerical difference between executed
rotation and reported angular velocity, even on a contact-free, gravity-free
rigid body. It does **not** qualify a robot task or justify changing the current
TGS32/8 robot settings.

All three physical runs used 2 ms steps for 3 s, the same undamped isotropic free
body, and two identical foot-supported articulations. One articulation began
1.736306 mm lower to expose initial contact resolution. Its ankle effort was
bounded by the same 30 Nm law throughout; all pose/velocity writes occurred only
at reset. Body frames were unscaled, the floor procedural, and COM/inertia,
quaternion dtype, configured timestep and authored settings were recorded.

| Actual run | Position / velocity iterations | Free-body rotation integral error at 3 s | Initial executed/reported rotation ratio | Supported root maximum error, clear / overlap |
|---|---:|---:|---:|---:|
| baseline-003 | 32 / 8 | 4.389194 mrad | 0.960915414 | 0.489 / 1.616 mrad |
| velocity4-001 | 32 / 4 | 4.389194 mrad | 0.960915414 | 0.932 / 1.438 mrad |
| position4-001 | 4 / 8 | 2.141285 mrad | 0.980934814 | 4.362 / 4.316 mrad |

Each completed run retains its original failed `free_body_endpoint_consistency`
check (100 microradian threshold). Its other six fixture checks passed. These
checks cover this deliberately small fixture, not the full robot. Earlier
baseline001 failed EULA bootstrap, baseline002 failed an external ground-asset
lookup, and arithmetic001 failed Triton compilation; all failed logs/reports
remain archived.

The follow-up **actual GPU arithmetic kernel**, small-angle002, directly compared
PTX `sin.approx.ftz.f32` with accurate libdevice sine and float64 math. At the
32-position-substep argument, 1.16926799e-6 rad, fast/accurate sine was
0.960915664. At the 4-position-substep argument, 9.35414391e-6 rad, it was
0.980934741. Those values reproduce the two physical first-step losses to less
than 3e-7 in ratio. Accurate libdevice sine differed from float64 by less than
1.5e-11 relatively at these angles. The kernel ran in 0.432 s on the L40S using
Triton3.3 / PyTorch2.7.0+cu128, with no physics or dependency installation.

The current [official PhysX GPU TGS source](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/gpusolver/src/CUDA/solverMultiBlockTGS.cu)
uses `__sincosf` in each position substep's quaternion integration. CUDA documents
its sine component's [absolute-error bound](https://docs.nvidia.com/cuda/archive/12.8.1/pdf/CUDA_C_Programming_Guide.pdf),
which can be significant relative to tiny angles. The exact arithmetic match
strongly identifies fast-intrinsic small-angle precision as this free-body
mechanism. The installed PhysX binary call graph has not been inspected; the
source-to-binary attribution is explicitly an inference, supported by independent
physical and arithmetic observations.

Supported articulations additionally involve [split-impulse contact solving](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Simulation.html):
position integration can use biased constraint velocities while reported final
velocities exclude that bias. The known warning about more than four TGS velocity
iterations describes a historical behavior change; the warning itself does not
establish a current defect. Reducing position iterations worsened this fixture's
supported-root discrepancy and is not adopted as a robot fix.

## Reproduction and archive

On an already prepared Isaac host, with the usual EULA environment and no other
changes:

```bash
/workspace/venv/bin/python scripts/dexterous/isaac_pose_velocity_fixture.py \
  --variant baseline --output out/pose-velocity-baseline
/workspace/venv/bin/python scripts/dexterous/cuda_small_angle_fixture.py \
  --output out/cuda-small-angle
```

Use a finite process timeout and an owned-node teardown guard. Other fixture
variants are explicit command choices; none changes application defaults.

The complete local archive is
`~/Desktop/Projects/DoorBench-runs/2026-09-08-shadow-loopback/pose-velocity-fixtures/`.
Each run has source, launch arguments, raw trace or failure log, final report,
and `archive-manifest.json` with every file independently SHA-verified against
remote bytes. The compact committed [receipt](evidence/pose-velocity-fixtures-001.json)
binds these manifests, reports, raw traces and numerical comparisons.

The sensor follow-up is separately scoped in
[POSE_DERIVED_GYROSCOPE.md](POSE_DERIVED_GYROSCOPE.md). No external issue was posted.
