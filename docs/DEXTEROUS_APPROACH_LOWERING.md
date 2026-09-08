# Continuous approach and lowering

The native H1/Shadow v2 controller walks from a separated start, stops and lowers
to a manipulation stance using motors throughout. Trial 004 reaches a quiet
0.8712 m pelvis height, 14 mm from the frozen Door55 body waypoint, with zero QP
failures and all physical gates clean. Its heading drifts 11.88°, so it **fails
the original 2° precision-approach criterion**. This is a measured handoff candidate
for adaptive acquisition, not a successful approach/opening benchmark.

`ApproachLoweringController` in
[`doorbench/dexterous/approach_lowering.py`](../doorbench/dexterous/approach_lowering.py)
accepts a native scene, official H1 checkpoint, world XY/yaw goal and optional
controller parameters. Call `command(foot_loads)` once per 2 ms step, apply the
returned original servo controls, then advance the existing plant. The callback
never steps the scene, changes poses or inserts foot constraints. It exposes the
live `.stance`, `.stage`, `.solver_failures` and failure timestamps for a seamless
next controller. Planned foot references stay at the physical landing poses.

The numerical iteration budget increases from 16,000 to 100,000 at unchanged
absolute/relative solver tolerances. Other stance callers retain their original
defaults. This resolves the 21 transient iteration failures of trial 001. Trying
to impose a strong yaw objective caused falls in 002/003; those failures remain
in the [complete trial inventory](../results/dexterous/2026-09-08/h1-approach-lowering.json).

Run from the repository root with the native asset environment:

```bash
PYTHONPATH=. /path/to/asset-python scripts/dexterous/probe_approach_lowering.py \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --door /path/to/doors/db0055_swing_single \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --checkpoint /path/to/unitree_rl_gym/deploy/pre_train/h1/motion.pt \
  --output out/approach-lowering-new
```

The nonzero exit from the retained heading failure is expected. Each output
includes source/archive hashes, exact controls and states, all gates and the
measured `landed-reference.json`; that snapshot is for analysis, never a runtime
reset. Rendering is available through `render_locomotion_approach.py --trial`.

The [frozen continuation protocol](../configs/dexterous/h1-door55-approach-lowering-development.json)
declares the different intermediate criteria before acquisition integration:
physical safety, zero solver failures, quiet support, height/XY accuracy and
collision-free arm preparation from the exact measured heading/root/legs.
It preserves the failed precision-heading score. No full approach-acquire-open-
traverse result has yet been produced by this component.

For a fixed explicit-force adapter, construct `NativeH1MotorAdapter` before the
reset-only affine-to-force representation change so it caches the original
servo limits/gains/bias. Convert returned walking and upper-body servo targets
using the original affine equation and original caps. A stance constructed after
that conversion reads the force-mode motor contract and returns leg forces;
those `.stance.act` entries must not be converted a second time. Record and test
this adapter separately; no physical motor cap is enlarged by the representation.
