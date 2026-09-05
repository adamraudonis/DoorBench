# Published reference-motion feasibility baseline

Frozen source: `54a5c7c771d8419c69bd94d432b0ff75a2016daa`; reference index SHA-256 `d9f5c60f6f76992e7a1a7efba0163bab63defe8e7bef0400cfd58aea0fbfe204`. This audit reads the published v1 recordings and matching assets without regenerating or modifying them. It covers all **1,000 doors, 30 families, and 273,050 actor frames**, including approach, manipulation, traversal and unsuccessful attempts.

**None of the 1,000 clips passes every feasibility screen.** Fixed limb lengths do pass. The native benchmark still reports 879 successes, 118 failures and three damaged episodes; 852 of those 879 successful episodes contain unintended avatar/collider penetration in this independent audit. Benchmark success therefore cannot stand in for feasible human motion.

## Results

| Criterion | Doors flagged | Interpretation |
|---|---:|---|
| Fixed limb length error > 0.01 mm | 0 | Exact Euclidean segment check on native actor positions. |
| Active wrist-target residual > 20 mm | 758 | Intended hand target is not reached at one or more active samples. |
| Declared planted-foot drift > 5 mm | 723 | World ankle position moves from its continuous stance anchor. |
| Declared support foot height error > 15 mm | 403 | Rendered sole is too far from its declared ground support. |
| Unintended avatar penetration > 3 mm | 962 | MuJoCo collision geometry query; floor/foot contact has a 5 mm tolerance. |
| Torso/head/leg/foot penetration > 3 mm | 669 | Excludes every arm and hand primitive, separating body clearance from grasp ambiguity. |
| Head penetration > 3 mm | 364 | Subset of the collision result; the avatar head is a 108 mm-radius sphere. |
| Joint-point speed > 3 m/s | 1,000 | Motion continuity screen, not a universal human speed limit. |
| Joint-point acceleration > 15 m/s² | 1,000 | Motion continuity screen. |
| Inferred joint angular speed > 720°/s | 897 | Finite differences of elbow/knee/hip angles. |
| Broad joint-range screen | 3 | Right-elbow flexion exceeds the chosen 150° screen. |
| Aperture smaller than rigid head diameter | 7 | An intrinsic obstruction for this fixed spherical head. |

The maximum native limb-length error is 0.000361 mm; four-decimal web coordinate rounding raises the maximum to 0.138 mm. The original fixed dimensions are upper arm 0.30 m, forearm 0.28 m, thigh/shin 0.43 m each, shoulder offsets ±0.18 m, pelvis height 0.94 m before up to 0.30 m crouch, and a 0.108 m rendered head radius. The generator calls the avatar nominally 1.72 m tall; its upright recorded head center is 1.58 m and rendered head top is 1.688 m. No limbs were rescaled for this audit.

## Concrete blockers

- `db0002_swing_single`: stance drift reaches **0.429 m**. At 5.10 s, a thigh intersects `leaf_slab` by 0.159 m. Its right wrist moves 0.433 m between 5.15 and 5.20 s (8.67 m/s).
- `db0579_swing_single`: the largest declared-stance drift is **0.458 m**.
- `db0602_sliding_single`: the right wrist jumps **0.560 m in 0.006 s** at the final transition (93.27 m/s). The worst-speed joint is the right wrist for 809 doors, an ankle for 175, and a knee for 16.
- `db0066_revolving`: the torso intersects `rotor_shaft` by **0.191 m** at 7.70 s; 31 samples have head intersection. A straight synthetic-base path does not establish a usable route through the rotating compartments.
- `db0017_hatch_ceiling`: active wrist error reaches **1.674 m**. This clip has no collision penetration, which illustrates why collision clearance alone does not establish reach or task feasibility.
- `db0241_hatch_floor`: active wrist error reaches **0.684 m**, stance drift 0.275 m, and a shin intersects `hatch_slab` by 0.173 m. Horizontal traversal needs a descent/support plan rather than planar walking.
- `db0239_gate_swing`: active wrist error reaches **3.886 m**. Wide assemblies need a working-position plan that respects hand reach throughout motion.
- `db0238_pet_door`, `db0627_pet_door`, `db0892_pet_door` have 0.19 m apertures; `db0743_pet_door`, `db0753_pet_door`, `db0901_pet_door`, `db0951_pet_door` have 0.16 m apertures. Each is narrower than the fixed **0.216 m** spherical head. All 15 pet doors require a nonwalking strategy; this report does **not** claim that every larger pet aperture is impossible for every crawling posture.

Frozen `humanoid.py` explains two common defects: the planted-foot branch relocates a foot once it is >0.32 m from the moving root, and the right hand switches directly between an active target and its rest target. The IK preserves segment lengths but does not impose continuous contact, a collision-free body path, or temporal derivative limits. Those are separate planner constraints.

## Methods and limits

Collision coverage uses **2,061,021 `mj_geomDistance` signed-distance calls** after setting recorded native `qpos` and refreshing kinematics on private MuJoCo data. The audit adds 33 private mocap primitives matching the published viewer: 15 cylindrical bones, 16 joint/head spheres and two oriented foot boxes. A conservative AABB broad phase only discards separated pairs; actual intersection depth always comes from MuJoCo. Moving-sphere/box and rotated-cylinder/box fixtures independently check world transforms and known penetration depths. Native box/sphere/capsule/cylinder colliders are queried directly; mesh contacts use MuJoCo’s convex geometry.

These are exact engine queries for the selected mathematical collision shapes, **not exact human-anatomy certification**. The avatar is stylized; feet are rigid boxes and cylinders have no soft tissue. Native NPZ actor coordinates are used to avoid web rounding. Visual-only door parts, avatar self-collision, continuous motion between samples, stability, center of mass, support friction, actuator limits and force closure are not certified. Floor supports are declared by the recorded `foot_contact` array, which the web JSON does not carry. Positive foot clearance above a declared ground stance is a support-height defect even when no collision occurs.

Only an active right-wrist intersection within 9 cm of its target on operator/lock/latch/sensor geometry is classified as intended contact; this narrow exemption does not prove a grasp. Other arm, torso, head and foot intersections remain flagged. Samples without penetration are not declared dynamically feasible.

Range screening uses elbow flexion ≤150°, knee flexion ≤155°, hip flexion −30°…130° and hip abduction ≤50°. These are explicit engineering screening thresholds, not medical limits or a specified robot joint model. Joint positions alone cannot recover shoulder/hip axial rotations, wrist orientation or anatomical hyperextension signs. The report preserves measured ranges rather than treating a passing broad screen as complete articulation validation.

## Acceptance criteria for improved motions

1. Keep the original fixed dimensions. Verify both native and exported poses with stated quantization tolerances.
2. Plan a reachable working pose before manipulation; maintain ≤20 mm wrist residual while contact is active, and make grasp/release transitions continuous.
3. Hold a planted foot within 5 mm of its stance anchor and provide genuine support. Introduce a swing phase before relocating it; validate sole height and avoid obstacles.
4. Eliminate unintended body penetrations beyond the explicit numerical tolerance, then add denser or continuous sweep checks and visual-only geometry where relevant.
5. Enforce motion derivatives and articulation limits for the chosen embodiment. Passing the broad audit screen is only a first gate.
6. Treat pet apertures, ceiling hatches, floor hatches and revolving compartments as different capability/planning problems. Return an honest infeasible/unsupported outcome when the fixed body cannot execute the requested traversal.
7. Validate contact forces, balance, friction and actuator feasibility separately before calling a humanoid policy dynamically feasible. Preserve native benchmark outcomes as a separate result.

## Family coverage

Every family is included in the all-door pass. Counts below are flagged doors; each row names one inspected diagnostic fixture.

| Family | Doors | Wrist >20 mm | Stance drift >5 mm | Body intersection | Fixture |
|---|---:|---:|---:|---:|---|
| accordion | 12 | 1 | 0 | 12 | `db0065_accordion` |
| automatic_sliding | 15 | 0 | 8 | 8 | `db0130_automatic_sliding` |
| automatic_swing | 10 | 6 | 4 | 9 | `db0011_automatic_swing` |
| baby_gate | 10 | 10 | 10 | 10 | `db0176_baby_gate` |
| bifold | 30 | 27 | 22 | 30 | `db0004_bifold` |
| blast | 6 | 6 | 6 | 6 | `db0288_blast` |
| cold_storage | 15 | 15 | 15 | 15 | `db0003_cold_storage` |
| dutch | 12 | 7 | 11 | 11 | `db0095_dutch` |
| elevator | 8 | 0 | 0 | 0 | `db0053_elevator` |
| garage_sectional | 18 | 17 | 3 | 16 | `db0148_garage_sectional` |
| garage_tiltup | 7 | 7 | 2 | 6 | `db0005_garage_tiltup` |
| gate_sliding | 10 | 9 | 9 | 10 | `db0033_gate_sliding` |
| gate_swing | 40 | 39 | 38 | 40 | `db0006_gate_swing` |
| hatch_ceiling | 8 | 8 | 2 | 0 | `db0017_hatch_ceiling` |
| hatch_floor | 10 | 10 | 2 | 10 | `db0241_hatch_floor` |
| pet_door | 15 | 1 | 0 | 15 | `db0045_pet_door` |
| pivot | 20 | 14 | 18 | 20 | `db0044_pivot` |
| revolving | 15 | 10 | 1 | 15 | `db0066_revolving` |
| rollup | 15 | 15 | 4 | 14 | `db0001_rollup` |
| saloon | 12 | 11 | 0 | 6 | `db0031_saloon` |
| ship_watertight | 10 | 10 | 10 | 10 | `db0168_ship_watertight` |
| sliding_bypass | 35 | 35 | 34 | 35 | `db0008_sliding_bypass` |
| sliding_single | 100 | 76 | 89 | 100 | `db0007_sliding_single` |
| stall | 15 | 10 | 13 | 15 | `db0054_stall` |
| strip_curtain | 8 | 6 | 0 | 7 | `db0037_strip_curtain` |
| swing_double | 76 | 51 | 53 | 76 | `db0010_swing_double` |
| swing_single | 440 | 342 | 358 | 438 | `db0002_swing_single` |
| turnstile_fullheight | 10 | 4 | 3 | 10 | `db0187_turnstile_fullheight` |
| turnstile_tripod | 10 | 3 | 0 | 10 | `db0202_turnstile_tripod` |
| vault | 8 | 8 | 8 | 8 | `db0124_vault` |

## Reproduction

```sh
.venv/bin/python scripts/audit_reference_feasibility.py --workers 6 --out out/reference-feasibility-v1
.venv/bin/python -m pytest -q tests/test_reference_feasibility_audit.py
```

Four focused fixtures pass. Machine-readable results are `out/reference-feasibility-v1/summary.json` and one JSON per door under `out/reference-feasibility-v1/doors/`, with source hashes, metrics, failing criteria, collision examples and worst-motion sample times. `--doors families` runs one representative per family; `--doors id1,id2` restricts a diagnostic run. No reference generator, recording, viewer or published asset was edited.
