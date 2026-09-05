# Physical realism audit — 2026-09-04

Snapshot: master `eb4729a86c7d2e964cd766d28c2a5666d74e100a`, before takeover fixes. Scope: original `prompt`, handoffs, generator/IR/export/QA/benchmark sources; all 1000 persisted specs, models and QA reports; fresh MuJoCo distance measurements, 159 optimized closer-loop sweeps, and two closer dynamics comparisons. Raw measurements: [physics_audit_metrics.json](physics_audit_metrics.json). These are geometric/dynamic findings, not a claim of measured agreement with real products.

## Highest-priority verified defects

1. **26/26 flat-track sliding doors lose rail coverage at full travel.** The rail is centered on the opening, despite one-sided travel (`doorbench/geometry/other.py:146`). On `db0079_sliding_single`, the rail ends at x=1.117 m; the leading wheel center reaches x=1.4805 m, **363.5 mm beyond the rail**, with a shortest wheel–rail gap of 313.6 mm. Even closed, both wheels have a 6 mm gap from the rail. Wheels are visual geoms rigidly attached to a mathematically constrained sliding leaf (`other.py:220`): it cannot fall off in simulation, so `free_opens` passing proves no physical support. Every affected door is signed off.
2. **130/130 generated wall-bumper stops are detached from static supports.** Measured closest signed distance from each `wall_bumper_stop` to every other world geom exceeds 3 mm. On `db0024_swing_single`, the nearest support is the floor, 325 mm away; the jamb is 738.9 mm away. The builder places one floating cylinder at a calculated strike position (`doorbench/geometry/hinged.py:593`), without a stem, mounting plate, or supporting wall. All pass clearance because separation is never a clearance failure.
3. **Five closer loops cannot close geometrically.** For all 159 connect-loop doors, swept 25 primary-angle configurations, enforced joint-equality couplings, and optimized both arm angles against the two MuJoCo connect anchors. Only `db0188`, `db0432`, `db0549`, `db0937` fail at 13.33 mm; `db0585` fails at 12.00 mm. Their rising carriers lift a planar linkage whose other endpoint is fixed (`doorbench/geometry/hinged.py:149`, `doorbench/geometry/common.py:1174`). `viewer/src/kinematics.test.ts:119` explicitly tolerates this defect; the dataset-wide test likewise adds allowable rise to its 1 mm threshold. An apparently green test therefore preserves known invalid geometry.
4. **Closer QA evaluates different damping physics from the benchmark.** `qa.py:250` steps raw MJCF; asymmetric closing and backcheck are installed only in `DoorEnv._install_passive_callback` (`benchmark/env.py:160`). Same exported door, same QA-style 60° initialization, same 12 s simulation: `db0012` reaches 6° in **1.212 s raw / 4.128 s with the environment callback**, peak speed 1.277 / 0.284 rad/s; `db0188` 0.888 / 1.900 s, 1.713 / 0.623 rad/s. The check only examines final angle. It does not validate time windows, latch speed, delayed action, hold-open, or backcheck.

## What signed_off currently establishes

All **1000/1000** stored QA files report all their present checks passing. Coverage is conditional, with no mandatory checklist completeness requirement (`doorbench/qa.py:299`):

| Present check | Doors |
|---|---:|
| Load full/simple/minimal, clearance, mass, settle full/simple/minimal, URDF load, USD open, RL USD open | 1000 each |
| Hold under applied opening effort | 601 |
| Operator actuation opens | 475 |
| Latch spring returns | 308 |
| Relatch | 297 |
| Closer returns | 263 |
| Free opening | 252 |
| Locked operator still holds | 40 |

There are **zero** sign-off gates for attachment, track support, handle return, keypad access-control correctness, each individual dog holding, closer-loop residuals, material appearance, or per-asset human/vision approval. `clearance.py:111` resolves polynomial equalities/tendons but not point connections, `clearance.py:241` skips unlimited mechanism joints (the closer pinion/elbow are unlimited), and export body-contact exclusions remain active. Its “exhaustive” wording overstates discrete independent-joint sampling; it cannot establish all multijoint configurations are collision-free. The 2 mm general and 12 mm hinge tolerances and hardware/leaf exemptions are explicit approximations (`clearance.py:30`, `clearance.py:179`).

## Mechanisms and export fidelity

- **Closer arms do not transmit the modeled closing spring.** Main/elbow joints have only 0.01 damping, no pinion spring (`common.py:1175`, `common.py:1193`); `hinged.py:46` puts spring/damping directly on the door joint. The 333 non-`none` closer specs include 183 surface, 21 concealed, 28 floor, 37 spring-hinge, 22 gate, 13 electromagnetic-hold, 15 automatic, 6 pneumatic and 8 gas-strut models. Automatic arms and pneumatic cylinders are static decorative geoms (`common.py:1135`, `common.py:1151`). Hold-open and delayed-action catalogue fields do not become a physical hold mechanism or delayed-action law in the callback.
- **URDF loading is structural validation only.** Springs, breakable welds and loop closures are namespaced extensions; unilateral latch tendon behavior is replaced with bilateral mimic (`export/urdf.py:1`, `export/urdf.py:160`). Generic URDF consumers do not reproduce full MuJoCo mechanics.
- **Full USD intentionally omits connect constraints and physical weld enforcement.** They become JSON metadata (`export/usd.py:585`), and one-sided tendons require environment logic. A USD joint-count check is not PhysX dynamic verification (`qa.py:272`). Canonical RL USD freezes extra mechanisms, releases some blocking parts, and omits leaves beyond the second (`export/usd.py:665`); scores from that representation are not proof of full-mechanism operation.
- **Mass agreement is internally enforced, not independent calibration.** `build.py:72` rescales leaf bodies to the same derived mass used by `qa.py:101`. Worst persisted discrepancy is only 0.091%, but this primarily verifies export bookkeeping. Friction, seal drag, armature floors and damage forces retain heuristic/calibration assumptions (`physics.py:84`, `physics.py:119`, `build.py:59`, `physics.py:267`), not per-hardware measured torque/force curves. Damage is threshold labeling, not plastic deformation or fracture (`benchmark/labels.py:300`).

## Handoff corrections and remaining behavior gaps

- The handoff says QA only drives dog 0. **Current master actually torques every dog** (`qa.py:171`). Six ship doors have independent levers and four have wheel dogging; an all-dogs actuation pass does not prove that each individual dog independently prevents opening when every other dog is released. No such negative test is signed off. Wheel mechanisms use ideal equality coupling without a fully modeled transmission.
- The handoff says keypad operation is only API driven. **Current master already detects physical button presses** (`benchmark/labels.py:255`) and unlocks on a matching suffix of the collected sequence. Across 28 keypad-lock doors there is no timeout, wrong-attempt counter, electronic lockout, mechanical combination/set semantics, or QA gate. A wrong prefix followed by the correct code succeeds under current logic; documentation must distinguish this permissive sequence model from real hardware.
- Many lever/knob operators already have springs (`common.py:891`), but `latch_returns` checks the bolt rather than the handle (`qa.py:204`). No measured settling-time/chatter check exists. Conversely, some no-return operators have positive catalogue spring fields that their generated joint does not use; a return taxonomy must separate knobs/levers from dog levers, slide bolts and handwheels before enforcing a blanket rule.

## Recommended order

First prevent false approval: add required geometry/support/loop checks, document untested behavior as unverified, and separate inspection status from simulation smoke tests. Fix rail coverage and actual roller contact, mount stops to existing supports, and make the five rising-hinge closer configurations mechanically compatible. Then calibrate closer dynamics through the actual linkage and validate both exported standalone behavior and each supported environment. Extend access-control/handle/multiple-lock tests with negative cases. Real-product source measurements and explicit tolerances are still needed before describing the dataset as physically accurate across all mechanisms.

## Takeover repair: rising-hinge closer (same session)

Implemented handoff option (b): the five cam-lift doors retain their rising hinges and existing closer parameters, with a **passive vertical shoe, visible guide/mounting plate, forearm neck and pivot pin**. The guide stroke is the actual maximum rise plus 2 mm assembly allowance. This is a generic mechanically compatible shoe, not a claim that the named commercial closer supplies this exact mounting kit. A gravity-only substitution was rejected because it would not close `db0432` against the current resistance: approximately 4.77 N m gravitational return versus 6.75 N m friction/stiction. No hidden compensating spring or extra gravity torque was introduced.

Added `doorbench/linkage_qa.py`, with native joint limits and every free mechanism joint on both sides of a connect constraint. Equality-driven carriers remain prescribed inputs. All active loop endpoints are solved together; the gate requires **less than 1 mm** separation across rest plus at least 25 positions per leaf joint, and errors fail. This **1 mm loop-closure tolerance** is separate from the **3 mm static-support distance used for the original audit findings**; it does not establish support, physical contact, or collision clearance. The original shipped dataset scores 995/1000 on this new gate, rejecting exactly the five known incompatible shoes. Its gate result has been handed to the maintainer for sign-off integration and full regeneration.

| Verification on five repaired temporary full exports | Result |
|---|---:|
| Original optimized loop residual | 12.00–13.33 mm |
| Repaired Python gate, 25 angle samples plus rest | all 5 pass; worst <0.000010 mm |
| Repaired viewer generic solver, 61 angle samples | all 5 pass; worst <0.000002 mm |
| Smooth torque-servo opening, 16 s / 8000 MuJoCo steps per door | worst loop separation <0.0194 mm |
| Existing load, clearance, mass, settle, closer-return and export QA | all 5 pass |
| Independent gate fixtures plus five-door regression tests | 18 passed |

The dynamic servo test reached 87.8–98.4° depending on the authored 90/100° limit; full endpoint coverage comes from the separate kinematic sweep. The synthetic fixed-shoe fixture independently fails at 10 mm, while adding a movable frame-side shoe passes. A virtual-work test verifies gravitational torque against simulated moving mass and the actual cam pitch. MJCF currently defaults to 9.81 m/s² while the spec reports 9.80665 m/s²; the torque comparison uses the actual simulated gravity, documenting this small pre-existing difference.

Removed the rising-hinge exception from `viewer/src/kinematics.test.ts`: the dataset assertion now enforces the same 1 mm limit, and the cold-storage regression requires the solver to own `closer_shoe_slide`. Those dataset-dependent assertions require regenerated assets; the repaired five models were separately verified through the actual viewer solver from temporary exports. Raw after results: [rising_closer/verification.json](rising_closer/verification.json). Personally inspected render evidence: [before at 90°](rising_closer/db0188_before_90_shoe.jpg), [after at 0°](rising_closer/db0188_0_shoe.jpg), [45°](rising_closer/db0188_45_shoe.jpg), [90°](rising_closer/db0188_90_shoe.jpg), and [full door at 90°](rising_closer/db0188_90_iso.jpg).

**Remaining coverage limits:** Retaining lips/end caps for the shoe are not modeled; its prismatic joint supplies that restraint. This is a simplified guide assembly.  this gate establishes point-loop feasibility, not continuous collision freedom, physical attachment of the rest of the door, force transmission, full parameter calibration, hold-open/backcheck behavior, or equivalent URDF/PhysX mechanics. The closer spring still acts at the door hinge. Full USD and URDF still require the export limitations above. Rail and stop repairs are coordinated separately; global regenerated counts and full-suite status belong in the maintainer's final report. No shared dataset assets, task board, README, branch, commit, cloud job, or remote service were changed by this audit task.
