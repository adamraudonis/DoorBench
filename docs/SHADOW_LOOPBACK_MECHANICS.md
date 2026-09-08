# Versioned Shadow loopback mechanics

`shadow-loopback-v2` adds the manufacturer's missing unilateral finger constraint
to the native H1 + Shadow model. The original `upstream-v1` model remains available
and is the generator default so earlier experiments remain reproducible.

The [Shadow finger documentation](https://shadow-robot-company-dexterous-hand.readthedocs-hosted.com/en/latest/user_guide/md_finger.html)
specifies **J1 ≤ J2**: the distal joint cannot flex beyond the middle joint. The
existing J0 motor drives their sum. The imported model had the sum transmission
but lacked the passive loopback. Several failed v1 grasp trials therefore folded
the distal link by over a radian while leaving the middle link almost straight.
Those trials do not establish behavior of the intended Shadow mechanism.

V2 adds an unactuated fixed tendon `J1 - J2` to each FF/MF/RF/LF finger of both
hands. It applies no force in the slack region, permits J1 < J2, and limits the
difference from above at zero. Its lower bound is one radian beyond the minimum
allowed by the individual joint ranges, so it cannot restrict legal slack. It is
not an equality constraint. The eight names are `lh_FF_loopback` through
`lh_LF_loopback` and `rh_FF_loopback` through `rh_LF_loopback`.

The generator checks that body masses/inertias, geometry, collision parameters,
joint ranges, original tendon coefficients, actuator transmissions and strength,
support structures, gravity, and solver settings remain unchanged. The resulting
robot still has 69 scalar joints, 61 motors, 53 tactile sensors, 53.239896 kg total
mass, no equality constraints, no mocap bodies, and no gravity compensation.
Default v1 XML was verified byte-identical to the previous generator using the
same pinned upstream checkout.

## Reproduce the model and fixture

From the repository root, with the project's native Python environment active:

```bash
python scripts/dexterous/setup_robot.py --mechanics-profile shadow-loopback-v2
python scripts/dexterous/probe_shadow_loopback.py --output out/dexterous/loopback-fixture-001
python -m pytest -q tests/test_shadow_loopback.py tests/test_dexterous_grasp_verification.py
```

The model defaults to `out/dexterous/robot/h1-shadow-loopback-v2.xml`; its adjacent
audit contains the pinned source revision, complete mechanics profile and XML
hash. Writing v2 to the historical basename `h1-shadow.xml` is rejected. The
profile is [shadow-loopback-v2.json](../configs/dexterous/shadow-loopback-v2.json).
Changing this profile changes mechanics and requires a new qualification receipt.

## Numerical compliance and limits

The manufacturer describes the inequality, not a measured cable stiffness. V2
uses [MuJoCo's fixed tendon limit](https://mujoco.readthedocs.io/en/stable/XMLreference.html#tendon-fixed)
with `solref=[0.004, 1]`, `solimp=[0.9, 0.95, 0.001, 0.5, 2]`, zero margin and no
spring or damping. The 4 ms time constant is two 2 ms physics steps; the generator
rejects a time constant shorter than two timesteps. This is a declared numerical
approximation, not a calibrated physical cable model or a MuJoCo/PhysX stiffness
parity claim.

The isolated fixture uses a serial J2/J1 chain with the original Shadow middle
and distal masses, centers of mass, inertias, armature, damping and frictionloss.
It contains no motors or contacts. For loading cases, an explicitly declared
**test excitation** ramps generalized torque from zero to +1 Nm on J1 and −1 Nm
on J2 during 0.5–0.7 s, then holds until 1.5 s. One Nm is the original J0 motor
force scale; this differential excitation is not the sum motor's transmission.
External fixture torques are not permitted assistance in a robot task.

| Fixture condition | Peak J1 − J2 | Final J1 − J2 | Meaning |
|---|---:|---:|---|
| V2 slack [0.2, 0.6], no excitation | −400.000 mrad | −400.000 mrad | Exactly unchanged, zero constraint force |
| No loopback, excitation from slack | 1842.643 mrad | 1758.021 mrad | Missing mechanism control |
| 20 ms prototype, excitation from slack | 193.989 mrad | 193.989 mrad | Rejected as too compliant |
| V2, excitation from equality [0.6, 0.6] | 7.760 mrad | 7.760 mrad | Within 20 mrad tolerance |
| V2, excitation from slack [0.2, 0.6] | **27.774 mrad** | 7.760 mrad | **Transient fails 20 mrad tolerance** |

At the final loaded state the V2 constraint produces approximately −1/+1 Nm,
opposing the excitation. In the slack-entry stress case the first discrete
crossing is already 23.27 mrad before the limit engages; the next tick reaches
27.77 mrad. The failed transient is retained as a regression test. V2 does not
provide an unconditional 20 mrad guarantee at 2 ms, and increasing stiffness alone
cannot prevent a zero-margin limit's first discrete crossing.

Every robot qualification must separately record `max(0, J1 - J2)` over all eight
pairs at reset and **every physics step**, rejecting any value above 20 mrad.
Individual joint-range checks are insufficient. Existing contact, anatomical pad,
native motor force, penetration and no-assistance gates remain necessary.

## Initialized closed-door hold, 2026-09-08

The first native V2 task diagnostic initialized the canonical opposed grip on
Door55 with leaf and operator exactly at rest. Its MF and LF initial paired joints
were projected to the J1 ≤ J2 halfspace while preserving each J0 sum; the largest
per-joint initializer change was 1.884 mrad. This was a documented reset, not a
runtime state correction. The free body then ran six seconds under native capped
motors with a landed-foot controller and 6 N grip motor feedforward.

The strict audit includes reset and all 3000 subsequent 2 ms steps. It passes
closed start, finite/upright behavior, motor/joint/penetration limits, zero external
assistance, sustained volar-pad opposition, and the new loopback gate. Peak
loopback violation is **1.266 mrad**, peak torso tilt **0.321°**, and final palm
error **1.424 mm**. Final FF/MF/RF/LF/TH loads are
**4.574/5.723/5.229/5.592/9.657 N**. Three hand perspectives at initial, middle and
final samples show the thumb on the opposite side from the fingers; full-body
views retain the original bent-knee stance with upright torso.

This establishes an **initialized hold only**. It does not establish contact-free
acquisition, opening, traversal, Isaac parity, or a vision/tactile-only actor.
The initialized audit exempts contact-free start explicitly; its Boolean field
must not be interpreted as evidence of actual contact freedom.

Local evidence is under `/tmp/doorbench-shadow-loopback/` and its durable copy
`~/Desktop/Projects/DoorBench-runs/2026-09-08-shadow-loopback/`:

- `final/`: versioned generated XML and audit; XML SHA-256
  `3148cbbefa04e65ae813080275008619a3d4ef94f1ca0887c8f51e79630c3c21`.
- `default-soft-prototype/` and `default-fixture.json`: retained failed prototypes.
- `fixture-002/`: exact shared native comparison XMLs, schedules, traces and report.
- `reproduced-fixture-003/`: same metrics from the committed probe script.
- `legacy-check/report.json`: v1 byte-preservation receipt.
- `initialized-hold-v2-001/`: complete native states, controls, 2 ms physical gates,
  strict initialized audit, closeups and source/dependency archive.
  `diagnostic-override.json` identifies the executed standalone script and command;
  it supersedes the archive's ordinary acquisition entry point and explicitly
  records the initialized scope. Generated assets are not committed.

The compact machine-readable summary is
[shadow-loopback-native-2026-09-08.json](evidence/shadow-loopback-native-2026-09-08.json).
