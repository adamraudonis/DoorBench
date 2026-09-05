# Planned reference style: controlled turn-step experiment

**Result, 2026-09-05:** increasing the maximum turn of one foot swing from 20° to 30° produced **11 independent passes out of 12 candidates**. The stall candidate failed clearance and stance checks. The passing candidates were shorter in aggregate, but every changed passing candidate had at least one higher final-clock joint-jerk peak. **There is no global adoption or naturalness claim.** The completed v2 corpus and its generator remain unchanged.

## Controls and scope

The experiment began after all 1,000 v2 jobs completed and the corpus runner exited. A fresh isolated copy included the exact v2 powered/manual-contact builders, original rig, interpolation clearance guard, solver, retimer and independent validator. Four fresh Python workers, each with one numerical-library thread, regenerated all 12 **20° controls first**. Every saved NPZ array exactly matched its completed v2 baseline, and every control passed the independent validator.

The same isolated source then produced 12 candidates. An externally hashed adapter changed only `max_step_yaw_deg` to 30° before IK. The `smooth` body guide, geometry, source outcomes, native recordings, rig, contact semantics and validation gates were retained. Native qpos was checked against the immutable recording at every declared native time. Source role/segment identities and contact geometry sets were checked separately from the changed footstep count and clock. All 24 generation/validation attempts completed in 178.89 seconds.

This is a deliberately varied convenience sample, not a random estimate of all-door acceptance. Selection includes manual manipulation, a powered route, folding/pivot/gate geometry, narrow passages and previously difficult clearance cases. The final v2 cold-room control is used throughout; an earlier v1 descriptive measurement is not an experimental control.

## All 12 results

Durations use authoritative `actor_time`. A short turn swing means authored ankle endpoint travel below 0.08 m and rotation of at least 5°; that definition is descriptive, not a comfort threshold.

| Exact door ID | Duration 20° → 30° (s) | Change | Foot swings | Short turn swings | 30° result |
|---|---:|---:|---:|---:|---|
| db0002_swing_single | 86.51 → 78.52 | −9.23% | 85 → 70 | 31 → 10 | Pass |
| db0079_sliding_single | 48.17 → 45.78 | −4.97% | 44 → 39 | 7 → 3 | Pass |
| db0193_sliding_single | 60.82 → 58.71 | −3.47% | 43 → 40 | 4 → 3 | Pass |
| db0153_automatic_sliding | 29.67 → 29.67 | 0.00% | 20 → 20 | 0 → 0 | Pass; unchanged |
| db0044_pivot | 104.11 → 99.21 | −4.71% | 95 → 88 | 17 → 8 | Pass |
| db0015_swing_double | 75.20 → 72.13 | −4.07% | 66 → 56 | 13 → 2 | Pass |
| db0372_bifold | 43.44 → 42.82 | −1.43% | 28 → 27 | 7 → 5 | Pass |
| db0335_gate_swing | 88.07 → 83.40 | −5.30% | 82 → 75 | 16 → 12 | Pass |
| db0894_cold_storage | 101.64 → 91.35 | −10.13% | 98 → 82 | 25 → 10 | Pass |
| db0103_stall | 62.78 → 59.43 | Rejected candidate | 57 → 50 | 9 → 1 | **Fail** |
| db0457_swing_single | 84.47 → 79.01 | −6.47% | 78 → 65 | 27 → 11 | Pass |
| db0139_swing_single | 88.56 → 77.82 | −12.13% | 88 → 70 | 38 → 13 | Pass |

Among the 11 passing cases, total duration is 810.65 → 758.40 s (−6.45%); median per-door reduction is 4.97%. The powered case has no relevant turn consolidation and is unchanged. These reductions do not imply that the new actor meets the original benchmark clock.

### Stall regression

`db0103_stall` first violates the 3 mm noncontact clearance gate at frame 752 (`operate`, actor time 18.426 s): `actor_geom_forearm_r` is 2.531 mm from `slide_latch_knob`, falling to **1.150 mm** at frame 757. The intentional hand grip exception does not exempt the forearm. This is a clearance failure; the measured minimum is still positive.

At frame 1298 (`traverse`, 30.014 s), left-foot target error reaches **12.04 mm** and **1.65°**, exceeding the 1 mm and 0.5° stance gates. Saved and interpolated stance drift also fail. The shorter candidate cannot replace the accepted 20° control.

## Jerk and tracking tradeoffs

All 31 scalar joints and the free root were analyzed, not just the 16 displayed landmarks. Jerk is the third finite difference on each clock: `proposal_time` attributes the solver path before retiming; `actor_time` measures the final playback path. Quantiles are time-weighted. Phase derivatives exclude boundaries whose four contributing poses span different phases. Jerk is **not** currently an independent acceptance gate or a validated physiological/perceptual threshold.

Every one of the **10 changed, passing candidates** has at least one named hinge whose final-clock peak jerk rises by more than 10%; the powered case is unchanged. The table exposes such regressions rather than presenting shorter duration as uniformly smoother motion. For each clock, the displayed joint is the largest relative increase in its whole-clip peak among joints whose control peak was at least 10 rad/s³. This avoids division by nearly zero baselines; it is a review statistic, not an overall quality score. All other joints and event frames remain in the evidence JSON.

| Door | Proposal-clock peak jerk: joint, 20° → 30° (rad/s³) | Final-clock peak jerk: joint, 20° → 30° (rad/s³) |
|---|---|---|
| db0002 | ankle L roll: 80.1 → 161.9 | hip L yaw: 604.6 → 1108.0 |
| db0079 | shoulder R pitch: 3427.4 → 3434.7 | shoulder R roll: 126.8 → 235.2 |
| db0193 | hip R yaw: 200.7 → 315.3 | hip R yaw: 422.9 → 1084.9 |
| db0153 | Unchanged | Unchanged |
| db0044 | hip L yaw: 237.7 → 304.9 | hip L roll: 364.8 → 632.3 |
| db0015 | shoulder L yaw: 741.0 → 1886.8 | elbow L: 199.5 → 601.3 |
| db0372 | shoulder R yaw: 2788.2 → 2936.0 | hip L yaw: 208.8 → 370.8 |
| db0335 | shoulder L yaw: 58.6 → 91.3 | hip R roll: 290.7 → 775.7 |
| db0894 | ankle R roll: 122.6 → 6878.4 | spine yaw: 26.0 → 225.9 |
| db0103, rejected | ankle L roll: 126.3 → 10810.0 | spine yaw: 40.3 → 388.8 |
| db0457 | hip L yaw: 235.6 → 295.2 | hip R roll: 488.1 → 786.2 |
| db0139 | spine roll: 760.2 → 1252.9 | ankle L roll: 151.9 → 371.2 |

The cold-room case is a particularly clear tradeoff. Besides the tabled ankle peak, its **spine-yaw proposal-clock peak rises 326.2 → 14009.7 rad/s³**, and its final-clock peak rises 26.0 → 225.9. Candidate source frames 2005–2008 in `traverse` contain spine-yaw coordinates approximately `[0.0000248, 0.0000248, −0.022694, −0.003271]` rad. This is a localized posture change, not merely a shifted whole-clip quantile. During `operate`, right-elbow jerk p95 also rises **101.61 → 151.16 rad/s³**, while that phase shortens 37.35 → 30.55 s. Existing velocity/acceleration and geometry checks still pass after retiming; that does not certify visual smoothness.

Tracking-error increases in the passing cases are much smaller in absolute units. The table reports independent maxima, including both feet where applicable. The largest accepted foot-target increase is cold-room's **0.0515 → 0.1159 mm**, still well below the 1 mm gate. No passing case has a comparable large active-hand tracking regression; the largest increase is about 0.0021 mm for the double door. The stall failure remains explicit.

| Door | Active-hand position error, 20° → 30° (mm) | Foot-target position error, 20° → 30° (mm) |
|---|---:|---:|
| db0002 | 0.2458 → 0.2460 | 0.0458 → 0.0506 |
| db0079 | 0.2924 → 0.2917 | 0.0675 → 0.0675 |
| db0193 | 1.3334 → 1.3245 | 0.0465 → 0.0465 |
| db0153 | No active hand | 0.0544 → 0.0544 |
| db0044 | 0.2674 → 0.2672 | 0.0603 → 0.0603 |
| db0015 | 0.2862 → 0.2883 | 0.0470 → 0.0470 |
| db0372 | 0.3209 → 0.3082 | 0.0520 → 0.0520 |
| db0335 | 0.2376 → 0.2375 | 0.0614 → 0.0614 |
| db0894 | 0.2883 → 0.2882 | 0.0515 → 0.1159 |
| db0103, rejected | 0.2688 → 0.2688 | 0.0501 → **12.0395** |
| db0457 | 0.2881 → 0.2881 | 0.0503 → 0.0503 |
| db0139 | 0.2612 → 0.2612 | 0.0647 → 0.0603 |

## Interpretation and separate work

The 30° option is worth retaining as an **unadopted experiment with per-candidate validation and full-clip visual review**. It is not a safe global replacement for 20°. No complete-clip personal style approval is claimed here. Human review must remain separate from sampled kinematic acceptance.

The explicit manual `db0193` schedule retains the same hasp/withdrawal/pull roles; the padlock was already unengaged, so this is not a new necessary-unlocking claim. The powered case retains zero hand contact and no trigger/motor-causality claim. All source outcomes remain separate from the new actor's completion evidence. See [scope and limitations](planned-reference-scope.md).

**Separate pelvis-continuity pilot:** its result is reserved for a later update. This experiment includes no temporal prior, landing-weight ramp, pelvis-height cap, rig/contact change or posthoc pose smoothing. Results from a separate pilot must not be attributed to turn30.

## Evidence and provenance

Local evidence root: `/tmp/doorbench-style30-batch-20260905/`. The experiment artifacts are not part of the published dataset or tracked asset tree. `turn20/` and `turn30/` contain each clip, NPZ, solver diagnostics, independent validation and an adjacent attempt receipt. `jobs/` preserves input bindings. `comparison.json` preserves complete scalar/root trajectory metrics on both clocks, phase metrics, validator maxima and exact event frames. Inputs and copied source were checked before and after each attempt. The runtime/source identity includes MuJoCo 3.12.0, Mink 1.3.0 and the frozen generator files.

| Evidence | SHA256 |
|---|---|
| Final v2 `baseline-index.json` | `3228ae28673d76ada4b7e6ce9d9fa487352c5687de50301a9be005b55e44a7a6` |
| Source/runtime identity | `954bf2e7e077866e0f9db8090f656c88cb3d5695722c54a995a042729a658c54` |
| `manifest.json` — 89 copied Python files and selection | `b77e8caa5f9677a587f89d94aa4ad70b46e4886c5d236a614dc9be15dc75d0bb` |
| `completed.json` — all 24 executions | `db5fd5a935f0e3eda9b44a817d815e977e969c0fd18c4155164e6049d7a0feb8` |
| `comparison.json` — both clocks, full joint/tracking metrics | `c9c608d258cd9225d482171df27d7e8be35d16f73b312199418c64f30ce84ad0` |
| `source/style_worker.py` — only turn-angle override | `b25feda09467ddd09272f1d6e878258e6907b331abbf1eadbedb94f9376e0b38` |
| `source/style_batch.py` — isolated four-worker controller | `4949cd25087be7c110ed816552ac9e57b6d98ff5bb456d02fd1e1f91aa8b6ed4` |
| `turn30/db0103_stall/validation.json` | `675e051ded17c927abfa322076accba4ba9b76abba7560801d43be2c47e099dd` |
| `turn30/db0894_cold_storage/validation.json` | `5c30fa7f90e8db263490d0a04a44e08fe8399f234076052a8f95c45cdb71aa71` |

The comparison code and its metric-helper hashes are recorded inside `comparison.json`; source and artifact hashes for every door are in its variant records and attempt receipts. Simple analytic checks verified the finite-difference clock scaling, phase-boundary exclusion and constant-velocity root translation/rotation descriptors. The 24 full independent validations supply the motion evidence; these analytic checks do not replace them.

Acceptance retains the existing sampled geometry/contact/derivative/task-evidence contract. It does not establish continuous dynamics, dynamic balance, force closure, mechanism causality, original benchmark timing or human naturalness.
