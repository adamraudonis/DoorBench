# Isaac parity gate

_Report generated 2026-09-05T10:38:45 by `scripts/isaaclab/parity_report.py` from `results/parity/`, repository commit `a60f32892`. Dataset: 1000 doors, manifest version `0.1.0` generated 2026-09-05T10:37:22, reference run commit `a60f32892`._

### Which runs this page compares

| run | file | doors | engine | dt | protocol / metrics | generated |
|---|---|---|---|---|---|---|
| MuJoCo reference | `results/parity/mujoco.json` | 1000 | mujoco `3.12.0` | 0.002 | 1.0 / 1.1 | 2026-09-05T10:38:34 |
| PhysX `full` | `results/parity/isaac_full.json` | 1000 | isaac_sim **not recorded**, isaac_lab **not recorded**, physx_dt `0.008333333333333333` | 0.008333333... | 1.0 / 1.0 (not recorded) | 2026-09-05T09:42:07 |
| PhysX `rl` | `results/parity/isaac_rl.json` | 1000 | isaac_sim **not recorded**, isaac_lab **not recorded**, physx_dt `0.008333333333333333` | 0.008333333... | 1.0 / 1.0 (not recorded) | 2026-09-05T09:06:30 |

> **Not comparable, and not counted as agreement.**
> * `full`: **256 doors stale** - the PhysX record and the MuJoCo reference were produced from different protocol inputs (`inputs_hash`), so nothing in them is comparable. They are grade **X** and published as *untested*, never as ok or fail. Example: `db0002_swing_single`; reason: inputs_hash mujoco ef6f8d01145e != physx c1386aeb5617.
> * `rl`: **256 doors stale** - the PhysX record and the MuJoCo reference were produced from different protocol inputs (`inputs_hash`), so nothing in them is comparable. They are grade **X** and published as *untested*, never as ok or fail. Example: `db0002_swing_single`; reason: inputs_hash mujoco ef6f8d01145e != physx c1386aeb5617.
> * `full`: **744 doors** carry metrics whose *definition* changed between the two runs (`arrival_speed`, `speed_at_latch`). Those metrics are reported and **not graded** until the older side is re-run; every other metric of the door is compared as usual.
> * `rl`: **744 doors** carry metrics whose *definition* changed between the two runs (`arrival_speed`, `speed_at_latch`). Those metrics are reported and **not graded** until the older side is re-run; every other metric of the door is compared as usual.
> * `full`: **76 doors** entered the relatch phase at different angles in the two runs (it continues from operate, where a leaf that coasts into its stop rebounds in MuJoCo and not in PhysX), so its *timing* metrics (`t_close`) measure two different experiments and are not graded. The phase's verdict metrics - `relatch_closed_angle` and `relatch_repush_angle`, both end states - are graded as usual.
> * `rl`: **80 doors** entered the relatch phase at different angles in the two runs (it continues from operate, where a leaf that coasts into its stop rebounds in MuJoCo and not in PhysX), so its *timing* metrics (`t_close`) measure two different experiments and are not graded. The phase's verdict metrics - `relatch_closed_angle` and `relatch_repush_angle`, both end states - are graded as usual.


Every door runs **one behavioural protocol** in MuJoCo (the reference physics, CPU) and in Isaac Sim / PhysX on the GPU pod, on both USD kinds (`door.usda` full fidelity, `door_rl.usda` canonical 8-link). The two runs are compared phase by phase: both simulators must reach the same pass / fail verdict (else grade **C**), and when they agree the metrics must be within tolerance (else grade **B**); **A** is parity, **X** means the door could not be compared (spawn / structure error). A disagreement is tagged with a discrepancy class whose likely root cause comes from the analysis of the first 40-door probe. The per-door verdict is published in `qa.json` (`isaac_parity`) and as a badge in the viewer.

## Headline

| USD kind | compared | parity (A) | same verdicts (A + B) | disagree (C) | not comparable (X) | of which stale | untested |
|---|---|---|---|---|---|---|---|
| `full` | 744 / 1000 | **549 / 1000** (74 % of compared) | 679 / 1000 (91 %) | 65 | 256 | 256 | 0 |
| `rl` | 744 / 1000 | **547 / 1000** (74 % of compared) | 711 / 1000 (96 %) | 33 | 256 | 256 | 0 |

Door badge (`qa.json.isaac_parity.ok`; viewer chip *Isaac parity*): **669 ok** (grade A or B in every tested kind), **75 fail** (a status disagreement or not comparable), 256 untested.

## What the gate runs

| phase | what is compared | pass criterion (per simulator) |
|---|---|---|
| `structure` | P0 structure (joints / limits / gains / mass match model.json) | joint set, ranges (2e-3), stiffness / friction (1 %), spring target (1e-3), moving mass (20 % / 0.5 kg) |
| `pose0` | P1 pose0 (body, COM and joint-anchor frames at q0) | body origins, COMs, joint anchors within 5 mm; axes dot >= 0.999 |
| `settle` | P2 settle (1 s free, spring targets kept) | primary drift < 0.05 rad / 0.01 m, every other joint < 0.02 rad / 2 mm, no MuJoCo warnings, penetration > -12 mm |
| `hold` | P3 hold / free_opens (adaptive QA push on the door joint only) | has_holding: q < 2 deg / 15 mm under the adaptive push; else opens > 10 deg / 5 cm within 6 s |
| `operate_open` | P4 operate + open (operator, thumbturn / aux / dogs, then push) | q > min(20 deg, 0.5 max_open) / 5 cm after operator + push (chain: inside the slack window) |
| `release` | P5 release (latch bolt re-extends) | bolt < 6 mm after the operator is released |
| `relatch` | P6 relatch (close and re-push) | closed < 2 deg after 6 s closing drive, re-push < 2.5 deg |
| `closer_return` | P7 closer return from 60 deg | abs(q) < 6 deg after 12 s from 60 deg |
| `locked_holds` | P8 locked holds (operator worked + push) | q < 2 deg / 15 mm (+ chain slack) with operator worked + push |
| `limits` | P9 limits (every joint inside its range) | every limited joint inside lo - tol .. hi + tol (2 deg / 5 mm) |
| `sanity` | P10 sanity (finite, no explosion) | finite state, no velocity cap hit, no body displaced > 5 m, no MuJoCo warnings |

<details><summary>Metric tolerances (a delta passes when within either bound)</summary>

| metric | hinge (rad, s) | slide (m, s) | relative | where the bound comes from |
|---|---|---|---|---|
| `settle_drift` | 0.02 | 0.005 | - | the QA settle gate itself allows 0.05 rad / 0.01 m of drift in 1 s; the comparison bound is under half of that, so a door that would still sign off cannot disagree here |
| `settle_drift_primary` | 0.02 | 0.005 | - | as settle_drift |
| `settle_drift_operator` | 0.02 | 0.005 | - | as settle_drift |
| `settle_drift_latch` | 0.002 | 0.002 | - | 2 mm: a third of the 6 mm 'bolt has returned' threshold, so latch slop cannot be confused with a bolt that did not re-extend |
| `pen0_m` | 0.003 | 0.003 | - | 3 mm: PhysX's 5 mm contactOffset vs MuJoCo's margin-free contacts; the sign-off gate rejects anything past -12 mm |
| `hold_displacement` | 0.01 | 0.003 | - | 0.01 rad / 3 mm on a door that must stay shut: half the 2 deg / 15 mm 'held' threshold, plus the leaf's authored locked play where it has any. On a door that must swing open the quantity is q at the free-swing crossing and the q_at_1s bound applies instead |
| `t_free` | 0.25 | 0.25 | 30 % | 0.25 s / 30 %: two 30 Hz samples plus the 1 s minimum push; the crossing time of a leaf accelerating from rest is dominated by the first tenth of a rad |
| `q_at_1s` | 0.1 | 0.05 | 20 % | 0.1 rad / 5 cm, 20 %: a free leaf covers ~0.1 rad in one 30 Hz sample at its typical 3 rad/s, so this is sampling noise, not travel; the door's own locked play is added |
| `opened` | 0.1 | 0.05 | 20 % | 0.1 rad / 5 cm, 20 %: as q_at_1s. Waived when both runs coasted into the same joint stop - MuJoCo's soft limit (solreflimit 0.005) returns ~17 % of the impact velocity and PhysX's articulation limit is inelastic, so the end-of-phase angle is a rebound; q_primary_max is graded instead |
| `actuate_displacement` | 0.1 | 0.05 | 20 % | as opened |
| `t_open` | 0.3 | 0.3 | 30 % | 0.3 s / 30 %: the operator is worked from 0.6 s and the push starts at 1.2 s, so a 0.3 s spread is the width of the drive ramp, not a difference in whether the door opens |
| `t_open_bench` | 0.3 | 0.3 | 30 % | as t_open |
| `q_primary_max` | 0.1 | 0.05 | 20 % | 0.1 rad / 5 cm, 20 %: the peak the leaf reaches, which both engines must agree on even when they bounce off the stop differently |
| `t_unlatch` | 0.2 | 0.2 | - | 0.2 s: six 30 Hz samples of bolt travel |
| `operator_travel_reached` | 0.05 | 0.005 | 10 % | 10 % of the operator's own travel (the hardware table's travel, or the joint range when the MJCF authors a larger one) |
| `bolt_retract_max_frac` | 0.15 | 0.15 | - | 15 % of the throw: the latch is judged unlatched at 80 % of throw, so this cannot flip that verdict |
| `curve_rmse_primary` | 0.15 | 0.05 | - | 0.15 rad / 5 cm RMS over the operate phase: about the 0.1 rad per-sample bound sustained over the whole curve |
| `bolt_after_release_m` | 0.002 | 0.002 | - | 2 mm: a third of the 6 mm 'bolt returned' threshold |
| `t_bolt_return` | 0.2 | 0.2 | - | 0.2 s: six 30 Hz samples |
| `operator_after_release_frac` | 0.1 | 0.1 | - | 10 % of the operator travel, as operator_travel_reached |
| `relatch_closed_angle` | 0.0175 | 0.005 | - | 1 deg / 5 mm: half the 2 deg 'closed' threshold of the relatch check |
| `relatch_repush_angle` | 0.0175 | 0.005 | - | 1 deg / 5 mm: as relatch_closed_angle against the 2.5 deg re-push threshold |
| `t_close` | 0.5 | 0.5 | 30 % | 0.5 s / 30 %: the closing drive is a constant effort from rest. Not compared when the two runs entered the phase at different angles - relatch continues from operate, and a leaf that starts 0.4 rad further open must take longer |
| `arrival_speed` | 0.2 | 0.1 | 30 % | 0.2 rad/s / 0.1 m/s, 30 %: peak |v| over the 100 ms before the latch crossing (metrics 1.1). The 1.0 definition - |v| at the single sample nearest the crossing - was a sampling artefact and is never compared against 1.1 |
| `closer_final_angle` | 0.0349 | 0.01 | - | 2 deg / 1 cm: a third of the 6 deg 'closed by the closer' threshold |
| `closer_t_close` | 0.5 | 0.5 | 30 % | 0.5 s / 30 %: a 12 s phase; the closer's own sweep takes 2-6 s |
| `peak_closing_speed` | 0.2 | 0.1 | 30 % | 0.2 rad/s / 0.1 m/s, 30 %: the slam threshold of the damage model is 2-4 rad/s, so this cannot hide a slam |
| `speed_at_latch` | 0.2 | 0.1 | 30 % | as arrival_speed (same definition, same 100 ms window) |
| `curve_rmse_closer` | 0.1 | 0.05 | - | 0.1 rad / 5 cm RMS over the 12 s closer sweep |
| `locked_displacement` | 0.01 | 0.003 | - | 0.01 rad / 3 mm: half the 2 deg / 15 mm 'locked holds' threshold, plus the chain / guard slack where the lock has any |
| *(any other metric)* | 0.05 | 0.02 | 20 % | fallback for a metric with no bound of its own; every metric that decides a grade should have an entry above |
| `velocity_cap_hit_primary` | - | - | - | boolean: did the door joint leave the physical velocity range (15 rad/s / 6 m/s) in one run and not the other |
| `settle_drift_joint` | - | - | - | per joint, over the joints the USD actually has: 0.02 rad / 5 mm, the same bound as settle_drift |

</details>

**When a delta is *not* graded.** Four cases, each of which would otherwise report the comparison rather than the door. None of them loosens a bound: each one says the two numbers do not measure the same thing.

1. **The two records are not the same door** (`inputs_hash` differs): grade **X**, published as untested. The hash covers the joints, the adaptive push, the thresholds, the couplings, the schedule and the flags each runner was handed.
2. **The metric's definition changed** between the two runs (`metrics_version`, `protocol.METRIC_DEF_CHANGED_IN`): reported, not graded, until the older side is re-run. Metrics 1.1 redefined `arrival_speed` / `speed_at_latch` from *|v| at the 30 Hz sample nearest the crossing* - which reads the post-impact velocity and lands either side of a millisecond-long impact at random - to the peak |v| over the 100 ms of approach.
3. **Both runs coasted into the same joint stop**: the value at the end of the operate phase is then a rebound (MuJoCo's soft limit, `solreflimit` 5 ms, returns about 17 % of the impact velocity; PhysX's articulation limit is inelastic), so `opened` is waived and `q_primary_max` - the peak both leaves reach, and the quantity that says how far the door actually swung - is graded in its place.
4. **The phase was entered from a different state**: `relatch` continues from `operate`, so when that rebound leaves the two leaves at different angles its *timing* metrics compare two different experiments. Its verdict metrics (`relatch_closed_angle`, `relatch_repush_angle`) are end states and stay graded.

**The push the gate applies.** Both simulators drive the door joint with the sign-off QA's adaptive push: twice the static resistance at rest (gravity bias + Coulomb friction + spring preload) plus a base sized by the leaf itself - `0.5 m g W`, half the moment gravity would exert on the leaf if it lay horizontal, clamped to [2, 60] N*m for a hinge and [2, 80] N for a slide (`doorbench.qa.push_base`). A flat 60 N*m is what a person applies to a 20-100 kg door; on a 0.14-1.4 kg pet flap it is about 100x the mechanism's own scale, accelerates it at ~2000 rad/s^2 and drives it to 30-85 rad/s in MuJoCo and to a non-finite state in PhysX.

## Discrepancy classes

| class | full | rl | doors | what it means | likely root cause | fix direction | examples |
|---|---|---|---|---|---|---|---|
| `METRICS_VERSION_SKEW` | 744 | 744 | 744 | the two records agree on the door but were reduced by different metric formulas; the affected metrics are reported and not graded | doorbench.parity.protocol.METRICS_VERSION was raised after one of the runs (see METRIC_DEF_CHANGED_IN) | re-run the older side; the other metrics of the door are still compared | `db0001_rollup`, `db0003_cold_storage`, `db0005_garage_tiltup`, `db0006_gate_swing` |
| `STALE_INPUTS` | 256 | 256 | 256 | the two records were produced from different protocol inputs (inputs_hash differs), so nothing in them is comparable | the dataset, the adaptive push, a threshold or a coupling changed between the MuJoCo reference run and the PhysX run | re-run the older side against the current dataset; the verdict is withheld (untested), never published | `db0002_swing_single`, `db0004_bifold`, `db0008_sliding_bypass`, `db0009_bifold` |
| `QUANT` | 152 | 182 | 181 | both simulators reach the same pass / fail verdicts but at least one metric is outside tolerance | soft MuJoCo contacts / limits (solref 0.005) vs rigid PhysX limits, constraint-based vs effort-based Coulomb friction, implicitfast vs implicit drives at high damping; solver dt | rerun at Isaac dt 1/240 (32/8 iterations) and MuJoCo dt 0.001; if the delta shrinks below tolerance tag SOLVER_SENSITIVITY, else triage by phase | `db0003_cold_storage`, `db0010_swing_double`, `db0015_swing_double`, `db0029_sliding_single` |
| `CONTACT_GEOMETRY` | 44 | 11 | 44 | the bolt retracted (or there is no bolt) yet the leaf did not move, or a latch that holds in MuJoCo does not engage in PhysX (convex hulls, strike lip, panel clearance) | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes. The global self-collision part is FIXED in the export: physxArticulation:enabledSelfCollisions is True and every pair MuJoCo suppresses (same weld body, weld parent/child, contact_excludes) is authored as PhysxFilteredPairsAPI, so a latch holding one moving link against another (swing pairs, lift pins, drop bolts) now touches in PhysX too | enable contact reporting; rerun with Env collision disabled, then without the hardware part, to bisect frame contact vs articulation; check the authored filtered pairs against validate_usd_static.py | `db0039_swing_single`, `db0062_swing_single`, `db0075_gate_swing`, `db0086_swing_single` |
| `EXPORT_WELD` | 19 | 17 | 14 | MuJoCo holds the leaf (weld / lock equality) but PhysX has nothing holding it, so the door opens under the push | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench:couplings JSON. Both USD kinds now carry a breakable UsdPhysics.FixedJoint base -> leaf with physics:excludeFromArticulation, breakForce == breakTorque == holding_force_N and physics:jointEnabled (doorbench:env_release). A remaining occurrence means the joint is absent (stale assets) or PhysX did not parse the loop joint | regenerate the dataset; if PhysX rejects an excludeFromArticulation joint between two articulation links, fall back to --emulate-weld and report it | `db0026_swing_single`, `db0158_swing_double`, `db0216_swing_single`, `db0316_swing_double` |
| `EXPORT_COUPLING` | 7 | 11 | 18 | the operator turns but the bolt does not retract (or does not return) in PhysX, so the door stays latched | H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate it as a kinematic clamp each step (the soft target offset under-retracts by 40-60 %) | shared clamp function (write_joint_state_to_sim(max(latch_q, scale * op_q))) in the parity runner and DoorMechanismAction; read scale from doorbench:rl.latch_coupling / doorbench:latch_coupling_scale | `db0124_vault`, `db0288_blast`, `db0296_sliding_single`, `db0331_sliding_single` |
| `RL_CANON` | 0 | 11 | 11 | door.usda agrees with MuJoCo but door_rl.usda does not: a welded lock / operator / panel or an empty operator slot changes the behaviour | H4: panic doors with robot outside and no far-side trim get operator_joint None (exit device welded, latch never retracts); engaged locks with no canonical slot welded engaged; extra leaves omitted. Parts the operator retracts (revolute hooks, cremone shoot bolts, wheel-driven dogs) are welded RELEASED since the export fix, and every decision is recorded in doorbench:rl (welded / released_parts / released_holding / welded_engaged) | the RL expectation is derived from that ground truth in protocol.expected_outcomes (hold -> na when the only holding part is welded released, operate -> stays_closed when a lock part is welded engaged); a remaining RL_CANON is a documented structural limit of the 8-link articulation | `db0296_sliding_single`, `db0331_sliding_single`, `db0548_swing_single`, `db0593_sliding_single` |
| `VELOCITY_EXPLOSION` | 0 | 2 | 2 | a joint left the physical velocity range in one simulator (velocity cap hit, non-finite state) - the drift or displacement it reports is the debris, not the behaviour | rigid PhysX limits plus a heavy / stiff mechanism at 120 Hz, initial penetration, or an effort far above the mechanism's own scale | dt <= 1/240; check the initial penetration and the applied effort against the leaf's inertia (doorbench.qa.push_base) | `db0213_garage_sectional`, `db0873_rollup` |

<details><summary><code>METRICS_VERSION_SKEW</code> - all 744 doors</summary>

`db0001_rollup` `db0003_cold_storage` `db0005_garage_tiltup` `db0006_gate_swing` `db0007_sliding_single` `db0010_swing_double` `db0011_automatic_swing` `db0012_swing_single` `db0015_swing_double` `db0016_swing_single` `db0018_sliding_single` `db0019_swing_double` `db0021_swing_single` `db0022_swing_single` `db0023_sliding_single` `db0024_swing_single` `db0025_swing_double` `db0026_swing_single` `db0027_swing_single` `db0028_swing_single` `db0029_sliding_single` `db0032_sliding_single` `db0033_gate_sliding` `db0035_swing_single` `db0036_swing_single` `db0038_sliding_single` `db0039_swing_single` `db0040_swing_single` `db0044_pivot` `db0047_swing_double` `db0048_swing_single` `db0049_swing_double` `db0050_swing_single` `db0053_elevator` `db0054_stall` `db0055_swing_single` `db0056_swing_single` `db0058_swing_single` `db0060_swing_single` `db0061_swing_single` `db0062_swing_single` `db0063_swing_single` `db0064_gate_sliding` `db0066_revolving` `db0067_sliding_single` `db0068_swing_single` `db0070_swing_single` `db0072_swing_single` `db0073_swing_single` `db0074_swing_single` `db0075_gate_swing` `db0076_swing_single` `db0079_sliding_single` `db0080_cold_storage` `db0081_swing_single` `db0082_swing_single` `db0083_swing_single` `db0085_revolving` `db0086_swing_single` `db0087_sliding_single` `db0088_pivot` `db0089_automatic_swing` `db0091_swing_single` `db0094_swing_single` `db0095_dutch` `db0096_swing_single` `db0097_swing_double` `db0098_gate_swing` `db0099_sliding_single` `db0101_swing_single` `db0102_automatic_swing` `db0103_stall` `db0104_garage_tiltup` `db0107_sliding_single` `db0108_revolving` `db0110_swing_double` `db0112_swing_double` `db0114_swing_single` `db0115_swing_single` `db0116_swing_single` `db0117_swing_single` `db0118_dutch` `db0119_swing_single` `db0120_swing_single` `db0122_swing_single` `db0124_vault` `db0125_sliding_single` `db0126_gate_swing` `db0127_swing_double` `db0128_swing_double` `db0129_sliding_single` `db0130_automatic_sliding` `db0131_sliding_single` `db0132_swing_double` `db0133_swing_single` `db0134_sliding_single` `db0135_swing_single` `db0136_automatic_swing` `db0137_bifold` `db0138_automatic_swing` `db0139_swing_single` `db0140_swing_single` `db0142_swing_single` `db0143_swing_single` `db0144_swing_double` `db0145_swing_single` `db0147_swing_single` `db0148_garage_sectional` `db0149_swing_double` `db0151_sliding_single` `db0152_swing_single` `db0153_automatic_sliding` `db0154_swing_single` `db0155_swing_single` `db0156_swing_single` `db0157_sliding_single` `db0158_swing_double` `db0159_automatic_swing` `db0160_swing_single` `db0161_pivot` `db0165_swing_single` `db0166_swing_single` `db0167_sliding_single` `db0168_ship_watertight` `db0169_swing_single` `db0171_swing_single` `db0172_swing_single` `db0173_garage_tiltup` `db0174_stall` `db0175_garage_sectional` `db0178_swing_single` `db0180_swing_double` `db0181_swing_single` `db0182_swing_single` `db0183_swing_double` `db0184_rollup` `db0185_swing_single` `db0188_cold_storage` `db0189_swing_single` `db0193_sliding_single` `db0195_swing_single` `db0196_rollup` `db0197_swing_double` `db0198_garage_sectional` `db0199_swing_single` `db0201_swing_single` `db0203_automatic_sliding` `db0204_dutch` `db0205_swing_single` `db0206_swing_single` `db0207_swing_single` `db0209_swing_single` `db0210_swing_single` `db0211_swing_double` `db0212_swing_single` `db0213_garage_sectional` `db0214_swing_single` `db0216_swing_single` `db0219_swing_single` `db0220_swing_single` `db0221_pivot` `db0222_swing_double` `db0224_swing_single` `db0225_automatic_swing` `db0227_swing_single` `db0228_swing_single` `db0229_swing_single` `db0230_swing_single` `db0232_swing_single` `db0233_swing_single` `db0234_swing_single` `db0235_sliding_single` `db0237_swing_single` `db0239_gate_swing` `db0240_swing_single` `db0242_sliding_single` `db0243_swing_single` `db0244_swing_single` `db0245_swing_single` `db0247_swing_single` `db0250_revolving` `db0252_swing_single` `db0254_swing_single` `db0255_swing_single` `db0256_swing_single` `db0257_swing_single` `db0258_rollup` `db0259_swing_single` `db0260_revolving` `db0261_swing_double` `db0262_sliding_single` `db0263_swing_single` `db0264_swing_single` `db0265_swing_single` `db0266_swing_single` `db0268_swing_single` `db0269_swing_single` `db0270_swing_single` `db0274_sliding_single` `db0275_swing_double` ... and 544 more

</details>

<details><summary><code>STALE_INPUTS</code> - all 256 doors</summary>

`db0002_swing_single` `db0004_bifold` `db0008_sliding_bypass` `db0009_bifold` `db0013_swing_single` `db0014_gate_swing` `db0017_hatch_ceiling` `db0020_sliding_bypass` `db0030_bifold` `db0031_saloon` `db0034_gate_swing` `db0037_strip_curtain` `db0041_sliding_single` `db0042_sliding_bypass` `db0043_bifold` `db0045_pet_door` `db0046_sliding_bypass` `db0051_swing_single` `db0052_pet_door` `db0057_sliding_bypass` `db0059_bifold` `db0065_accordion` `db0069_bifold` `db0071_bifold` `db0077_bifold` `db0078_bifold` `db0084_sliding_single` `db0090_bifold` `db0092_sliding_bypass` `db0093_sliding_single` `db0100_bifold` `db0105_swing_single` `db0106_gate_swing` `db0109_swing_single` `db0111_swing_single` `db0113_bifold` `db0121_hatch_ceiling` `db0123_saloon` `db0141_bifold` `db0146_gate_sliding` `db0150_sliding_single` `db0162_gate_swing` `db0163_strip_curtain` `db0164_bifold` `db0170_saloon` `db0176_baby_gate` `db0177_accordion` `db0179_vault` `db0186_swing_single` `db0187_turnstile_fullheight` `db0190_turnstile_fullheight` `db0191_gate_swing` `db0192_gate_swing` `db0194_bifold` `db0200_gate_swing` `db0202_turnstile_tripod` `db0208_hatch_ceiling` `db0215_accordion` `db0217_sliding_bypass` `db0218_sliding_bypass` `db0223_swing_single` `db0226_sliding_single` `db0231_bifold` `db0236_swing_single` `db0238_pet_door` `db0241_hatch_floor` `db0246_sliding_single` `db0248_swing_single` `db0249_accordion` `db0251_sliding_single` `db0253_gate_swing` `db0267_sliding_bypass` `db0271_sliding_single` `db0272_turnstile_tripod` `db0273_turnstile_fullheight` `db0292_accordion` `db0301_gate_swing` `db0303_swing_single` `db0314_ship_watertight` `db0317_swing_single` `db0319_swing_single` `db0321_sliding_bypass` `db0336_baby_gate` `db0337_sliding_bypass` `db0338_turnstile_fullheight` `db0340_sliding_single` `db0343_bifold` `db0344_turnstile_tripod` `db0345_sliding_single` `db0350_strip_curtain` `db0352_blast` `db0357_hatch_ceiling` `db0360_hatch_floor` `db0368_gate_swing` `db0370_sliding_single` `db0372_bifold` `db0373_sliding_single` `db0375_swing_single` `db0377_sliding_bypass` `db0379_swing_single` `db0380_hatch_floor` `db0381_accordion` `db0382_sliding_bypass` `db0387_sliding_bypass` `db0389_hatch_ceiling` `db0393_turnstile_tripod` `db0394_swing_single` `db0399_pet_door` `db0401_sliding_bypass` `db0402_swing_single` `db0406_strip_curtain` `db0412_hatch_floor` `db0422_swing_single` `db0426_vault` `db0430_bifold` `db0431_sliding_single` `db0441_swing_single` `db0442_hatch_floor` `db0445_sliding_bypass` `db0447_sliding_bypass` `db0449_hatch_floor` `db0452_hatch_floor` `db0457_swing_single` `db0458_vault` `db0459_sliding_bypass` `db0473_accordion` `db0483_baby_gate` `db0486_bifold` `db0491_sliding_bypass` `db0495_sliding_bypass` `db0497_turnstile_fullheight` `db0499_sliding_bypass` `db0500_sliding_bypass` `db0505_baby_gate` `db0516_turnstile_tripod` `db0521_swing_single` `db0524_bifold` `db0528_turnstile_fullheight` `db0529_hatch_floor` `db0533_accordion` `db0535_strip_curtain` `db0537_pet_door` `db0541_bifold` `db0542_sliding_bypass` `db0547_swing_single` `db0550_sliding_bypass` `db0558_stall` `db0559_hatch_floor` `db0566_gate_swing` `db0567_swing_single` `db0573_stall` `db0578_swing_single` `db0579_swing_single` `db0586_sliding_bypass` `db0589_turnstile_fullheight` `db0598_hatch_ceiling` `db0600_ship_watertight` `db0606_swing_single` `db0609_pet_door` `db0621_sliding_bypass` `db0624_saloon` `db0627_pet_door` `db0628_strip_curtain` `db0630_sliding_single` `db0632_swing_single` `db0641_strip_curtain` `db0642_swing_single` `db0647_sliding_single` `db0649_sliding_bypass` `db0652_sliding_bypass` `db0661_baby_gate` `db0665_bifold` `db0669_sliding_bypass` `db0679_bifold` `db0683_bifold` `db0687_strip_curtain` `db0691_sliding_single` `db0694_stall` `db0698_baby_gate` `db0703_swing_single` `db0707_swing_double` `db0716_saloon` `db0720_sliding_bypass` `db0722_swing_single` `db0728_swing_single` `db0729_ship_watertight` `db0731_sliding_bypass` `db0736_swing_single` `db0738_saloon` `db0740_swing_single` `db0743_pet_door` `db0744_ship_watertight` `db0745_swing_single` `db0747_sliding_bypass` `db0748_vault` `db0750_bifold` `db0751_gate_swing` `db0753_pet_door` `db0757_sliding_single` `db0760_swing_single` ... and 56 more

</details>

<details><summary><code>QUANT</code> - all 181 doors</summary>

`db0003_cold_storage` `db0010_swing_double` `db0015_swing_double` `db0029_sliding_single` `db0036_swing_single` `db0050_swing_single` `db0054_stall` `db0060_swing_single` `db0062_swing_single` `db0080_cold_storage` `db0082_swing_single` `db0083_swing_single` `db0086_swing_single` `db0088_pivot` `db0095_dutch` `db0097_swing_double` `db0112_swing_double` `db0114_swing_single` `db0117_swing_single` `db0124_vault` `db0125_sliding_single` `db0129_sliding_single` `db0130_automatic_sliding` `db0132_swing_double` `db0135_swing_single` `db0143_swing_single` `db0165_swing_single` `db0168_ship_watertight` `db0182_swing_single` `db0188_cold_storage` `db0193_sliding_single` `db0203_automatic_sliding` `db0204_dutch` `db0211_swing_double` `db0214_swing_single` `db0224_swing_single` `db0227_swing_single` `db0234_swing_single` `db0237_swing_single` `db0242_sliding_single` `db0252_swing_single` `db0262_sliding_single` `db0263_swing_single` `db0270_swing_single` `db0276_cold_storage` `db0283_automatic_sliding` `db0285_ship_watertight` `db0288_blast` `db0291_stall` `db0294_swing_single` `db0302_swing_double` `db0304_swing_single` `db0306_sliding_single` `db0309_gate_sliding` `db0311_swing_single` `db0323_automatic_sliding` `db0325_swing_single` `db0330_swing_single` `db0341_swing_double` `db0354_pivot` `db0366_gate_swing` `db0384_ship_watertight` `db0385_swing_single` `db0386_swing_single` `db0407_swing_double` `db0408_swing_single` `db0409_cold_storage` `db0414_automatic_sliding` `db0415_gate_sliding` `db0418_swing_single` `db0420_stall` `db0424_swing_single` `db0429_swing_double` `db0432_cold_storage` `db0440_turnstile_fullheight` `db0443_swing_single` `db0444_swing_single` `db0446_swing_single` `db0451_swing_single` `db0453_sliding_single` `db0462_swing_single` `db0469_cold_storage` `db0471_sliding_single` `db0474_automatic_sliding` `db0484_swing_single` `db0493_swing_single` `db0507_cold_storage` `db0508_swing_single` `db0509_swing_double` `db0514_automatic_sliding` `db0518_swing_single` `db0519_swing_single` `db0520_swing_double` `db0523_swing_single` `db0530_vault` `db0536_swing_double` `db0544_swing_single` `db0545_sliding_single` `db0549_cold_storage` `db0555_sliding_single` `db0557_stall` `db0561_swing_single` `db0562_swing_double` `db0565_swing_single` `db0568_swing_single` `db0571_swing_single` `db0582_swing_single` `db0585_cold_storage` `db0587_swing_double` `db0599_swing_single` `db0610_swing_single` `db0611_automatic_sliding` `db0613_gate_sliding` `db0623_blast` `db0625_swing_single` `db0631_sliding_single` `db0634_swing_double` `db0637_pivot` `db0638_swing_single` `db0645_swing_single` `db0656_sliding_single` `db0663_stall` `db0664_swing_single` `db0671_pivot` `db0672_blast` `db0673_cold_storage` `db0674_ship_watertight` `db0680_swing_single` `db0689_swing_single` `db0699_swing_single` `db0700_dutch` `db0706_swing_double` `db0717_swing_single` `db0734_swing_single` `db0764_automatic_sliding` `db0767_swing_single` `db0769_sliding_single` `db0770_automatic_sliding` `db0771_swing_single` `db0772_blast` `db0773_swing_single` `db0787_swing_double` `db0789_swing_single` `db0791_swing_single` `db0799_swing_single` `db0801_automatic_sliding` `db0803_swing_single` `db0807_swing_double` `db0812_swing_single` `db0814_swing_single` `db0819_swing_single` `db0833_swing_double` `db0836_swing_single` `db0850_swing_single` `db0852_cold_storage` `db0858_stall` `db0863_automatic_sliding` `db0865_cold_storage` `db0871_swing_double` `db0889_swing_single` `db0894_cold_storage` `db0898_ship_watertight` `db0902_pivot` `db0903_sliding_single` `db0904_swing_single` `db0905_swing_single` `db0906_dutch` `db0909_swing_single` `db0911_ship_watertight` `db0934_swing_single` `db0935_gate_sliding` `db0937_cold_storage` `db0948_swing_single` `db0955_swing_single` `db0960_blast` `db0961_sliding_single` `db0966_sliding_single` `db0970_sliding_single` `db0973_swing_single` `db0984_swing_single` `db0990_automatic_sliding`

</details>

<details><summary><code>CONTACT_GEOMETRY</code> - all 44 doors</summary>

`db0039_swing_single` `db0062_swing_single` `db0075_gate_swing` `db0086_swing_single` `db0122_swing_single` `db0149_swing_double` `db0182_swing_single` `db0222_swing_double` `db0255_swing_single` `db0263_swing_single` `db0287_gate_swing` `db0304_swing_single` `db0332_baby_gate` `db0454_swing_double` `db0464_swing_single` `db0531_swing_single` `db0540_gate_swing` `db0561_swing_single` `db0577_swing_double` `db0601_swing_single` `db0604_swing_double` `db0610_swing_single` `db0633_swing_single` `db0664_swing_single` `db0675_baby_gate` `db0682_swing_double` `db0714_swing_double` `db0725_swing_single` `db0761_swing_single` `db0763_swing_single` `db0808_swing_single` `db0812_swing_single` `db0832_swing_double` `db0854_gate_swing` `db0889_swing_single` `db0925_swing_single` `db0929_swing_double` `db0930_sliding_single` `db0944_swing_double` `db0948_swing_single` `db0949_swing_single` `db0955_swing_single` `db0963_swing_double` `db0984_swing_single`

</details>

<details><summary><code>EXPORT_WELD</code> - all 14 doors</summary>

`db0026_swing_single` `db0158_swing_double` `db0216_swing_single` `db0316_swing_double` `db0334_swing_double` `db0413_swing_double` `db0448_sliding_single` `db0534_swing_double` `db0591_pivot` `db0597_sliding_single` `db0702_swing_double` `db0733_swing_double` `db0792_swing_double` `db0897_swing_single`

</details>

<details><summary><code>EXPORT_COUPLING</code> - all 18 doors</summary>

`db0124_vault` `db0288_blast` `db0296_sliding_single` `db0331_sliding_single` `db0530_vault` `db0548_swing_single` `db0593_sliding_single` `db0594_sliding_single` `db0620_sliding_single` `db0623_blast` `db0639_sliding_single` `db0672_blast` `db0690_sliding_single` `db0724_sliding_single` `db0772_blast` `db0792_swing_double` `db0804_sliding_single` `db0960_blast`

</details>

<details><summary><code>RL_CANON</code> - all 11 doors</summary>

`db0296_sliding_single` `db0331_sliding_single` `db0548_swing_single` `db0593_sliding_single` `db0594_sliding_single` `db0620_sliding_single` `db0639_sliding_single` `db0690_sliding_single` `db0724_sliding_single` `db0792_swing_double` `db0804_sliding_single`

</details>

<details><summary><code>VELOCITY_EXPLOSION</code> - all 2 doors</summary>

`db0213_garage_sectional` `db0873_rollup`

</details>

## By family

| family | doors | compared | stale | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|---|---|
| accordion | 12 | 0 | 12 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x12 |
| automatic_sliding | 15 | 15 | 0 | 15 | 0 | 5 / 15 | 3 / 15 | - |
| automatic_swing | 10 | 10 | 0 | 10 | 0 | 10 / 10 | 10 / 10 | - |
| baby_gate | 10 | 2 | 8 | 0 | 2 | 0 / 0 | 2 / 2 | STALE_INPUTS x8, CONTACT_GEOMETRY x2 |
| bifold | 30 | 2 | 28 | 2 | 0 | 2 / 2 | 2 / 2 | STALE_INPUTS x28 |
| blast | 6 | 5 | 1 | 0 | 5 | 0 / 0 | 0 / 5 | EXPORT_COUPLING x5, STALE_INPUTS x1 |
| cold_storage | 15 | 15 | 0 | 15 | 0 | 0 / 15 | 4 / 15 | - |
| dutch | 12 | 12 | 0 | 12 | 0 | 8 / 12 | 8 / 12 | - |
| elevator | 8 | 8 | 0 | 8 | 0 | 8 / 8 | 8 / 8 | - |
| garage_sectional | 18 | 18 | 0 | 18 | 0 | 18 / 18 | 17 / 18 | VELOCITY_EXPLOSION x1 |
| garage_tiltup | 7 | 7 | 0 | 7 | 0 | 7 / 7 | 7 / 7 | - |
| gate_sliding | 10 | 8 | 2 | 8 | 0 | 5 / 8 | 4 / 8 | STALE_INPUTS x2 |
| gate_swing | 40 | 23 | 17 | 19 | 4 | 19 / 19 | 22 / 23 | STALE_INPUTS x17, CONTACT_GEOMETRY x4 |
| hatch_ceiling | 8 | 0 | 8 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x8 |
| hatch_floor | 10 | 0 | 10 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x10 |
| pet_door | 15 | 0 | 15 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x15 |
| pivot | 20 | 20 | 0 | 19 | 1 | 14 / 19 | 14 / 19 | EXPORT_WELD x1 |
| revolving | 15 | 15 | 0 | 15 | 0 | 15 / 15 | 15 / 15 | - |
| rollup | 15 | 15 | 0 | 15 | 0 | 15 / 15 | 14 / 15 | VELOCITY_EXPLOSION x1 |
| saloon | 12 | 4 | 8 | 4 | 0 | 4 / 4 | 4 / 4 | STALE_INPUTS x8 |
| ship_watertight | 10 | 6 | 4 | 6 | 0 | 2 / 6 | 0 / 6 | STALE_INPUTS x4 |
| sliding_bypass | 35 | 0 | 35 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x35 |
| sliding_single | 100 | 79 | 21 | 67 | 12 | 74 / 76 | 52 / 70 | STALE_INPUTS x21, EXPORT_COUPLING x9, RL_CANON x9 |
| stall | 15 | 11 | 4 | 11 | 0 | 5 / 11 | 5 / 11 | STALE_INPUTS x4 |
| strip_curtain | 8 | 0 | 8 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x8 |
| swing_double | 76 | 74 | 2 | 55 | 19 | 34 / 55 | 36 / 55 | CONTACT_GEOMETRY x11, EXPORT_WELD x8, STALE_INPUTS x2 |
| swing_single | 440 | 392 | 48 | 362 | 30 | 303 / 363 | 320 / 388 | STALE_INPUTS x48, CONTACT_GEOMETRY x26, EXPORT_WELD x3 |
| turnstile_fullheight | 10 | 1 | 9 | 1 | 0 | 1 / 1 | 0 / 1 | STALE_INPUTS x9 |
| turnstile_tripod | 10 | 0 | 10 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x10 |
| vault | 8 | 2 | 6 | 0 | 2 | 0 / 0 | 0 / 2 | STALE_INPUTS x6, EXPORT_COUPLING x2 |

## By hardware

### latch kind

| kind | compared | stale | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|---|
| deadlatch | 88 | 0 | 84 | 4 | 66 / 84 | 67 / 88 | CONTACT_GEOMETRY x4 |
| dogs | 13 | 4 | 6 | 7 | 2 / 6 | 0 / 13 | EXPORT_COUPLING x7, STALE_INPUTS x4 |
| electric_bolt | 11 | 0 | 8 | 3 | 8 / 8 | 11 / 11 | EXPORT_WELD x2, CONTACT_GEOMETRY x1 |
| gravity_bar | 19 | 11 | 19 | 0 | 19 / 19 | 18 / 19 | STALE_INPUTS x11 |
| hook | 19 | 12 | 4 | 15 | 13 / 13 | 10 / 10 | STALE_INPUTS x12, EXPORT_COUPLING x9, RL_CANON x9 |
| magnetic | 16 | 25 | 16 | 0 | 4 / 16 | 8 / 16 | STALE_INPUTS x25 |
| mortise_latch | 74 | 0 | 68 | 6 | 54 / 68 | 57 / 74 | CONTACT_GEOMETRY x6 |
| multi_bolt | 0 | 7 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x7 |
| none | 238 | 139 | 232 | 6 | 198 / 232 | 186 / 232 | STALE_INPUTS x139, EXPORT_WELD x6, VELOCITY_EXPLOSION x2 |
| rim_latch | 42 | 0 | 41 | 1 | 39 / 42 | 37 / 41 | EXPORT_COUPLING x1, RL_CANON x1 |
| roller | 8 | 0 | 8 | 0 | 3 / 8 | 3 / 8 | - |
| slide_bolt | 20 | 10 | 20 | 0 | 14 / 20 | 12 / 20 | STALE_INPUTS x10 |
| tubular_latch | 165 | 48 | 132 | 33 | 113 / 132 | 121 / 148 | STALE_INPUTS x48, CONTACT_GEOMETRY x27, EXPORT_WELD x6 |
| vertical_rods | 31 | 0 | 31 | 0 | 16 / 31 | 17 / 31 | - |

### lock kind

| kind | compared | stale | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|---|
| card_reader | 20 | 0 | 20 | 0 | 17 / 20 | 17 / 20 | - |
| chain | 4 | 0 | 4 | 0 | 3 / 4 | 3 / 4 | - |
| child_lock_cover | 5 | 3 | 5 | 0 | 4 / 5 | 4 / 5 | STALE_INPUTS x3 |
| deadbolt_double | 6 | 0 | 6 | 0 | 4 / 6 | 4 / 6 | - |
| deadbolt_single | 32 | 0 | 22 | 10 | 15 / 22 | 22 / 26 | CONTACT_GEOMETRY x6, EXPORT_WELD x4, EXPORT_COUPLING x1 |
| delayed_egress | 16 | 0 | 16 | 0 | 16 / 16 | 16 / 16 | - |
| dogs | 13 | 4 | 6 | 7 | 2 / 6 | 0 / 13 | EXPORT_COUPLING x7, STALE_INPUTS x4 |
| electric_strike | 22 | 1 | 22 | 0 | 14 / 22 | 14 / 22 | STALE_INPUTS x1 |
| hook_lock | 21 | 7 | 18 | 3 | 20 / 21 | 16 / 18 | STALE_INPUTS x7, EXPORT_COUPLING x3, RL_CANON x3 |
| interlock | 8 | 0 | 8 | 0 | 8 / 8 | 8 / 8 | - |
| jam_stuck | 6 | 6 | 6 | 0 | 5 / 6 | 5 / 6 | STALE_INPUTS x6 |
| keyed_cylinder | 26 | 0 | 24 | 2 | 20 / 24 | 21 / 26 | EXPORT_WELD x2 |
| keypad_code | 28 | 0 | 18 | 10 | 16 / 18 | 21 / 28 | CONTACT_GEOMETRY x10 |
| mag_lock | 35 | 12 | 29 | 6 | 24 / 29 | 21 / 29 | STALE_INPUTS x12, EXPORT_WELD x6 |
| multipoint | 6 | 1 | 1 | 5 | 0 / 1 | 2 / 5 | CONTACT_GEOMETRY x4, EXPORT_WELD x1, STALE_INPUTS x1 |
| night_latch | 4 | 0 | 0 | 4 | 0 / 0 | 1 / 4 | CONTACT_GEOMETRY x4 |
| none | 370 | 174 | 351 | 19 | 281 / 356 | 272 / 358 | STALE_INPUTS x174, CONTACT_GEOMETRY x14, EXPORT_COUPLING x5 |
| padlock | 31 | 9 | 31 | 0 | 23 / 31 | 24 / 31 | STALE_INPUTS x9, VELOCITY_EXPLOSION x1 |
| privacy_button | 29 | 14 | 29 | 0 | 28 / 29 | 28 / 29 | STALE_INPUTS x14 |
| slide_bolt | 40 | 14 | 38 | 2 | 38 / 40 | 31 / 38 | STALE_INPUTS x14, EXPORT_COUPLING x2, RL_CANON x2 |
| swing_bar_guard | 2 | 0 | 2 | 0 | 2 / 2 | 2 / 2 | - |
| thumbturn_only | 20 | 4 | 13 | 7 | 9 / 13 | 15 / 17 | CONTACT_GEOMETRY x6, STALE_INPUTS x4, EXPORT_WELD x1 |
| vault_wheel | 0 | 7 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x7 |

### closer kind

| kind | compared | stale | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|---|
| auto_operator_full | 4 | 0 | 4 | 0 | 4 / 4 | 4 / 4 | - |
| auto_operator_low_energy | 11 | 0 | 10 | 1 | 8 / 10 | 9 / 10 | EXPORT_WELD x1 |
| concealed_overhead | 21 | 0 | 19 | 2 | 15 / 19 | 15 / 19 | EXPORT_WELD x2 |
| electromagnetic_hold | 13 | 0 | 13 | 0 | 9 / 13 | 9 / 13 | - |
| floor_spring | 28 | 0 | 26 | 2 | 15 / 26 | 17 / 27 | CONTACT_GEOMETRY x1, EXPORT_WELD x1 |
| gas_strut | 0 | 8 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x8 |
| gate | 10 | 12 | 5 | 5 | 5 / 5 | 9 / 10 | STALE_INPUTS x12, CONTACT_GEOMETRY x5 |
| none | 446 | 221 | 387 | 59 | 342 / 396 | 324 / 419 | STALE_INPUTS x221, CONTACT_GEOMETRY x34, EXPORT_COUPLING x17 |
| pneumatic | 6 | 0 | 6 | 0 | 6 / 6 | 6 / 6 | - |
| spring_hinge | 22 | 15 | 19 | 3 | 15 / 19 | 15 / 22 | STALE_INPUTS x15, CONTACT_GEOMETRY x3 |
| surface_overhead | 183 | 0 | 180 | 3 | 130 / 181 | 139 / 181 | CONTACT_GEOMETRY x1, EXPORT_WELD x1, EXPORT_COUPLING x1 |

### operator kind

| kind | compared | stale | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|---|
| card_lever | 20 | 0 | 20 | 0 | 15 / 20 | 15 / 20 | - |
| cremone | 1 | 2 | 1 | 0 | 0 / 1 | 1 / 1 | STALE_INPUTS x2 |
| flush_pull | 22 | 50 | 22 | 0 | 21 / 22 | 17 / 22 | STALE_INPUTS x50 |
| gate_latch_fork | 4 | 8 | 4 | 0 | 4 / 4 | 4 / 4 | STALE_INPUTS x8 |
| handleset | 13 | 0 | 8 | 5 | 0 / 8 | 0 / 10 | CONTACT_GEOMETRY x4, EXPORT_WELD x1 |
| hasp | 6 | 3 | 6 | 0 | 5 / 6 | 6 / 6 | STALE_INPUTS x3 |
| hook_lock_slider | 13 | 2 | 4 | 9 | 13 / 13 | 4 / 4 | EXPORT_COUPLING x9, RL_CANON x9, STALE_INPUTS x2 |
| keypad_deadbolt | 9 | 0 | 1 | 8 | 1 / 1 | 4 / 9 | CONTACT_GEOMETRY x8 |
| keypad_lever | 19 | 0 | 17 | 2 | 15 / 17 | 17 / 19 | CONTACT_GEOMETRY x2 |
| knob | 75 | 60 | 68 | 7 | 63 / 68 | 67 / 72 | STALE_INPUTS x60, CONTACT_GEOMETRY x7 |
| lever | 200 | 17 | 173 | 27 | 123 / 173 | 126 / 189 | STALE_INPUTS x17, CONTACT_GEOMETRY x15, EXPORT_COUPLING x8 |
| lift_latch | 6 | 10 | 0 | 6 | 0 / 0 | 6 / 6 | STALE_INPUTS x10, CONTACT_GEOMETRY x6 |
| none | 44 | 58 | 43 | 1 | 33 / 43 | 30 / 44 | STALE_INPUTS x58, VELOCITY_EXPLOSION x1, CONTACT_GEOMETRY x1 |
| paddle | 11 | 0 | 9 | 2 | 8 / 9 | 8 / 10 | CONTACT_GEOMETRY x1, EXPORT_WELD x1 |
| panic_crossbar | 6 | 0 | 6 | 0 | 6 / 6 | 6 / 6 | - |
| panic_touchbar | 73 | 0 | 72 | 1 | 54 / 73 | 53 / 72 | EXPORT_COUPLING x1, RL_CANON x1 |
| pull | 140 | 16 | 134 | 6 | 111 / 134 | 108 / 136 | STALE_INPUTS x16, EXPORT_WELD x6 |
| push_button_screen | 7 | 0 | 7 | 0 | 6 / 7 | 7 / 7 | - |
| push_plate | 24 | 0 | 23 | 1 | 22 / 23 | 23 / 23 | EXPORT_WELD x1 |
| ring_pull | 19 | 10 | 19 | 0 | 19 / 19 | 17 / 19 | STALE_INPUTS x10 |
| slide_bolt_handle | 13 | 6 | 13 | 0 | 11 / 13 | 11 / 13 | STALE_INPUTS x6 |
| t_handle | 10 | 0 | 10 | 0 | 10 / 10 | 9 / 10 | VELOCITY_EXPLOSION x1 |
| thumb_latch | 9 | 3 | 9 | 0 | 9 / 9 | 8 / 9 | STALE_INPUTS x3 |
| wheel | 0 | 11 | 0 | 0 | 0 / 0 | 0 / 0 | STALE_INPUTS x11 |

## By kinematics

| kinematics | compared | stale | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|---|
| hinge_horizontal | 7 | 41 | 7 | 0 | 7 / 7 | 7 / 7 | STALE_INPUTS x41 |
| hinge_vertical | 578 | 138 | 515 | 63 | 401 / 516 | 427 / 554 | STALE_INPUTS x138, CONTACT_GEOMETRY x43, EXPORT_WELD x12 |
| rotor | 16 | 19 | 16 | 0 | 16 / 16 | 15 / 16 | STALE_INPUTS x19 |
| slide_horizontal | 110 | 58 | 98 | 12 | 92 / 107 | 67 / 101 | STALE_INPUTS x58, EXPORT_COUPLING x9, RL_CANON x9 |
| slide_vertical | 33 | 0 | 33 | 0 | 33 / 33 | 31 / 33 | VELOCITY_EXPLOSION x2 |

## Metric deltas

Every graded metric, per USD kind and phase: how far apart the two simulators are, against the bound. `median |delta|` and `p95 |delta|` are over the doors where the metric exists in both runs; `outside tol` is how many of them decide a grade **B**. A metric whose deltas pile up inside the band is solver noise; one whose deltas are spread far wider is a behavioural difference the class table should already name.

| kind | phase | metric | unit | n | median \|delta\| | p95 \|delta\| | tol | outside tol | worst door |
|---|---|---|---|---|---|---|---|---|---|
| `rl` | `operate_open` | `q_primary_max` | hinge | 358 | 0.0202 | 1.747 | 0.1 | 77 | `db0384_ship_watertight` (-2.368) |
| `full` | `operate_open` | `q_primary_max` | hinge | 364 | 0.01807 | 1.566 | 0.1 | 69 | `db0664_swing_single` (-2.353) |
| `full` | `hold` | `hold_displacement` | hinge | 601 | 0.0001779 | 0.7728 | 0.1 | 43 | `db0413_swing_double` (1.917) |
| `full` | `hold` | `q_at_1s` | hinge | 601 | 0.0001779 | 0.7728 | 0.1 | 43 | `db0413_swing_double` (1.917) |
| `full` | `settle` | `settle_drift_joint` | hinge | 41 | 0.08329 | 0.8234 | 0.02 | 41 | `db0124_vault` (0.8482) |
| `rl` | `hold` | `hold_displacement` | hinge | 583 | 0.0001787 | 0.4876 | 0.1 | 38 | `db0413_swing_double` (1.917) |
| `rl` | `hold` | `q_at_1s` | hinge | 583 | 0.0001787 | 0.4876 | 0.1 | 37 | `db0413_swing_double` (1.917) |
| `rl` | `operate_open` | `opened` | hinge | 358 | 0.1623 | 1.524 | 0.1 | 36 | `db0674_ship_watertight` (-2.322) |
| `rl` | `hold` | `hold_displacement` | slide | 127 | 4.939e-05 | 0.624 | 0.05 | 34 | `db0125_sliding_single` (-1.257) |
| `rl` | `hold` | `q_at_1s` | slide | 127 | 4.939e-05 | 0.624 | 0.05 | 34 | `db0125_sliding_single` (-1.257) |
| `rl` | `settle` | `settle_drift_joint` | hinge | 29 | 0.1173 | 1.015 | 0.02 | 29 | `db0124_vault` (1.029) |
| `full` | `operate_open` | `opened` | hinge | 364 | 0.1437 | 1.156 | 0.1 | 27 | `db0304_swing_single` (-1.956) |
| `full` | `relatch` | `relatch_repush_angle` | hinge | 257 | 0.0001712 | 1.512 | 0.01745 | 18 | `db0763_swing_single` (1.743) |
| `full` | `hold` | `hold_displacement` | slide | 143 | 2.334e-05 | 0.1996 | 0.05 | 16 | `db0597_sliding_single` (0.9099) |
| `full` | `hold` | `q_at_1s` | slide | 143 | 2.334e-05 | 0.1996 | 0.05 | 16 | `db0597_sliding_single` (0.9099) |
| `full` | `relatch` | `relatch_closed_angle` | hinge | 257 | 4.222e-05 | 0.04676 | 0.01745 | 14 | `db0633_swing_single` (0.07426) |
| `full` | `closer_return` | `closer_final_angle` | hinge | 252 | 3.757e-05 | 0.01592 | 0.03491 | 10 | `db0799_swing_single` (0.06114) |
| `rl` | `operate_open` | `opened` | slide | 22 | 2.334e-05 | 1.376 | 0.05 | 10 | `db0213_garage_sectional` (-2.077) |
| `rl` | `operate_open` | `q_primary_max` | slide | 22 | 0.008558 | 1.377 | 0.05 | 10 | `db0213_garage_sectional` (-2.079) |
| `full` | `operate_open` | `operator_travel_reached` | hinge | 364 | 0.0002261 | 0.03414 | 0.05 | 9 | `db0836_swing_single` (0.5349) |
| `rl` | `operate_open` | `operator_travel_reached` | hinge | 358 | 0.0002217 | 0.0343 | 0.05 | 9 | `db0836_swing_single` (0.5349) |
| `full` | `hold` | `secondary_drift` | slide | 14 | 0.1398 | 0.1433 | 0.02 | 9 | `db0474_automatic_sliding` (-0.1433) |
| `rl` | `hold` | `secondary_drift` | slide | 14 | 0.1398 | 0.1433 | 0.02 | 9 | `db0474_automatic_sliding` (-0.1433) |
| `full` | `settle` | `settle_drift` | hinge | 601 | 2.721e-07 | 6.395e-05 | 0.02 | 6 | `db0291_stall` (-0.2686) |
| `rl` | `settle` | `settle_drift` | hinge | 601 | 3.211e-07 | 6.395e-05 | 0.02 | 6 | `db0291_stall` (-0.2686) |
| `full` | `closer_return` | `peak_closing_speed` | hinge | 252 | 0.01337 | 0.5079 | 0.2 | 6 | `db0010_swing_double` (-0.9712) |
| `rl` | `closer_return` | `peak_closing_speed` | hinge | 252 | 0.01932 | 0.5079 | 0.2 | 6 | `db0010_swing_double` (-0.9658) |
| `full` | `release` | `operator_after_release_frac` | hinge | 258 | 4.134e-05 | 0.003422 | 0.1 | 5 | `db0432_cold_storage` (0.1297) |
| `full` | `locked_holds` | `locked_displacement` | hinge | 29 | 0.0001949 | 1.655 | 0.01 | 5 | `db0413_swing_double` (1.917) |
| `rl` | `locked_holds` | `locked_displacement` | hinge | 29 | 0.0001949 | 1.655 | 0.01 | 5 | `db0413_swing_double` (1.917) |
| `rl` | `relatch` | `relatch_repush_angle` | hinge | 253 | 0.0001638 | 0.0002648 | 0.01745 | 4 | `db0507_cold_storage` (-0.6015) |
| `full` | `locked_holds` | `operator_travel_reached` | hinge | 29 | 0.0003571 | 0.5076 | 0.05 | 3 | `db0114_swing_single` (0.5298) |
| `rl` | `locked_holds` | `operator_travel_reached` | hinge | 29 | 0.0003569 | 0.5076 | 0.05 | 3 | `db0114_swing_single` (0.5298) |
| `full` | `settle` | `settle_drift_joint` | slide | 3 | 0.04 | 0.04 | 0.005 | 3 | `db0415_gate_sliding` (-0.04) |
| `rl` | `hold` | `t_free` | slide | 97 | 0.032 | 0.232 | 0.25 | 2 | `db0631_sliding_single` (1.166) |
| `rl` | `closer_return` | `closer_final_angle` | hinge | 252 | 3.788e-05 | 0.01525 | 0.03491 | 2 | `db0366_gate_swing` (-0.05527) |
| `rl` | `relatch` | `t_close` | hinge | 173 | 0.06667 | 0.332 | 0.5 | 2 | `db0568_swing_single` (0.5653) |
| `rl` | `operate_open` | `bolt_retract_max_frac` | hinge | 270 | 0.02689 | 0.05317 | 0.15 | 1 | `db0548_swing_single` (-1.036) |
| `full` | `relatch` | `t_close` | hinge | 170 | 0.068 | 0.3 | 0.5 | 1 | `db0568_swing_single` (0.532) |
| `full` | `settle` | `settle_drift` | slide | 143 | 4.479e-14 | 5.167e-06 | 0.005 | 0 | - |
| `full` | `hold` | `t_free` | slide | 97 | 0.001333 | 0.1347 | 0.25 | 0 | - |
| `rl` | `settle` | `settle_drift` | slide | 143 | 4.531e-14 | 5.902e-06 | 0.005 | 0 | - |
| `full` | `hold` | `t_free` | hinge | 138 | 0.000667 | 0.034 | 0.25 | 0 | - |
| `full` | `operate_open` | `bolt_retract_max_frac` | hinge | 270 | 0.02689 | 0.05273 | 0.15 | 0 | - |
| `full` | `operate_open` | `t_open` | hinge | 343 | 0.000667 | 0.034 | 0.3 | 0 | - |
| `full` | `operate_open` | `t_open_bench` | hinge | 343 | 0.032 | 0.066 | 0.3 | 0 | - |
| `full` | `operate_open` | `t_unlatch` | hinge | 270 | 0.032 | 0.032 | 0.2 | 0 | - |
| `full` | `release` | `bolt_after_release_m` | hinge | 258 | 3.183e-05 | 4.19e-05 | 0.002 | 0 | - |
| `full` | `release` | `t_bolt_return` | hinge | 258 | 0.066 | 0.1327 | 0.2 | 0 | - |
| `full` | `relatch` | `bolt_max_during_close` | hinge | 257 | 0.003141 | 0.01075 | 0.05 | 0 | - |
| `full` | `relatch` | `bolt_min_during_close` | hinge | 257 | 0.0002686 | 0.001489 | 0.05 | 0 | - |
| `rl` | `hold` | `t_free` | hinge | 138 | 0.000667 | 0.034 | 0.25 | 0 | - |
| `rl` | `operate_open` | `t_open` | hinge | 327 | 0.000667 | 0.03333 | 0.3 | 0 | - |
| `rl` | `operate_open` | `t_open_bench` | hinge | 327 | 0.032 | 0.066 | 0.3 | 0 | - |
| `rl` | `operate_open` | `t_unlatch` | hinge | 269 | 0.032 | 0.032 | 0.2 | 0 | - |
| `rl` | `release` | `bolt_after_release_m` | hinge | 254 | 3.183e-05 | 4.19e-05 | 0.002 | 0 | - |
| `rl` | `release` | `operator_after_release_frac` | hinge | 254 | 6.256e-05 | 0.003422 | 0.1 | 0 | - |
| `rl` | `release` | `t_bolt_return` | hinge | 254 | 0.066 | 0.1333 | 0.2 | 0 | - |
| `rl` | `relatch` | `bolt_max_during_close` | hinge | 253 | 0.00324 | 0.009707 | 0.05 | 0 | - |
| `rl` | `relatch` | `bolt_min_during_close` | hinge | 253 | 0.0002774 | 0.001489 | 0.05 | 0 | - |
| `rl` | `relatch` | `relatch_closed_angle` | hinge | 253 | 2.784e-05 | 0.0001756 | 0.01745 | 0 | - |
| `full` | `hold` | `secondary_drift` | hinge | 87 | 4.958e-05 | 0.00142 | 0.05 | 0 | - |
| `full` | `closer_return` | `closer_t_close` | hinge | 213 | 0.001333 | 0.1013 | 0.5 | 0 | - |
| `rl` | `hold` | `secondary_drift` | hinge | 87 | 4.958e-05 | 0.00142 | 0.05 | 0 | - |
| `rl` | `closer_return` | `closer_t_close` | hinge | 219 | 0.001333 | 0.1333 | 0.5 | 0 | - |
| `full` | `locked_holds` | `locked_displacement` | slide | 5 | 1.001e-05 | 6.381e-05 | 0.003 | 0 | - |
| `full` | `locked_holds` | `operator_travel_reached` | slide | 5 | 0.0004613 | 0.001265 | 0.005 | 0 | - |
| `rl` | `locked_holds` | `locked_displacement` | slide | 5 | 0.0002499 | 0.0002614 | 0.003 | 0 | - |
| `rl` | `locked_holds` | `operator_travel_reached` | slide | 5 | 0.0004613 | 0.0005502 | 0.005 | 0 | - |
| `full` | `operate_open` | `opened` | slide | 22 | 7.672e-06 | 2.18e-05 | 0.05 | 0 | - |
| `full` | `operate_open` | `operator_travel_reached` | slide | 22 | 0.000324 | 0.0003896 | 0.005 | 0 | - |
| `full` | `operate_open` | `q_primary_max` | slide | 22 | 0.000505 | 0.004124 | 0.05 | 0 | - |
| `full` | `operate_open` | `t_open` | slide | 22 | 0.001333 | 0.03333 | 0.3 | 0 | - |
| `full` | `operate_open` | `t_open_bench` | slide | 22 | 0.032 | 0.1 | 0.3 | 0 | - |
| `rl` | `operate_open` | `operator_travel_reached` | slide | 22 | 0.0003241 | 0.0003924 | 0.005 | 0 | - |
| `rl` | `operate_open` | `t_open` | slide | 12 | 0.032 | 0.06667 | 0.3 | 0 | - |
| `rl` | `operate_open` | `t_open_bench` | slide | 12 | 0.03333 | 0.2333 | 0.3 | 0 | - |

<details><summary>Delta histograms (green = inside the tolerance band)</summary>

![rl operate_open.q_primary_max hinge](media/parity/hist_rl_operate_open_q_primary_max_hinge.png)

![full operate_open.q_primary_max hinge](media/parity/hist_full_operate_open_q_primary_max_hinge.png)

![full hold.hold_displacement hinge](media/parity/hist_full_hold_hold_displacement_hinge.png)

![full hold.q_at_1s hinge](media/parity/hist_full_hold_q_at_1s_hinge.png)

![full settle.settle_drift_joint hinge](media/parity/hist_full_settle_settle_drift_joint_hinge.png)

![rl hold.hold_displacement hinge](media/parity/hist_rl_hold_hold_displacement_hinge.png)

![rl hold.q_at_1s hinge](media/parity/hist_rl_hold_q_at_1s_hinge.png)

![rl operate_open.opened hinge](media/parity/hist_rl_operate_open_opened_hinge.png)

![rl hold.hold_displacement slide](media/parity/hist_rl_hold_hold_displacement_slide.png)

![rl hold.q_at_1s slide](media/parity/hist_rl_hold_q_at_1s_slide.png)

![rl settle.settle_drift_joint hinge](media/parity/hist_rl_settle_settle_drift_joint_hinge.png)

![full operate_open.opened hinge](media/parity/hist_full_operate_open_opened_hinge.png)

![full relatch.relatch_repush_angle hinge](media/parity/hist_full_relatch_relatch_repush_angle_hinge.png)

![full hold.hold_displacement slide](media/parity/hist_full_hold_hold_displacement_slide.png)

![full hold.q_at_1s slide](media/parity/hist_full_hold_q_at_1s_slide.png)

![full relatch.relatch_closed_angle hinge](media/parity/hist_full_relatch_relatch_closed_angle_hinge.png)

![full closer_return.closer_final_angle hinge](media/parity/hist_full_closer_return_closer_final_angle_hinge.png)

![rl operate_open.opened slide](media/parity/hist_rl_operate_open_opened_slide.png)

![rl operate_open.q_primary_max slide](media/parity/hist_rl_operate_open_q_primary_max_slide.png)

![full operate_open.operator_travel_reached hinge](media/parity/hist_full_operate_open_operator_travel_reached_hinge.png)

![rl operate_open.operator_travel_reached hinge](media/parity/hist_rl_operate_open_operator_travel_reached_hinge.png)

![full hold.secondary_drift slide](media/parity/hist_full_hold_secondary_drift_slide.png)

![rl hold.secondary_drift slide](media/parity/hist_rl_hold_secondary_drift_slide.png)

![full settle.settle_drift hinge](media/parity/hist_full_settle_settle_drift_hinge.png)

![rl settle.settle_drift hinge](media/parity/hist_rl_settle_settle_drift_hinge.png)

![full closer_return.peak_closing_speed hinge](media/parity/hist_full_closer_return_peak_closing_speed_hinge.png)

![rl closer_return.peak_closing_speed hinge](media/parity/hist_rl_closer_return_peak_closing_speed_hinge.png)

![full release.operator_after_release_frac hinge](media/parity/hist_full_release_operator_after_release_frac_hinge.png)

![full locked_holds.locked_displacement hinge](media/parity/hist_full_locked_holds_locked_displacement_hinge.png)

![rl locked_holds.locked_displacement hinge](media/parity/hist_rl_locked_holds_locked_displacement_hinge.png)

![rl relatch.relatch_repush_angle hinge](media/parity/hist_rl_relatch_relatch_repush_angle_hinge.png)

![full locked_holds.operator_travel_reached hinge](media/parity/hist_full_locked_holds_operator_travel_reached_hinge.png)

![rl locked_holds.operator_travel_reached hinge](media/parity/hist_rl_locked_holds_operator_travel_reached_hinge.png)

![rl hold.t_free slide](media/parity/hist_rl_hold_t_free_slide.png)

![rl closer_return.closer_final_angle hinge](media/parity/hist_rl_closer_return_closer_final_angle_hinge.png)

![rl relatch.t_close hinge](media/parity/hist_rl_relatch_t_close_hinge.png)

![rl operate_open.bolt_retract_max_frac hinge](media/parity/hist_rl_operate_open_bolt_retract_max_frac_hinge.png)

![full relatch.t_close hinge](media/parity/hist_full_relatch_t_close_hinge.png)

![full settle.settle_drift slide](media/parity/hist_full_settle_settle_drift_slide.png)

![full hold.t_free slide](media/parity/hist_full_hold_t_free_slide.png)

![rl settle.settle_drift slide](media/parity/hist_rl_settle_settle_drift_slide.png)

![full hold.t_free hinge](media/parity/hist_full_hold_t_free_hinge.png)

![full operate_open.bolt_retract_max_frac hinge](media/parity/hist_full_operate_open_bolt_retract_max_frac_hinge.png)

![full operate_open.t_open hinge](media/parity/hist_full_operate_open_t_open_hinge.png)

![full operate_open.t_open_bench hinge](media/parity/hist_full_operate_open_t_open_bench_hinge.png)

![full operate_open.t_unlatch hinge](media/parity/hist_full_operate_open_t_unlatch_hinge.png)

![full release.bolt_after_release_m hinge](media/parity/hist_full_release_bolt_after_release_m_hinge.png)

![full release.t_bolt_return hinge](media/parity/hist_full_release_t_bolt_return_hinge.png)

![full relatch.bolt_max_during_close hinge](media/parity/hist_full_relatch_bolt_max_during_close_hinge.png)

![full relatch.bolt_min_during_close hinge](media/parity/hist_full_relatch_bolt_min_during_close_hinge.png)

![rl hold.t_free hinge](media/parity/hist_rl_hold_t_free_hinge.png)

![rl operate_open.t_open hinge](media/parity/hist_rl_operate_open_t_open_hinge.png)

![rl operate_open.t_open_bench hinge](media/parity/hist_rl_operate_open_t_open_bench_hinge.png)

![rl operate_open.t_unlatch hinge](media/parity/hist_rl_operate_open_t_unlatch_hinge.png)

![rl release.bolt_after_release_m hinge](media/parity/hist_rl_release_bolt_after_release_m_hinge.png)

![rl release.operator_after_release_frac hinge](media/parity/hist_rl_release_operator_after_release_frac_hinge.png)

![rl release.t_bolt_return hinge](media/parity/hist_rl_release_t_bolt_return_hinge.png)

![rl relatch.bolt_max_during_close hinge](media/parity/hist_rl_relatch_bolt_max_during_close_hinge.png)

![rl relatch.bolt_min_during_close hinge](media/parity/hist_rl_relatch_bolt_min_during_close_hinge.png)

![rl relatch.relatch_closed_angle hinge](media/parity/hist_rl_relatch_relatch_closed_angle_hinge.png)

![full hold.secondary_drift hinge](media/parity/hist_full_hold_secondary_drift_hinge.png)

![full closer_return.closer_t_close hinge](media/parity/hist_full_closer_return_closer_t_close_hinge.png)

![rl hold.secondary_drift hinge](media/parity/hist_rl_hold_secondary_drift_hinge.png)

![rl closer_return.closer_t_close hinge](media/parity/hist_rl_closer_return_closer_t_close_hinge.png)

![full operate_open.opened slide](media/parity/hist_full_operate_open_opened_slide.png)

![full operate_open.operator_travel_reached slide](media/parity/hist_full_operate_open_operator_travel_reached_slide.png)

![full operate_open.q_primary_max slide](media/parity/hist_full_operate_open_q_primary_max_slide.png)

![full operate_open.t_open slide](media/parity/hist_full_operate_open_t_open_slide.png)

![full operate_open.t_open_bench slide](media/parity/hist_full_operate_open_t_open_bench_slide.png)

![rl operate_open.operator_travel_reached slide](media/parity/hist_rl_operate_open_operator_travel_reached_slide.png)

![rl operate_open.t_open slide](media/parity/hist_rl_operate_open_t_open_slide.png)

![rl operate_open.t_open_bench slide](media/parity/hist_rl_operate_open_t_open_bench_slide.png)

</details>

## Top offenders (20)

| door | family | grade full / rl | phase | MuJoCo | PhysX full | PhysX rl | classes | likely root cause |
|---|---|---|---|---|---|---|---|---|
| `db0334_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.002866 | 1.658 (disagree) | 1.658 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0413_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.002976 | 1.92 (disagree) | 1.92 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0534_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003278 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0702_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003719 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0733_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003324 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0792_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003315 | 1.658 (disagree) | 1.658 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW`, `EXPORT_COUPLING`, `RL_CANON` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0026_swing_single` | swing_single | C / C | `hold` | hold_displacement=1.194e-06 | 1.138 (disagree) | 1.137 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0149_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.004243 | 1.92 (disagree) | 1.92 (disagree) | `CONTACT_GEOMETRY`, `METRICS_VERSION_SKEW` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes. The global self-collision part is FIXED in... |
| `db0158_swing_double` | swing_double | C / C | `hold` | hold_displacement=1.727e-06 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0216_swing_single` | swing_single | C / C | `hold` | hold_displacement=4.645e-06 | 1.78 (disagree) | 1.78 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0222_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003401 | 1.571 (disagree) | 1.571 (disagree) | `CONTACT_GEOMETRY`, `METRICS_VERSION_SKEW` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes. The global self-collision part is FIXED in... |
| `db0316_swing_double` | swing_double | C / C | `hold` | hold_displacement=2.281e-06 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0454_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003065 | 1.658 (disagree) | 1.658 (disagree) | `CONTACT_GEOMETRY`, `METRICS_VERSION_SKEW` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes. The global self-collision part is FIXED in... |
| `db0577_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.00314 | 1.571 (disagree) | 1.571 (disagree) | `CONTACT_GEOMETRY`, `METRICS_VERSION_SKEW` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes. The global self-collision part is FIXED in... |
| `db0591_pivot` | pivot | C / C | `hold` | hold_displacement=5.598e-06 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |
| `db0604_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003248 | 1.92 (disagree) | 1.92 (disagree) | `CONTACT_GEOMETRY`, `METRICS_VERSION_SKEW` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes. The global self-collision part is FIXED in... |
| `db0682_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003026 | 1.571 (disagree) | 1.571 (disagree) | `CONTACT_GEOMETRY`, `METRICS_VERSION_SKEW` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes. The global self-collision part is FIXED in... |
| `db0714_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.00318 | 1.571 (disagree) | 1.571 (disagree) | `CONTACT_GEOMETRY`, `METRICS_VERSION_SKEW` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes. The global self-collision part is FIXED in... |
| `db0832_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003402 | 1.92 (disagree) | 1.92 (disagree) | `CONTACT_GEOMETRY`, `METRICS_VERSION_SKEW` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes. The global self-collision part is FIXED in... |
| `db0897_swing_single` | swing_single | C / C | `hold` | hold_displacement=3.934e-06 | 1.725 (disagree) | 1.725 (disagree) | `EXPORT_WELD`, `METRICS_VERSION_SKEW` | H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench... |

### `db0334_swing_double` - grade C (swing_double, tubular_latch latch, deadbolt_single lock engaged, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.658 in PhysX vs 0.002867
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.658 in PhysX vs 0.002867
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0413_swing_double` - grade C (swing_double, tubular_latch latch, deadbolt_single lock engaged, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.92 in PhysX vs 0.002975
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.92 in PhysX vs 0.002975
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0534_swing_double` - grade C (swing_double, tubular_latch latch, thumbturn_only lock engaged, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged thumbturn_only holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003277
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged thumbturn_only holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003277
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0702_swing_double` - grade C (swing_double, tubular_latch latch, multipoint lock engaged, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged multipoint holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003719
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged multipoint holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003719
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0733_swing_double` - grade C (swing_double, tubular_latch latch, deadbolt_single lock engaged, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003324
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003324
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0792_swing_double` - grade C (swing_double, tubular_latch latch, deadbolt_single lock engaged, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open **disagree**, release na, relatch na, closer_return na, locked_holds na
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * operate_open: operator moved (travel 0.96) but bolt retracted n/a of its throw; MuJoCo opened 1.454
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
  * operate_open: operate_open agrees in door.usda but not in door_rl.usda

### `db0026_swing_single` - grade C (swing_single, none latch, mag_lock lock engaged, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (1.194e-06), PhysX opened 1.138
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (1.194e-06), PhysX opened 1.137
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0149_swing_double` - grade C (swing_double, tubular_latch latch, none lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.004243), PhysX opened 1.92: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.004243), PhysX opened 1.92: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0158_swing_double` - grade C (swing_double, none latch, mag_lock lock engaged, auto_operator_low_energy closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (1.727e-06), PhysX opened 1.571
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (1.727e-06), PhysX opened 1.571
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0216_swing_single` - grade C (swing_single, none latch, mag_lock lock engaged, concealed_overhead closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (4.645e-06), PhysX opened 1.78
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (4.645e-06), PhysX opened 1.78
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0222_swing_double` - grade C (swing_double, tubular_latch latch, none lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003401), PhysX opened 1.571: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003401), PhysX opened 1.571: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0316_swing_double` - grade C (swing_double, none latch, mag_lock lock engaged, surface_overhead closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (2.281e-06), PhysX opened 1.571
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (2.281e-06), PhysX opened 1.571
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0454_swing_double` - grade C (swing_double, tubular_latch latch, none lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003065), PhysX opened 1.658: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003065), PhysX opened 1.658: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0577_swing_double` - grade C (swing_double, tubular_latch latch, none lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.00314), PhysX opened 1.571: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.00314), PhysX opened 1.571: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0591_pivot` - grade C (pivot, none latch, mag_lock lock engaged, floor_spring closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (5.598e-06), PhysX opened 1.571
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (5.598e-06), PhysX opened 1.571
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0604_swing_double` - grade C (swing_double, tubular_latch latch, none lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003248), PhysX opened 1.92: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003248), PhysX opened 1.92: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0682_swing_double` - grade C (swing_double, tubular_latch latch, deadbolt_single lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003026), PhysX opened 1.571: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003026), PhysX opened 1.571: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0714_swing_double` - grade C (swing_double, tubular_latch latch, thumbturn_only lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.00318), PhysX opened 1.571: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.00318), PhysX opened 1.571: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0832_swing_double` - grade C (swing_double, tubular_latch latch, none lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003402), PhysX opened 1.92: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open agree, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003402), PhysX opened 1.92: bolt / strike contact not engaging
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

### `db0897_swing_single` - grade C (swing_single, none latch, mag_lock lock engaged, concealed_overhead closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (3.934e-06), PhysX opened 1.725
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (3.934e-06), PhysX opened 1.725
  * -: metrics arrival_speed, speed_at_latch not graded: mujoco 1.1 vs physx 1.0

## Known not-comparable categories

* **Env-release locks** (mag lock, delayed egress, card reader, electric strike, interlock): MuJoCo holds them with a `<weld>`; the USD carries the PhysX counterpart since the export fix - a breakable `FixedJoint` base -> leaf with `physics:excludeFromArticulation` and `breakForce` = the holding force - so both simulators must *hold*. A door that *opens* here in PhysX means the joint is missing or was not parsed (class `EXPORT_WELD`).
* **Panic doors with the robot outside and no far-side trim**: `operator_joint` is None, the exit device is welded in `door_rl.usda`; both simulators must *hold*.
* **Welded releases in `door_rl.usda`**: parts the operator retracts (hooks, cremone bolts, wheel-driven dogs) are welded RELEASED and the door must open; an engaged lock with no canonical slot and no operator coupling (thumbturns, aux bolts, extra dogs) is welded ENGAGED and the RL expectation for `operate_open` flips to 'stays closed'. The split is ground truth from `doorbench:rl["welded"]`, not a guess; a `full` / `rl` disagreement there is `RL_CANON`.
* **Free-swing families** (saloon, bifold, accordion, bypass, pet door, strip curtain, revolving, turnstiles): no qa.py behavioural check exists, so their MuJoCo reference is itself unvalidated; their push phase is informational.
* **Closer-arm loop closures** (`connect` equalities) are not exported: the pinion / elbow joints swing freely in `door.usda` and are excluded from the limit check.

## Reproduce

```bash
# 1. run the shared protocol (doorbench/parity/protocol.py) in MuJoCo on the CPU and in Isaac Sim on the GPU pod:
#    -> results/parity/mujoco.json, results/parity/isaac_full.json, results/parity/isaac_rl.json
#    (optional sensitivity reruns: results/parity/isaac_<kind>_<variant>.json, e.g. isaac_full_dt240.json)
# 2. join, classify, render this page + summary.json (no simulator needed)
PYTHONPATH=$PWD python scripts/isaaclab/parity_report.py            # --results DIR --top N --no-plots
# 3. publish per door: qa.json isaac_parity + manifest badge (idempotent; --check for CI)
PYTHONPATH=$PWD python scripts/merge_isaac_results.py
# legacy: render from the 40-door probe instead of protocol results
PYTHONPATH=$PWD python scripts/isaaclab/probe_to_parity.py && PYTHONPATH=$PWD python scripts/isaaclab/parity_report.py --results results/parity/probe
```
