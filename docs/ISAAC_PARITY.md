# Isaac parity gate

_Generated 2026-09-04T23:12:04 by `scripts/isaaclab/parity_report.py` from `results/parity/probe/` (commit `99dd62c74`). Reference: MuJoCo mujoco 3.12.0; PhysX: isaac_sim 5.1.0, isaac_lab 2.3.2, api 2.3._

> **Note.** These inputs come from the legacy 40-door probe (`validate_usd_isaacsim.py`: a fixed 60 N*m / 60 N push plus 8 N*m on the operator for 400 steps, position targets zeroed every step) adapted to the parity schema by `scripts/isaaclab/probe_to_parity.py`, with qa.json as the MuJoCo side. The protocols differ (adaptive QA push, latch coupling, spring targets), so every class below is a hypothesis about the probe as much as about the door; the numbers are replaced when `doorbench/parity/protocol.py` runs on the GPU.

Every door runs **one behavioural protocol** in MuJoCo (the reference physics, CPU) and in Isaac Sim / PhysX on the GPU pod, on both USD kinds (`door.usda` full fidelity, `door_rl.usda` canonical 8-link). The two runs are compared phase by phase: both simulators must reach the same pass / fail verdict (else grade **C**), and when they agree the metrics must be within tolerance (else grade **B**); **A** is parity, **X** means the door could not be compared (spawn / structure error). A disagreement is tagged with a discrepancy class whose likely root cause comes from the analysis of the first 40-door probe. The per-door verdict is published in `qa.json` (`isaac_parity`) and as a badge in the viewer.

## Headline

| USD kind | tested | parity (A) | same verdicts (A + B) | disagree (C) | not comparable (X) | untested |
|---|---|---|---|---|---|---|
| `full` | 40 / 1000 | **21 / 1000** (52 % of tested) | 26 / 1000 (65 %) | 14 | 0 | 960 |
| `rl` | 40 / 1000 | **21 / 1000** (52 % of tested) | 26 / 1000 (65 %) | 14 | 0 | 960 |

Door badge (`qa.json.isaac_parity.ok`; viewer chip *Isaac parity*): **26 ok** (grade A or B in every tested kind), **14 fail** (a status disagreement or not comparable), 960 untested.

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

| metric | hinge (rad, s) | slide (m, s) | relative |
|---|---|---|---|
| `settle_drift` | 0.02 | 0.005 | - |
| `settle_drift_primary` | 0.02 | 0.005 | - |
| `settle_drift_operator` | 0.02 | 0.005 | - |
| `settle_drift_latch` | 0.002 | 0.002 | - |
| `pen0_m` | 0.003 | 0.003 | - |
| `hold_displacement` | 0.01 | 0.003 | - |
| `t_free` | 0.25 | 0.25 | 30 % |
| `q_at_1s` | 0.1 | 0.05 | 20 % |
| `opened` | 0.1 | 0.05 | 20 % |
| `actuate_displacement` | 0.1 | 0.05 | 20 % |
| `t_open` | 0.3 | 0.3 | 30 % |
| `t_open_bench` | 0.3 | 0.3 | 30 % |
| `t_unlatch` | 0.2 | 0.2 | - |
| `operator_travel_reached` | 0.05 | 0.005 | 10 % |
| `bolt_retract_max_frac` | 0.15 | 0.15 | - |
| `curve_rmse_primary` | 0.15 | 0.05 | - |
| `bolt_after_release_m` | 0.002 | 0.002 | - |
| `t_bolt_return` | 0.2 | 0.2 | - |
| `operator_after_release_frac` | 0.1 | 0.1 | - |
| `relatch_closed_angle` | 0.0175 | 0.005 | - |
| `relatch_repush_angle` | 0.0175 | 0.005 | - |
| `t_close` | 0.5 | 0.5 | 30 % |
| `arrival_speed` | 0.2 | 0.1 | 30 % |
| `closer_final_angle` | 0.0349 | 0.01 | - |
| `closer_t_close` | 0.5 | 0.5 | 30 % |
| `peak_closing_speed` | 0.2 | 0.1 | 30 % |
| `curve_rmse_closer` | 0.1 | 0.05 | - |
| `locked_displacement` | 0.01 | 0.003 | - |
| *(any other metric)* | 0.05 | 0.02 | 20 % |

</details>

## Discrepancy classes

| class | full | rl | doors | what it means | likely root cause | fix direction | examples |
|---|---|---|---|---|---|---|---|
| `PHYSICS_PARAM_PRELOAD` | 7 | 7 | 7 | settle drift or a false opening that matches a spring whose target was zeroed (operator sag q = tau_g / k, closer preload gone) | H2 / D1: Isaac Lab forwards zero position targets to the PhysX drives; the USD targetPosition (springref) is lost while the stiffness stays | restore doorbench:target_si / rl joints[*].target each step (as DoorMechanismAction does); report drift per joint | `db0003_cold_storage`, `db0017_hatch_ceiling`, `db0021_swing_single`, `db0032_sliding_single` |
| `EXPORT_COUPLING` | 6 | 6 | 6 | the operator turns but the bolt does not retract (or does not return) in PhysX, so the door stays latched | H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate it as a kinematic clamp each step (the soft target offset under-retracts by 40-60 %) | shared clamp function (write_joint_state_to_sim(max(latch_q, scale * op_q))) in the parity runner and DoorMechanismAction; read scale from doorbench:rl.latch_coupling / doorbench:latch_coupling_scale | `db0002_swing_single`, `db0013_swing_single`, `db0027_swing_single`, `db0036_swing_single` |
| `QUANT` | 5 | 5 | 5 | both simulators reach the same pass / fail verdicts but at least one metric is outside tolerance | soft MuJoCo contacts / limits (solref 0.005) vs rigid PhysX limits, constraint-based vs effort-based Coulomb friction, implicitfast vs implicit drives at high damping; solver dt | rerun at Isaac dt 1/240 (32/8 iterations) and MuJoCo dt 0.001; if the delta shrinks below tolerance tag SOLVER_SENSITIVITY, else triage by phase | `db0005_garage_tiltup`, `db0006_gate_swing`, `db0014_gate_swing`, `db0015_swing_double` |
| `PHYSICS_PARAM_FRICTION` | 3 | 3 | 3 | a free-swinging door opens in one simulator but not the other (timing or threshold), pointing at Coulomb friction or gravity bias mapping | H3: physxJointAxis friction efforts vs MuJoCo frictionloss (stick / slip differs), legacy jointFriction coefficient adding load-dependent friction, or an effort below the adaptive QA push | measure breakaway effort on one door in both sims; use the per-door qa_push; zero the legacy coefficient | `db0001_rollup`, `db0017_hatch_ceiling`, `db0029_sliding_single` |
| `EXPORT_WELD` | 1 | 1 | 1 | MuJoCo holds the leaf (weld / lock equality) but PhysX has nothing holding it, so the door opens under the push | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active with its full range | author a FixedJoint / locked drive (or a D6 joint with breakForce = holding_force_N) tagged doorbench:env_release that DoorMechanismAction disables on REX / badge / timer; classify env_release_only and test 'holds' in both sims | `db0026_swing_single` |
| `CONTACT_GEOMETRY` | 1 | 1 | 1 | the bolt retracted (or there is no bolt) yet the leaf did not move, or a latch that holds in MuJoCo does not engage in PhysX (convex hulls, strike lip, panel clearance) | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes, self-collision disabled globally vs MuJoCo's selective exclusion | enable contact reporting; rerun with Env collision disabled, then without the hardware part, to bisect frame contact vs articulation; author PhysxFilteredPairsAPI for model.contact_excludes | `db0033_gate_sliding` |

## By family

| family | doors | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|---|
| automatic_swing | 10 | 1 | 1 | 0 | 1 / 1 | 1 / 1 | - |
| bifold | 30 | 3 | 3 | 0 | 3 / 3 | 3 / 3 | - |
| cold_storage | 15 | 1 | 0 | 1 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_PRELOAD x1 |
| garage_tiltup | 7 | 1 | 1 | 0 | 0 / 1 | 0 / 1 | - |
| gate_sliding | 10 | 1 | 0 | 1 | 0 / 0 | 0 / 0 | CONTACT_GEOMETRY x1 |
| gate_swing | 40 | 3 | 3 | 0 | 1 / 3 | 1 / 3 | - |
| hatch_ceiling | 8 | 1 | 0 | 1 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_PRELOAD x1, PHYSICS_PARAM_FRICTION x1 |
| rollup | 15 | 1 | 0 | 1 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_FRICTION x1 |
| saloon | 12 | 1 | 1 | 0 | 1 / 1 | 1 / 1 | - |
| sliding_bypass | 35 | 2 | 2 | 0 | 2 / 2 | 2 / 2 | - |
| sliding_single | 100 | 6 | 4 | 2 | 3 / 4 | 3 / 4 | PHYSICS_PARAM_FRICTION x1, PHYSICS_PARAM_PRELOAD x1 |
| strip_curtain | 8 | 1 | 1 | 0 | 1 / 1 | 1 / 1 | - |
| swing_double | 76 | 4 | 4 | 0 | 3 / 4 | 3 / 4 | - |
| swing_single | 440 | 14 | 6 | 8 | 6 / 6 | 6 / 6 | EXPORT_COUPLING x6, PHYSICS_PARAM_PRELOAD x4, EXPORT_WELD x1 |

## By hardware

### latch kind

| kind | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|
| deadlatch | 2 | 0 | 2 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_PRELOAD x2, EXPORT_COUPLING x1 |
| gravity_bar | 1 | 1 | 0 | 1 / 1 | 1 / 1 | - |
| hook | 2 | 1 | 1 | 0 / 1 | 0 / 1 | PHYSICS_PARAM_PRELOAD x1 |
| magnetic | 1 | 1 | 0 | 1 / 1 | 1 / 1 | - |
| mortise_latch | 2 | 1 | 1 | 1 / 1 | 1 / 1 | PHYSICS_PARAM_PRELOAD x1, EXPORT_COUPLING x1 |
| none | 21 | 17 | 4 | 14 / 17 | 14 / 17 | PHYSICS_PARAM_FRICTION x3, PHYSICS_PARAM_PRELOAD x1, EXPORT_WELD x1 |
| rim_latch | 2 | 2 | 0 | 2 / 2 | 2 / 2 | - |
| roller | 1 | 0 | 1 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_PRELOAD x1 |
| slide_bolt | 2 | 1 | 1 | 0 / 1 | 0 / 1 | CONTACT_GEOMETRY x1 |
| tubular_latch | 4 | 0 | 4 | 0 / 0 | 0 / 0 | EXPORT_COUPLING x4, PHYSICS_PARAM_PRELOAD x1 |
| vertical_rods | 2 | 2 | 0 | 2 / 2 | 2 / 2 | - |

### lock kind

| kind | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|
| deadbolt_single | 2 | 2 | 0 | 2 / 2 | 2 / 2 | - |
| delayed_egress | 1 | 0 | 1 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_PRELOAD x1 |
| hook_lock | 2 | 1 | 1 | 1 / 1 | 1 / 1 | PHYSICS_PARAM_PRELOAD x1 |
| mag_lock | 1 | 0 | 1 | 0 / 0 | 0 / 0 | EXPORT_WELD x1 |
| none | 26 | 18 | 8 | 15 / 18 | 15 / 18 | EXPORT_COUPLING x4, PHYSICS_PARAM_FRICTION x3, PHYSICS_PARAM_PRELOAD x3 |
| padlock | 3 | 2 | 1 | 1 / 2 | 1 / 2 | PHYSICS_PARAM_PRELOAD x1, EXPORT_COUPLING x1 |
| slide_bolt | 3 | 2 | 1 | 1 / 2 | 1 / 2 | CONTACT_GEOMETRY x1 |
| thumbturn_only | 2 | 1 | 1 | 1 / 1 | 1 / 1 | PHYSICS_PARAM_PRELOAD x1, EXPORT_COUPLING x1 |

### closer kind

| kind | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|
| auto_operator_low_energy | 1 | 1 | 0 | 1 / 1 | 1 / 1 | - |
| concealed_overhead | 2 | 1 | 1 | 0 / 1 | 0 / 1 | PHYSICS_PARAM_PRELOAD x1, EXPORT_COUPLING x1 |
| electromagnetic_hold | 2 | 2 | 0 | 2 / 2 | 2 / 2 | - |
| floor_spring | 1 | 0 | 1 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_PRELOAD x1, EXPORT_COUPLING x1 |
| gate | 1 | 1 | 0 | 0 / 1 | 0 / 1 | - |
| none | 25 | 14 | 11 | 11 / 14 | 11 / 14 | EXPORT_COUPLING x4, PHYSICS_PARAM_PRELOAD x4, PHYSICS_PARAM_FRICTION x3 |
| spring_hinge | 1 | 1 | 0 | 1 / 1 | 1 / 1 | - |
| surface_overhead | 7 | 6 | 1 | 6 / 6 | 6 / 6 | PHYSICS_PARAM_PRELOAD x1 |

### operator kind

| kind | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|
| flush_pull | 2 | 2 | 0 | 2 / 2 | 2 / 2 | - |
| gate_latch_fork | 1 | 1 | 0 | 1 / 1 | 1 / 1 | - |
| hook_lock_slider | 1 | 0 | 1 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_PRELOAD x1 |
| knob | 7 | 4 | 3 | 4 / 4 | 4 / 4 | EXPORT_COUPLING x3 |
| lever | 4 | 0 | 4 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_PRELOAD x4, EXPORT_COUPLING x2 |
| lift_latch | 1 | 1 | 0 | 0 / 1 | 0 / 1 | - |
| none | 3 | 3 | 0 | 2 / 3 | 2 / 3 | - |
| paddle | 1 | 0 | 1 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_PRELOAD x1, EXPORT_COUPLING x1 |
| panic_touchbar | 5 | 5 | 0 | 5 / 5 | 5 / 5 | - |
| pull | 10 | 7 | 3 | 5 / 7 | 5 / 7 | PHYSICS_PARAM_FRICTION x2, EXPORT_WELD x1 |
| push_plate | 2 | 2 | 0 | 2 / 2 | 2 / 2 | - |
| ring_pull | 1 | 0 | 1 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_PRELOAD x1, PHYSICS_PARAM_FRICTION x1 |
| slide_bolt_handle | 2 | 1 | 1 | 0 / 1 | 0 / 1 | CONTACT_GEOMETRY x1 |

## By kinematics

| kinematics | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|
| hinge_horizontal | 3 | 2 | 1 | 1 / 2 | 1 / 2 | PHYSICS_PARAM_PRELOAD x1, PHYSICS_PARAM_FRICTION x1 |
| hinge_vertical | 27 | 18 | 9 | 15 / 18 | 15 / 18 | EXPORT_COUPLING x6, PHYSICS_PARAM_PRELOAD x5, EXPORT_WELD x1 |
| slide_horizontal | 9 | 6 | 3 | 5 / 6 | 5 / 6 | PHYSICS_PARAM_FRICTION x1, PHYSICS_PARAM_PRELOAD x1, CONTACT_GEOMETRY x1 |
| slide_vertical | 1 | 0 | 1 | 0 / 0 | 0 / 0 | PHYSICS_PARAM_FRICTION x1 |

## Top offenders (19)

| door | family | grade full / rl | phase | MuJoCo | PhysX full | PhysX rl | classes | likely root cause |
|---|---|---|---|---|---|---|---|---|
| `db0017_hatch_ceiling` | hatch_ceiling | C / C | `settle` | settle_drift=1.699e-05 | 1.566 (disagree) | 1.566 (disagree) | `PHYSICS_PARAM_PRELOAD`, `PHYSICS_PARAM_FRICTION` | H2 / D1: Isaac Lab forwards zero position targets to the PhysX drives; the USD targetPosition (springref) is lost while the stiffness stays |
| `db0036_swing_single` | swing_single | C / C | `settle` | settle_drift=3.129e-06 | 0.2905 (disagree) | 0.301 (disagree) | `PHYSICS_PARAM_PRELOAD`, `EXPORT_COUPLING` | H2 / D1: Isaac Lab forwards zero position targets to the PhysX drives; the USD targetPosition (springref) is lost while the stiffness stays |
| `db0039_swing_single` | swing_single | C / C | `settle` | settle_drift=3.457e-05 | 0.4 (disagree) | 0.4 (disagree) | `PHYSICS_PARAM_PRELOAD`, `EXPORT_COUPLING` | H2 / D1: Isaac Lab forwards zero position targets to the PhysX drives; the USD targetPosition (springref) is lost while the stiffness stays |
| `db0040_swing_single` | swing_single | C / C | `settle` | settle_drift=2.65e-11 | 0.2282 (disagree) | 0.2356 (disagree) | `PHYSICS_PARAM_PRELOAD`, `EXPORT_COUPLING` | H2 / D1: Isaac Lab forwards zero position targets to the PhysX drives; the USD targetPosition (springref) is lost while the stiffness stays |
| `db0001_rollup` | rollup | C / C | `hold` | hold_displacement=2.39 | 9.071e-08 (disagree) | 8.083e-08 (disagree) | `PHYSICS_PARAM_FRICTION` | H3: physxJointAxis friction efforts vs MuJoCo frictionloss (stick / slip differs), legacy jointFriction coefficient adding load-dependent friction, or an effor... |
| `db0002_swing_single` | swing_single | C / C | `operate_open` | opened=1.692 | 0.001957 (disagree) | 0.001957 (disagree) | `EXPORT_COUPLING` | H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate i... |
| `db0003_cold_storage` | cold_storage | C / C | `settle` | settle_drift=2.997e-12 | 0.4887 (disagree) | 0.4836 (disagree) | `PHYSICS_PARAM_PRELOAD` | H2 / D1: Isaac Lab forwards zero position targets to the PhysX drives; the USD targetPosition (springref) is lost while the stiffness stays |
| `db0013_swing_single` | swing_single | C / C | `operate_open` | opened=0.9718 | 0.001835 (disagree) | 0.001835 (disagree) | `EXPORT_COUPLING` | H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate i... |
| `db0021_swing_single` | swing_single | C / C | `settle` | settle_drift=6.53e-07 | 0.2909 (disagree) | 0.2941 (disagree) | `PHYSICS_PARAM_PRELOAD` | H2 / D1: Isaac Lab forwards zero position targets to the PhysX drives; the USD targetPosition (springref) is lost while the stiffness stays |
| `db0026_swing_single` | swing_single | C / C | `hold` | hold_displacement=1.194e-06 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0027_swing_single` | swing_single | C / C | `operate_open` | opened=1.556 | 0.001833 (disagree) | 0.001833 (disagree) | `EXPORT_COUPLING` | H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate i... |
| `db0029_sliding_single` | sliding_single | C / C | `hold` | hold_displacement=1.21 | 0.007471 (disagree) | 0.005962 (disagree) | `PHYSICS_PARAM_FRICTION` | H3: physxJointAxis friction efforts vs MuJoCo frictionloss (stick / slip differs), legacy jointFriction coefficient adding load-dependent friction, or an effor... |
| `db0032_sliding_single` | sliding_single | C / C | `settle` | settle_drift=4.931e-14 | 0.02694 (disagree) | 0.04431 (disagree) | `PHYSICS_PARAM_PRELOAD` | H2 / D1: Isaac Lab forwards zero position targets to the PhysX drives; the USD targetPosition (springref) is lost while the stiffness stays |
| `db0033_gate_sliding` | gate_sliding | C / C | `operate_open` | opened=3.6 | 0.002 (disagree) | 0.001998 (disagree) | `CONTACT_GEOMETRY` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes, self-collision disabled globally vs MuJoCo... |
| `db0005_garage_tiltup` | garage_tiltup | B / B | `hold` | hold_displacement=0.8581 | 1.536 (quant) | 1.536 (quant) | `QUANT` | soft MuJoCo contacts / limits (solref 0.005) vs rigid PhysX limits, constraint-based vs effort-based Coulomb friction, implicitfast vs implicit drives at high ... |
| `db0006_gate_swing` | gate_swing | B / B | `hold` | hold_displacement=0.1748 | 0.4132 (quant) | 0.4129 (quant) | `QUANT` | soft MuJoCo contacts / limits (solref 0.005) vs rigid PhysX limits, constraint-based vs effort-based Coulomb friction, implicitfast vs implicit drives at high ... |
| `db0014_gate_swing` | gate_swing | B / B | `operate_open` | opened=0.8732 | 1.571 (quant) | 1.571 (quant) | `QUANT` | soft MuJoCo contacts / limits (solref 0.005) vs rigid PhysX limits, constraint-based vs effort-based Coulomb friction, implicitfast vs implicit drives at high ... |
| `db0015_swing_double` | swing_double | B / B | `hold` | hold_displacement=1.571 | 1.172 (quant) | 1.026 (quant) | `QUANT` | soft MuJoCo contacts / limits (solref 0.005) vs rigid PhysX limits, constraint-based vs effort-based Coulomb friction, implicitfast vs implicit drives at high ... |
| `db0038_sliding_single` | sliding_single | B / B | `hold` | hold_displacement=0.3784 | 1.028 (quant) | 0.8678 (quant) | `QUANT` | soft MuJoCo contacts / limits (solref 0.005) vs rigid PhysX limits, constraint-based vs effort-based Coulomb friction, implicitfast vs implicit drives at high ... |

### `db0017_hatch_ceiling` - grade C (hatch_ceiling, none latch, none lock, none closer)

* `full` grade C: structure agree, settle **disagree**, hold na, operate_open **disagree**, sanity agree
  * settle: drift mujoco=1.699e-05 physx=1.566
  * operate_open: nothing holds this door, yet PhysX opened only 7.548e-12 vs 1.005: push below the gravity / friction load, or friction mapped differently
* `rl` grade C: structure agree, settle **disagree**, hold na, operate_open **disagree**, sanity agree
  * settle: drift mujoco=1.699e-05 physx=1.566
  * operate_open: nothing holds this door, yet PhysX opened only 9.248e-08 vs 1.005: push below the gravity / friction load, or friction mapped differently

### `db0036_swing_single` - grade C (swing_single, tubular_latch latch, none lock, concealed_overhead closer)

* `full` grade C: structure agree, settle **disagree**, hold na, operate_open **disagree**, release na, relatch na, closer_return na, sanity agree
  * settle: drift mujoco=3.129e-06 physx=0.2905
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 0.9402
* `rl` grade C: structure agree, settle **disagree**, hold na, operate_open **disagree**, release na, relatch na, closer_return na, sanity agree
  * settle: drift mujoco=3.129e-06 physx=0.301
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 0.9402

### `db0039_swing_single` - grade C (swing_single, mortise_latch latch, thumbturn_only lock, floor_spring closer)

* `full` grade C: structure agree, settle **disagree**, hold na, operate_open **disagree**, release na, relatch na, closer_return na, sanity agree
  * settle: drift mujoco=3.457e-05 physx=0.4
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 0.865
* `rl` grade C: structure agree, settle **disagree**, hold na, operate_open **disagree**, release na, relatch na, closer_return na, sanity agree
  * settle: drift mujoco=3.457e-05 physx=0.4
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 0.865

### `db0040_swing_single` - grade C (swing_single, deadlatch latch, padlock lock, none closer)

* `full` grade C: structure agree, settle **disagree**, hold na, operate_open **disagree**, release na, relatch na, sanity agree
  * settle: drift mujoco=2.65e-11 physx=0.2282
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 1.69
* `rl` grade C: structure agree, settle **disagree**, hold na, operate_open **disagree**, release na, relatch na, sanity agree
  * settle: drift mujoco=2.65e-11 physx=0.2356
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 1.69

### `db0001_rollup` - grade C (rollup, none latch, none lock, none closer)

* `full` grade C: structure agree, settle agree, hold **disagree**, sanity agree
  * hold: free push: mujoco opened, physx stuck (hold_displacement 2.39 vs 9.071e-08)
* `rl` grade C: structure agree, settle agree, hold **disagree**, sanity agree
  * hold: free push: mujoco opened, physx stuck (hold_displacement 2.39 vs 8.083e-08)

### `db0002_swing_single` - grade C (swing_single, tubular_latch latch, none lock, none closer)

* `full` grade C: structure agree, settle agree, hold na, operate_open **disagree**, release na, relatch na, sanity agree
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 1.692
* `rl` grade C: structure agree, settle agree, hold na, operate_open **disagree**, release na, relatch na, sanity agree
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 1.692

### `db0003_cold_storage` - grade C (cold_storage, roller latch, none lock, none closer)

* `full` grade C: structure agree, settle **disagree**, hold na, operate_open quant, sanity agree
  * settle: drift mujoco=2.997e-12 physx=0.4887
* `rl` grade C: structure agree, settle **disagree**, hold na, operate_open quant, sanity agree
  * settle: drift mujoco=2.997e-12 physx=0.4836

### `db0013_swing_single` - grade C (swing_single, tubular_latch latch, none lock, none closer)

* `full` grade C: structure agree, settle agree, hold na, operate_open **disagree**, release na, relatch na, sanity agree
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 0.9718
* `rl` grade C: structure agree, settle agree, hold na, operate_open **disagree**, release na, relatch na, sanity agree
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 0.9718

### `db0021_swing_single` - grade C (swing_single, deadlatch latch, delayed_egress lock engaged, surface_overhead closer)

* `full` grade C: structure agree, settle **disagree**, hold agree, sanity agree
  * settle: drift mujoco=6.53e-07 physx=0.2909
* `rl` grade C: structure agree, settle **disagree**, hold agree, sanity agree
  * settle: drift mujoco=6.53e-07 physx=0.2941

### `db0026_swing_single` - grade C (swing_single, none latch, mag_lock lock engaged, none closer)

* `full` grade C: structure agree, settle agree, hold **disagree**, sanity agree
  * hold: mag_lock engaged: MuJoCo weld holds (1.194e-06), PhysX opened 1.571
* `rl` grade C: structure agree, settle agree, hold **disagree**, sanity agree
  * hold: mag_lock engaged: MuJoCo weld holds (1.194e-06), PhysX opened 1.571

### `db0027_swing_single` - grade C (swing_single, tubular_latch latch, none lock, none closer)

* `full` grade C: structure agree, settle agree, hold na, operate_open **disagree**, release na, relatch na, sanity agree
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 1.556
* `rl` grade C: structure agree, settle agree, hold na, operate_open **disagree**, release na, relatch na, sanity agree
  * operate_open: operator moved (travel n/a) but bolt retracted n/a of its throw; MuJoCo opened 1.556

### `db0029_sliding_single` - grade C (sliding_single, none latch, none lock, none closer)

* `full` grade C: structure agree, settle agree, hold **disagree**, sanity agree
  * hold: free push: mujoco opened, physx stuck (hold_displacement 1.21 vs 0.007471)
* `rl` grade C: structure agree, settle agree, hold **disagree**, sanity agree
  * hold: free push: mujoco opened, physx stuck (hold_displacement 1.21 vs 0.005962)

### `db0032_sliding_single` - grade C (sliding_single, hook latch, hook_lock lock engaged, none closer)

* `full` grade C: structure agree, settle **disagree**, hold na, locked_holds agree, sanity agree
  * settle: drift mujoco=4.931e-14 physx=0.02694
* `rl` grade C: structure agree, settle **disagree**, hold na, locked_holds agree, sanity agree
  * settle: drift mujoco=4.931e-14 physx=0.04431

### `db0033_gate_sliding` - grade C (gate_sliding, slide_bolt latch, slide_bolt lock engaged, none closer)

* `full` grade C: structure agree, settle agree, hold na, operate_open **disagree**, sanity agree
  * operate_open: latch released (bolt n/a) but the leaf did not open in PhysX (0.002 vs 3.6)
* `rl` grade C: structure agree, settle agree, hold na, operate_open **disagree**, sanity agree
  * operate_open: latch released (bolt n/a) but the leaf did not open in PhysX (0.001998 vs 3.6)

### `db0005_garage_tiltup` - grade B (garage_tiltup, none latch, none lock, none closer)

* `full` grade B: structure agree, settle agree, hold quant, sanity agree
  * hold: hold_displacement: mujoco 0.8581 vs physx 1.536 (tol 0.1)
* `rl` grade B: structure agree, settle agree, hold quant, sanity agree
  * hold: hold_displacement: mujoco 0.8581 vs physx 1.536 (tol 0.1)

### `db0006_gate_swing` - grade B (gate_swing, slide_bolt latch, padlock lock, none closer)

* `full` grade B: structure agree, settle agree, hold quant, sanity agree
  * hold: hold_displacement: mujoco 0.1748 vs physx 0.4132 (tol 0.1)
* `rl` grade B: structure agree, settle agree, hold quant, sanity agree
  * hold: hold_displacement: mujoco 0.1748 vs physx 0.4129 (tol 0.1)

### `db0014_gate_swing` - grade B (gate_swing, hook latch, none lock, gate closer)

* `full` grade B: structure agree, settle agree, hold na, operate_open quant, closer_return na, sanity agree
  * operate_open: opened: mujoco 0.8732 vs physx 1.571 (tol 0.1)
* `rl` grade B: structure agree, settle agree, hold na, operate_open quant, closer_return na, sanity agree
  * operate_open: opened: mujoco 0.8732 vs physx 1.571 (tol 0.1)

### `db0015_swing_double` - grade B (swing_double, none latch, none lock, concealed_overhead closer)

* `full` grade B: structure agree, settle agree, hold quant, closer_return na, sanity agree
  * hold: hold_displacement: mujoco 1.571 vs physx 1.172 (tol 0.1)
* `rl` grade B: structure agree, settle agree, hold quant, closer_return na, sanity agree
  * hold: hold_displacement: mujoco 1.571 vs physx 1.026 (tol 0.1)

### `db0038_sliding_single` - grade B (sliding_single, none latch, slide_bolt lock, none closer)

* `full` grade B: structure agree, settle agree, hold quant, sanity agree
  * hold: hold_displacement: mujoco 0.3784 vs physx 1.028 (tol 0.05)
* `rl` grade B: structure agree, settle agree, hold quant, sanity agree
  * hold: hold_displacement: mujoco 0.3784 vs physx 0.8678 (tol 0.05)

## Known not-comparable categories

* **Env-release locks** (mag lock, delayed egress, card reader, electric strike, interlock): MuJoCo holds them with a `<weld>` that has no PhysX counterpart; the runner emulates the hold or marks the phase `na_env_logic`. A door that *opens* here in PhysX is class `EXPORT_WELD`.
* **Panic doors with the robot outside and no far-side trim**: `operator_joint` is None, the exit device is welded in `door_rl.usda`; both simulators must *hold*.
* **Welded releases in `door_rl.usda`** (thumbturns, aux bolts, extra dogs): the RL expectation for `operate_open` flips to 'stays closed'; a `full` / `rl` disagreement there is `RL_CANON`.
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

## The protocol

Phases follow `doorbench.qa.run_qa` (the dataset sign-off) expressed in simulated time, so a 500 Hz MuJoCo run and a
120 Hz PhysX run apply the same schedule.  All joint values are DoorBench coordinates (MuJoCo `q`; USD `q` +
`doorbench:zero_offset`).  Curves are recorded at 30 Hz; metrics and pass/fail come from one shared code path.

| phase | drive | duration | expectation (from `qa.door_flags` + the RL slot metadata) |
|---|---|---|---|
| settle | none | 1 s | primary drift < 0.05 rad / 0.01 m, no MuJoCo warnings, initial penetration > -12 mm |
| hold | adaptive push on the primary joint: `min(2(bias + friction + preload) + 60 N·m \| 80 N, 800 \| 4000)` | 1 s (holding) / ≤ 6 s (free) | `hold` (< 2°/15 mm) for latched / locked doors (locked rotors and bolted flaps: their locked play + 1°), `free_opens` (> 10°/5 cm) otherwise - free-swing families included, a leaf nothing holds must move; qa.py additionally grades them with the jam gate (`no_jam`: no static geometry may press on a moving part with > 200 N during the push - the check that caught the revolving wings jammed on the wall header) |
| operate | thumbturn 2 N·m (t < 1.2 s), aux bolts 3 N·m / 60 N, dogs 14 N·m, operator 4 / 8 / 10 / 14 N·m or 120 N from 0.6 s, push from 1.2 s while q < 50° | 6.4 s | `opens` (> min(20°, ½ max_open) / 5 cm; chain guards inside the slack window); RL: `stays_closed` when the release parts are welded engaged |
| release | none; primary joint pinned | 0.8 s | `bolt_returns` (< 6 mm) |
| relatch | −min(½ push, 1.5·static + 40) for 6 s, then +push 1 s | 7 s | `relatches` (closed < 2°, re-push < 2.5°) |
| closer | none, from min(60°, 0.8 max_open) | 12 s | `closes` (< 6°) |
| locked | operator 6 N·m / 150 N + push | 2 s | `locked_holds` |

Per-door inputs (`results/parity/mujoco.json` → `doors.<id>.inputs`) carry the forces measured in MuJoCo at
`qpos0` (gravity bias, Coulomb friction, spring preload), the thresholds, the couplings (one-sided latch tendon,
mimic equalities, welds, loop closures, MJCF servos) and the expected outcome per phase for `mjcf`, `usd_full`,
`usd_rl`.  The Isaac runner never recomputes a force; it only maps MJCF joint names to the joints of the file it runs.

### What the PhysX runner emulates (and records in `emulations_used`)

* **spring targets restored every step** — Isaac Lab zero-initialises position targets, which erased every USD spring
  preload in the first probe (closer doors opened under 60 N·m, levers sagged under gravity); the runner writes
  `doorbench:target_si` each step and applies efforts only through `set_joint_effort_target`
* **latch clamp (+ target)** — the one-sided MJCF tendon `bolt ≥ scale · operator` has no PhysX counterpart; the latch
  joint state is clamped to the tendon minimum every step (`write_joint_state_to_sim`) and, by default
  (`--latch-mode clamp+target`), the latch drive target follows that minimum while the tendon pulls — otherwise the
  300 N/m latch spring re-extends a 0.04 kg bolt by ~2.5 mm within one 1/120 s step and the recorded retraction
  chatters below the tendon minimum (the strike gap is 3 mm); `--latch-mode clamp` keeps the pure clamp
* **batch layout** — the doors of a batch sit on a centred 20 × 14 m grid (`--spacing`) on a ground plane sized to
  the grid: gate leaves sweep / slide up to 8.2 m from their origin and fences / floor-hatch decks extend up to
  9.9 m, so the 6 m grid of the first probe let neighbouring doors collide; batches group doors with the same phase
  schedule (`--no-group` to keep the `--doors` order) so a batch does not step 12 s of `closer` for one door
* **servo emulated** — MJCF position actuators of automatic doors (`ctrl = 0`) as clipped feed-forward effort
* **weld pinned hold** (opt-in `--emulate-weld`) — mag locks / delayed egress are MuJoCo `<weld>` equalities not
  exported to USD; by default the door is left free and the verdict reports `EXPORT_WELD_MISSING`

## Verdict and discrepancy codes

`compare_door` first compares the pass/fail status of every applicable phase, then the metrics of agreeing phases
against tolerances (hold 0.01 rad / 3 mm; opened within 20 % or 0.1 rad / 5 cm, `t_open` within 30 % or 0.3 s,
operator travel 10 %, bolt retraction 15 % of throw; release 2 mm / 0.2 s; relatch 1°; closer 2° / 30 % of closing
time; per-joint settle drift 0.02 rad / 5 mm).

| code | meaning |
|---|---|
| `OK` | every applicable phase agrees within tolerance |
| `PHYSX_NO_OPEN` | MuJoCo opens (free push or operator + push), PhysX does not |
| `PHYSX_HOLD_FAIL` | MuJoCo holds (latch / lock / locked handle), PhysX opens |
| `EXPORT_WELD_MISSING` | the hold relied on a MuJoCo weld (env-released lock) that is not in the USD |
| `LATCH_NO_RETURN`, `RELATCH_FAIL`, `CLOSER_NO_RETURN` | phase-specific PhysX failures with a MuJoCo pass |
| `SETTLE_DRIFT` | a joint moves during the free settle in one simulator only (e.g. lost spring preload) |
| `LIMIT_VIOLATION` | PhysX leaves an authored joint range that MuJoCo respects (MuJoCo's soft limits overshoot by a few degrees under hard pushes; only a PhysX overshoot > 2× MuJoCo's counts) |
| `NAN`, `LOAD_FAIL`, `STRUCTURE_FAIL` | non-finite state; spawn / inspection error; joint set, limits, gains or spring targets differ from model.json |
| `METRIC_DELTA` | statuses agree but a metric is outside tolerance (quantitative) |
| `RL_CANON` | door_rl.usda behaves differently by construction (welded lock parts, empty operator slot); informational |
| `MUJOCO_FAIL` | the reference itself fails a phase qa.json passed (protocol bug or nondeterminism, not a PhysX bug) |
| `INFO_DISAGREE` | disagreement on an informational phase (roller / magnetic catches; the free-swing push stopped being informational when it turned out to be the only thing that pushed 12 locked accordion folds and 10 header-jammed revolving doors) |

Grades per door and kind: **A** all phases agree within tolerance, **B** statuses agree but a metric is off
(`METRIC_DELTA`, `SETTLE_DRIFT`, `INFO_DISAGREE`), **C** a status disagreement or a limits / NaN failure,
**X** not comparable.  A door's grade is the worst of `full` and `rl`.

## Status

* MuJoCo reference: 1000/1000 doors pass every applicable phase and reproduce their qa.json metrics (`qa_push`,
  `hold_displacement`, `actuate_displacement`, `closer_final_angle`, ...) to 1e-3.  Verified independently on a
  seeded 61-door sample (2 per family): bit-identical records across worker counts, resumes and machines.
* Informational phases that fail in MuJoCo (`mujoco_summary.json` → `informational_fails`; families qa.py never
  pushed, so they are reported, not graded) — these are **door bugs the reference surfaced, not protocol bugs**:
  * accordion, 12/12: the panel couplings alternate `panel_i = ∓2·panel_0` but every panel hinge is authored with range
    `[-π, 0]`, so the even panels sit on their limit and the whole fold is kinematically locked (65 N·m moves the lead
    hinge 0.0009 rad; `qfrc_constraint` absorbs the full push, contacts carry no force)
  * revolving, 8/15: a wing stile touches `wall_header` at q0 (gap 0) and jams against it as the rotor turns
    (8.6 kN contact normal force; 3 doors do not move at all, 5 crawl < 0.12 rad in 6 s; the other 7 turn normally)
  * bifold, 3/30: the panel tops rub on `wall_header` (20–40 N normal force, zero gap) and the fold crawls ~0.1 rad in 6 s
  * cold-storage roller relatches (5): correct — a roller latch does not hold a re-push
* Behaviour that is *by construction* in both simulators and worth knowing when reading the metrics: closer doors run
  with the symmetric MJCF damping (`damping_opening` + air), because the asymmetric `damping_closing` / backcheck live
  only in `DoorEnv`'s passive callback and the USD carries them only as `doorbench:damping_closing` attributes — so
  `closer_t_close` is 0.6–1.8 s (median 1.07 s over 263 doors) against `closing_time_est_s` of 2–5 s and `slam` never
  fires; turnstile rotors have no indexing detent (`ratchet_deg` is spec-only), so a 68 N·m push spins a full-height
  rotor 1.5 rev/s; the qa.py push counts a closer's spring preload twice (`bias` = |qfrc_bias − qfrc_passive| already
  contains it), e.g. 311 N·m on db0012 — kept, because the gate must reproduce qa.json; doors with `rest_angle_deg`
  (10 stall doors) pass `hold` / `operate` trivially since they start open.
* PhysX side: written against Isaac Lab 2.3.2, **not executed on this machine** (no NVIDIA GPU).  First run:
  `bash isaaclab/cloud/parity.sh --limit 40`, then `scripts/parity_compare.py`.  Expect roughly 5-7 min per batch of
  20 doors (up to ~4200 physics steps per batch with a per-door Python loop) — about 8-10 h for 1000 doors × 2 kinds;
  `--retry-errors` re-runs doors whose record is a spawn / batch error, everything else is resumable.
* Known limits of the PhysX emulation (to verify on the first GPU run): joint Coulomb friction is authored twice in
  the USD (`physxJointAxis:*:staticFrictionEffort` and the legacy `physxJoint:jointFriction` coefficient; Isaac Lab
  exposes only the latter as `joint_friction_coeff`, recorded in `structure.friction_coeff_readback`); mimic-joint
  gearing units for revolute→prismatic couplings; `PhysxJointAxisAPI` friction efforts being honoured at all.
* Planned: write `isaac_parity` into each door's qa.json, a viewer badge, and fixing the export gaps the gate finds
  (latch tendon and welds as native PhysX constraints where possible).
