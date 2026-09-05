# Isaac parity gate

_Generated 2026-09-05T03:04:35 by `scripts/isaaclab/parity_report.py` from `results/parity/` (commit `0d849b618`). Reference: MuJoCo mujoco 3.12.0; PhysX: isaac_sim None, isaac_lab None, physx_dt 0.0083333333333333...._

Every door runs **one behavioural protocol** in MuJoCo (the reference physics, CPU) and in Isaac Sim / PhysX on the GPU pod, on both USD kinds (`door.usda` full fidelity, `door_rl.usda` canonical 8-link). The two runs are compared phase by phase: both simulators must reach the same pass / fail verdict (else grade **C**), and when they agree the metrics must be within tolerance (else grade **B**); **A** is parity, **X** means the door could not be compared (spawn / structure error). A disagreement is tagged with a discrepancy class whose likely root cause comes from the analysis of the first 40-door probe. The per-door verdict is published in `qa.json` (`isaac_parity`) and as a badge in the viewer.

## Headline

| USD kind | tested | parity (A) | same verdicts (A + B) | disagree (C) | not comparable (X) | untested |
|---|---|---|---|---|---|---|
| `full` | 1000 / 1000 | **27 / 1000** (3 % of tested) | 904 / 1000 (90 %) | 96 | 0 | 0 |
| `rl` | 1000 / 1000 | **32 / 1000** (3 % of tested) | 954 / 1000 (95 %) | 46 | 0 | 0 |

Door badge (`qa.json.isaac_parity.ok`; viewer chip *Isaac parity*): **890 ok** (grade A or B in every tested kind), **110 fail** (a status disagreement or not comparable), 0 untested.

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
| `QUANT` | 2206 | 2283 | 934 | both simulators reach the same pass / fail verdicts but at least one metric is outside tolerance | soft MuJoCo contacts / limits (solref 0.005) vs rigid PhysX limits, constraint-based vs effort-based Coulomb friction, implicitfast vs implicit drives at high damping; solver dt | rerun at Isaac dt 1/240 (32/8 iterations) and MuJoCo dt 0.001; if the delta shrinks below tolerance tag SOLVER_SENSITIVITY, else triage by phase | `db0001_rollup`, `db0002_swing_single`, `db0003_cold_storage`, `db0004_bifold` |
| `CONTACT_GEOMETRY` | 57 | 11 | 57 | the bolt retracted (or there is no bolt) yet the leaf did not move, or a latch that holds in MuJoCo does not engage in PhysX (convex hulls, strike lip, panel clearance) | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes, self-collision disabled globally vs MuJoCo's selective exclusion | enable contact reporting; rerun with Env collision disabled, then without the hardware part, to bisect frame contact vs articulation; author PhysxFilteredPairsAPI for model.contact_excludes | `db0014_gate_swing`, `db0039_swing_single`, `db0051_swing_single`, `db0062_swing_single` |
| `EXPORT_WELD` | 21 | 17 | 16 | MuJoCo holds the leaf (weld / lock equality) but PhysX has nothing holding it, so the door opens under the push | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active with its full range | author a FixedJoint / locked drive (or a D6 joint with breakForce = holding_force_N) tagged doorbench:env_release that DoorMechanismAction disables on REX / badge / timer; classify env_release_only and test 'holds' in both sims | `db0026_swing_single`, `db0158_swing_double`, `db0187_turnstile_fullheight`, `db0216_swing_single` |
| `EXPORT_COUPLING` | 14 | 20 | 27 | the operator turns but the bolt does not retract (or does not return) in PhysX, so the door stays latched | H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate it as a kinematic clamp each step (the soft target offset under-retracts by 40-60 %) | shared clamp function (write_joint_state_to_sim(max(latch_q, scale * op_q))) in the parity runner and DoorMechanismAction; read scale from doorbench:rl.latch_coupling / doorbench:latch_coupling_scale | `db0124_vault`, `db0179_vault`, `db0288_blast`, `db0296_sliding_single` |
| `RL_CANON` | 0 | 15 | 15 | door.usda agrees with MuJoCo but door_rl.usda does not: a welded lock / operator / panel or an empty operator slot changes the behaviour | H4: panic doors with robot outside and no far-side trim get operator_joint None (exit device welded, latch never retracts); engaged locks welded; world-mounted latches welded released; extra leaves omitted | derive the RL expectation from doorbench:rl (lock.engaged, operator_slot_joint) and document 'holds by construction'; or map the release to a canonical slot | `db0296_sliding_single`, `db0331_sliding_single`, `db0345_sliding_single`, `db0373_sliding_single` |
| `PHYSICS_PARAM_FRICTION` | 7 | 4 | 9 | a free-swinging door opens in one simulator but not the other (timing or threshold), pointing at Coulomb friction or gravity bias mapping | H3: physxJointAxis friction efforts vs MuJoCo frictionloss (stick / slip differs), legacy jointFriction coefficient adding load-dependent friction, or an effort below the adaptive QA push | measure breakaway effort on one door in both sims; use the per-door qa_push; zero the legacy coefficient | `db0052_pet_door`, `db0273_turnstile_fullheight`, `db0338_turnstile_fullheight`, `db0528_turnstile_fullheight` |
| `PHYSICS_PARAM_PRELOAD` | 5 | 0 | 5 | settle drift or a false opening that matches a spring whose target was zeroed (operator sag q = tau_g / k, closer preload gone) | H2 / D1: Isaac Lab forwards zero position targets to the PhysX drives; the USD targetPosition (springref) is lost while the stiffness stays | restore doorbench:target_si / rl joints[*].target each step (as DoorMechanismAction does); report drift per joint | `db0187_turnstile_fullheight`, `db0497_turnstile_fullheight`, `db0528_turnstile_fullheight`, `db0880_turnstile_fullheight` |

## By family

| family | doors | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|---|
| accordion | 12 | 12 | 11 | 1 | 0 / 11 | 0 / 12 | PHYSICS_PARAM_FRICTION x1 |
| automatic_sliding | 15 | 15 | 15 | 0 | 0 / 15 | 0 / 15 | - |
| automatic_swing | 10 | 10 | 10 | 0 | 0 / 10 | 0 / 10 | - |
| baby_gate | 10 | 10 | 0 | 10 | 0 / 0 | 8 / 10 | CONTACT_GEOMETRY x10 |
| bifold | 30 | 30 | 30 | 0 | 0 / 30 | 0 / 30 | - |
| blast | 6 | 6 | 0 | 6 | 0 / 0 | 0 / 5 | EXPORT_COUPLING x6 |
| cold_storage | 15 | 15 | 15 | 0 | 0 / 15 | 0 / 15 | - |
| dutch | 12 | 12 | 12 | 0 | 0 / 12 | 0 / 12 | - |
| elevator | 8 | 8 | 8 | 0 | 0 / 8 | 0 / 8 | - |
| garage_sectional | 18 | 18 | 18 | 0 | 0 / 18 | 0 / 18 | - |
| garage_tiltup | 7 | 7 | 7 | 0 | 1 / 7 | 1 / 7 | - |
| gate_sliding | 10 | 10 | 10 | 0 | 1 / 10 | 0 / 10 | - |
| gate_swing | 40 | 40 | 34 | 6 | 2 / 34 | 2 / 40 | CONTACT_GEOMETRY x6 |
| hatch_ceiling | 8 | 8 | 8 | 0 | 0 / 8 | 0 / 8 | - |
| hatch_floor | 10 | 10 | 10 | 0 | 0 / 10 | 0 / 10 | - |
| pet_door | 15 | 15 | 12 | 3 | 0 / 12 | 0 / 15 | PHYSICS_PARAM_FRICTION x3 |
| pivot | 20 | 20 | 19 | 1 | 0 / 19 | 0 / 19 | EXPORT_WELD x1 |
| revolving | 15 | 15 | 15 | 0 | 0 / 15 | 0 / 15 | - |
| rollup | 15 | 15 | 15 | 0 | 0 / 15 | 0 / 15 | - |
| saloon | 12 | 12 | 12 | 0 | 0 / 12 | 0 / 12 | - |
| ship_watertight | 10 | 10 | 10 | 0 | 0 / 10 | 0 / 10 | - |
| sliding_bypass | 35 | 35 | 35 | 0 | 0 / 35 | 0 / 35 | - |
| sliding_single | 100 | 100 | 86 | 14 | 10 / 97 | 7 / 89 | EXPORT_COUPLING x11, RL_CANON x11, EXPORT_WELD x2 |
| stall | 15 | 15 | 15 | 0 | 0 / 15 | 0 / 15 | - |
| strip_curtain | 8 | 8 | 8 | 0 | 0 / 8 | 0 / 8 | - |
| swing_double | 76 | 76 | 55 | 21 | 0 / 57 | 0 / 55 | CONTACT_GEOMETRY x11, EXPORT_WELD x8, RL_CANON x3 |
| swing_single | 440 | 440 | 407 | 33 | 6 / 408 | 7 / 436 | CONTACT_GEOMETRY x29, EXPORT_WELD x3, EXPORT_COUPLING x1 |
| turnstile_fullheight | 10 | 10 | 3 | 7 | 0 / 3 | 0 / 8 | PHYSICS_PARAM_PRELOAD x5, PHYSICS_PARAM_FRICTION x3, EXPORT_WELD x2 |
| turnstile_tripod | 10 | 10 | 10 | 0 | 7 / 10 | 7 / 10 | - |
| vault | 8 | 8 | 0 | 8 | 0 / 0 | 0 / 2 | EXPORT_COUPLING x8 |

## By hardware

### latch kind

| kind | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|
| deadlatch | 88 | 84 | 4 | 1 / 84 | 1 / 88 | CONTACT_GEOMETRY x4 |
| dogs | 17 | 10 | 7 | 0 / 10 | 0 / 17 | EXPORT_COUPLING x7 |
| electric_bolt | 11 | 8 | 3 | 0 / 8 | 2 / 11 | EXPORT_WELD x2, CONTACT_GEOMETRY x1 |
| gravity_bar | 30 | 30 | 0 | 0 / 30 | 4 / 30 | - |
| hook | 31 | 4 | 27 | 7 / 15 | 8 / 20 | CONTACT_GEOMETRY x16, EXPORT_COUPLING x11, RL_CANON x11 |
| magnetic | 41 | 38 | 3 | 0 / 38 | 0 / 41 | PHYSICS_PARAM_FRICTION x3 |
| mortise_latch | 74 | 68 | 6 | 1 / 68 | 1 / 74 | CONTACT_GEOMETRY x6 |
| multi_bolt | 7 | 0 | 7 | 0 / 0 | 0 / 0 | EXPORT_COUPLING x7 |
| none | 377 | 361 | 16 | 15 / 363 | 14 / 367 | EXPORT_WELD x8, PHYSICS_PARAM_FRICTION x6, PHYSICS_PARAM_PRELOAD x5 |
| rim_latch | 42 | 41 | 1 | 0 / 42 | 0 / 41 | EXPORT_COUPLING x1, RL_CANON x1 |
| roller | 8 | 8 | 0 | 0 / 8 | 0 / 8 | - |
| slide_bolt | 30 | 30 | 0 | 3 / 30 | 2 / 30 | - |
| tubular_latch | 213 | 177 | 36 | 0 / 177 | 0 / 196 | CONTACT_GEOMETRY x30, EXPORT_WELD x6, EXPORT_COUPLING x1 |
| vertical_rods | 31 | 31 | 0 | 0 / 31 | 0 / 31 | - |

### lock kind

| kind | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|
| card_reader | 20 | 20 | 0 | 1 / 20 | 1 / 20 | - |
| chain | 4 | 4 | 0 | 0 / 4 | 0 / 4 | - |
| child_lock_cover | 8 | 8 | 0 | 0 / 8 | 0 / 8 | - |
| deadbolt_double | 6 | 6 | 0 | 1 / 6 | 1 / 6 | - |
| deadbolt_single | 32 | 22 | 10 | 2 / 22 | 3 / 26 | CONTACT_GEOMETRY x6, EXPORT_WELD x4, EXPORT_COUPLING x1 |
| delayed_egress | 16 | 16 | 0 | 0 / 16 | 0 / 16 | - |
| dogs | 17 | 10 | 7 | 0 / 10 | 0 / 17 | EXPORT_COUPLING x7 |
| electric_strike | 23 | 23 | 0 | 1 / 23 | 1 / 23 | - |
| hook_lock | 28 | 24 | 4 | 5 / 28 | 1 / 24 | EXPORT_COUPLING x4, RL_CANON x4 |
| interlock | 8 | 8 | 0 | 0 / 8 | 0 / 8 | - |
| jam_stuck | 12 | 12 | 0 | 0 / 12 | 0 / 12 | - |
| keyed_cylinder | 26 | 24 | 2 | 0 / 24 | 1 / 26 | EXPORT_WELD x2 |
| keypad_code | 28 | 18 | 10 | 0 / 18 | 0 / 28 | CONTACT_GEOMETRY x10 |
| mag_lock | 47 | 38 | 9 | 7 / 38 | 7 / 41 | EXPORT_WELD x8, PHYSICS_PARAM_PRELOAD x3 |
| multipoint | 7 | 1 | 6 | 0 / 2 | 0 / 5 | CONTACT_GEOMETRY x4, EXPORT_WELD x1, PHYSICS_PARAM_FRICTION x1 |
| night_latch | 4 | 0 | 4 | 0 / 0 | 0 / 4 | CONTACT_GEOMETRY x4 |
| none | 544 | 507 | 37 | 3 / 513 | 12 / 529 | CONTACT_GEOMETRY x24, PHYSICS_PARAM_FRICTION x7, RL_CANON x6 |
| padlock | 40 | 40 | 0 | 2 / 40 | 2 / 40 | - |
| privacy_button | 43 | 43 | 0 | 0 / 43 | 0 / 43 | - |
| slide_bolt | 54 | 50 | 4 | 5 / 53 | 3 / 51 | EXPORT_COUPLING x3, RL_CANON x3, PHYSICS_PARAM_FRICTION x1 |
| swing_bar_guard | 2 | 2 | 0 | 0 / 2 | 0 / 2 | - |
| thumbturn_only | 24 | 14 | 10 | 0 / 14 | 0 / 21 | CONTACT_GEOMETRY x9, EXPORT_WELD x1 |
| vault_wheel | 7 | 0 | 7 | 0 / 0 | 0 / 0 | EXPORT_COUPLING x7 |

### closer kind

| kind | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|
| auto_operator_full | 4 | 4 | 0 | 0 / 4 | 0 / 4 | - |
| auto_operator_low_energy | 11 | 10 | 1 | 0 / 10 | 0 / 10 | EXPORT_WELD x1 |
| concealed_overhead | 21 | 19 | 2 | 0 / 19 | 0 / 19 | EXPORT_WELD x2 |
| electromagnetic_hold | 13 | 13 | 0 | 0 / 13 | 0 / 13 | - |
| floor_spring | 28 | 26 | 2 | 0 / 26 | 0 / 27 | CONTACT_GEOMETRY x1, EXPORT_WELD x1 |
| gas_strut | 8 | 8 | 0 | 0 / 8 | 0 / 8 | - |
| gate | 22 | 11 | 11 | 0 / 11 | 3 / 22 | CONTACT_GEOMETRY x11 |
| none | 667 | 579 | 88 | 26 / 592 | 28 / 627 | CONTACT_GEOMETRY x41, EXPORT_COUPLING x26, RL_CANON x14 |
| pneumatic | 6 | 6 | 0 | 0 / 6 | 0 / 6 | - |
| spring_hinge | 37 | 34 | 3 | 0 / 34 | 0 / 37 | CONTACT_GEOMETRY x3 |
| surface_overhead | 183 | 180 | 3 | 1 / 181 | 1 / 181 | CONTACT_GEOMETRY x1, EXPORT_WELD x1, EXPORT_COUPLING x1 |

### operator kind

| kind | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|
| card_lever | 20 | 20 | 0 | 1 / 20 | 1 / 20 | - |
| cremone | 3 | 1 | 2 | 0 / 3 | 0 / 1 | PHYSICS_PARAM_FRICTION x2, RL_CANON x2 |
| flush_pull | 72 | 72 | 0 | 1 / 72 | 0 / 72 | - |
| gate_latch_fork | 12 | 12 | 0 | 0 / 12 | 0 / 12 | - |
| handleset | 13 | 8 | 5 | 0 / 8 | 0 / 10 | CONTACT_GEOMETRY x4, EXPORT_WELD x1 |
| hasp | 9 | 9 | 0 | 0 / 9 | 0 / 9 | - |
| hook_lock_slider | 15 | 4 | 11 | 7 / 15 | 0 / 4 | EXPORT_COUPLING x11, RL_CANON x11 |
| keypad_deadbolt | 9 | 1 | 8 | 0 / 1 | 0 / 9 | CONTACT_GEOMETRY x8 |
| keypad_lever | 19 | 17 | 2 | 0 / 17 | 0 / 19 | CONTACT_GEOMETRY x2 |
| knob | 135 | 127 | 8 | 0 / 127 | 0 / 132 | CONTACT_GEOMETRY x8 |
| lever | 217 | 188 | 29 | 2 / 188 | 2 / 206 | CONTACT_GEOMETRY x17, EXPORT_COUPLING x8, EXPORT_WELD x5 |
| lift_latch | 16 | 0 | 16 | 0 / 0 | 8 / 16 | CONTACT_GEOMETRY x16 |
| none | 102 | 91 | 11 | 8 / 91 | 9 / 100 | PHYSICS_PARAM_FRICTION x6, PHYSICS_PARAM_PRELOAD x5, EXPORT_WELD x2 |
| paddle | 11 | 9 | 2 | 0 / 9 | 0 / 10 | CONTACT_GEOMETRY x1, EXPORT_WELD x1 |
| panic_crossbar | 6 | 6 | 0 | 0 / 6 | 0 / 6 | - |
| panic_touchbar | 73 | 72 | 1 | 0 / 73 | 0 / 72 | EXPORT_COUPLING x1, RL_CANON x1 |
| pull | 156 | 149 | 7 | 5 / 149 | 10 / 152 | EXPORT_WELD x6, PHYSICS_PARAM_FRICTION x1 |
| push_button_screen | 7 | 7 | 0 | 0 / 7 | 0 / 7 | - |
| push_plate | 24 | 23 | 1 | 0 / 23 | 0 / 23 | EXPORT_WELD x1 |
| ring_pull | 29 | 29 | 0 | 0 / 29 | 0 / 29 | - |
| slide_bolt_handle | 19 | 19 | 0 | 3 / 19 | 2 / 19 | - |
| t_handle | 10 | 10 | 0 | 0 / 10 | 0 / 10 | - |
| thumb_latch | 12 | 12 | 0 | 0 / 12 | 0 / 12 | - |
| wheel | 11 | 4 | 7 | 0 / 4 | 0 / 4 | EXPORT_COUPLING x7 |

## By kinematics

| kinematics | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |
|---|---|---|---|---|---|---|
| hinge_horizontal | 48 | 45 | 3 | 1 / 45 | 1 / 48 | PHYSICS_PARAM_FRICTION x3 |
| hinge_vertical | 716 | 630 | 86 | 8 / 633 | 17 / 683 | CONTACT_GEOMETRY x56, EXPORT_COUPLING x16, EXPORT_WELD x12 |
| rotor | 35 | 28 | 7 | 7 / 28 | 7 / 33 | PHYSICS_PARAM_PRELOAD x5, PHYSICS_PARAM_FRICTION x3, EXPORT_WELD x2 |
| slide_horizontal | 168 | 154 | 14 | 11 / 165 | 7 / 157 | EXPORT_COUPLING x11, RL_CANON x11, EXPORT_WELD x2 |
| slide_vertical | 33 | 33 | 0 | 0 / 33 | 0 / 33 | - |

## Top offenders (20)

| door | family | grade full / rl | phase | MuJoCo | PhysX full | PhysX rl | classes | likely root cause |
|---|---|---|---|---|---|---|---|---|
| `db0334_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.002866 | 1.658 (disagree) | 1.658 (disagree) | `EXPORT_WELD` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0413_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.002976 | 1.92 (disagree) | 1.92 (disagree) | `EXPORT_WELD` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0534_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003278 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0702_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003719 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0733_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003324 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0792_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003315 | 1.658 (disagree) | 1.658 (disagree) | `EXPORT_WELD`, `EXPORT_COUPLING`, `RL_CANON` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0026_swing_single` | swing_single | C / C | `hold` | hold_displacement=1.194e-06 | 1.138 (disagree) | 1.137 (disagree) | `EXPORT_WELD` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0149_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.004243 | 1.92 (disagree) | 1.92 (disagree) | `CONTACT_GEOMETRY` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes, self-collision disabled globally vs MuJoCo... |
| `db0158_swing_double` | swing_double | C / C | `hold` | hold_displacement=1.727e-06 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0179_vault` | vault | C / C | `operate_open` | opened=1.725 | 0.001912 (disagree) | 0.001912 (disagree) | `EXPORT_COUPLING` | H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate i... |
| `db0216_swing_single` | swing_single | C / C | `hold` | hold_displacement=4.645e-06 | 1.78 (disagree) | 1.78 (disagree) | `EXPORT_WELD` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0222_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003401 | 1.571 (disagree) | 1.571 (disagree) | `CONTACT_GEOMETRY` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes, self-collision disabled globally vs MuJoCo... |
| `db0273_turnstile_fullheight` | turnstile_fullheight | C / C | `hold` | hold_displacement=9.491 | -3.569e-07 (disagree) | -0.0007435 (disagree) | `PHYSICS_PARAM_FRICTION` | H3: physxJointAxis friction efforts vs MuJoCo frictionloss (stick / slip differs), legacy jointFriction coefficient adding load-dependent friction, or an effor... |
| `db0316_swing_double` | swing_double | C / C | `hold` | hold_displacement=2.281e-06 | 1.571 (disagree) | 1.571 (disagree) | `EXPORT_WELD` | H5: weld-type lock equalities (mag_lock / delayed_egress leaf -> world) are exported only as doorbench:couplings JSON; write_usd_rl keeps door_hinge active wit... |
| `db0338_turnstile_fullheight` | turnstile_fullheight | C / C | `hold` | hold_displacement=9.685 | 3.594e-07 (disagree) | -0.0005956 (disagree) | `PHYSICS_PARAM_FRICTION` | H3: physxJointAxis friction efforts vs MuJoCo frictionloss (stick / slip differs), legacy jointFriction coefficient adding load-dependent friction, or an effor... |
| `db0352_blast` | blast | C / C | `operate_open` | opened=1.7 | 0.002101 (disagree) | 0.002101 (disagree) | `EXPORT_COUPLING` | H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate i... |
| `db0426_vault` | vault | C / C | `operate_open` | opened=1.7 | 0.002132 (disagree) | 0.002131 (disagree) | `EXPORT_COUPLING` | H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate i... |
| `db0454_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.003065 | 1.658 (disagree) | 1.658 (disagree) | `CONTACT_GEOMETRY` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes, self-collision disabled globally vs MuJoCo... |
| `db0458_vault` | vault | C / C | `operate_open` | opened=1.739 | 0.001894 (disagree) | 0.001894 (disagree) | `EXPORT_COUPLING` | H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate i... |
| `db0577_swing_double` | swing_double | C / C | `hold` | hold_displacement=0.00314 | 1.571 (disagree) | 1.571 (disagree) | `CONTACT_GEOMETRY` | H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes, self-collision disabled globally vs MuJoCo... |

### `db0334_swing_double` - grade C (swing_double, tubular_latch latch, deadbolt_single lock engaged, none closer)

* `full` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.658 in PhysX vs 0.002867
* `rl` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.658 in PhysX vs 0.002867

### `db0413_swing_double` - grade C (swing_double, tubular_latch latch, deadbolt_single lock engaged, none closer)

* `full` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.92 in PhysX vs 0.002975
* `rl` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.92 in PhysX vs 0.002975

### `db0534_swing_double` - grade C (swing_double, tubular_latch latch, thumbturn_only lock engaged, none closer)

* `full` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged thumbturn_only holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003277
* `rl` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged thumbturn_only holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003277

### `db0702_swing_double` - grade C (swing_double, tubular_latch latch, multipoint lock engaged, none closer)

* `full` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged multipoint holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003719
* `rl` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged multipoint holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003719

### `db0733_swing_double` - grade C (swing_double, tubular_latch latch, deadbolt_single lock engaged, none closer)

* `full` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003324
* `rl` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds **disagree**
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * locked_holds: locked door moved 1.571 in PhysX vs 0.003324

### `db0792_swing_double` - grade C (swing_double, tubular_latch latch, deadbolt_single lock engaged, none closer)

* `full` grade C: settle quant, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
* `rl` grade C: settle quant, hold **disagree**, operate_open **disagree**, release na, relatch na, closer_return na, locked_holds na
  * hold: engaged deadbolt_single holds in MuJoCo, not in PhysX (lock constraint not exported)
  * operate_open: operator moved (travel 0.96) but bolt retracted n/a of its throw; MuJoCo opened 1.454
  * operate_open: operate_open agrees in door.usda but not in door_rl.usda

### `db0026_swing_single` - grade C (swing_single, none latch, mag_lock lock engaged, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (1.194e-06), PhysX opened 1.138
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (1.194e-06), PhysX opened 1.137

### `db0149_swing_double` - grade C (swing_double, tubular_latch latch, none lock, none closer)

* `full` grade C: settle quant, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.004243), PhysX opened 1.92: bolt / strike contact not engaging
* `rl` grade C: settle quant, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.004243), PhysX opened 1.92: bolt / strike contact not engaging

### `db0158_swing_double` - grade C (swing_double, none latch, mag_lock lock engaged, auto_operator_low_energy closer)

* `full` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (1.727e-06), PhysX opened 1.571
* `rl` grade C: settle quant, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (1.727e-06), PhysX opened 1.571

### `db0179_vault` - grade C (vault, multi_bolt latch, vault_wheel lock engaged, none closer)

* `full` grade C: settle agree, hold agree, operate_open **disagree**, release na, relatch na, closer_return na, locked_holds na
  * operate_open: operator moved (travel 6.283) but bolt retracted n/a of its throw; MuJoCo opened 1.725
* `rl` grade C: settle agree, hold agree, operate_open **disagree**, release na, relatch na, closer_return na, locked_holds na
  * operate_open: operator moved (travel 6.283) but bolt retracted n/a of its throw; MuJoCo opened 1.725

### `db0216_swing_single` - grade C (swing_single, none latch, mag_lock lock engaged, concealed_overhead closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (4.645e-06), PhysX opened 1.78
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (4.645e-06), PhysX opened 1.78

### `db0222_swing_double` - grade C (swing_double, tubular_latch latch, none lock, none closer)

* `full` grade C: settle quant, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003401), PhysX opened 1.571: bolt / strike contact not engaging
* `rl` grade C: settle quant, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003401), PhysX opened 1.571: bolt / strike contact not engaging

### `db0273_turnstile_fullheight` - grade C (turnstile_fullheight, none latch, none lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: free push: mujoco opened, physx stuck (hold_displacement 9.491 vs -3.569e-07)
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: free push: mujoco opened, physx stuck (hold_displacement 9.491 vs -0.0007435)

### `db0316_swing_double` - grade C (swing_double, none latch, mag_lock lock engaged, surface_overhead closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (2.281e-06), PhysX opened 1.571
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: mag_lock engaged: MuJoCo weld holds (2.281e-06), PhysX opened 1.571

### `db0338_turnstile_fullheight` - grade C (turnstile_fullheight, none latch, none lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: free push: mujoco opened, physx stuck (hold_displacement 9.685 vs 3.594e-07)
* `rl` grade C: settle agree, hold **disagree**, operate_open na, release na, relatch na, closer_return na, locked_holds na
  * hold: free push: mujoco opened, physx stuck (hold_displacement 9.685 vs -0.0005956)

### `db0352_blast` - grade C (blast, multi_bolt latch, vault_wheel lock engaged, none closer)

* `full` grade C: settle agree, hold agree, operate_open **disagree**, release na, relatch na, closer_return na, locked_holds na
  * operate_open: operator moved (travel 6.283) but bolt retracted n/a of its throw; MuJoCo opened 1.7
* `rl` grade C: settle agree, hold agree, operate_open **disagree**, release na, relatch na, closer_return na, locked_holds na
  * operate_open: operator moved (travel 6.283) but bolt retracted n/a of its throw; MuJoCo opened 1.7

### `db0426_vault` - grade C (vault, multi_bolt latch, vault_wheel lock engaged, none closer)

* `full` grade C: settle agree, hold agree, operate_open **disagree**, release na, relatch na, closer_return na, locked_holds na
  * operate_open: operator moved (travel 6.283) but bolt retracted n/a of its throw; MuJoCo opened 1.7
* `rl` grade C: settle agree, hold agree, operate_open **disagree**, release na, relatch na, closer_return na, locked_holds na
  * operate_open: operator moved (travel 6.283) but bolt retracted n/a of its throw; MuJoCo opened 1.7

### `db0454_swing_double` - grade C (swing_double, tubular_latch latch, none lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003065), PhysX opened 1.658: bolt / strike contact not engaging
* `rl` grade C: settle agree, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.003065), PhysX opened 1.658: bolt / strike contact not engaging

### `db0458_vault` - grade C (vault, multi_bolt latch, vault_wheel lock engaged, none closer)

* `full` grade C: settle agree, hold agree, operate_open **disagree**, release na, relatch na, closer_return na, locked_holds na
  * operate_open: operator moved (travel 6.283) but bolt retracted n/a of its throw; MuJoCo opened 1.739
* `rl` grade C: settle agree, hold agree, operate_open **disagree**, release na, relatch na, closer_return na, locked_holds na
  * operate_open: operator moved (travel 6.283) but bolt retracted n/a of its throw; MuJoCo opened 1.739

### `db0577_swing_double` - grade C (swing_double, tubular_latch latch, none lock, none closer)

* `full` grade C: settle agree, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.00314), PhysX opened 1.571: bolt / strike contact not engaging
* `rl` grade C: settle agree, hold **disagree**, operate_open quant, release na, relatch na, closer_return na, locked_holds na
  * hold: latch (tubular_latch) holds in MuJoCo (0.00314), PhysX opened 1.571: bolt / strike contact not engaging

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
