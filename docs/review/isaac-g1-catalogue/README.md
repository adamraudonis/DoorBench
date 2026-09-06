# Native G1 catalogue results — September 6, 2026

UTC start: **2026-09-06T20:17:17.122684+00:00**. UTC completion: **2026-09-06T22:37:41.749705+00:00**.

| Scope | Attempted | Audited opening crossings | Native errors |
|---|---:|---:|---:|
| Complete non-pet collection | 985 / 985 | 44 / 985 (4.5%) | 10 |
| Applicable upright doorways | 967 / 967 | 44 / 967 (4.6%) | 10 |
| Horizontal hatches | 18 / 18 | Not an upright transit task | 0 |

The 15 supplementary pet doors were excluded before export. Errors remain in the denominator. This is one closed-start, seed-0 attempt per door, using the unchanged Unitree locomotion checkpoint in native Isaac Sim on an L40S. It is **not** the assigned core manipulation, unlocking, closing or damage benchmark. There were **14 isolated retries** for invalid/missing native results, and no retries of ordinary policy failures.

Raw far-side goals counted 80 successes: 24 vertical cases missed a valid opening crossing and 12 were horizontal hatches. The reported 44 additionally require a trace crossing of the actual opening plane, within its width and height. This excludes walking around or below an opening and non-applicable horizontal hatches. The metric checks root traversal, not complete body clearance or safe contact forces. The canonical exporter marks **25 fixtures** with unsupported spatial elements; those limitations remain visible and their inclusion does not certify mechanical parity.

[Reproduction instructions](../../ISAAC_G1_CATALOGUE.md) · [Download native evidence, frozen inputs and video](https://github.com/adamraudonis/DoorBench/releases/tag/g1-isaac-2026-09-06) · [Per-door CSV with review links](per-door.csv) · [Source/trace audit](traversal-audit.json)

## Per-family outcomes

| Family | Attempted | Audited crossings | Native errors |
|---|---:|---:|---:|
| accordion | 12 | 0 | 0 |
| automatic_sliding | 15 | 9 | 0 |
| automatic_swing | 10 | 0 | 0 |
| baby_gate | 10 | 1 | 0 |
| bifold | 30 | 0 | 0 |
| blast | 6 | 0 | 0 |
| cold_storage | 15 | 3 | 0 |
| dutch | 12 | 0 | 0 |
| elevator | 8 | 0 | 0 |
| garage_sectional | 18 | 0 | 0 |
| garage_tiltup | 7 | 0 | 0 |
| gate_sliding | 10 | 0 | 0 |
| gate_swing | 40 | 4 | 0 |
| hatch_ceiling | 8 | N/A | 0 |
| hatch_floor | 10 | N/A | 0 |
| pivot | 20 | 2 | 1 |
| revolving | 15 | 6 | 0 |
| rollup | 15 | 0 | 0 |
| saloon | 12 | 6 | 0 |
| ship_watertight | 10 | 0 | 0 |
| sliding_bypass | 35 | 0 | 0 |
| sliding_single | 100 | 0 | 0 |
| stall | 15 | 1 | 0 |
| strip_curtain | 8 | 3 | 0 |
| swing_double | 76 | 6 | 0 |
| swing_single | 440 | 3 | 9 |
| turnstile_fullheight | 10 | 0 | 0 |
| turnstile_tripod | 10 | 0 | 0 |
| vault | 8 | 0 | 0 |

## Native failures requiring repair

These states remained invalid on an isolated rerun. They are candidates for revision-specific native-engine quarantine and mechanical/export diagnosis, not evidence that the locomotion policy needs stronger actions. The score above retains them. The receipts, original batch attempts, retries and exact fixture hashes are in the evidence archive.

| Door | Native error | Retry simulation time (s) |
|---|---|---:|
| [db0056_swing_single](https://adamraudonis.github.io/DoorBench/#/review?door=db0056_swing_single) | nonfinite_native_state | 0.324 |
| [db0269_swing_single](https://adamraudonis.github.io/DoorBench/#/review?door=db0269_swing_single) | nonfinite_native_state | 0.244 |
| [db0416_swing_single](https://adamraudonis.github.io/DoorBench/#/review?door=db0416_swing_single) | nonfinite_native_state | 0.032 |
| [db0443_swing_single](https://adamraudonis.github.io/DoorBench/#/review?door=db0443_swing_single) | nonfinite_native_state | 0.706 |
| [db0485_swing_single](https://adamraudonis.github.io/DoorBench/#/review?door=db0485_swing_single) | nonfinite_native_state | 0.206 |
| [db0554_swing_single](https://adamraudonis.github.io/DoorBench/#/review?door=db0554_swing_single) | nonfinite_native_state | 2.424 |
| [db0671_pivot](https://adamraudonis.github.io/DoorBench/#/review?door=db0671_pivot) | nonfinite_native_state | 0.068 |
| [db0701_swing_single](https://adamraudonis.github.io/DoorBench/#/review?door=db0701_swing_single) | nonfinite_native_state | 0.042 |
| [db0872_swing_single](https://adamraudonis.github.io/DoorBench/#/review?door=db0872_swing_single) | nonfinite_native_state | 0.044 |
| [db0988_swing_single](https://adamraudonis.github.io/DoorBench/#/review?door=db0988_swing_single) | nonfinite_native_state | 0.168 |

The trained policy controls the legs and holds a fixed upper body. It does not perceive or manipulate handles/locks. Ordinary falls, stalled contact and timeouts therefore remain useful baseline failures; they do not establish a defective test. See the separate [scripted-hand failure review](../../SCRIPTED_FAILURE_REVIEW.md) before filtering training demonstrations or scenarios.

## Hero recording

The README video is a separate, simultaneous **16/16 audited** rerun, recorded from 2026-09-06 21:21:51 UTC with the last traversal receipt at 21:26:34 UTC. The selected cases are illustrative successes, not a random sample. Two narrow cases failed the crossing margin on a pilot rerun and were replaced for this illustration; their original catalogue outcomes are unchanged. The release includes the selection and pilot evidence. The final camera uses different lighting, floor appearance and 7.5 m spacing, with unchanged physical materials, door inputs and policy. The main evaluation uses 12 m spacing.

The recorded mechanical fixtures match generator revision `e0ea25ab0`; the runner is frozen at `a0d8248cc`. The website and Blender images describe their separately versioned public dataset snapshot. Do not mix those revisions when reproducing or interpreting an outcome. Native traces retain root pose and door joint positions, not complete retargetable human or robot motion.
