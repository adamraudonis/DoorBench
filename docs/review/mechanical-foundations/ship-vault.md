# Marine dogs, wheel transmissions and heavy boltwork

The ten ship-door dogging mechanisms now have native load paths and pass their component checks. This is **not whole-door signoff**: their required open-door holdbacks are tracked separately in [ship-holdbacks.md](ship-holdbacks.md). The original 14 vault/blast boltwork and crane defects were repaired and independently tested; see [vaults.md](vaults.md). Subsequent perimeter rebate and whole-door return work is tracked separately and does not establish an enclosure seal or security rating.

## Physical findings and correction

The previous ship model had a 20 mm gap between each rear cleat base and its outboard bridge. Its rotating dogs lacked through-leaf spindles, bored bearings and connecting webs. The four wheel variants drove separate dogs through four remote equalities without connecting rods. All 24 original marine/vault/blast models passed closed-pose collision checks, illustrating why collision clearance alone does not establish structural attachment.

Each ship dog now has a retained 14 mm spindle through a machined leaf aperture, two bored bearing assemblies, keyed wedge web and physically connected cleat base. Individual dogs have usable levers on both declared faces. Wheel variants have supported keyed 20:120 gears and four real pin-connected parallelogram rods. Only the spur pair uses an explicitly ideal rigid gear relation; its teeth do not claim native tooth contact, strength or wear simulation. Rod lengths, pins, bores and native connection constraints are real modeled components.

Correct material masses exposed another fault: on four individual-dog variants, the 2.514 N·m gravitational torque exceeded the original 1.5 N·m bearing friction. Returned levers fell toward approximately 55° release after the hand moved away. An original over-centre extension-spring mechanism now holds each individual dog's endpoints without increasing friction. Its 35 mm crank, 43 mm fixed-anchor radius, 50 mm free length and 10,000 N/m stiffness are explicit engineering choices. The spring lies in front of the spindle, and its supported bracket routes around the entire wedge sweep. Hook fatigue, spring rating and environmental degradation are not certified.

The old operator allowance is replaced only for these complete ship operator assemblies: 1.2 kg per individual-dog door or 5 kg per wheel door. Their complete geometry-backed operator masses are counted instead. Existing hinge, lock and placard allowances remain. Native moving mass equals the reconciled geometry/BOM mass; no arm or rod is scaled down to an obsolete hardware budget.

## Independent native evidence

`doorbench.marine_dog_qa.run_marine_dog_qa(model, meta)` performs a bounded component service cycle. It applies an actual 80 N·m leaf-hinge load while dogged, releases and returns individual dogs sequentially or drives the actual wheel grip, removes all hand forces for two seconds at each endpoint, then loads the returned lock again. Manual force remains capped at 120 N. No inspection resolver or remote follower force is used while stepping.

| Door | Dog input | Native assembly kg | Peak hand N | Maximum penetration mm |
|---|---|---:|---:|---:|
| db0168 | 6 independent | 104.634 | 46.035 | 0.118 |
| db0285 | 4 independent | 82.493 | 32.109 | 0.297 |
| db0314 | wheel / 4 rods | 114.947 | 11.704 | 0.176 |
| db0384 | 6 independent | 147.461 | 32.109 | 0.095 |
| db0600 | wheel / 4 rods | 103.043 | 11.703 | 0.176 |
| db0674 | 8 independent | 157.302 | 32.109 | 0.067 |
| db0729 | wheel / 4 rods | 98.881 | 11.703 | 0.150 |
| db0744 | wheel / 4 rods | 95.400 | 11.704 | 0.083 |
| db0898 | 4 independent | 104.787 | 46.035 | 0.128 |
| db0911 | 6 independent | 105.252 | 32.109 | 0.243 |

All ten pass 91-sample static and running clearance checks. Maximum wheel linkage residual is 7.655 micrometres; maximum gear residual is 0.00000706 rad. Loaded leaf displacement remains below 0.004555 rad before and after the cycle. Native warnings are zero. A disconnected rod plus removal of its actual pin-contact path leaves a dog engaged and blocks opening. Removing one real retention spring reproduces gravity-driven undogging and causes the gate to fail.

Evidence is retained under ignored `out/mechanical-foundations/ship-vault/final/`: `report.json` binds every XML/model/spec file and records source hashes; `audit.py` rebuilds the ten fixtures. A concurrent, unrelated tripod allocation edit changed the shared `physics.py` hash, so the receipt additionally compares a fresh-source rebuild against the exact tested input bytes. The full geometry/native suite passed 58 tests; the subsequent explicit-incomplete-metadata regression passed separately.

Reproduction:

```sh
PYTHONPATH=. python -m pytest tests/test_marine_dog_mounts.py tests/test_marine_linkage.py --basetemp=out/marine-test
```

## Historical vault/blast baseline defects

The earlier eight vault doors (0124, 0179, 0426, 0458, 0530, 0748, 0913, 0921) and six blast doors (0288, 0352, 0623, 0672, 0772, 0960) had separate bolts coupled remotely to their operator. Physical tie rods/racks, shaft journals and bolt-guide bores were absent; the bolt shafts passed into uncut leaf stock. Crane-hinge visuals had leaf-side arms but no frame journal. These baseline defects were repaired and tested as documented in [vaults.md](vaults.md); they are retained here as historical findings. The three previously mislabeled operators 0124/0672/0960 were normalized to their actual heavy dog lever, after seeded generation without extra random draws.

## Primary design references

[NAVSHIPS 316-0042, original Heintz instruction book](https://maritime.org/doc/doors/index.php) describes supported dog spindles, connecting rods and crank/rack mechanisms, as well as springs that hold extreme operating positions. The generic DoorBench reduction and spring layout are original designs, not copies of its dimensions.

[Juniper dog assemblies](https://www.juniperindustries.com/doors/dogqarfq.cfm) document spindle-mounted dogs with straight/flanged bearings, springs, washers and retainers. [Juniper connecting rods](https://www.juniperindustries.com/doors/conrodsrfq.cfm) identify rods, bearings, studs and retaining hardware, with configurations tied to door size and hand. [Juniper complete quick-acting doors](https://www.juniperindustries.com/doors/doorqarfq.cfm) provide the corresponding assembly context.

None of this evidence establishes human reachability, embodied motion feasibility, complete traversal, gasket compression, fluid leakage, rated pressure resistance, material strength or durability.
