# DoorBench taxonomy

The classification behind the 1000 doors: how the leaf moves (**motion class**), what kind of door it is
(**family**, 30), which real-world setting or construction variant it represents (**variant**, 89), and which
mechanisms (operators, latches, locks, closers, hinges) the families share.  This page is the human-readable
companion of three artefacts that are generated from the same source:

| artefact | what | how it is made |
|---|---|---|
| `doorbench/taxonomy.py` | `FAMILIES` + `*_CONTEXTS` (sampler inputs) and, below them, the hierarchy metadata: `MOTION_CLASSES`, `KINEMATICS_TYPES`, `FAMILY_INFO`, `CONTEXT_INFO` / `SETTINGS`, `FAMILY_VARIANTS`, `variant_of()`, `build_hierarchy()` | hand-written; the metadata is never read by the sampler, so editing it cannot change a door |
| `viewer/public/taxonomy.json` | the tree with per-node counts, representative doors + thumbnails, hardware / size / task summaries, the families × mechanism-kind matrices and the shared-mechanism list | `python scripts/taxonomy_report.py` after every dataset build (committed like `assets/manifest.json`) |
| site page **Hierarchy** (`viewer/src/Hierarchy.tsx`) | collapsible tree with thumbnails and badges, relationship heat matrix, links into the catalogue (`#/?family=…&context=…`) | reads the JSON; no per-door fetches |
| `tests/test_taxonomy.py` | every door in exactly one leaf, counts add up to 1000, catalogue filters reproduce each leaf, the JSON is fresh, this document names every family and context | `pytest -q tests/test_taxonomy.py` |

`python scripts/taxonomy_report.py --tree` prints the tree, `--md` the tables reproduced further down.

## Scope

DoorBench models **openings a human or an animal passes through**, including the barriers that are not strictly
"doors" but that a robot moving through buildings meets: gates, turnstiles, hatches, strip curtains, pet flaps and
elevator landing doors.  Deliberately **out of scope** (they are manipulation objects, not passages): cabinet and
appliance doors (fridge, oven, dishwasher, lockers), vehicle / train / bus doors, windows and shutters that are not
walked through.  The audit below lists passable door types that are still missing.

## Structure of the hierarchy

```
motion class (9)  ->  family (30)  ->  variant (89)  ->  door (1000)
```

* **Motion class** — the kinematic nature of the leaf, following the vocabulary of EN 12519 (door and window
  terminology) and the hardware standards: *Hinged swing*, *Pivot*, *Sliding*, *Folding*, *Overhead*, *Roll-up*,
  *Rotary*, *Hatches & flaps*, *Flexible*.  `taxonomy.MOTION_CLASSES`.
* **Family** — the 30 sampler families (`taxonomy.FAMILIES`), each with a reference card
  (`taxonomy.FAMILY_INFO`: real examples, standards, what makes it hard for a robot).
* **Variant** — the leaf level, `taxonomy.FAMILY_VARIANTS`: the sampler **context** where a family has several
  (swing single / pair, sliding, gates, automatic swing), otherwise the physical variant that changes the mechanism
  the robot faces (panel count, bi-parting / single slide, centre / side opening, dog levers / handwheel, floor spring /
  free pivot, cooler / freezer slab, credential-locked / free-spinning …).  Every variant carries a catalogue filter
  (`family` + `context` / `tag` / `slab` / `operator` / `lock` / `closer`) that selects exactly its doors.
* **Cross-cuts** (not part of the tree, but attached to every node): the precise **kinematics type** written into
  `spec.kinematics.type`, the **setting** of the context (residential / commercial / institutional / industrial /
  outdoor / security / marine), the **mechanism kinds** from `doorbench/hardware.py`, the benchmark scenarios and the
  conditions.

### Kinematics types (cross-cut)

| `kinematics.type` | meaning | families | doors |
|---|---|---|---:|
| `hinge_vertical` | rotation about a vertical axis (hinged / pivoted / folding leaf) | swing single & pair, dutch, saloon, automatic swing, cold storage, stall, gates (swing), baby gate, watertight, vault, blast, pivot, bifold, accordion | 716 |
| `slide_horizontal` | translation along the wall | sliding, bypass, automatic sliding, sliding gate, elevator | 168 |
| `hinge_horizontal` | rotation about a horizontal axis | tilt-up garage, floor / ceiling hatch, pet door, strip curtain | 48 |
| `rotor` | continuous rotation of a compartmented rotor | revolving, tripod and full-height turnstile | 35 |
| `slide_vertical` | vertical lift | sectional garage, roll-up | 33 |

Note that `FAMILIES[...]` carries only a coarse `hinge` / `slide` tag (revolving doors and turnstiles are `hinge`,
roll-ups `slide`); the motion classes and `KINEMATICS_TYPES` above are the authoritative grouping (finding T-01).

### Settings (cross-cut over contexts)

| setting | doors | contexts |
|---|---:|---|
| Residential | 475 | `residential_interior`, `residential_exterior`, `heritage_rustic`, `french`, `pocket`, `barn`, `patio_glass`, `shoji_fusuma`, `residential`, `closet`, `partition`, `garage` |
| Commercial | 284 | `commercial_office`, `fire_egress`, `storefront_glass`, `commercial_pair_panic`, `storefront_pair`, `hospitality`, `architectural`, `commercial_entry`, `restroom`, `food_service`, `vertical_transport` |
| Industrial | 65 | `industrial_utility`, `cell_industrial`, `utility`, `industrial` |
| Institutional | 59 | `institutional`, `double_egress` |
| Outdoor | 58 | `garden_picket`, `chain_link`, `wrought_iron`, `pool_safety`, `ranch_tube`, `barn_pair`, `outdoor` |
| Security | 49 | `security_detention`, `access_control`, `security` |
| Marine | 10 | `marine` |

## Compact tree

Counts are doors in the current build (1000).  Variant ids in `code` are the catalogue filter values.

```
Hinged swing  [654]
├─ Swing (single) (swing_single)  [440]
│    ├─ Residential interior (residential_interior)  [130]
│    ├─ Residential exterior (residential_exterior)  [60]
│    ├─ Commercial office (commercial_office)  [60]
│    ├─ Fire / egress (fire_egress)  [55]
│    ├─ Institutional (institutional)  [45]
│    ├─ Industrial / utility (industrial_utility)  [25]
│    ├─ Storefront glass (storefront_glass)  [25]
│    ├─ Heritage / rustic (heritage_rustic)  [25]
│    └─ Security / detention (security_detention)  [15]
├─ Swing (pair) (swing_double)  [76]
│    ├─ Commercial panic pair (commercial_pair_panic)  [26]
│    ├─ French pair (french)  [20]
│    ├─ Storefront pair (storefront_pair)  [12]
│    ├─ Double egress (double_egress)  [10]
│    └─ Barn pair (barn_pair)  [8]
├─ Dutch (dutch)  [12]
│    └─ Residential (residential)  [12]
├─ Saloon (saloon)  [12]
│    └─ Hospitality (hospitality)  [12]
├─ Automatic swing (automatic_swing)  [10]
│    ├─ Institutional (institutional)  [4]
│    ├─ Commercial office (commercial_office)  [4]
│    └─ Storefront glass (storefront_glass)  [2]
├─ Cold storage (cold_storage)  [15]
│    ├─ Cooler (100 mm) (cold_storage_100)  [9]
│    └─ Freezer (150 mm) (freezer_150)  [6]
├─ Toilet stall (stall)  [15]
│    ├─ HPL partition (hpl_partition)  [6]
│    ├─ Powder-coated steel (phenolic_partition)  [6]
│    └─ Stainless (stainless_hollow)  [3]
├─ Gate (swing) (gate_swing)  [40]
│    ├─ Garden picket (garden_picket)  [12]
│    ├─ Chain-link (chain_link)  [8]
│    ├─ Wrought iron (wrought_iron)  [8]
│    ├─ Ranch tube (ranch_tube)  [6]
│    └─ Pool safety (pool_safety)  [6]
├─ Baby gate (baby_gate)  [10]
│    ├─ Self-closing (gate_spring)  [5]
│    └─ Manual (none)  [5]
├─ Watertight (marine) (ship_watertight)  [10]
│    ├─ Individually dogged (dog_lever)  [6]
│    └─ Quick-acting (handwheel) (wheel_ship_hatch)  [4]
├─ Vault (vault)  [8]
│    ├─ Handwheel boltwork (wheel_vault)  [6]
│    ├─ Lever bolt (lever_straight)  [1]
│    └─ Lever dogs (dog_lever)  [1]
├─ Blast door (blast)  [6]
│    ├─ Lever dogs (dog_lever)  [3]
│    ├─ Lever bolt (lever_straight)  [2]
│    └─ Handwheel boltwork (wheel_vault)  [1]
Pivot  [20]
├─ Pivot (architectural) (pivot)  [20]
│    ├─ Free pivot (none)  [8]
│    ├─ Floor spring (hold-open) (floor_spring)  [8]
│    └─ Floor spring (floor_spring_nohold)  [4]
Sliding  [168]
├─ Sliding (sliding_single)  [100]
│    ├─ Barn (surface track) (barn)  [26]
│    ├─ Pocket (pocket)  [22]
│    ├─ Patio glass (patio_glass)  [22]
│    ├─ Shoji / fusuma (shoji_fusuma)  [16]
│    └─ Cell / industrial (cell_industrial)  [14]
├─ Bypass closet (sliding_bypass)  [35]
│    ├─ Wood closet (closet_wood)  [15]
│    ├─ Mirrored wardrobe (mirror)  [10]
│    ├─ Shoji pair (oshiire) (shoji_pair)  [5]
│    └─ Frameless glass (glass_frameless)  [5]
├─ Automatic sliding (automatic_sliding)  [15]
│    ├─ Bi-parting (bi_parting)  [9]
│    └─ Single slide (single_slide)  [6]
├─ Gate (sliding) (gate_sliding)  [10]
│    ├─ Chain-link (chain_link_gate)  [4]
│    ├─ Wrought iron (wrought_iron_gate)  [2]
│    ├─ Bar grille (steel_bar_grille)  [2]
│    └─ Expanded metal (expanded_metal_gate)  [2]
├─ Elevator (elevator)  [8]
│    ├─ Centre opening (center_opening)  [5]
│    └─ Side opening (side_opening)  [3]
Folding  [42]
├─ Bifold (bifold)  [30]
│    ├─ 2 panels (2_panel)  [18]
│    └─ 4 panels (4_panel)  [12]
├─ Accordion (accordion)  [12]
│    ├─ 8 panels (8_panel)  [5]
│    ├─ 6 panels (6_panel)  [4]
│    └─ 10 panels (10_panel)  [3]
Overhead  [25]
├─ Garage (sectional) (garage_sectional)  [18]
│    ├─ Steel, non-insulated (garage_steel_single)  [8]
│    ├─ Steel, insulated (garage_steel_insulated)  [7]
│    └─ Wood carriage-house (garage_wood_carriage)  [3]
├─ Garage (tilt-up) (garage_tiltup)  [7]
│    ├─ Steel (garage_steel_single)  [4]
│    └─ Wood carriage-house (garage_wood_carriage)  [3]
Roll-up  [15]
├─ Roll-up (rollup)  [15]
│    ├─ Steel slat curtain (rollup_steel)  [11]
│    └─ Aluminium grille (rollup_alu_grille)  [4]
Rotary  [35]
├─ Revolving (revolving)  [15]
│    ├─ 4 wings (4_wing)  [9]
│    └─ 3 wings (3_wing)  [6]
├─ Tripod turnstile (turnstile_tripod)  [10]
│    ├─ Credential-locked (mag_lock)  [7]
│    └─ Free-spinning (none)  [3]
├─ Full-height turnstile (turnstile_fullheight)  [10]
│    ├─ Credential-locked (mag_lock)  [6]
│    └─ Free-spinning (none)  [4]
Hatches & flaps  [33]
├─ Floor hatch (hatch_floor)  [10]
│    ├─ Oak cellar trapdoor (cellar_trapdoor)  [4]
│    ├─ Steel plate hatch (steel_plate_security)  [4]
│    └─ Plywood hatch (attic_hatch)  [2]
├─ Ceiling hatch (hatch_ceiling)  [8]
│    ├─ Plywood attic hatch (attic_hatch)  [5]
│    ├─ Steel plate hatch (steel_plate_security)  [2]
│    └─ Hollow-metal scuttle (hollow_metal_18ga)  [1]
├─ Pet door (pet_door)  [15]
│    ├─ Medium dog (medium_dog)  [4]
│    ├─ Cat (cat)  [4]
│    ├─ Small dog (small_dog)  [3]
│    ├─ Large dog (large_dog)  [2]
│    └─ XL dog (xl_dog)  [2]
Flexible  [8]
├─ Strip curtain (strip_curtain)  [8]
│    └─ Industrial (industrial)  [8]
```

## Audit findings

Method: the taxonomy was rebuilt bottom-up from `taxonomy.py`, `spec.py` (every generator's draws) and the
generated `assets/manifest.json` + `spec.json` of all 1000 doors, then compared with the door / hardware
classification used by the industry (EN 12519 terminology, ANSI/BHMA A156.x, EN 1154 / 1155 / 1125 / 179,
EN 16005 / 17352, DASMA 102 / 108, ASME A17.1, SOLAS II-1/13, UL 608, ASTM F1004 / F2247, ISPSC pool barriers) and
with what a mobile robot meets in buildings.  Every hardware combination in the dataset was cross-tabulated
(family × operator × latch × lock × closer × hinge × condition) and screened for combinations that do not occur in
the real world, and specs were hashed to find duplicates.

Severity: **High** = misleads users of the dataset or invalidates a benchmark result; **Medium** = a class of doors is
mis-grouped, mislabelled or physically implausible; **Low** = naming / metadata / a handful of doors.
"Dataset change" says whether fixing it changes generated doors (ids or specs); everything marked *metadata* is
fixed on this branch without touching a door (the sampler output hash is verified identical).

### A. Structure and grouping

| id | sev. | finding | evidence | dataset change | status |
|---|---|---|---|---|---|
| T-01 | Medium | `FAMILIES` labels revolving doors and turnstiles as `hinge`, bifold / accordion as `hinge`, tilt-up as `hinge` and roll-ups as `slide`; the column is coarse and misleading (and unused by code). | `taxonomy.FAMILIES` vs `spec.kinematics.type` (`rotor`, `slide_vertical`, `hinge_horizontal`) | metadata | fixed: `MOTION_CLASSES` + `KINEMATICS_TYPES`; proposal: replace the column in v0.2 |
| T-02 | Medium | *Powered* is a hardware attribute, not a motion class, yet `automatic_swing` is a family (generated by `gen_swing_single` + closer swap) while `automatic_sliding` has its own geometry and elevators / turnstiles are powered too.  Users looking for "automatic doors" find 25 doors and miss 8 elevators + 13 locked turnstiles. | 10 + 15 + 8 + 13 doors; `kinematics.actuator` present on 33 | no (facet only) | hierarchy exposes a `powered` flag per node; proposal: `powered` facet in the manifest |
| T-03 | Medium | Pivot-hung doors are split: family `pivot` (20, oversized architectural) vs 20 of 25 `storefront_glass` swing singles, 9 storefront pairs and 2 automatic swings on `pivot_center` / `pivot_offset` hinges. | hinge kind `pivot_*` on 51 doors in 4 families | no | family relabelled *Pivot (architectural)*; the hinge-kind matrix shows the cross-link |
| T-04 | Medium | `sliding_single / cell_industrial` conflates detention cell sliders, industrial sliding fire doors, cold-room sliders and freight-elevator manual gates, and its slab pool includes `elevator_landing` and `cold_storage_100`: a "freight elevator manual gate" is built from a cold-storage panel and a "detention cell sliding door" from an elevator landing panel. | 14 doors, 11 distinct (slab, use case) pairs | yes | proposal: split into `detention_cell` and `industrial_heavy`; move cold-room sliders to `cold_storage` |
| T-05 | Low | Screen and storm doors (12) are hidden inside `residential_exterior`; they are a distinct type (a second leaf in front of the entry door: pneumatic closer, push-button latch, 20-30 mm frame) and the two-doors-in-series case is not modelled. | slabs `screen_alu` 4, `screen_wood` 4, `storm_alu_glass` 4 | yes | proposal: `screen_storm` context (+ an "entry behind storm door" scenario) |
| T-06 | Low | Pet flaps appear twice: family `pet_door` (15 stand-alone flaps in a host panel) and the `pet_flap` extra on 14 swing singles. Intentional (the extra tests a door with a hole in it), but the hierarchy counts only the family. | 15 + 14 | no | documented |
| T-07 | Low | Turnstile lock is `mag_lock` on 13 turnstiles only because the post-processing rule "an electric strike needs a latch bolt" converts `electric_strike` → `mag_lock`; the real mechanism is a solenoid-locked ratchet. | `turnstile_*` lock counts | yes (name only) | proposal: `ratchet_solenoid` lock kind |
| T-08 | Low | Latch model `dogs_6` on every watertight door regardless of `kinematics.dogs` = 4 / 6 / 8 (the geometry follows the count; only the name is wrong). | 10 doors | yes (name only) | proposal: `dogs_4` / `dogs_8` models |
| T-09 | Medium | Context ids are inconsistent across families: `security_detention` / `security` / `cell_industrial` / `access_control`; `residential` vs `residential_interior` / `_exterior`; `commercial_entry` vs `storefront_*`; `industrial` vs `industrial_utility`; and roll-ups carry context `garage` although 12 of 15 are shop shutters, docks, self-storage units and parking grilles. | 45 context ids in the manifest | yes (context strings) | mitigated: `CONTEXT_INFO` maps every context to one of 7 settings; proposal: setting / sub-context vocabulary in v0.2, roll-up contexts `storage`, `dock`, `shopfront`, `parking` |
| T-10 | Medium | `use_case` titles are drawn independently of the physical variant in several generators, so the catalogue lies: 4 wood bypass closets are titled "mirrored wardrobe doors", 4 more "shoji closet (oshiire)", 2 mirror doors "shoji closet", a pine bifold "utility closet bifold (louvered)"; pet doors "in screen door" for an XL dog. | `sliding_bypass` 35, `bifold` 30, `pet_door` 15, hatches | yes (`use_case` only) | proposal: derive `use_case` from the variant |
| T-11 | High | Two task vocabularies disagree: the legacy `spec.task` / manifest `task` (9 values, shown as a catalogue chip) and the benchmark scenarios (`benchmark.primary`).  12 credential-locked turnstiles are tagged `push_through` (free passage) but their primary scenario is `locked_recognize`; 100 `hold_and_pass` and 50 `peek` doors have no such scenario. | `task` × `benchmark.core` cross-tab (see *Tasks vs scenarios*) | yes (drop / recompute `task`) | proposal: make `benchmark.primary` the only task field; mapping table below |

### B. Gaps: door types a robot meets that are missing or thin

| id | sev. | finding | dataset change | proposal |
|---|---|---|---|---|
| T-12 | High | **Speed gates / optical turnstiles** (swing-wing or retracting glass barriers, EN 17352) — the standard office-lobby, metro and airport barrier — are absent while tripod + full-height turnstiles have 20 doors. | yes | new family `speed_gate` (swing wing / retracting wing / flap gate), 15-20 doors, credential-gated with tailgating logic |
| T-13 | Medium | **Shower / sauna / bathroom glass doors** (frameless pivot or slider with a magnetic or roller catch, 0.6-0.9 m) — met daily by a home robot. | yes | contexts `shower_pivot` under `pivot` / `swing_single` and `shower_slider` under `sliding_single` |
| T-14 | Medium | **Vestibules and airlocks** (two doors in series, optionally interlocked: clean rooms, banks, cold chains) and **hotel connecting doors** (back-to-back pair). | yes | scenario-level composition of two doors + `interlock` lock; a `series` field in the scenario |
| T-15 | Medium | **Exterior folding-sliding glass walls** ("bifold patio doors", 3-6 panels top-hung) and **manual bi-parting sliders** (only automatic bi-parting exists). | yes | `bifold` context `exterior_glass`; `sliding_bypass` variant `biparting_manual` |
| T-16 | Low | **Counter / bar flap gates**, half-height swing gates (reception counters, checkout aisles), **kissing / wicket gates** and cattle guards outdoors. | yes | `gate_swing` contexts `counter_flap`, `kissing_gate` |
| T-17 | Low | **High-speed fabric roll-ups**, **four-fold doors** (fire stations, car washes) and **breakaway ICU sliding glass doors** (telescopic, swing-out). | yes | `rollup` variant `fabric_high_speed`; `automatic_sliding` variant `icu_breakaway` |
| T-18 | Low | Automatic **revolving doors**: 4 of 15 are `manual: False` but carry no actuator block (speed governor only); security revolving doors (interlocking, one-person) absent. | yes | `kinematics.actuator` for powered revolving doors |
| T-19 | Low | Cat flaps have 4-way locks and microchip readers; modelled only as magnet + slide bolt.  Elevator models the hoistway door only (car door / clutch not modelled) — fine, documented. | no | documented |

### C. Counts and balance

| id | sev. | finding | evidence | dataset change | proposal |
|---|---|---|---|---|---|
| T-20 | Medium | **Exact duplicates**: 6 turnstiles are byte-identical to another (`db0344 / db0393 / db0516 / db0946`; `db0896 / db0994`), and 16 near-duplicate groups (22 redundant doors: identical discrete dims + size) in turnstiles, patio sliders, mirror bypass, strip curtains, bifolds — small-quota generators vary too few dimensions. | spec hash / discrete-dimension signature | yes | vary arm length, cabinet width, lane width, direction, drop arm, glass tint; or lower the quota |
| T-21 | Low | Quotas favour exotic heavy doors: vault + blast 14, marine 10, turnstiles 20, revolving 15 (59 doors, 5.9 %) while common types are missing (T-12/13) and swing singles are 44 % (real buildings: > 90 % hinged swing).  Not wrong for a benchmark that wants coverage, but should be a deliberate choice. | `FAMILIES` quotas | yes | v0.2 quota table: vault 5, blast 3, turnstiles 12, speed gates 15, shower 15, automatic swing 15 |
| T-22 | Medium | **Traverse scenarios a humanoid cannot complete**: `pet_door` open-and-traverse × 14 through a 0.16-0.39 m pass plane; `hatch_ceiling` open-and-traverse × 7 with the goal 2.9 m up and no ladder; `hatch_floor` traverse with the goal 1 m below the floor. | `benchmark.scenarios[0]` of those doors | yes (scenario assignment) | score these as *operate* (open / close) or tag `non_traversable_humanoid` so the core number excludes them |
| T-23 | Low | `difficulty` 1.0-1.1 for strip curtains and pet doors, 4.6-4.75 for marine / vault doors: the scale measures hardware effort, not passability; fine, but the catalogue sorts by it. | `difficulty` per family | no | rename to *hardware difficulty* in the UI |

### D. Hardware combinations that would not occur

| id | sev. | finding | evidence | dataset change | proposal |
|---|---|---|---|---|---|
| T-24 | Medium | **Deadbolts / thumbturn deadbolts on frameless glass leaves** (`db0011`, `db0016`, `db0039`, `db0161`, `db0479`, `db0817`, `db0919`): a 10-19 mm glass leaf cannot house a mortise or tubular bolt; `gen_swing_double` already omits them (`frameless_glass_patch_lock_omitted`) but `swing_single`, `pivot` and `automatic_swing` do not.  Thumbturn deadbolts on **aluminium storefront** leaves (5) are realistic (Adams Rite hook-bolt). | 7 doors | yes | apply the pair rule to all glass leaves, or add a `patch_lock` model |
| T-25 | Low | `spring_hinge_single` closer on 24 doors whose hinge model is a plain butt, and 2 screen doors (`db0101`, `db0794`) with `spring_single` hinges but a pneumatic / no closer: the spring hinge *is* the hinge. | hinge × closer cross-tab | yes | tie `hinge.model = spring_single` to `closer = spring_hinge_single` |
| T-26 | Medium | **Maglocks without a release device**: 18 of the 47 maglock doors carry neither a `rex_button` nor a reader extra (pairs, pivots, sliders), and 6 of them start *engaged* with `robot_side_release = True` although `model.json` has no REX body to press (`db0026`, `db0158`, `db0216`, `db0261`, `db0316`, `db0897`) — the `unlock_and_traverse` scenario then has no physical release on the door. | `lock = mag_lock` ∧ extras ∩ {rex_button, keypad_reader_wall} = ∅; body list of `model.json` | yes | always attach a REX + reader with a maglock (codes require an inside release) |
| T-27 | Low | Panic exit device + **padlock** (`db0789`): a padlock on an exit-device door violates IBC 1010.1.9; keep as a locked-recognize case but tag non-compliant. | 1 door | no | tag `non_compliant` |
| T-28 | Low | `keyed_cylinder` lock with no bolt: pull-only sliding cell door `db0732`, garage doors `db0829` (operator none) and `db0964` (lift handle) — a cylinder needs a bolt (hook-bolt / T-handle lock). | 3 doors | yes | hook-bolt or T-handle |
| T-29 | Low | `slide_bolt` lock on 4 pet doors (real pet doors use a slide-in locking panel) and on 8 sliders / sliding gates (plausible as a foot bolt / security pin, wrong name). | 12 doors | yes (name) | `locking_panel`, `security_pin` |
| T-30 | Low | Condition / material mismatches: `swollen` on fiberglass / aluminium-screen leaves (`db0503`, `db0714`, `db0794`); `rusty` on wood / fiberglass leaves whose hinge model is not `butt_rusty` (`db0086`, `db0263`, `db0280`, `db0339`, `db0599`, `db0725`, `db0988`); `jam_stuck` with condition `new` / `well_oiled` (`db0223`, `db0571`). | 12 doors | yes | condition draw conditioned on slab material; `rusty` ⇒ `butt_rusty` |
| T-31 | Low | Vault / blast doors with a plain commercial `lever_straight` operator (`db0124`, `db0672`, `db0960`) — should be a bolt-work lever handle. | 3 doors | yes | `lever_bolt_handle` operator |
| T-32 | Low | Roll-up curtains with a `pull_ring` (`db0196`, `db0506`, `db0983`); privacy-button locks on commercial office doors (`db0646`, `db0225`); `cold_storage` doors with a surface closer *and* cam-lift hinges (5, tracked in G5). | 10 doors | yes | lift handle / pull strap; occupancy indicator instead |
| T-33 | Low | Full-height turnstile rotor mass 14-18 kg (3-4 wings × 8 stainless bars are 40-60 kg); tripod arms 7-11 kg are plausible. | `physics.mass` | yes (physics) | for the physics owner |
| T-34 | Low | `EXTRAS` table: 13 (extra, family) pairs occurred in the data but not in the table (all turnstile / revolving / automatic-door extras, kick plates on saloons, warning placards on swing and sliding doors); `warning_placard` listed a *context*; 5 extras are never attached, 2 of which duplicate `LOCKS` entries. | `EXTRAS` vs manifest | metadata | fixed |
| T-35 | Low | Catalogue hygiene: never-used operators `cane_bolt_drop`, `mail_slot`; latches `ball_catch`, `electric_strike`, `garage_slide_lock`, `mag_lock_600/1200`; locks `hasp`, `mortise_deadbolt`; slabs `cardboard`, `polycarbonate_panel`, `solid_wood_cherry`; panel styles `2_panel`, `canvas_flap`, `glass_sidelite_style`, `steel_half_glass`; stops `overhead_110_hold`, `wedge_jammed`; and `kinematics.stop` values `track_end`, `hook_holdback`, `prop_arm` that are not in `hardware.STOPS`. | catalogue diff | no | prune or use in v0.2 |

### What was *not* found

* Every door references valid catalogue entries; every family meets its quota exactly; the four context tables sum to
  their family quotas; the seeded sampler reproduces the on-disk specs (0 mismatches).
* No closers on sliding doors, no closers on saloon doors other than their double-acting spring hinges, no knobs on
  fire doors, no fire door without a closer, no keypad lock without a keypad, no hook lock on a hinged door.
* `mass_kg` is consistently the moving mass of **one leaf** (pairs, bypass, bi-parting and revolving alike).

## Proposals for the next dataset release (v0.2)

1. **Vocabulary** — replace the coarse `hinge` / `slide` column of `FAMILIES` with the kinematics type; make the motion
   class, setting and `powered` facets first-class manifest fields; retire `spec.task` in favour of `benchmark.primary`
   (T-01, T-02, T-09, T-11).
2. **New families / variants** (T-12 … T-17): `speed_gate` (15-20), shower / sauna glass doors (15), vestibule pairs
   as scenario compositions, exterior folding-sliding glass, manual bi-parting sliders, counter flaps and kissing gates,
   high-speed fabric roll-ups, ICU breakaway sliders.  Funded by trimming vault / blast / turnstile / revolving quotas
   and the duplicated turnstiles (T-20, T-21).
3. **Contexts** — split `cell_industrial`; add `screen_storm`; give roll-ups real contexts; derive `use_case` from the
   variant (T-04, T-05, T-09, T-10).
4. **Hardware rules** in `spec.generate_all` post-processing (T-24 … T-32): no bolt locks in glass leaves without a
   patch lock; spring hinge ⇔ spring closer; maglock ⇒ REX + reader; cylinder ⇒ bolt; `rusty` ⇒ rusty hinge model;
   `swollen` ⇒ wood; bolt-work lever handle on vaults; names `ratchet_solenoid`, `dogs_4/8`, `locking_panel`.
5. **Benchmark** — mark pet doors and ceiling / floor hatches non-traversable for humanoids or score them as
   operate-only (T-22); rename `difficulty` to hardware difficulty (T-23).

## Family reference cards

One card per family.  *Hardware* rows are counted from the generated doors (kinds from `doorbench/hardware.py`); *sizes* are leaf width × height and the moving mass of one leaf.

### Swing (single) — `swing_single` (440 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Single-leaf hinged door (residential, commercial, fire, institutional, industrial, security, storefront) |
| Real examples | bedroom / office door; front entry door; stairwell fire door; hospital patient-room door; storefront glass door; detention cell door; cottage plank door |
| Kinematics | `hinge_vertical`; leaves: 1; flags: – |
| Variants | Residential interior `residential_interior` (130); Residential exterior `residential_exterior` (60); Commercial office `commercial_office` (60); Fire / egress `fire_egress` (55); Institutional `institutional` (45); Industrial / utility `industrial_utility` (25); Storefront glass `storefront_glass` (25); Heritage / rustic `heritage_rustic` (25); Security / detention `security_detention` (15) |
| Operators | lever (160), knob (96), panic touchbar (42), pull (39), keypad lever (19), +10 more |
| Latches | tubular latch (183), deadlatch (85), mortise latch (63), rim latch (39), +3 more |
| Locks | privacy button (42), keypad code (28), deadbolt single (24), mag lock (20), keyed cylinder (20), +13 more; engaged at start: 133 (57 without a robot-side release) |
| Closers | surface overhead (148), spring hinge (25), concealed overhead (18), floor spring (10), +3 more |
| Hinges / bearings | butt (333), continuous (35), rising butt (16), strap (16), +5 more |
| Sizes | leaf 0.61–1.22 × 1.90–2.74 m; mass 10–166 kg (median 53) |
| Conditions | normal (133), new (111), worn (78), old dry (35), swollen (20), +4 more |
| Benchmark scenarios (core) | open and traverse (307), unlock and traverse (76), locked recognize (57), open then close (50), close only (25); mean difficulty 3.41/5 |
| Standards / references | ANSI/BHMA A156.1 hinges; A156.2 bored locks; A156.3 exit devices; A156.4 closers; A156.13 mortise locks; EN 12519 terminology; EN 1154 / EN 1155 / EN 1125 / EN 179; NFPA 80 / UL 10C fire doors; ADA 2010 §404; IBC §1010 |
| Hard for a robot because | The reference problem: locate and operate the operator (lever, knob, panic bar, pull), overcome latch preload and closer spring, keep the body clear of the swing, hold a self-closing leaf while passing, re-latch on close.  Handing (left/right), push/pull side and lock state change the whole plan. |

### Swing (pair) — `swing_double` (76 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Pair of hinged leaves (french, commercial pair w/ panic hardware, double egress, storefront pair) |
| Real examples | french patio doors; auditorium exit pair with vertical rods; hospital double-egress cross-corridor pair; mall entrance storefront pair; barn / carriage-house double doors |
| Kinematics | `hinge_vertical`; leaves: 2 (active + inactive, or both active); flags: pair (76), double_egress (10) |
| Variants | Commercial panic pair `commercial_pair_panic` (26); French pair `french` (20); Storefront pair `storefront_pair` (12); Double egress `double_egress` (10); Barn pair `barn_pair` (8) |
| Operators | panic touchbar (30), pull (14), lever (11), push plate (5), handleset (3), +5 more |
| Latches | vertical rods (27), tubular latch (17), mortise latch (3), rim latch (2) |
| Locks | mag lock (8), deadbolt single (6), delayed egress (5), slide bolt (4), multipoint (3), +2 more; engaged at start: 16 (7 without a robot-side release) |
| Closers | surface overhead (30), electromagnetic hold (7), floor spring (6), concealed overhead (3), +1 more |
| Hinges / bearings | butt (46), continuous (10), strap (8), pivot offset (6), +2 more |
| Sizes | leaf 0.61–1.22 × 2.00–2.44 m; mass 23–134 kg (median 58) |
| Conditions | normal (27), new (25), worn (12), old dry (6), swollen (3), +2 more |
| Benchmark scenarios (core) | open and traverse (60), unlock and traverse (9), locked recognize (7), close only (4), open then close (3); mean difficulty 3.51/5 |
| Standards / references | ANSI/BHMA A156.3 (exit devices, removable mullions); A156.16 flush bolts / coordinators; EN 1125 panic pairs; IBC §1010.1.9 (inactive leaves); NFPA 80 §6.4 pairs (astragals, coordinators) |
| Hard for a robot because | Which leaf is active?  Inactive leaves are held by flush bolts, cane bolts or a removable mullion; double-egress leaves swing opposite ways; the clear opening may need both leaves; vertical-rod devices latch top and bottom. |

### Dutch — `dutch` (12 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Dutch door: independently hinged upper & lower halves with joining bolt |
| Real examples | kitchen / nursery dutch door; stable half-door; daycare reception door |
| Kinematics | `hinge_vertical`; leaves: 1 leaf split into 2 half-leaves; flags: dutch (12) |
| Variants | Residential `residential` (12) |
| Operators | lever (6), knob (4), handleset (2) |
| Latches | tubular latch (12) |
| Locks | none; engaged at start: 0 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | butt (12) |
| Sizes | leaf 0.76–0.91 × 2.03–2.13 m; mass 22–63 kg (median 39) |
| Conditions | normal (6), worn (3), new (3) |
| Benchmark scenarios (core) | open and traverse (12), open then close (3), close only (2); mean difficulty 2.17/5 |
| Standards / references | ANSI/BHMA A156.16 (joining / dutch-door bolt) |
| Hard for a robot because | Two independently hinged halves joined by a bolt: with the bolt thrown the door behaves as one leaf, otherwise only the upper half opens; the lower half must be opened separately to pass. |

### Saloon — `saloon` (12 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Double-acting spring-hinged leaves swinging both ways (cafe / saloon / kitchen pass) |
| Real examples | cafe kitchen pass doors; restaurant kitchen swing door; hospital utility double-acting door; saloon bar doors |
| Kinematics | `hinge_vertical`; leaves: 1 or 2 (double-acting); flags: pair (9), both_ways (12) |
| Variants | Hospitality `hospitality` (12) |
| Operators | push plate (3) |
| Latches | none |
| Locks | none; engaged at start: 0 (0 without a robot-side release) |
| Closers | spring hinge (12) |
| Hinges / bearings | double action (12) |
| Sizes | leaf 0.45–0.90 × 0.90–2.03 m; mass 8–50 kg (median 22) |
| Conditions | normal (5), old dry (3), new (2), worn (2) |
| Benchmark scenarios (core) | open and traverse (12); mean difficulty 2.17/5 |
| Standards / references | ANSI/BHMA A156.17 (double-acting spring hinges); EN 1935 (hinge grades) |
| Hard for a robot because | No latch and no operator: push through in either direction against the spring hinges and stay clear of the return swing, which can strike the robot from behind. |

### Automatic swing — `automatic_swing` (10 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Automatic swing operators (low-energy push-to-open / full-energy) |
| Real examples | low-energy push-to-open office / hotel door; full-energy hospital corridor door |
| Kinematics | `hinge_vertical`; leaves: 1; flags: powered (7), unpowered_operator (3) |
| Variants | Institutional `institutional` (4); Commercial office `commercial_office` (4); Storefront glass `storefront_glass` (2) |
| Operators | lever (5), card lever (2), panic touchbar (1), pull (1), push plate (1) |
| Latches | deadlatch (3), mortise latch (3), rim latch (1), tubular latch (1) |
| Locks | card reader (2), deadbolt single (2), thumbturn only (1), swing bar guard (1), privacy button (1); engaged at start: 4 (3 without a robot-side release) |
| Closers | auto operator low energy (6), auto operator full (4) |
| Hinges / bearings | butt (6), continuous (2), pivot offset (1), pivot center (1) |
| Sizes | leaf 0.81–1.22 × 2.03–2.44 m; mass 37–92 kg (median 60) |
| Conditions | new (4), worn (3), normal (3) |
| Benchmark scenarios (core) | open and traverse (6), locked recognize (3), unlock and traverse (1); mean difficulty 4.2/5 |
| Standards / references | ANSI/BHMA A156.19 (low-energy / power-assist); A156.10 (full-energy); EN 16005 |
| Hard for a robot because | Press the wall button or wave, or start the swing by hand (push-and-go); the operator then swings and holds the leaf; unpowered it behaves as a heavy closer.  Card readers and maglocks sit on the same doors. |

### Cold storage — `cold_storage` (15 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Walk-in cooler / freezer doors (cam-lift hinges, gasket, inside release) |
| Real examples | walk-in cooler door; walk-in freezer door; lab cold-storage door; florist cooler |
| Kinematics | `hinge_vertical`; leaves: 1 (100-150 mm insulated); flags: self_closing (15) |
| Variants | Cooler (100 mm) `cold_storage_100` (9); Freezer (150 mm) `freezer_150` (6) |
| Operators | lever (15) |
| Latches | magnetic (10), roller (5) |
| Locks | padlock (4); engaged at start: 0 (0 without a robot-side release) |
| Closers | surface overhead (5) |
| Hinges / bearings | cam lift (15) |
| Sizes | leaf 0.86–1.22 × 1.98–2.44 m; mass 44–78 kg (median 57) |
| Conditions | normal (6), damaged (3), worn (3), new (3) |
| Benchmark scenarios (core) | open and traverse (15); mean difficulty 2.67/5 |
| Standards / references | NSF/ANSI 7 (commercial refrigerators); industry hardware: Kason 1245 cam-lift hinges, Kason 58 SafeGuard latch |
| Hard for a robot because | Cam-lift hinges raise the leaf as it opens (self-closing by gravity), a magnetic gasket holds it shut, the SafeGuard handle needs a pull-and-lift; an inside release must always work. |

### Toilet stall — `stall` (15 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Toilet partition stall door (gravity hinge, slide latch) |
| Real examples | public restroom stall; ADA outswing stall; locker-room changing stall |
| Kinematics | `hinge_vertical`; leaves: 1 (partition door, 0.6-0.86 m); flags: self_closing (15) |
| Variants | HPL partition `hpl_partition` (6); Powder-coated steel `phenolic_partition` (6); Stainless `stainless_hollow` (3) |
| Operators | slide bolt handle (11), pull (4) |
| Latches | slide bolt (15) |
| Locks | slide bolt (5); engaged at start: 2 (1 without a robot-side release) |
| Closers | none |
| Hinges / bearings | gravity pivot (15) |
| Sizes | leaf 0.61–0.86 × 1.47–2.00 m; mass 11–32 kg (median 19) |
| Conditions | normal (5), worn (4), damaged (3), new (3) |
| Benchmark scenarios (core) | open and traverse (13), locked recognize (1), unlock and traverse (1); mean difficulty 2.0/5 |
| Standards / references | ADA §604.8 (toilet compartments); manufacturer hardware (Bobrick gravity hinges, slide latches) |
| Hard for a robot because | Gravity hinges hold the door ajar when vacant; an occupied stall is latched from inside with a slide latch and must be recognised as such; narrow leaf and pilaster gap. |

### Gate (swing) — `gate_swing` (40 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Outdoor swing gates: picket, chain-link, wrought iron, pool, ranch |
| Real examples | garden picket gate; schoolyard chain-link gate; estate wrought-iron gate; pool safety gate; ranch tube gate |
| Kinematics | `hinge_vertical`; leaves: 1 (outdoor, 0.9-4.8 m); flags: – |
| Variants | Garden picket `garden_picket` (12); Chain-link `chain_link` (8); Wrought iron `wrought_iron` (8); Ranch tube `ranch_tube` (6); Pool safety `pool_safety` (6) |
| Operators | gate latch fork (12), slide bolt handle (6), lift latch (6), thumb latch (5), ring pull (4), +3 more |
| Latches | gravity bar (17), slide bolt (6), hook (6), mortise latch (4) |
| Locks | padlock (9), slide bolt (7), deadbolt double (2), electric strike (1); engaged at start: 7 (2 without a robot-side release) |
| Closers | gate (17) |
| Hinges / bearings | strap (37), butt (3) |
| Sizes | leaf 0.90–4.80 × 0.90–2.40 m; mass 4–96 kg (median 14) |
| Conditions | worn (10), rusty (10), normal (10), new (5), sagging (5) |
| Benchmark scenarios (core) | open and traverse (33), open then close (7), unlock and traverse (5), close only (4), locked recognize (2); mean difficulty 3.17/5 |
| Standards / references | ISPSC §305 / ASTM F1908 pool barriers (self-closing, self-latching, latch at 1.37 m); EN 12209 / BS 3621 gate locks; ASTM F2200 (automated gates) |
| Hard for a robot because | Outdoor: uneven ground clearance, sagging hinges, gravity latches that must be lifted, hasps and padlocks, pool latches mounted at 1.5 m out of a child's reach; wide farm gates swing through large arcs. |

### Baby gate — `baby_gate` (10 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Pressure- or hardware-mounted child safety gate |
| Real examples | stair-top gate; kitchen doorway gate; hallway pet gate |
| Kinematics | `hinge_vertical`; leaves: 1 (0.75-1.1 m wide, waist high); flags: both_ways (5), auto_close (5) |
| Variants | Self-closing `gate_spring` (5); Manual `none` (5) |
| Operators | lift latch (10) |
| Latches | hook (10) |
| Locks | none; engaged at start: 0 (0 without a robot-side release) |
| Closers | gate (5) |
| Hinges / bearings | butt (10) |
| Sizes | leaf 0.75–1.10 × 0.75–1.00 m; mass 5–12 kg (median 11) |
| Conditions | new (4), normal (4), worn (2) |
| Benchmark scenarios (core) | open and traverse (10); mean difficulty 2.5/5 |
| Standards / references | ASTM F1004 (expansion gates and expandable enclosures); EN 1930 (child safety barriers) |
| Hard for a robot because | A lift-and-swing latch designed to defeat toddlers, a trip bar at floor level, spring return; the robot can also step over it. |

### Watertight (marine) — `ship_watertight` (10 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Marine watertight door with dogging levers or central wheel |
| Real examples | ship bulkhead WT door; engine-room WT door; offshore weathertight door; submarine bulkhead hatch |
| Kinematics | `hinge_vertical`; leaves: 1 (dogged); flags: – |
| Variants | Individually dogged `dog_lever` (6); Quick-acting (handwheel) `wheel_ship_hatch` (4) |
| Operators | lever (6), wheel (4) |
| Latches | dogs (10) |
| Locks | dogs (10); engaged at start: 10 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | butt (10) |
| Sizes | leaf 0.65–0.80 × 1.50–1.90 m; mass 64–119 kg (median 80) |
| Conditions | rusty (4), normal (3), worn (2), well oiled (1) |
| Benchmark scenarios (core) | unlock and traverse (10); mean difficulty 4.6/5 |
| Standards / references | SOLAS II-1 Reg. 13 (watertight doors); ISO 6042 (weathertight steel doors); class rules (ABS / DNV / LR) |
| Hard for a robot because | Release 4-8 wedge dogs one by one (or spin a central handwheel on quick-acting doors), pull a 64-120 kg leaf off a compressed gasket, step over a 150-450 mm coaming, re-dog behind. |

### Vault — `vault` (8 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Vault / safe-room door with handwheel boltwork |
| Real examples | bank vault door; safe-room door; data-centre vault; gun vault |
| Kinematics | `hinge_vertical`; leaves: 1 (0.8-1.5 t); flags: – |
| Variants | Handwheel boltwork `wheel_vault` (6); Lever bolt `lever_straight` (1); Lever dogs `dog_lever` (1) |
| Operators | wheel (6), lever (2) |
| Latches | multi bolt (6), dogs (2) |
| Locks | vault wheel (6), dogs (2); engaged at start: 8 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | butt (8) |
| Sizes | leaf 0.90–1.20 × 1.90–2.10 m; mass 784–1459 kg (median 1016) |
| Conditions | normal (4), well oiled (2), old dry (1), new (1) |
| Benchmark scenarios (core) | unlock and traverse (8); mean difficulty 4.75/5 |
| Standards / references | UL 608 (burglary-resistant vault doors); EN 1143-1 (secure storage units) |
| Hard for a robot because | Turn a handwheel one to two full turns to retract 4-8 bolts, then move a tonne of steel on crane hinges: high inertia, very low friction, a step sill. |

### Blast door — `blast` (6 doors, motion class *Hinged swing*)

| | |
|---|---|
| What it is | Blast door (very heavy, multi-hinge, lever bolts) |
| Real examples | bunker / shelter blast door; test-cell blast door |
| Kinematics | `hinge_vertical`; leaves: 1 (0.7-1.2 t); flags: – |
| Variants | Lever dogs `dog_lever` (3); Lever bolt `lever_straight` (2); Handwheel boltwork `wheel_vault` (1) |
| Operators | lever (5), wheel (1) |
| Latches | dogs (5), multi bolt (1) |
| Locks | dogs (5), vault wheel (1); engaged at start: 6 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | butt (6) |
| Sizes | leaf 0.90–1.20 × 1.90–2.10 m; mass 673–1164 kg (median 1012) |
| Conditions | well oiled (2), normal (2), old dry (1), new (1) |
| Benchmark scenarios (core) | unlock and traverse (6); mean difficulty 4.17/5 |
| Standards / references | ASTM F2247 (blast-resistant doors); UFC 4-010-01 |
| Hard for a robot because | Like a vault door but latched by lever dogs or a wheel; heavy gaskets and a raised sill. |

### Pivot (architectural) — `pivot` (20 doors, motion class *Pivot*)

| | |
|---|---|
| What it is | Pivot door: center or offset floor pivot, often oversized / heavy |
| Real examples | modern residence pivot entry; hotel lobby / museum pivot door; boutique frameless-glass pivot |
| Kinematics | `hinge_vertical`; leaves: 1 (oversized, 0.9-1.8 m); flags: – |
| Variants | Free pivot `none` (8); Floor spring (hold-open) `floor_spring` (8); Floor spring `floor_spring_nohold` (4) |
| Operators | pull (14), lever (3) |
| Latches | magnetic (5), mortise latch (1) |
| Locks | thumbturn only (4), mag lock (4); engaged at start: 1 (1 without a robot-side release) |
| Closers | floor spring (12) |
| Hinges / bearings | pivot center (15), pivot offset (5) |
| Sizes | leaf 0.90–1.80 × 2.13–3.00 m; mass 56–262 kg (median 108) |
| Conditions | new (10), normal (7), well oiled (3) |
| Benchmark scenarios (core) | open and traverse (19), close only (2), locked recognize (1); mean difficulty 3.95/5 |
| Standards / references | ANSI/BHMA A156.4 (floor closers / pivots); A156.17 pivots; EN 1154 floor springs |
| Hard for a robot because | Heavy leaf (56-262 kg) on a centre or offset pivot: part of the leaf swings towards the robot while the rest swings away; the floor spring's hold-open at 90 deg and large inertia dominate. |

### Sliding — `sliding_single` (100 doors, motion class *Sliding*)

| | |
|---|---|
| What it is | Single sliding leaf: pocket, barn (surface track), patio glass, shoji/fusuma, cell, industrial |
| Real examples | bathroom pocket door; barn door on a flat track; patio sliding glass door; shoji / fusuma; detention cell slider; industrial sliding fire door |
| Kinematics | `slide_horizontal`; leaves: 1 (+ fixed panel for patio / shoji); flags: – |
| Variants | Barn (surface track) `barn` (26); Pocket `pocket` (22); Patio glass `patio_glass` (22); Shoji / fusuma `shoji_fusuma` (16); Cell / industrial `cell_industrial` (14) |
| Operators | flush pull (41), pull (36), hook lock slider (15), ring pull (4), lever (2) |
| Latches | hook (15), gravity bar (6), slide bolt (3), electric bolt (3) |
| Locks | hook lock (28), slide bolt (6), keyed cylinder (4), padlock (3), mag lock (2), +1 more; engaged at start: 18 (10 without a robot-side release) |
| Closers | none |
| Hinges / bearings | none |
| Sizes | leaf 0.71–1.50 × 1.76–2.44 m; mass 8–214 kg (median 46) |
| Conditions | normal (34), new (25), worn (23), old dry (10), damaged (8) |
| Benchmark scenarios (core) | open then close (90), open and traverse (82), close only (39), locked recognize (10), unlock and traverse (8); mean difficulty 2.63/5 |
| Standards / references | ANSI/BHMA A156.14 sliding & folding door hardware; AAMA/WDMA/CSA 101 (patio doors); NFPA 80 (sliding fire doors); IBC §1010.1.2 (sliding doors not in egress except auto) |
| Hard for a robot because | Grasp a flush pull or edge and translate the leaf along its track against roller friction; a hook lock or teardrop latch must be lifted first; the leaf disappears into a pocket or behind a fixed panel so the grasp point moves. |

### Bypass closet — `sliding_bypass` (35 doors, motion class *Sliding*)

| | |
|---|---|
| What it is | Two or three overlapping leaves on parallel tracks (closet, mirrored, shoji pair) |
| Real examples | bedroom closet bypass doors; mirrored wardrobe doors; shoji closet (oshiire) |
| Kinematics | `slide_horizontal`; leaves: 2 or 3 on parallel tracks; flags: – |
| Variants | Wood closet `closet_wood` (15); Mirrored wardrobe `mirror` (10); Shoji pair (oshiire) `shoji_pair` (5); Frameless glass `glass_frameless` (5) |
| Operators | flush pull (28), knob (3) |
| Latches | none |
| Locks | none; engaged at start: 0 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | none |
| Sizes | leaf 0.61–1.22 × 2.03–2.40 m; mass 6–77 kg (median 19) |
| Conditions | normal (15), worn (10), damaged (5), new (5) |
| Benchmark scenarios (core) | open and traverse (35), open then close (35), close only (9); mean difficulty 1.94/5 |
| Standards / references | ANSI/BHMA A156.14 |
| Hard for a robot because | Leaves overlap: only one track's leaf can be moved from a given side, and moving one leaf can uncover or cover the other; finger cups give little purchase. |

### Automatic sliding — `automatic_sliding` (15 doors, motion class *Sliding*)

| | |
|---|---|
| What it is | Sensor-activated sliding doors (single / bi-parting) with manual breakout |
| Real examples | supermarket / pharmacy entrance; hospital entrance; airport and office-lobby sliders |
| Kinematics | `slide_horizontal`; leaves: 1 or 2 (bi-parting) + fixed sidelites; flags: bi_parting (9), breakout (15), unpowered_operator (3), powered (12) |
| Variants | Bi-parting `bi_parting` (9); Single slide `single_slide` (6) |
| Operators | none |
| Latches | none |
| Locks | electric strike (4); engaged at start: 0 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | none |
| Sizes | leaf 0.90–1.20 × 2.13–2.40 m; mass 38–101 kg (median 74) |
| Conditions | normal (6), new (6), worn (3) |
| Benchmark scenarios (core) | open and traverse (15); mean difficulty 2.07/5 |
| Standards / references | ANSI/BHMA A156.10 (full-energy power-operated doors); EN 16005; IBC §1010.1.4.3 (breakout for egress) |
| Hard for a robot because | The door opens itself when the sensor fires: approach into the detection zone, wait, pass before hold-open time expires; if the power is off, break out the leaf manually (220 N). |

### Gate (sliding) — `gate_sliding` (10 doors, motion class *Sliding*)

| | |
|---|---|
| What it is | Cantilever / track sliding vehicle & pedestrian gates |
| Real examples | cantilever driveway gate (manual); pedestrian sliding gate; warehouse yard gate |
| Kinematics | `slide_horizontal`; leaves: 1 (cantilever or bottom rail); flags: – |
| Variants | Chain-link `chain_link_gate` (4); Wrought iron `wrought_iron_gate` (2); Bar grille `steel_bar_grille` (2); Expanded metal `expanded_metal_gate` (2) |
| Operators | pull (5), hasp (3), slide bolt handle (2) |
| Latches | slide bolt (2) |
| Locks | padlock (3), electric strike (3), slide bolt (2); engaged at start: 3 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | none |
| Sizes | leaf 1.20–4.80 × 1.50–2.10 m; mass 6–329 kg (median 62) |
| Conditions | normal (4), rusty (4), worn (2) |
| Benchmark scenarios (core) | open then close (10), open and traverse (7), unlock and traverse (3); mean difficulty 3.3/5 |
| Standards / references | ASTM F2200; EN 13241 / EN 12453 (power-operated gates) |
| Hard for a robot because | Long, heavy leaves (up to 330 kg) on cantilever rollers: large start force, long travel, pinch zones at the posts. |

### Elevator — `elevator` (8 doors, motion class *Sliding*)

| | |
|---|---|
| What it is | Elevator landing doors (center or side opening, interlocked) |
| Real examples | office / residential-tower landing doors; hospital and freight elevator doors |
| Kinematics | `slide_horizontal`; leaves: 1 (side) or 2 (centre-opening) hoistway panels; flags: center_opening (5), interlocked (8), powered (8) |
| Variants | Centre opening `center_opening` (5); Side opening `side_opening` (3) |
| Operators | none |
| Latches | electric bolt (8) |
| Locks | interlock (8); engaged at start: 8 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | none |
| Sizes | leaf 0.45–1.07 × 2.10–2.40 m; mass 28–76 kg (median 37) |
| Conditions | new (4), normal (3), worn (1) |
| Benchmark scenarios (core) | unlock and traverse (8); mean difficulty 3.0/5 |
| Standards / references | ASME A17.1 / CSA B44 (hoistway door interlocks); EN 81-20 / EN 81-50 |
| Hard for a robot because | The robot cannot open a hoistway door: it presses the call button, waits for the car, and passes during the door-open dwell; doors reopen on obstruction. |

### Bifold — `bifold` (30 doors, motion class *Folding*)

| | |
|---|---|
| What it is | Bi-fold closet doors (2 or 4 panels) pivoting with guided free edge |
| Real examples | bedroom closet bifold; louvered utility-closet bifold |
| Kinematics | `hinge_vertical`; leaves: 2 or 4 panels (coupled); flags: fold (30) |
| Variants | 2 panels `2_panel` (18); 4 panels `4_panel` (12) |
| Operators | knob (24), pull (6) |
| Latches | magnetic (8) |
| Locks | none; engaged at start: 0 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | butt (30) |
| Sizes | leaf 0.15–0.76 × 2.03–2.40 m; mass 2–33 kg (median 7) |
| Conditions | normal (10), worn (10), new (5), damaged (5) |
| Benchmark scenarios (core) | open and traverse (30), open then close (30), close only (7); mean difficulty 1.83/5 |
| Standards / references | ANSI/BHMA A156.14 (bifold hardware) |
| Hard for a robot because | Pulling the knob rotates the pivot panel while the guide panel's free edge slides in the head track: a closed kinematic chain whose panels fold towards the robot. |

### Accordion — `accordion` (12 doors, motion class *Folding*)

| | |
|---|---|
| What it is | Accordion / concertina folding partition door |
| Real examples | room-divider accordion; laundry-nook accordion; office partition |
| Kinematics | `hinge_vertical`; leaves: 6-10 narrow panels (coupled); flags: fold (12), accordion (12) |
| Variants | 8 panels `8_panel` (5); 6 panels `6_panel` (4); 10 panels `10_panel` (3) |
| Operators | pull (6), flush pull (3), knob (3) |
| Latches | magnetic (6) |
| Locks | none; engaged at start: 0 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | continuous (12) |
| Sizes | leaf 0.09–0.30 × 2.03–2.40 m; mass 10–16 kg (median 12) |
| Conditions | normal (5), worn (5), new (2) |
| Benchmark scenarios (core) | open and traverse (12), open then close (12), close only (3); mean difficulty 1.5/5 |
| Standards / references | ANSI/BHMA A156.14 |
| Hard for a robot because | Many light panels on piano hinges concertina together; the pull travels the whole opening width and the stack can jam. |

### Garage (sectional) — `garage_sectional` (18 doors, motion class *Overhead*)

| | |
|---|---|
| What it is | Overhead sectional garage door (vertical lift approximated) |
| Real examples | residential single / double garage door; townhouse garage door |
| Kinematics | `slide_vertical`; leaves: 1 door of 4-5 hinged sections (vertical lift); flags: – |
| Variants | Steel, non-insulated `garage_steel_single` (8); Steel, insulated `garage_steel_insulated` (7); Wood carriage-house `garage_wood_carriage` (3) |
| Operators | pull (9), t handle (6) |
| Latches | none |
| Locks | slide bolt (3), padlock (3), keyed cylinder (2); engaged at start: 4 (2 without a robot-side release) |
| Closers | none |
| Hinges / bearings | none |
| Sizes | leaf 2.44–5.49 × 2.00–2.44 m; mass 54–188 kg (median 84) |
| Conditions | normal (6), worn (5), new (3), damaged (2), rusty (2) |
| Benchmark scenarios (core) | open then close (16), open and traverse (14), close only (8), unlock and traverse (2), locked recognize (2); mean difficulty 2.94/5 |
| Standards / references | ANSI/DASMA 102 (sectional garage doors); UL 325 (operators); EN 13241 / EN 12604 |
| Hard for a robot because | Lift a 2.4-5.5 m wide, 55-190 kg leaf from a low handle: the torsion-spring counterbalance carries most of the weight but a slack or disengaged opener changes the force; the leaf moves overhead towards the robot. |

### Garage (tilt-up) — `garage_tiltup` (7 doors, motion class *Overhead*)

| | |
|---|---|
| What it is | One-piece tilt-up garage door (offset pivot) |
| Real examples | 1960s tilt-up garage door; carport tilt-up door |
| Kinematics | `hinge_horizontal`; leaves: 1 one-piece panel; flags: – |
| Variants | Steel `garage_steel_single` (4); Wood carriage-house `garage_wood_carriage` (3) |
| Operators | t handle (4), pull (3) |
| Latches | none |
| Locks | padlock (2), slide bolt (1); engaged at start: 1 (1 without a robot-side release) |
| Closers | none |
| Hinges / bearings | pivot offset (7) |
| Sizes | leaf 2.44–5.49 × 2.00–2.44 m; mass 54–213 kg (median 84) |
| Conditions | worn (2), normal (2), new (1), rusty (1), damaged (1) |
| Benchmark scenarios (core) | open and traverse (6), open then close (6), locked recognize (1); mean difficulty 3.14/5 |
| Standards / references | ANSI/DASMA 102; EN 13241 |
| Hard for a robot because | The whole panel swings out at the bottom before it rises overhead, sweeping the approach area; extension springs counterbalance. |

### Roll-up — `rollup` (15 doors, motion class *Roll-up*)

| | |
|---|---|
| What it is | Roll-up / coiling steel curtain or grille |
| Real examples | self-storage unit door; shop-front security shutter; loading-dock coiling door; parking-garage grille |
| Kinematics | `slide_vertical`; leaves: 1 coiling curtain / grille; flags: – |
| Variants | Steel slat curtain `rollup_steel` (11); Aluminium grille `rollup_alu_grille` (4) |
| Operators | pull (9), ring pull (3) |
| Latches | none |
| Locks | slide bolt (3), padlock (3); engaged at start: 2 (1 without a robot-side release) |
| Closers | none |
| Hinges / bearings | none |
| Sizes | leaf 1.20–3.66 × 2.13–3.66 m; mass 28–129 kg (median 48) |
| Conditions | worn (5), normal (4), damaged (2), rusty (2), new (2) |
| Benchmark scenarios (core) | open then close (14), open and traverse (13), close only (3), locked recognize (1), unlock and traverse (1); mean difficulty 2.47/5 |
| Standards / references | ANSI/DASMA 108 (rolling doors); EN 13241; UL 325 (motorised) |
| Hard for a robot because | Lift the bottom bar of a curtain that coils overhead; manual curtains need a strong pull to start (counterbalance, slat friction) and chain hoists need many turns. |

### Revolving — `revolving` (15 doors, motion class *Rotary*)

| | |
|---|---|
| What it is | Revolving door: 3 or 4 wings on a central rotor inside a drum |
| Real examples | office tower / hotel / department store revolving door; airport and hospital lobby revolving doors |
| Kinematics | `rotor`; leaves: 3 or 4 wings; flags: breakout (15) |
| Variants | 4 wings `4_wing` (9); 3 wings `3_wing` (6) |
| Operators | pull (6), push plate (3) |
| Latches | none |
| Locks | electric strike (3); engaged at start: 0 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | rotor (15) |
| Sizes | leaf 0.87–1.77 × 2.13–2.70 m; mass 50–110 kg (median 62) |
| Conditions | normal (6), new (6), worn (3) |
| Benchmark scenarios (core) | open and traverse (15); mean difficulty 2.8/5 |
| Standards / references | ANSI/BHMA A156.27 (power & manual revolving doors); IBC §1010.1.4.1 (breakout, speed, adjacent swing door); EN 16005 (power-operated) |
| Hard for a robot because | Enter a moving compartment, keep pace with the wing (speed governor), and exit on the far side without touching the drum; breakout wings and a possible electric bolt at night. |

### Tripod turnstile — `turnstile_tripod` (10 doors, motion class *Rotary*)

| | |
|---|---|
| What it is | Waist-high tripod turnstile (ratcheting rotor) |
| Real examples | metro / subway turnstile; office lobby tripod turnstile; gym or stadium entrance |
| Kinematics | `rotor`; leaves: 3 arms (tripod); flags: one_way (10), locked_until_credential (7) |
| Variants | Credential-locked `mag_lock` (7); Free-spinning `none` (3) |
| Operators | none |
| Latches | none |
| Locks | mag lock (7); engaged at start: 7 (7 without a robot-side release) |
| Closers | none |
| Hinges / bearings | rotor (10) |
| Sizes | leaf 0.50 × 1.00 m; mass 8–11 kg (median 11) |
| Conditions | new (6), normal (3), worn (1) |
| Benchmark scenarios (core) | locked recognize (7), open and traverse (3); mean difficulty 2.0/5 |
| Standards / references | EN 17352 (pedestrian entrance control); IBC §1010.3 turnstiles; ADA §404 (turnstiles are not accessible routes) |
| Hard for a robot because | Present a credential, then push a waist-high arm that ratchets one way; the next arm rises into the path.  One-way and drop-arm variants. |

### Full-height turnstile — `turnstile_fullheight` (10 doors, motion class *Rotary*)

| | |
|---|---|
| What it is | Full-height rotating turnstile (3-4 wing rotor in cage) |
| Real examples | stadium / factory / metro full-height turnstile; parking-garage pedestrian turnstile |
| Kinematics | `rotor`; leaves: 3-4 wings of 8 bars; flags: one_way (7), locked_until_credential (6) |
| Variants | Credential-locked `mag_lock` (6); Free-spinning `none` (4) |
| Operators | none |
| Latches | none |
| Locks | mag lock (6); engaged at start: 6 (6 without a robot-side release) |
| Closers | none |
| Hinges / bearings | rotor (10) |
| Sizes | leaf 0.65–0.75 × 2.10 m; mass 14–18 kg (median 17) |
| Conditions | normal (5), worn (3), new (2) |
| Benchmark scenarios (core) | locked recognize (6), open and traverse (4); mean difficulty 2.0/5 |
| Standards / references | EN 17352; IBC §1010.3 |
| Hard for a robot because | Walk inside a rotating cage compartment while pushing the bars; the rotor indexes 90-120 deg per passage and cannot be reversed. |

### Floor hatch — `hatch_floor` (10 doors, motion class *Hatches & flaps*)

| | |
|---|---|
| What it is | Floor hatch / cellar trapdoor (horizontal hinge, lift up) |
| Real examples | cellar trapdoor; utility floor hatch; ship deck hatch; stage trapdoor; storm-shelter hatch |
| Kinematics | `hinge_horizontal`; leaves: 1 (lift up); flags: gravity_assisted_close (10) |
| Variants | Oak cellar trapdoor `cellar_trapdoor` (4); Steel plate hatch `steel_plate_security` (4); Plywood hatch `attic_hatch` (2) |
| Operators | ring pull (8), pull (2) |
| Latches | slide bolt (2) |
| Locks | slide bolt (2), padlock (2); engaged at start: 1 (0 without a robot-side release) |
| Closers | gas strut (4) |
| Hinges / bearings | butt (10) |
| Sizes | leaf 0.76–1.20 × 0.76–1.50 m; mass 9–66 kg (median 28) |
| Conditions | worn (4), normal (3), old dry (2), rusty (1) |
| Benchmark scenarios (core) | open then close (10), open and traverse (9), close only (7), unlock and traverse (1); mean difficulty 1.7/5 |
| Standards / references | IBC §1011.12 (roof / floor access); manufacturer data (Bilco floor doors) |
| Hard for a robot because | Lift a horizontal leaf against gravity from a ring pull, hold it past its balance point or onto a prop arm; gas struts assist; the robot stands next to the leaf's own edge. |

### Ceiling hatch — `hatch_ceiling` (8 doors, motion class *Hatches & flaps*)

| | |
|---|---|
| What it is | Ceiling / attic hatch, roof scuttle (push up) |
| Real examples | attic access hatch; roof scuttle; ceiling maintenance hatch |
| Kinematics | `hinge_horizontal`; leaves: 1 (push up); flags: gravity_assisted_close (8) |
| Variants | Plywood attic hatch `attic_hatch` (5); Steel plate hatch `steel_plate_security` (2); Hollow-metal scuttle `hollow_metal_18ga` (1) |
| Operators | ring pull (2), pull (2) |
| Latches | slide bolt (2) |
| Locks | slide bolt (2), padlock (1); engaged at start: 1 (0 without a robot-side release) |
| Closers | gas strut (4) |
| Hinges / bearings | butt (8) |
| Sizes | leaf 0.56–0.90 × 0.76–1.20 m; mass 4–52 kg (median 6) |
| Conditions | worn (3), normal (2), rusty (2), old dry (1) |
| Benchmark scenarios (core) | open then close (8), open and traverse (7), close only (2), unlock and traverse (1); mean difficulty 1.5/5 |
| Standards / references | IBC §1011.12 |
| Hard for a robot because | Overhead push at 2.4 m; passing through needs a ladder, so the realistic task is open / close only. |

### Pet door — `pet_door` (15 doors, motion class *Hatches & flaps*)

| | |
|---|---|
| What it is | Standalone pet door flap in a wall/door panel (dog & cat sizes) |
| Real examples | cat flap in a back door; large-dog flap in a wall or garage door |
| Kinematics | `hinge_horizontal`; leaves: 1 flap (cat to XL dog); flags: both_ways (15), flap (15) |
| Variants | Medium dog `medium_dog` (4); Cat `cat` (4); Small dog `small_dog` (3); Large dog `large_dog` (2); XL dog `xl_dog` (2) |
| Operators | none |
| Latches | magnetic (12) |
| Locks | slide bolt (4); engaged at start: 1 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | flap pin (15) |
| Sizes | leaf 0.15–0.38 × 0.17–0.64 m; mass 0–1 kg (median 0) |
| Conditions | new (6), normal (6), worn (3) |
| Benchmark scenarios (core) | open and traverse (14), unlock and traverse (1); mean difficulty 1.07/5 |
| Standards / references | no code; manufacturer size classes (small / medium / large / XL) |
| Hard for a robot because | Too small for a humanoid: a flap swinging both ways on a top pin with a weak magnet, sometimes closed by a slide-in locking panel.  Relevant for quadrupeds and for recognising a non-passable opening. |

### Strip curtain — `strip_curtain` (8 doors, motion class *Flexible*)

| | |
|---|---|
| What it is | PVC strip curtain doorway (many hinged strips) |
| Real examples | walk-in cooler strip curtain; warehouse dock strip door; food-processing strip curtain |
| Kinematics | `hinge_horizontal`; leaves: 5-18 overlapping PVC strips; flags: both_ways (8), strips (8) |
| Variants | Industrial `industrial` (8) |
| Operators | none |
| Latches | none |
| Locks | none; engaged at start: 0 (0 without a robot-side release) |
| Closers | none |
| Hinges / bearings | continuous (8) |
| Sizes | leaf 0.20–0.40 × 2.08–2.98 m; mass 2–4 kg (median 3) |
| Conditions | normal (3), damaged (2), worn (2), new (1) |
| Benchmark scenarios (core) | open and traverse (8); mean difficulty 1.0/5 |
| Standards / references | no code; OSHA / food-safety guidance |
| Hard for a robot because | Deformable strips that wrap around the body and obscure vision; no mechanism, but contact along the whole body. |

## Families × mechanism kinds

Rows are families in hierarchy order, columns the mechanism *kinds* (a kind groups catalogue models: e.g. `surface_overhead` = LCN 4040, Norton 1600, residential light …).  A cell is the number of doors of that family carrying that kind; `none` is omitted.  The same matrices drive the *Relationships* panel of the Hierarchy page.

### Families × closer kinds

| family | auto operator full | auto operator low energy | concealed overhead | electromagnetic hold | floor spring | gas strut | gate | pneumatic | spring hinge | surface overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `swing_single` | · | 3 | 18 | 6 | 10 | · | · | 6 | 25 | 148 |
| `swing_double` | · | 2 | 3 | 7 | 6 | · | · | · | · | 30 |
| `dutch` | · | · | · | · | · | · | · | · | · | · |
| `saloon` | · | · | · | · | · | · | · | · | 12 | · |
| `automatic_swing` | 4 | 6 | · | · | · | · | · | · | · | · |
| `cold_storage` | · | · | · | · | · | · | · | · | · | 5 |
| `stall` | · | · | · | · | · | · | · | · | · | · |
| `gate_swing` | · | · | · | · | · | · | 17 | · | · | · |
| `baby_gate` | · | · | · | · | · | · | 5 | · | · | · |
| `ship_watertight` | · | · | · | · | · | · | · | · | · | · |
| `vault` | · | · | · | · | · | · | · | · | · | · |
| `blast` | · | · | · | · | · | · | · | · | · | · |
| `pivot` | · | · | · | · | 12 | · | · | · | · | · |
| `sliding_single` | · | · | · | · | · | · | · | · | · | · |
| `sliding_bypass` | · | · | · | · | · | · | · | · | · | · |
| `automatic_sliding` | · | · | · | · | · | · | · | · | · | · |
| `gate_sliding` | · | · | · | · | · | · | · | · | · | · |
| `elevator` | · | · | · | · | · | · | · | · | · | · |
| `bifold` | · | · | · | · | · | · | · | · | · | · |
| `accordion` | · | · | · | · | · | · | · | · | · | · |
| `garage_sectional` | · | · | · | · | · | · | · | · | · | · |
| `garage_tiltup` | · | · | · | · | · | · | · | · | · | · |
| `rollup` | · | · | · | · | · | · | · | · | · | · |
| `revolving` | · | · | · | · | · | · | · | · | · | · |
| `turnstile_tripod` | · | · | · | · | · | · | · | · | · | · |
| `turnstile_fullheight` | · | · | · | · | · | · | · | · | · | · |
| `hatch_floor` | · | · | · | · | · | 4 | · | · | · | · |
| `hatch_ceiling` | · | · | · | · | · | 4 | · | · | · | · |
| `pet_door` | · | · | · | · | · | · | · | · | · | · |
| `strip_curtain` | · | · | · | · | · | · | · | · | · | · |

### Families × latch kinds

| family | deadlatch | dogs | electric bolt | gravity bar | hook | magnetic | mortise latch | multi bolt | rim latch | roller | slide bolt | tubular latch | vertical rods |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `swing_single` | 85 | · | · | 7 | · | · | 63 | · | 39 | 3 | · | 183 | 4 |
| `swing_double` | · | · | · | · | · | · | 3 | · | 2 | · | · | 17 | 27 |
| `dutch` | · | · | · | · | · | · | · | · | · | · | · | 12 | · |
| `saloon` | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `automatic_swing` | 3 | · | · | · | · | · | 3 | · | 1 | · | · | 1 | · |
| `cold_storage` | · | · | · | · | · | 10 | · | · | · | 5 | · | · | · |
| `stall` | · | · | · | · | · | · | · | · | · | · | 15 | · | · |
| `gate_swing` | · | · | · | 17 | 6 | · | 4 | · | · | · | 6 | · | · |
| `baby_gate` | · | · | · | · | 10 | · | · | · | · | · | · | · | · |
| `ship_watertight` | · | 10 | · | · | · | · | · | · | · | · | · | · | · |
| `vault` | · | 2 | · | · | · | · | · | 6 | · | · | · | · | · |
| `blast` | · | 5 | · | · | · | · | · | 1 | · | · | · | · | · |
| `pivot` | · | · | · | · | · | 5 | 1 | · | · | · | · | · | · |
| `sliding_single` | · | · | 3 | 6 | 15 | · | · | · | · | · | 3 | · | · |
| `sliding_bypass` | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `automatic_sliding` | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `gate_sliding` | · | · | · | · | · | · | · | · | · | · | 2 | · | · |
| `elevator` | · | · | 8 | · | · | · | · | · | · | · | · | · | · |
| `bifold` | · | · | · | · | · | 8 | · | · | · | · | · | · | · |
| `accordion` | · | · | · | · | · | 6 | · | · | · | · | · | · | · |
| `garage_sectional` | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `garage_tiltup` | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `rollup` | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `revolving` | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `turnstile_tripod` | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `turnstile_fullheight` | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `hatch_floor` | · | · | · | · | · | · | · | · | · | · | 2 | · | · |
| `hatch_ceiling` | · | · | · | · | · | · | · | · | · | · | 2 | · | · |
| `pet_door` | · | · | · | · | · | 12 | · | · | · | · | · | · | · |
| `strip_curtain` | · | · | · | · | · | · | · | · | · | · | · | · | · |

### Families × lock kinds

| family | card reader | chain | child lock cover | deadbolt double | deadbolt single | delayed egress | dogs | electric strike | hook lock | interlock | jam stuck | keyed cylinder | keypad code | mag lock | multipoint | night latch | padlock | privacy button | slide bolt | swing bar guard | thumbturn only | vault wheel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `swing_single` | 18 | 4 | 8 | 4 | 24 | 11 | · | 11 | · | · | 12 | 20 | 28 | 20 | 4 | 4 | 9 | 42 | 15 | 1 | 16 | · |
| `swing_double` | · | · | · | · | 6 | 5 | · | · | · | · | · | · | · | 8 | 3 | · | 1 | · | 4 | · | 3 | · |
| `dutch` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `saloon` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `automatic_swing` | 2 | · | · | · | 2 | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | 1 | 1 | · |
| `cold_storage` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 4 | · | · | · | · | · |
| `stall` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 5 | · | · | · |
| `gate_swing` | · | · | · | 2 | · | · | · | 1 | · | · | · | · | · | · | · | · | 9 | · | 7 | · | · | · |
| `baby_gate` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `ship_watertight` | · | · | · | · | · | · | 10 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `vault` | · | · | · | · | · | · | 2 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 6 |
| `blast` | · | · | · | · | · | · | 5 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 |
| `pivot` | · | · | · | · | · | · | · | · | · | · | · | · | · | 4 | · | · | · | · | · | · | 4 | · |
| `sliding_single` | · | · | · | · | · | · | · | 1 | 28 | · | · | 4 | · | 2 | · | · | 3 | · | 6 | · | · | · |
| `sliding_bypass` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `automatic_sliding` | · | · | · | · | · | · | · | 4 | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `gate_sliding` | · | · | · | · | · | · | · | 3 | · | · | · | · | · | · | · | · | 3 | · | 2 | · | · | · |
| `elevator` | · | · | · | · | · | · | · | · | · | 8 | · | · | · | · | · | · | · | · | · | · | · | · |
| `bifold` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `accordion` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `garage_sectional` | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | 3 | · | 3 | · | · | · |
| `garage_tiltup` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | 1 | · | · | · |
| `rollup` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | 3 | · | · | · |
| `revolving` | · | · | · | · | · | · | · | 3 | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `turnstile_tripod` | · | · | · | · | · | · | · | · | · | · | · | · | · | 7 | · | · | · | · | · | · | · | · |
| `turnstile_fullheight` | · | · | · | · | · | · | · | · | · | · | · | · | · | 6 | · | · | · | · | · | · | · | · |
| `hatch_floor` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | 2 | · | · | · |
| `hatch_ceiling` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | 2 | · | · | · |
| `pet_door` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 4 | · | · | · |
| `strip_curtain` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |

### Families × operator kinds

| family | card lever | cremone | flush pull | gate latch fork | handleset | hasp | hook lock slider | keypad deadbolt | keypad lever | knob | lever | lift latch | paddle | panic crossbar | panic touchbar | pull | push button screen | push plate | ring pull | slide bolt handle | t handle | thumb latch | wheel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `swing_single` | 18 | · | · | · | 8 | 3 | · | 9 | 19 | 96 | 160 | · | 9 | 4 | 42 | 39 | 7 | 12 | 5 | · | · | 7 | · |
| `swing_double` | · | 3 | · | · | 3 | · | · | · | · | 3 | 11 | · | 2 | 2 | 30 | 14 | · | 5 | 3 | · | · | · | · |
| `dutch` | · | · | · | · | 2 | · | · | · | · | 4 | 6 | · | · | · | · | · | · | · | · | · | · | · | · |
| `saloon` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | · |
| `automatic_swing` | 2 | · | · | · | · | · | · | · | · | · | 5 | · | · | · | 1 | 1 | · | 1 | · | · | · | · | · |
| `cold_storage` | · | · | · | · | · | · | · | · | · | · | 15 | · | · | · | · | · | · | · | · | · | · | · | · |
| `stall` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 4 | · | · | · | 11 | · | · | · |
| `gate_swing` | · | · | · | 12 | · | 3 | · | · | · | 2 | 2 | 6 | · | · | · | · | · | · | 4 | 6 | · | 5 | · |
| `baby_gate` | · | · | · | · | · | · | · | · | · | · | · | 10 | · | · | · | · | · | · | · | · | · | · | · |
| `ship_watertight` | · | · | · | · | · | · | · | · | · | · | 6 | · | · | · | · | · | · | · | · | · | · | · | 4 |
| `vault` | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · | · | · | · | · | · | 6 |
| `blast` | · | · | · | · | · | · | · | · | · | · | 5 | · | · | · | · | · | · | · | · | · | · | · | 1 |
| `pivot` | · | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | 14 | · | · | · | · | · | · | · |
| `sliding_single` | · | · | 41 | · | · | · | 15 | · | · | · | 2 | · | · | · | · | 36 | · | · | 4 | · | · | · | · |
| `sliding_bypass` | · | · | 28 | · | · | · | · | · | · | 3 | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `automatic_sliding` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `gate_sliding` | · | · | · | · | · | 3 | · | · | · | · | · | · | · | · | · | 5 | · | · | · | 2 | · | · | · |
| `elevator` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `bifold` | · | · | · | · | · | · | · | · | · | 24 | · | · | · | · | · | 6 | · | · | · | · | · | · | · |
| `accordion` | · | · | 3 | · | · | · | · | · | · | 3 | · | · | · | · | · | 6 | · | · | · | · | · | · | · |
| `garage_sectional` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 9 | · | · | · | · | 6 | · | · |
| `garage_tiltup` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | 4 | · | · |
| `rollup` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 9 | · | · | 3 | · | · | · | · |
| `revolving` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 6 | · | 3 | · | · | · | · | · |
| `turnstile_tripod` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `turnstile_fullheight` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `hatch_floor` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | 8 | · | · | · | · |
| `hatch_ceiling` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | 2 | · | · | · | · |
| `pet_door` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `strip_curtain` | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |

### Families × hinge kinds

| family | butt | cam lift | concealed | continuous | double action | flap pin | gravity pivot | lift off | pivot center | pivot offset | rising butt | rotor | spring | strap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `swing_single` | 333 | · | 13 | 35 | · | · | · | 4 | 12 | 8 | 16 | · | 3 | 16 |
| `swing_double` | 46 | · | · | 10 | · | · | · | 3 | 3 | 6 | · | · | · | 8 |
| `dutch` | 12 | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `saloon` | · | · | · | · | 12 | · | · | · | · | · | · | · | · | · |
| `automatic_swing` | 6 | · | · | 2 | · | · | · | · | 1 | 1 | · | · | · | · |
| `cold_storage` | · | 15 | · | · | · | · | · | · | · | · | · | · | · | · |
| `stall` | · | · | · | · | · | · | 15 | · | · | · | · | · | · | · |
| `gate_swing` | 3 | · | · | · | · | · | · | · | · | · | · | · | · | 37 |
| `baby_gate` | 10 | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `ship_watertight` | 10 | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `vault` | 8 | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `blast` | 6 | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `pivot` | · | · | · | · | · | · | · | · | 15 | 5 | · | · | · | · |
| `sliding_single` | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `sliding_bypass` | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `automatic_sliding` | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `gate_sliding` | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `elevator` | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `bifold` | 30 | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `accordion` | · | · | · | 12 | · | · | · | · | · | · | · | · | · | · |
| `garage_sectional` | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `garage_tiltup` | · | · | · | · | · | · | · | · | · | 7 | · | · | · | · |
| `rollup` | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `revolving` | · | · | · | · | · | · | · | · | · | · | · | 15 | · | · |
| `turnstile_tripod` | · | · | · | · | · | · | · | · | · | · | · | 10 | · | · |
| `turnstile_fullheight` | · | · | · | · | · | · | · | · | · | · | · | 10 | · | · |
| `hatch_floor` | 10 | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `hatch_ceiling` | 8 | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `pet_door` | · | · | · | · | · | 15 | · | · | · | · | · | · | · | · |
| `strip_curtain` | · | · | · | 8 | · | · | · | · | · | · | · | · | · | · |

### Mechanism kinds shared by the most families

| mechanism | kind | families | doors |
|---|---|---:|---:|
| operator | `pull` | 15 (swing_single, swing_double, automatic_swing, stall, pivot, sliding_single…) | 156 |
| hinge | `butt` | 12 (swing_single, swing_double, dutch, automatic_swing, gate_swing, baby_gate…) | 482 |
| lock | `slide_bolt` | 12 (swing_single, swing_double, stall, gate_swing, sliding_single, gate_sliding…) | 54 |
| lock | `padlock` | 11 (swing_single, swing_double, cold_storage, gate_swing, sliding_single, gate_sliding…) | 40 |
| operator | `lever` | 11 (swing_single, swing_double, dutch, automatic_swing, cold_storage, gate_swing…) | 217 |
| operator | `knob` | 7 (swing_single, swing_double, dutch, gate_swing, sliding_bypass, bifold…) | 135 |
| operator | `ring_pull` | 7 (swing_single, swing_double, gate_swing, sliding_single, rollup, hatch_floor…) | 29 |
| latch | `slide_bolt` | 6 (stall, gate_swing, sliding_single, gate_sliding, hatch_floor, hatch_ceiling) | 30 |
| lock | `electric_strike` | 6 (swing_single, gate_swing, sliding_single, automatic_sliding, gate_sliding, revolving) | 23 |
| lock | `mag_lock` | 6 (swing_single, swing_double, pivot, sliding_single, turnstile_tripod, turnstile_fullheight) | 47 |
| hinge | `continuous` | 5 (swing_single, swing_double, automatic_swing, accordion, strip_curtain) | 67 |
| hinge | `pivot_offset` | 5 (swing_single, swing_double, automatic_swing, pivot, garage_tiltup) | 27 |
| latch | `magnetic` | 5 (cold_storage, pivot, bifold, accordion, pet_door) | 41 |
| latch | `mortise_latch` | 5 (swing_single, swing_double, automatic_swing, gate_swing, pivot) | 74 |
| operator | `push_plate` | 5 (swing_single, swing_double, saloon, automatic_swing, revolving) | 24 |
| hinge | `pivot_center` | 4 (swing_single, swing_double, automatic_swing, pivot) | 31 |
| latch | `tubular_latch` | 4 (swing_single, swing_double, dutch, automatic_swing) | 213 |
| lock | `thumbturn_only` | 4 (swing_single, swing_double, automatic_swing, pivot) | 24 |
| closer | `auto_operator_low_energy` | 3 (swing_single, swing_double, automatic_swing) | 11 |
| closer | `floor_spring` | 3 (swing_single, swing_double, pivot) | 28 |
| closer | `surface_overhead` | 3 (swing_single, swing_double, cold_storage) | 183 |
| hinge | `rotor` | 3 (revolving, turnstile_tripod, turnstile_fullheight) | 35 |
| hinge | `strap` | 3 (swing_single, swing_double, gate_swing) | 61 |
| latch | `dogs` | 3 (ship_watertight, vault, blast) | 17 |
| latch | `gravity_bar` | 3 (swing_single, gate_swing, sliding_single) | 30 |

## Cross-check against real-world classification

### EN 12519 door terminology → DoorBench

| EN 12519 / trade term | DoorBench | notes |
|---|---|---|
| single-leaf hinged (single-action) door | `swing_single` | handing (left / right), push / pull side in `spec.hinge`, `spec.robot` |
| double-leaf door, pair (active / inactive leaf, double egress) | `swing_double` | astragal, flush bolts / cane bolts / removable mullion in `leaf.inactive_leaf` |
| double-action (double-swing) door | `saloon` | `kinematics.both_ways`; spring hinges |
| stable door | `dutch` | joining bolt |
| pivot door | `pivot` (+ storefront pivots inside `swing_single`) | T-03 |
| sliding door: pocket, surface (barn), patio (lift-slide not modelled), bypass, telescopic (not modelled) | `sliding_single`, `sliding_bypass` | |
| folding door: bi-fold, multi-fold (concertina) | `bifold`, `accordion` | exterior folding-sliding walls missing (T-15) |
| revolving door | `revolving` | ANSI/BHMA A156.27 |
| turnstile, full-height turnstile, speed gate | `turnstile_tripod`, `turnstile_fullheight`; speed gates missing (T-12) | EN 17352 |
| up-and-over door (canopy / retractable), sectional door | `garage_tiltup`, `garage_sectional` | DASMA 102 |
| roller shutter / rolling grille | `rollup` | DASMA 108 |
| hatch, trapdoor, roof access | `hatch_floor`, `hatch_ceiling` | |
| power-operated pedestrian door (swing, sliding) | `automatic_swing`, `automatic_sliding` | A156.10 / A156.19 / EN 16005 |
| lift landing door | `elevator` | ASME A17.1 / EN 81-20 |
| watertight / weathertight door | `ship_watertight` | SOLAS II-1/13, ISO 6042 |
| vault door, blast door | `vault`, `blast` | UL 608, ASTM F2247 |
| gate (pedestrian, vehicle; swing / sliding / cantilever) | `gate_swing`, `gate_sliding` | pool barriers: ISPSC §305 |
| safety barrier (child) | `baby_gate` | ASTM F1004 / EN 1930 |
| toilet partition door | `stall` | ADA §604.8 |
| strip curtain, pet flap | `strip_curtain`, `pet_door` | no standard |

### Hardware standards → `doorbench/hardware.py` kinds

| standard | scope | DoorBench kinds |
|---|---|---|
| ANSI/BHMA A156.1, EN 1935 | butt hinges | hinge `butt`, `rising_butt`, `lift_off` |
| A156.26 | continuous hinges | `continuous` |
| A156.17 | self-closing hinges and pivots | `spring`, `double_action`, `gravity_pivot`, `cam_lift`, `pivot_center`, `pivot_offset` |
| A156.2 (bored), A156.13 (mortise), EN 12209 | latches and locksets | latch `tubular_latch`, `deadlatch`, `mortise_latch`; operator `lever`, `knob`, `paddle`, `handleset` |
| A156.5, A156.36 | auxiliary locks (deadbolts, night latches) | lock `deadbolt_single`, `deadbolt_double`, `night_latch`, `thumbturn_only`, `multipoint` |
| A156.3, UL 305, EN 1125, EN 179 | exit devices, panic / emergency hardware | operator `panic_touchbar`, `panic_crossbar`; latch `rim_latch`, `vertical_rods`; lock `delayed_egress` |
| A156.4, EN 1154, EN 1155 | door closers, electrically powered hold-open | closer `surface_overhead`, `concealed_overhead`, `floor_spring`, `electromagnetic_hold` |
| A156.10, A156.19, EN 16005 | power-operated pedestrian doors | closer `auto_operator_full`, `auto_operator_low_energy`; `kinematics.actuator` |
| A156.25, EN 14846 | electrified locking (maglock, strike, bolt, readers) | lock `mag_lock`, `electric_strike`, `card_reader`, `keypad_code`; latch `electric_bolt` |
| A156.14 | sliding and folding door hardware | operator `flush_pull`, `hook_lock_slider`; lock `hook_lock`; rollers / tracks in `kinematics` |
| A156.16 | auxiliary hardware (flush bolts, coordinators, stops) | `leaf.inactive_leaf.lock`, `kinematics.stop`, extras |
| A156.27 | revolving doors | `revolving` (speed governor, breakout) |
| EN 17352 | entrance control (turnstiles, speed gates) | `turnstile_*` (ratchet, credential) |
| ASME A17.1 | hoistway door interlocks | latch `elevator_interlock`, lock `interlock` |
| DASMA 102 / 108, UL 325 | sectional / rolling doors, operators | `garage_*`, `rollup` (counterbalance, opener) |
| SOLAS II-1/13 | watertight doors | operator `dog_lever`, `wheel`; latch `dogs` |
| UL 608, EN 1143-1 | vault doors | operator `wheel`; latch `multi_bolt`; lock `vault_wheel` |
| ISPSC §305, ASTM F1908 | pool gates | operator `lift_latch` (MagnaLatch), closer `gate` |
| ASTM F1004, EN 1930 | child safety gates | `baby_gate` |

## Tasks vs scenarios

`spec.task` (and the manifest `task` chip) is the legacy 9-value vocabulary of `taxonomy.TASKS`; the benchmark uses
the scenario names of `doorbench/benchmark/scenarios.py`.  Cross-tabulation over the 1000 doors:

| legacy `task` | doors | primary benchmark scenario | comment |
|---|---:|---|---|
| `open_and_traverse` | 280 | `open_and_traverse` | identical |
| `unlock_open_traverse` | 140 | `unlock_and_traverse` | identical |
| `locked_recognize` | 85 | `locked_recognize` | identical |
| `hold_and_pass` | 100 | `open_and_traverse` | self-closing doors; the benchmark has no hold-and-pass scenario (hold-open exists only in the human suite) |
| `open_only` | 116 | `open_and_traverse` (+ `open_then_close`) | benchmark always traverses |
| `peek` | 81 | `open_and_traverse` | no peek scenario |
| `close` | 86 | `open_and_traverse` (+ `open_then_close`, `close_only`) | 16 of 86 have `close_only` |
| `traverse_open` | 56 | `open_and_traverse` (1 × `locked_recognize`) | no start-open traverse scenario |
| `push_through` | 56 | `open_and_traverse` (43), **`locked_recognize` (12)**, `unlock_and_traverse` (1) | the 12 credential-locked turnstiles contradict their "push through" task (T-11) |

