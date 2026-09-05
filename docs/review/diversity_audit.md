# DoorBench takeover: dataset diversity and specification audit

Audit date: 2026-09-04 (local). Read-only inspection of **all 1,000 generated specs and all 1,000 model JSON files**, the manifest, original `prompt`, `handoffs/README.md`, taxonomy, sampler, material catalogue, geometry builders and material exporters. This report does not substitute for the maintainer's visual review or a fresh physics gate run.

Baseline: assets generated `2026-09-04T11:00:16`, dataset version `0.1.0`, seed `20260903`; checkout HEAD `eb4729a86c7d2e964cd766d28c2a5666d74e100a` at audit start. The working branch is `codex/takeover-inspection`; other takeover work may change source after these observations. No assets were edited or regenerated for this audit.

- Manifest SHA-256: `262b30494c5ad4b6aa5586feca551228802f07940bef4972af97453077bcbb87`.
- SHA-256 of concatenated `spec.json` bytes in sorted path order: `26a2c168db9ba725de5662cc89bb8f60b30dcc7299b68e389b1a31f783188367`.
- Exactly 1,000 specs, 1,000 distinct IDs, and the same ID set as the manifest. All family and explicit context quotas match the taxonomy.
- The manifest claims 1,000 signed off. This audit independently finds material, geometry and label defects that the existing sign-off does not exclude.

## Assessment

The dataset has substantial catalogue breadth: 30 families, 52 family/context combinations, 73 slab constructions, 52 panel styles and 59 primary operator models. The requested broad categories—hinged exit and entry doors, sliders, closets, overhead doors, turnstiles, heavy security doors, paper screens, locked doors and pet doors—are present. The current assets nevertheless do **not** meet the original prompt's highest-fidelity requirement. Numerical breadth overstates realized visual and mechanical diversity: transparent inserts retain opaque backing, advertised shapes collapse to primitive substitutes, textures are references without files or bindings, and some sampled descriptions contradict the construction.

Correct construction and conditional sampling before expanding the number of doors. Preserve this baseline and version any regenerated dataset; fixes to mass, material, geometry or taxonomy change the benchmark input.

## Findings requiring correction

### D-01 · P1 · Glazed apertures and louvers are surface decoration on solid slabs

**171 assets** have generated `Glazing` geoms and an opaque, visible `Leaf slab` in the same body. There are 173 assets with any `Glazing` geom; two have non-opaque backing. **44 assets** contain `Louver slat` geoms with a `Leaf slab` in the same body. In **43 of these**, every slat is entirely buried within the slab's thickness, including 13 bifold assets. For a typical 35 mm leaf, the slab half-thickness is 17.5 mm while the rotated slats extend only about 16.7 mm from the center plane. They are invisible, explaining the maintainer's flat-looking louver bifolds. These are counts from the actual generated `model.json`, not an estimate from labels. `db0524_bifold` is a separate mismatch: `mirror_bypass` plus `louver_half` takes the monolithic-glass early return and emits zero slats.

[`add_leaf_geoms`](../../doorbench/geometry/common.py#L493) calls `_slab_boxes` for the full visible slab, then overlays panes at a thickness of `leaf.thickness + 0.0016 m`. It reads `glazing.thickness` into `gt` but does not use `gt` for the pane geometry. Louver slats are likewise added after a full slab at lines 511–535. This prevents the intended open visual aperture and makes window geometry thickness unrelated to the specified glass thickness. A simplified collision proxy is reasonable in a performance tier, but the full visible asset needs real frame members, cutouts and glazing of the stated thickness.

Examples: `db0021_swing_single`, `db0027_swing_single`, `db0036_swing_single`, `db0019_swing_double`, `db0025_swing_double`. Review both faces and a grazing side view. Add a full-tier gate that samples the center of every intended glazed/louver aperture, checks the surrounding visual geometry, and verifies pane thickness separately from any documented collision proxy.

### D-02 · P1 · 27 monolithic glass doors have inconsistent thickness and mass

**27 `glass_frameless_*` assets** have `leaf.glazing.thickness != leaf.thickness`. The geometry builds one monolithic glass box at `leaf.thickness` in [`common.py:473`](../../doorbench/geometry/common.py#L473), while [`physics.leaf_mass`](../../doorbench/physics.py#L19) removes a glazing-area fraction from that slab and adds glass of a different thickness. Thus the material composition used for mass does not match the uniform glass geometry. This is an internal consistency defect using the project's own glass density, not an external standards judgement.

**Seven of those 27** also disagree with the nominal slab catalogue thickness:

| IDs | Slab | Leaf thickness | Glazing thickness |
|---|---|---:|---:|
| `db0067`, `db0274`, `db0766` (all `sliding_single`) | `glass_frameless_10` | 35 mm | 6 mm |
| `db0125`, `db0373`, `db0681`, `db0903` (all `sliding_single`) | `glass_frameless_12` | 45 mm | 19 mm |

Example `db0067_sliding_single`: slab plus glazing mass is **26.91 kg**, but its rendered uniform glass dimensions imply **126.42 kg** at the repository's 2,500 kg/m³. Hardware is excluded from both quantities. `db0039_swing_single` renders 19 mm glass implying **92.65 kg**, but its slab plus glazing mass is **60.22 kg**. For `db0125_sliding_single`, the total reported mass including hardware is **213.49 kg**, driven by an inconsistent 45/19 mm construction rather than the stated 12 mm slab.

Affected family counts: sliding single 10, swing single 5, sliding bypass 5, swing double 3, automatic sliding 3, automatic swing 1. Root causes include fixed thicknesses in [`gen_sliding_single`](../../doorbench/spec.py#L751), the common glazing assignment at line 831, fixed glass thickness in `gen_swing_single`, `gen_swing_double` and `gen_sliding_bypass`, and automatic slider glazing at line 1315. Gate the monolithic material volume/mass against one authoritative thickness; do not treat a monolithic glass leaf as a separate slab plus window.

### D-03 · P2 · At least 28 use-case descriptions contradict sampled construction

The following conservative text/construction checks flag **28 distinct assets**. This is a lower bound, not a complete semantic validator. `use_case` is sampled independently from slab and panel choices.

| Incompatible description | Count | Example |
|---|---:|---|
| `porch screen door` without a screen construction | 4 | `db0824_swing_single`: steel entry slab, six-panel style |
| `mirrored wardrobe doors` without mirror construction | 5 | `db0020_sliding_bypass`: hollow-core molded six-panel leaves |
| Description says shoji, slab is not shoji | 10 | `db0621_sliding_bypass`: louvered wood; `db0447_sliding_bypass`: mirror |
| Description says fusuma, slab is shoji | 1 | `db0226_sliding_single` |
| `cold room sliding door` without cold-storage construction | 4 | `db0966_sliding_single`: open steel bar grille, brush seal |
| Description explicitly says louvered, panel is not louvered | 4 | `db0071_bifold`: flat MDF |

Sources: [`gen_swing_single`](../../doorbench/spec.py#L274), [`gen_sliding_single`](../../doorbench/spec.py#L798), [`gen_sliding_bypass`](../../doorbench/spec.py#L876), [`gen_bifold`](../../doorbench/spec.py#L901). Derive descriptions from the completed construction, or sample a compatible product archetype before selecting variants. Do not count these descriptions as evidence of category coverage.

### D-04 · P2 · Cold-storage panel and glazing metadata disagree on 7/15 doors

All four cold-storage leaves labeled `glass_vision` have no glazing: `db0080`, `db0409`, `db0549`, `db0894`. Three leaves labeled `steel_flush` do have glazing: `db0673`, `db0865`, `db0937`. IDs all end `_cold_storage`.

[`gen_cold_storage:1293`](../../doorbench/spec.py#L1293) samples `panel_style` and the decision to add `glazing_for("glass_vision", ...)` independently. Sample the vision choice once and derive both fields from it. Gate panel/glazing agreement before export.

### D-05 · P1 · Conditional option cycles can return an option no longer allowed

[`Balanced.pick`](../../doorbench/spec.py#L29) caches a shuffled list by key and does not revalidate it when the caller supplies a different set of allowed levels. In [`gen_pivot:742`](../../doorbench/spec.py#L742), `mortise_euro` is allowed only with `lever_l_shape`, but a leftover cycle item can escape that restriction. Generated **`db0161_pivot` has `pull_ladder_full` and `mortise_euro`**. This pairs a fixed pull with a positive mortise latch without the intended actuating lever.

This is a general sampler defect, not a request to replace this one ID manually. Make conditional keys include their eligibility state, or sample from a compatibility matrix and revalidate every choice. Other dynamically changing `Balanced.pick` option sets should be audited at the same time.

### D-06 · P2 · 24 spring-closer specs name a non-spring hinge; five automatic closer specs have no motor actuator

Of 25 `spring_hinge_single` closers, **24** are paired with a hinge other than `spring_single`. Examples: `db0736_swing_single` uses `concealed_soss`, `db0645_swing_single` uses `piano`, and `db0062_swing_single` uses `butt_35_worn`. `add_closer` returns without geometry for the `spring_hinge` kind at [`common.py:1134`](../../doorbench/geometry/common.py#L1134). These require a visible and correctly identified source of spring torque rather than an unrelated hinge label plus passive force.

**Five** `auto_low_energy` closer specs have neither `kinematics.actuator` nor generated `model.meta.actuators`: `db0158_swing_double`, `db0608_swing_double`, `db0224_swing_single`, `db0411_swing_single`, `db0689_swing_single`. Motor construction is tied to the `automatic_swing` family in [`build.py:54`](../../doorbench/build.py#L54), and benchmark automatic operation is family-gated in [`runner.py:287`](../../doorbench/benchmark/runner.py#L287). If these represent disconnected automatic operators, explicitly label that state; otherwise generate and drive the mechanism from hardware capability rather than family name.

### D-07 · P2 · Surface and silhouette catalogue diversity is not fully realized

**208 leaf finishes reference 16 distinct texture IDs**, but `assets/textures` contains **zero files**. [`viewer/src/scene.ts:76`](../../viewer/src/scene.ts#L76) builds color/roughness/metalness materials with no texture maps. [`export/mjcf.py:173`](../../doorbench/export/mjcf.py#L173) likewise omits texture binding, and [`export/usd.py:178`](../../doorbench/export/usd.py#L178) sets constant PreviewSurface values. Current wood species, weathering, canvas and leather therefore have less appearance variety than their material names imply. The wall and floor texture references are also not loaded. Finish tags should distinguish available assets from future references.

The full geometry also reduces several advertised shape categories to substitutes:

- Six `arched_top` leaves remain rectangular slabs with plank lines (`common.py:536`); no arched silhouette is built.
- Twenty-one `2_panel_arch` and six `carved_ornate` entries use the same rectangular two-panel layout (`panels.py:99`, `panels.py:109`).
- Four `glass_oval`, two `glass_fan`, and two `porthole` entries use rectangular glazing regions (`panels.py:53–63` and the box-only glazing loop).
- Eight `ornamental_scroll` gates generate straight vertical cylinders using the same branch as bar grilles (`common.py:452`).
- Thirty `mesh_panel` leaves use a translucent filled box for infill (`common.py:465`); insect screen, chain link and expanded metal are not geometrically distinguished there.

Representative IDs: `db0147_swing_single` (arched), `db0139_swing_single` (carved), `db0126_gate_swing` (ornamental scroll). These observations are from generated primitive types and source, with image review remaining the maintainer's responsibility. Add actual silhouettes and infill/texture construction before advertising these as realized visual distinctions.

### D-08 · P2 · Repetition is concentrated in the minority families

There are **825 distinct discrete design signatures** when grouped by family, context, slab, panel style, operator, latch, lock, closer and hinge. The remaining 175 instances repeat one of those signatures; 88 signatures have multiple instances. This deliberately excludes size, finish, condition and kinematic parameters, so it is **not** a claim of 175 identical assets.

Grouping by family and the five hardware model fields yields **576 signatures**, with 424 repeated instances. Grouping by all physical input blocks (`family`, `context`, `leaf`, `opening`, `hinge`, `kinematics`, `operator`, `latch`, `lock`, `closer`, `seal`, `condition`, `extras`) yields **998 unique configurations**. One exact physical-input group has three members: `db0393_turnstile_tripod`, `db0516_turnstile_tripod`, `db0946_turnstile_tripod`; the differences are identity/use-case/robot start metadata.

| Family | Assets | Distinct discrete design signatures |
|---|---:|---:|
| Swing single | 440 | 432 |
| Swing double | 76 | 73 |
| Sliding single | 100 | 67 |
| Sliding bypass | 35 | 15 |
| Bifold | 30 | 20 |
| Accordion | 12 | 8 |
| Strip curtain | 8 | 1 |
| Elevator | 8 | 1 |

The latter two still vary sizes/kinematics/conditions. This grouping identifies where to invest further structural diversity. A family-aware train/test split should group repeated physical configurations and closely related templates together; random ID splits can test memorization of the same mechanism. Report macro averages by family alongside the aggregate: swing single alone contributes 44% of the doors, and 22 families have 15 or fewer examples, collectively only 241 doors.

### D-09 · P2 · Family names hide important motion approximations

All 18 `garage_sectional` assets use `sectional_vertical_lift`; their decorative sections share one rigid body and a vertical slider in [`build_vertical`](../../doorbench/geometry/other.py#L594). The 15 rollups also use a translating curtain body. The taxonomy documents the garage approximation, but these are not evidence of section-by-section curved-track articulation or curtain coiling. The 12 accordion assets share a rigid-panel, alternating equality-coupled mechanism; two have a canvas slab but no flexible fabric model. These distinctions matter for the original request for fully working doors and interior mechanisms.

Name the actual modeled motion in the viewer and benchmark metadata. Treat curved-track sectional, coiling curtain, flexible flap/curtain and physically constrained guided folds as explicit capabilities with individual gates. New category expansion candidates include telescoping sliders, balanced doors, folding/sliding storefront systems, speed gates with retracting flaps, and high-speed flexible industrial doors. This candidate list is a proposal, not a claim that the original prompt enumerated each item or that this audit established an exhaustive worldwide taxonomy.

### D-10 · P1 · All eight screen doors are assigned solid-aluminum slab mass

The maintainer flagged `db0083_swing_single` at 149.03 kg and `db0101_swing_single` at 152.51 kg during image review. The cause is deterministic: [`materials.py:346`](../../doorbench/materials.py#L346) declares both `screen_alu` and `screen_wood` as `monolithic=True`, with `core_material="insect_screen"`. That material has solid aluminum density **2,700 kg/m³**. [`SlabConstruction.area_density:229`](../../doorbench/materials.py#L229) returns `density * full_frame_thickness` immediately for a monolithic slab and ignores `extra_area_density` and the actual open mesh/frame construction.

For `db0083`, 28 mm × 2,700 = **75.6 kg/m²**, multiplied by 0.914 × 2.134 m = **147.456 kg slab**, plus 1.57 kg hardware = **149.026 kg**. Its own catalogue source note says approximately **8–10 kg** for a wood screen door. For `db0101`, 25 mm × 2,700 = **67.5 kg/m²**, multiplied by 0.914 × 2.438 m = **150.412 kg slab**, plus 2.10 kg hardware = **152.512 kg**. Its catalogue source note says approximately **4–6 kg total**. Those notes are approximate references, not a new calibrated replacement formula; the order-of-magnitude disagreement is indisputable within the project's own data.

All affected IDs are swing singles: `db0083`, `db0101`, `db0369`, `db0450`, `db0794`, `db0795`, `db0827`, `db0886`. Reported totals range **126.48–165.73 kg**. Correct with perimeter frame geometry and screen wire/open-area mass, with an area-density regression check. The resulting hinge friction, inertia, damage thresholds and benchmark opening difficulty must be regenerated. A blanket density tweak would also affect unrelated uses of `insect_screen` and should not be used as a shortcut.

### D-11 · P1 · Sectional high-lift wall opening and fixed tilt-up pivot fail visual/functional expectations

The maintainer's new closed inspection renders show sectional doors filling only the lower half of an opening. This is **not a pose bug**. `scripts/render_inspection.py` uses `m.qpos0` for the closed view and resolves equalities without advancing simulation. The generated section geometry is complete: `db0148_garage_sectional` has five contiguous slabs from **z=0.010 to 2.450 m**, matching its 2.44 m leaf. However, `build_vertical` deliberately cuts the wall from the ground to `Ho + Hh + 0.08`—**5.01 m** in this asset—so the area above the closed leaf is open sky. Tracks rise through that artificial opening. All 18 sectional assets use this branch. The same source explains `db0175` and `db0198`.

I independently viewed the maintainer's [`db0148` front image](../../out/inspection/db0148_garage_sectional/front.jpg) and confirmed that this geometry matches the observed empty upper half. Even a deliberately chosen vertical-lift installation requires an enclosure/upper wall appropriate to its stowed door and should expose the actual passage aperture separately. This is a structural geometry correction, not a camera adjustment.

All seven tilt-up doors use a hinge body fixed at `Hh * 0.5`, with no linkage that raises the pivot during opening (`build_horizontal`, `garage_tiltup` branch). The near-horizontal leaf therefore stays around **half the closed-door height**. The maintainer's waist-height platform observation follows directly from the joint transform. Model the actual support linkage/track and ensure the open leaf provides the intended clear passage before this family is considered functional.

### D-12 · P2 · Subdivision replays complete panel layouts on every part

All 12 Dutch doors pass the original `panel_style` and glazing block to both half-leaf builders, replacing only height ([`hinged.py:715`](../../doorbench/geometry/hinged.py#L715)). The six glazed Dutch assets therefore produce **twice the stated glazing count**: `db0118`/`db0626` have 12 panes for `count=6`; `db0333`/`db0391`/`db0460`/`db0906` have 18 for `count=9`. IDs end `_dutch`. The style also puts the same full-door design on the lower and upper halves, confirming the maintainer's repetitive visual observation. Explicitly model upper/lower panel designs and sum their actual glazing areas/counts for physics.

The same issue affects the six `sectional_long_windows` garage assets: the complete four-window style is applied independently to every section, producing **16 glazing geoms per asset** while the spec says `count=4`. `db0175_garage_sectional` has four windows on each of four sections, including the bottom one. Derive panel placement in whole-door coordinates and clip it to the real sections, or assign per-section designs deliberately.

## Coverage inventory

Counts are doors using a primary field, not hardware instances. Inactive-leaf hardware, extras, far-side operators and geometry can implement an item absent from the primary catalogue field.

| Dimension | Used / defined | Defined entries absent from this primary field |
|---|---:|---|
| Families | 30 / 30 | None |
| Slab constructions | 73 / 76 | `cardboard`, `polycarbonate_panel`, `solid_wood_cherry` |
| Panel styles | 52 / 56 | `2_panel`, `canvas_flap`, `glass_sidelite_style`, `steel_half_glass` |
| Primary operators | 59 / 61 | `cane_bolt_drop`, `mail_slot` |
| Primary latches | 26 / 31 | `ball_catch`, `electric_strike`, `garage_slide_lock`, `mag_lock_1200`, `mag_lock_600` |
| Locks | 27 / 29 | `hasp`, `mortise_deadbolt` |
| Closers | 17 / 17 | None |
| Hinges | 30 / 30 | None |
| Conditions | 9 / 9 | None |
| Spec task labels | 9 / 9 | None |
| Finish kinds | 7 / 12 | `anodized`, `galvanized`, `laminate`, `mirror`, `paper` |

Absence of a finish tag does not mean absence of its material: 13 mirror slabs use the `glass` finish kind, and 21 paper-face constructions use `natural`, `stain` or `weathered`. Seven mail-slot extras and inactive cane bolts are present despite missing primary operator selections. The distinction between physical material, surface treatment and visual substrate needs a consistent schema.

### Original prompt coverage

| Requested category | Evidence in this dataset |
|---|---|
| Exit doors and push-to-exit bars | 55 fire-egress swing singles, 26 commercial panic pairs; 79 primary panic operators overall |
| Sliding and closet doors | 100 sliding single, 35 bypass, 30 bifold, 12 accordion |
| Entry doors | 60 residential exterior swing singles, plus storefront/automatic/pivot families |
| Pull-up/overhead doors | 18 sectional, 7 tilt-up, 15 rollup; motion limitations in D-09 |
| Turnstiles | 10 tripod, 10 full-height |
| Heavy doors | 8 vault, 6 blast, 10 ship watertight; mass field spans 0.1375–1,459.412 kg per modeled leaf/unit across all families |
| Light paper doors | 15 shoji and 6 fusuma slab constructions |
| Locked doors | 239 engaged locks: 141 with robot-side release, 98 without; seven disengaged turnstiles retain `robot_side_release=false` |
| Keypad locks | 28 locks of `keypad_code` kind: 10 four-digit, 10 six-digit, 8 mechanical. Only 15 currently have a stored code because code assignment follows engaged-lock selection; the other 13 do not |
| Pet doors | 15 standalone pet doors plus 14 `pet_flap` extras |
| Glass and mirrors | 300 specs have `leaf.glazing`; 125 glass-face constructions include 13 mirrors. D-01/D-02 limit validity |

### Families and realized variant counts

| Family | Doors | Slabs | Primary operators | Panel styles |
|---|---:|---:|---:|---:|
| swing_single | 440 | 43 | 36 | 37 |
| swing_double | 76 | 17 | 15 | 13 |
| dutch | 12 | 5 | 4 | 5 |
| saloon | 12 | 6 | 2 | 5 |
| pivot | 20 | 8 | 5 | 4 |
| sliding_single | 100 | 18 | 11 | 15 |
| sliding_bypass | 35 | 7 | 5 | 7 |
| bifold | 30 | 5 | 2 | 5 |
| accordion | 12 | 4 | 3 | 1 |
| revolving | 15 | 1 | 3 | 1 |
| turnstile_tripod | 10 | 1 | 1 | 1 |
| turnstile_fullheight | 10 | 1 | 1 | 1 |
| garage_sectional | 18 | 3 | 3 | 4 |
| garage_tiltup | 7 | 2 | 2 | 3 |
| rollup | 15 | 2 | 4 | 2 |
| pet_door | 15 | 2 | 1 | 1 |
| hatch_floor | 10 | 3 | 3 | 3 |
| hatch_ceiling | 8 | 3 | 3 | 3 |
| ship_watertight | 10 | 2 | 2 | 4 |
| vault | 8 | 1 | 3 | 2 |
| blast | 6 | 1 | 3 | 2 |
| gate_swing | 40 | 4 | 9 | 7 |
| gate_sliding | 10 | 4 | 3 | 2 |
| baby_gate | 10 | 3 | 1 | 2 |
| stall | 15 | 3 | 2 | 1 |
| strip_curtain | 8 | 1 | 1 | 1 |
| cold_storage | 15 | 2 | 1 | 2 |
| automatic_sliding | 15 | 3 | 1 | 1 |
| automatic_swing | 10 | 7 | 6 | 5 |
| elevator | 8 | 1 | 1 | 1 |

### Materials, conditions and motion

| Face-material family | Count |
|---|---:|
| Wood | 307 |
| Metal | 306 |
| Composite | 172 |
| Glass | 125 |
| Plastic | 38 |
| Mesh | 27 |
| Paper | 21 |
| Fabric | 4 |

Face classification is `slab_face_material(SLABS[id]).family`, not the core's family: hollow-core paper honeycomb does not make a paper-faced door, and mineral fire cores do not imply stone-looking doors. The four fabric-faced entries are two padded leather and two canvas accordion constructions.

- Finishes: paint 306, powder coat 240, natural 149, glass 125, bare metal 66, stain 64, weathered 50.
- Most common named finish colors/materials: white 118, glass clear 112, grey 102, beige 53, pine 46, stainless 45. Finish RGBA/roughness variations add continuous diversity.
- Most common slab constructions: hollow metal 18ga 67, hollow core 60, solid-core particleboard 58, solid pine 46, solid MDF 40, solid oak 29, hollow-core molded 28, hollow metal 16ga 28, storefront aluminum 24, louvered wood 22.
- Conditions: normal 332, new 246, worn 204, old/dry 59, damaged 45, rusty 45, well-oiled 28, swollen 23, sagging 18. Condition names are not a verified visual damage inventory. The independent condition selection even labels `db0794_swing_single`'s aluminum screen construction as swollen wood; validate damage mechanisms against constituent materials.
- Kinematic type: vertical-axis hinge 716, horizontal slide 168, horizontal-axis hinge 48, rotor 35, vertical slide 33. The folding families are included under vertical hinges.
- Primary benchmark scenario: open-and-traverse 761, unlock-and-traverse 141, recognize-locked 98. The nine `spec.task` labels do not imply nine equally represented primary benchmark tasks.
- Automatic actuator annotations and generated actuator lists: 33 assets, comprising 15 automatic sliders, 10 automatic swing doors and 8 elevators. Powered state is true for 27 and false for 6.

## Reproduce the audit

Run this from the repository root with the environment's Python. It does not mutate files. It recomputes quotas, every primary-field distribution, absent catalogue entries, material counts, grouped diversity and the decisive defect counts/IDs from the generated inputs. Read the source references above for the interpretation of each flag.

```python
import collections as C
import hashlib
import json
from pathlib import Path
from doorbench import taxonomy as T, materials as M, hardware as H

paths = sorted(Path("assets/doors").glob("*/spec.json"))
specs = [json.loads(p.read_text()) for p in paths]
models = [json.loads((p.parent / "model.json").read_text()) for p in paths]
manifest = json.loads(Path("assets/manifest.json").read_text())
assert len(specs) == len({s["id"] for s in specs}) == 1000
assert {s["id"] for s in specs} == {d["id"] for d in manifest["doors"]}
assert C.Counter(s["family"] for s in specs) == {
    f: info[0] for f, info in T.FAMILIES.items()
}
for family, quotas in [
    ("swing_single", T.SWING_SINGLE_CONTEXTS),
    ("swing_double", T.SWING_DOUBLE_CONTEXTS),
    ("sliding_single", T.SLIDING_SINGLE_CONTEXTS),
    ("gate_swing", T.GATE_SWING_CONTEXTS),
]:
    assert C.Counter(s["context"] for s in specs if s["family"] == family) == quotas
print("manifest_sha256", hashlib.sha256(Path("assets/manifest.json").read_bytes()).hexdigest())
print("specs_sha256", hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest())

dimensions = {
    "family": (T.FAMILIES, lambda s: s["family"]),
    "slab": (M.SLABS, lambda s: s["leaf"]["slab"]),
    "panel": (T.PANEL_STYLES, lambda s: s["leaf"]["panel_style"]),
    "finish": (T.FINISH_KINDS, lambda s: s["leaf"]["finish"]["kind"]),
    "condition": (T.CONDITIONS, lambda s: s["condition"]),
    "task": (T.TASKS, lambda s: s["task"]),
}
for field, registry in [("operator", H.OPERATORS), ("latch", H.LATCHES),
                        ("lock", H.LOCKS), ("closer", H.CLOSERS), ("hinge", H.HINGES)]:
    dimensions[field] = (registry, lambda s, field=field: s[field]["model"])
for name, (registry, getter) in dimensions.items():
    counts = C.Counter(map(getter, specs))
    print(name, dict(counts.most_common()), "missing", sorted(set(registry) - counts.keys()))
print("face_materials", C.Counter(M.slab_face_material(M.SLABS[s["leaf"]["slab"]]).family for s in specs))
print("extras", C.Counter(e for s in specs for e in s["extras"]))
print("kinematics", C.Counter(s["kinematics"]["type"] for s in specs))
print("lock_states", C.Counter((s["lock"]["engaged"], s["lock"]["robot_side_release"]) for s in specs))
print("benchmark", C.Counter(s["benchmark"]["primary_scenario"] for s in specs))

hardware = ["operator", "latch", "lock", "closer", "hinge"]
def signature(s):
    return (s["family"], s["context"], s["leaf"]["slab"],
            s["leaf"]["panel_style"], *(s[k]["model"] for k in hardware))
print("design_signatures", len({signature(s) for s in specs}))
print("hardware_signatures", len({(s["family"], *(s[k]["model"] for k in hardware)) for s in specs}))
physical = ["family", "context", "leaf", "opening", "hinge", "kinematics",
            "operator", "latch", "lock", "closer", "seal", "condition", "extras"]
groups = C.defaultdict(list)
for s in specs:
    groups[json.dumps({k: s[k] for k in physical}, sort_keys=True)].append(s["id"])
print("physical_signatures", len(groups), "repeated", [v for v in groups.values() if len(v) > 1])
for family in T.FAMILIES:
    subset = [s for s in specs if s["family"] == family]
    print("family_variants", family, len(subset), len({s["leaf"]["slab"] for s in subset}),
          len({s["operator"]["model"] for s in subset}),
          len({s["leaf"]["panel_style"] for s in subset}), len({signature(s) for s in subset}))

def flags(name, predicate):
    found = [s["id"] for s in specs if predicate(s)]
    print(name, len(found), found)

flags("glass_nominal_thickness", lambda s: s["leaf"]["slab"].startswith("glass_frameless")
      and s["leaf"]["thickness"] not in M.SLABS[s["leaf"]["slab"]].typical_thickness)
flags("glass_mixed_thickness", lambda s: s["leaf"]["slab"].startswith("glass_frameless")
      and s["leaf"].get("glazing")
      and s["leaf"]["glazing"]["thickness"] != s["leaf"]["thickness"])
flags("cold_storage_disagreement", lambda s: s["family"] == "cold_storage"
      and (s["leaf"]["panel_style"] == "glass_vision") != bool(s["leaf"].get("glazing")))
flags("stale_pivot_latch", lambda s: s["family"] == "pivot"
      and s["latch"]["model"] == "mortise_euro" and s["operator"]["model"] != "lever_l_shape")
flags("spring_hinge_disagreement", lambda s: s["closer"]["model"] == "spring_hinge_single"
      and s["hinge"]["model"] != "spring_single")
flags("automatic_closer_without_actuator", lambda s: s["closer"]["model"].startswith("auto_")
      and not s["kinematics"].get("actuator"))

label_checks = {
    "screen": lambda s: s["use_case"] == "porch screen door" and not s["leaf"]["slab"].startswith("screen"),
    "mirror": lambda s: "mirrored" in s["use_case"] and s["leaf"]["slab"] != "mirror_bypass",
    "shoji": lambda s: "shoji" in s["use_case"].lower() and s["leaf"]["slab"] != "shoji",
    "fusuma": lambda s: "fusuma" in s["use_case"].lower() and s["leaf"]["slab"] != "fusuma",
    "cold_room": lambda s: s["context"] == "cell_industrial" and "cold room" in s["use_case"] and s["leaf"]["slab"] != "cold_storage_100",
    "louver": lambda s: "louvered" in s["use_case"] and "louver" not in s["leaf"]["panel_style"],
}
for name, predicate in label_checks.items():
    flags("label_" + name, predicate)
flags("label_union", lambda s: any(p(s) for p in label_checks.values()))

backed_glazing, backed_louvers, any_glazing = [], [], []
for s, model in zip(specs, models):
    opaque_backed = louver_backed = has_glazing = False
    for body in model["bodies"]:
        geoms = body["geoms"]
        glazed = any(g.get("part_label") == "Glazing" for g in geoms)
        slab = [g for g in geoms if g.get("part_label") == "Leaf slab"]
        opaque = any(g["visual"] and model["materials"][g["material"]]["rgba"][3] >= .99 for g in slab)
        has_glazing |= glazed
        opaque_backed |= glazed and opaque
        louver_backed |= bool(slab) and any(g.get("part_label") == "Louver slat" for g in geoms)
    if opaque_backed: backed_glazing.append(s["id"])
    if louver_backed: backed_louvers.append(s["id"])
    if has_glazing: any_glazing.append(s["id"])
for name, ids in [("opaque_backed_glazing", backed_glazing), ("backed_louvers", backed_louvers), ("any_glazing_geom", any_glazing)]:
    print(name, len(ids), ids)
textures = C.Counter(s["leaf"]["finish"]["texture"] for s in specs if s["leaf"]["finish"].get("texture"))
print("texture_references", sum(textures.values()), len(textures), dict(textures))
print("texture_files", len([p for p in Path("assets/textures").rglob("*") if p.is_file()]))

# Follow-up checks requested by the maintainer's visual inspection.
flags("screens", lambda s: s["leaf"]["slab"] in ("screen_alu", "screen_wood"))
for s, model in zip(specs, models):
    if s["leaf"]["slab"] in ("screen_alu", "screen_wood"):
        print("screen_mass", s["id"], s["physics"]["mass"], M.SLABS[s["leaf"]["slab"]])
    if s["family"] in ("dutch", "garage_sectional") and s["leaf"].get("glazing"):
        actual = sum(g.get("part_label") == "Glazing" for b in model["bodies"] for g in b["geoms"])
        print("subdivision_glazing", s["id"], s["leaf"]["glazing"]["count"], actual)
    if s["family"] == "garage_sectional":
        slabs = [g for b in model["bodies"] for g in b["geoms"] if g["name"].startswith("section_") and g.get("part_label") == "Leaf slab"]
        print("sectional_closed_extent", s["id"], min(g["pos"][2] - g["size"][2] for g in slabs),
              max(g["pos"][2] + g["size"][2] for g in slabs),
              "wall_cutout_top", s["opening"]["height"] + s["leaf"]["height"] + .08)
buried = []
for s, model in zip(specs, models):
    for body in model["bodies"]:
        slabs = [g for g in body["geoms"] if g.get("part_label") == "Leaf slab"]
        slats = [g for g in body["geoms"] if g.get("part_label") == "Louver slat"]
        if not slabs or not slats: continue
        fully_inside = []
        for g in slats:
            w, x, y, z = g["quat"]
            half_y = (abs(2*(x*y+w*z))*g["size"][0]
                      + abs(1-2*(x*x+z*z))*g["size"][1]
                      + abs(2*(y*z-w*x))*g["size"][2])
            fully_inside.append(any(abs(g["pos"][1]-slab["pos"][1]) + half_y <= slab["size"][1] for slab in slabs))
        if all(fully_inside):
            buried.append(s["id"])
            break
print("buried_louvers", len(buried), buried)
```

## Handoff status

- Delivered: complete generated-input coverage audit, reproducible counts, twelve actionable defect classes, prioritization, examples and implementation references.
- Manifest sign-off: 1,000/1,000 claimed by existing assets; fresh sign-off/clearance/tests/viewer build were not run because this task only inspected and documented current inputs.
- No renders produced in this subtask; one maintainer-generated sectional image independently inspected to establish its source. No per-door visual sign-off claimed. The parent maintainer performs the complete image review.
- Only this report was written. No dataset/code/README/TASKS changes, branch switches, commits, pushes or cloud operations were performed by this subtask.
