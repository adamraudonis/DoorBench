# DoorBench takeover review

I read the original [`prompt`](../../../prompt), the [handoff](../../../handoffs/README.md), and the relevant generator, physics, export and viewer code. I personally visually screened **every one of the 1,000 doors**, using 50 contact sheets with closed front, closed reverse, hardware and open views: **4,000 reviewed images**. Independent agents audited all specs/models and the physics gates, and checked the two repairs described below.

**The collection does not yet meet the original prompt's physical or visual fidelity bar.** It has useful breadth and working interaction infrastructure, but the current `signed_off` field means the implemented automated checks passed. It does not establish correct construction, plausible mass, complete mechanisms, realistic materials, or successful operation by a robot. I am not granting new physical or artistic approvals to these assets.

## Inspection evidence and limits

Browse the [50 contact sheets](atlas/CONTACT_SHEETS.md). The [atlas index](atlas/index.json) enumerates all 1,000 IDs and the four source image hashes per door. [Page 1](atlas/page_01.jpg) begins the sequence; [page 50](atlas/page_50.jpg) ends it. The accompanying [door review ledger](door_review_ledger.json) records the complete screening coverage, page/cell location and applicable audit flags for each ID. Family observations below are my visual observations, with source checks used to distinguish modeling defects from rendering artifacts.

The atlas preserves the **pre-repair dataset** generated `2026-09-04T11:00:16`, from baseline commit `eb4729a86c7d2e964cd766d28c2a5666d74e100a`. Its manifest SHA-256 is `262b30494c5ad4b6aa5586feca551228802f07940bef4972af97453077bcbb87`. Regeneration changes that snapshot; use the repair images and final verification record for the repaired geometry.

The inspection renderer frames the full assembly and numerically solves connected arms while preserving coupled leaf motion. Open positions are **prescribed kinematic inspection poses**, not actuation demonstrations. Forty-nine baseline doors require moving beyond an engaged lock's range to show nominal travel; their images are marked hypothetical. Five cold-room mechanisms cannot satisfy their connection at open travel and are marked infeasible. These flags must not be interpreted as visually approved motion. Hardware images cannot expose every concealed or reverse-side component; dense mechanisms still require individual close-up and dynamic review. Screening at contact-sheet scale is sufficient to find broad construction problems, but is not an engineering certification of each part.

## Most consequential findings

Counts overlap and must not be added into a rejected-door total. The [specification audit](../diversity_audit.md) contains the detailed measurements and reproduction code; the [physics audit](../physics_audit.md) contains baseline mechanism measurements and QA coverage.

| Finding | Measured scope | Consequence / next correction |
|---|---:|---|
| Screen doors use solid-aluminum slab mass | 8 doors, 126.48–165.73 kg | Replace the construction mass model with frame and mesh quantities; regenerate inertia, friction and benchmark inputs. |
| Monolithic glass thickness differs between geometry and mass calculation | 27 doors | Use a single authoritative glass thickness and volume, including seven catalogue-thickness contradictions. |
| Glazing overlays opaque slabs; louvers are buried | 171 backed glazed assets; 43 fully buried louver assets | Build real full-tier apertures, stiles and rails; validate pane thickness and open regions. A 44th louver asset retains backing too. |
| Wall stops are suspended away from all static support | 130 stops | Model a mounted stop, stem/bracket and load path; add an attachment gate. `db0024` is 325 mm from the nearest support. |
| Sectional and tilt-up garage geometry/motion is implausible | 18 sectional; 7 tilt-up | Fix the artificial upper wall aperture and model the actual overhead support/track motion. Tilt-ups currently rotate at half-height. |
| Missing or contradictory mechanisms | 5 automatic closer specs without actuators; 24 spring-closer specs with non-spring hinge labels | Generate hardware from compatible capabilities and test its operation. Pair closers and internal closer mechanics need further work. |
| Named appearance variety is not realized | 208 leaf finishes refer to absent textures | Add real materials/texture bindings and accurate silhouettes. Wood, weathering, mesh types, arches and ornate styles need construction-specific treatment. |
| Sampling/layout inconsistencies | At least 28 label conflicts; glazing-count mismatches on 6 Dutch and 6 sectional assets | Validate conditional choices; define panel layouts across the whole assembly rather than replaying them on each part. |

## What I saw across the collection

| Families / atlas pages | Personal visual assessment |
|---|---|
| Accordion, bifold, saloon (1, 3–4, 14–15) | Folding/swinging structures are recognizable. Many purported louver leaves look like flat slabs; source inspection confirmed buried slats. Closed accordion surfaces give little visual distinction between rigid folds and fabric. |
| Automatic sliding/swing and elevator (1–2, 6) | Head boxes and panel arrangements distinguish the broad types. Detailed suspension and powered operator construction are incomplete; opening renders alone do not establish automation. |
| Baby gates and swinging/sliding gates (2–3, 8–10) | Bars, frames and operator variants provide useful visible variety. Mesh/chain-link/ornamental names often reduce to filled panels or straight rods. |
| Cold storage, blast, ship, vault (4–5, 15, 50) | Thick leaves, release hardware and dogs are identifiable. Cold-room construction repeats strongly. The five rising-hinge closer defects are real; visible dogs still need individual holding/release checks. |
| Dutch doors (5–6) | Two independently opening halves are visible, but both repeat the full-door panel/window design. Source inspection confirmed doubled glazing counts on the glazed variants. |
| Garage sectional/tilt-up and rollup (6–8, 14) | Closed sectional leaves leave an artificial open bay above them. Tilt-ups form a roughly waist-height platform. Rollups translate a rigid curtain rather than coil. These are modeled approximations, not camera failures. |
| Floor/ceiling hatches and pet doors (10–13) | Broad forms are recognizable. Hatches need closer examination of strut/linkage attachment and force behavior; small flap variants are visually repetitive and do not establish flexible-material behavior. |
| Pivot, revolving and turnstiles (12–14, 49–50) | Motion families and main hardware silhouettes are recognizable. Repetition and primitive construction remain apparent; revolving shells and turnstile cages need detailed operational review. |
| Horizontal sliders and bypass doors (15–22) | Rail/hanger overhang is visible at open travel. The repair below corrects the travel envelope and modeled tread contact. Rail-only mechanisms remain incomplete. |
| Stalls and strip curtains (22–23) | Stalls read as partitions with hinged leaves. Strip curtains fan rigid strips upward; this is not flexible curtain deformation. |
| Double and single swing doors (23–49) | The strongest visible variety is in knobs, levers, paddles, pulls, panic bars and panel proportions. Flat materials, backed windows, hidden louvers, floating stops and incomplete closer arms recur. The implausible screen masses were obvious in the atlas labels and confirmed in source. |

## Diversity assessment

All **30 families**, **73/76 slab constructions**, **59/61 primary operators**, **30/30 hinges** and **17/17 closer labels** appear. This is substantial coverage of the original requested categories. It does not demonstrate that each named construction is faithfully realized. Forty-four percent of the dataset is single swing doors. There are 825 distinct discrete design signatures, with the remaining 175 instances repeating a signature while potentially varying dimensions, finishes or physics. Three tripod turnstiles have identical physical input blocks.

Prioritize structural correctness and compatibility before adding more IDs. Future evaluation should report per-family results and group closely related templates together when splitting train/test data. The previously reported benchmark scores remain historical baseline results; this takeover did not rerun training, certify those results on the repaired assets, or validate the old RunPod job.

## Repairs completed in this takeover

**Horizontal tracks:** rails now cover nominal travel with the correct center and lane. Flat-track hangers meet the rail, axle/strap geometry connects, standoffs reach the wall, stops meet terminal wheels and floor guides are mounted. The `db0079` rail was already 2.234 m long; its center and contact alignment were wrong. The new gate sweeps nominal unlocked travel and catches shortened rails, displaced wheels and misplaced stops. See the [repair report](../sliding_track_fix.md) and [before](../track_db0079_before.jpg)/[after](../track_db0079_after.jpg) images.

Floor-guide presence, mounting and engagement are checked on 40 doors / 60 stations, including the restored requested guides on 14 bypass doors. See the [bypass guide detail](../bypass_guides/db0008_station_detail.jpg).

This covers **168 horizontal sliders**. Actual modeled wheel contact is checked on **48 doors / 96 wheels**; the other **120 doors** are explicitly reported as rail-only coverage. This does not complete their suspension, bearings, wheel rotation or force transmission.

**Rising-hinge cold-room closers:** a passive vertical shoe accommodates the hinge's 12–13.3 mm rise. The mounted guide, sliding shoe block, forearm neck and pin provide visible connection geometry. Retaining lips/end caps are not yet modeled; the prismatic joint supplies that restraint, so this remains a simplified guide assembly. The actual cam rise and gravitational load are preserved. A generic connection-feasibility gate solves mechanisms across at least 25 positions and rejects endpoint separation of 1 mm or more. A fixed-shoe negative fixture reproduces the original failure; the viewer no longer waives these five geometry failures. This is a modeled slotted-shoe accommodation, not a claim of matching a specific manufacturer's assembly. Hydraulic closer force remains a joint-level approximation; the full internal closer mechanism is still outstanding.

See the [cold-room repair evidence and measurements](../rising_closer/verification.json) and the [open shoe detail](../rising_closer/db0188_90_shoe.jpg).

## Verification and continuation

Final regeneration passes all implemented checks for **1,000/1,000 doors**, and independent clearance reports **1,000/1,000 clean**. **387 Python tests** and **15 viewer tests** pass, together with viewer typecheck/build. Both new QA hooks pass on all doors; their substantive coverage is 168 sliders and 159 connect-loop mechanisms. The maximum connection residual is 9.962e-9 m. Detailed coverage and provenance are in the [verification record](verification.json). Baseline tests passed before edits: 351 Python tests and 15 viewer tests, plus viewer typecheck/build.

I also personally screened all **327 source-changed models** on [17 post-repair sheets](../takeover_after/CONTACT_SHEETS.md), reviewing another **1,308 images**. This includes all 173 geometry changes and 154 closer-connection description changes. The other 673 model hashes remain unchanged from the baseline screening. Broad construction and appearance defects remain visible; these repairs do not establish full physical or artistic approval.

The next work should first correct mass/construction and attachment defects, then complete overhead motion and missing hardware. Retain a separate human review state rather than promoting the existing automated `signed_off` field to artistic/physical approval. New gates must expose their coverage and omissions, and dataset changes must trigger regeneration and benchmark versioning.

Source, tests, reports and compact review media are on `codex/takeover-inspection`. Generated `assets/` are local verification outputs and are excluded from the commit. `README.md` and `TASKS.md` are unchanged.

Reproduce the inspection on an explicitly chosen generated dataset:

```sh
.venv/bin/python scripts/render_inspection.py --assets assets --out out/inspection-new --workers 6
.venv/bin/python scripts/inspection_atlas.py --assets assets --renders out/inspection-new --out out/inspection-new-atlas
```

The renderer records source hashes; the atlas rejects stale records and retains forced/unsolved pose flags. Generating images does not create personal review verdicts. Keep this baseline atlas intact when inspecting a regenerated dataset.
