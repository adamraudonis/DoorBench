# Moving mass and framed-glass construction

The independent inventory covers all 1,000 current models and their locally restored published counterparts. It found two distinct defects: one panel's mass was being shared across several physical panels, and 54 framed-glass doors used their approximately 45 mm frame depth as solid glass thickness. These defects can cancel numerically in a paired storefront door; a plausible assembly total alone does not establish correct construction.

## Scope and independent checks

`doorbench/mechanical_audit.py` derives projected material area from actual IR panel geometry. Overlapping collision rectangles are unioned within a panel body, while separate bodies retain their separate areas. Revolving wings are counted inside their single rotor body. A tripod's existing three-arm formula is not multiplied again. Framed glass is checked from actual primitive material volumes and densities, ignoring mass overrides. Hollow turnstile-arm assumptions are reported separately from solid collision-hull volume.

The initial stock-area audit found 209 models whose entire moving mass was less than 90% of the estimated panel material alone: 30 bifolds, 35 bypasses, 76 swing pairs, nine saloon pairs, eight strip curtains, five elevators, 12 accordions, 15 revolving doors, nine automatic sliders and ten full-height turnstiles. This threshold misses some count errors where unusually heavy hardware masks the missing panel mass. The later, stricter audit also checks the published thick-glass volume directly; it flags 247 published models. These are different screening measurements, not counts from repeated benchmark evaluations.

The current inventory reports no mass/material/operator-inventory findings under those checks. This is **not a mechanical certification**. In particular, the separate review identified 33 overhead-door models with rigid-motion approximations and hatch support/operation gaps. An inventory cannot establish collision-free motion, access, linkage forces, structural strength, glass breakage or realistic behavior.

## Correct mass contract

`physics.mass.total_kg` now means the complete moving assembly. `reference_unit` preserves the explicitly labeled mass of the declared reference panel. `per_body` lists each physical panel body, its actual authored width/height, material mass and installed hardware budget. `dynamics_mass_kg` and `per_body_dynamics` keep the mass carried by each driven panel separate from assembly mass.

- Swing pairs, saloon pairs, bypass lanes and bi-parting sliders contain multiple full panels. The inactive member of a swing pair has its own material and hinge budget, with no fabricated operator/closer installation.
- Folding widths describe individual panels. Their small hinge/jamb gaps are accounted for; shared hinge hardware and each leading-panel pull are allocated once.
- Dutch halves divide one full-height slab. Strip-curtain widths describe individual overlapping strips. An embedded pet flap has its own material body and the containing slab loses the cutout area.
- Sectional garage panels divide one full-door envelope. Revolving/full-height turnstile rotors contain all their wings. The tripod reference formula already includes all three arms.

`mass_reconciliation.py` applies these explicit panel targets. It does not distribute one assembly budget by the sum of visual and collision geometry volumes. Proxy operator hardware consumes its owner's catalog budget, with the calibration scale recorded. New bodies declared in `meta.mechanism_mass_bodies` instead retain their actual authored material/assembly inertia and add their full mass to the reported total. This prevents a 3.71 kg steel lifting arm from being reduced to approximately 0.84 kg by an obsolete hardware allowance. The per-body bill of materials is recorded as `geometry_backed_bodies_kg`; repeated reconciliation does not add it twice. Empty or zero-mass declarations fail. Leaf material is never reduced to pay for another leaf's mechanism. A documented 1 gram per-link minimum reserve remains for otherwise unbudgeted proxy coordinates.

Mass metadata distinguishes `slab_and_catalogue_hardware_budget_kg` and `geometry_backed_mechanisms_kg`. Analytical panel friction/inertia and initial counterbalance sizing use the former; they are not silently relabeled as complete articulated force curves. Native generalized forces include all bodies. A linkage's counterbalance calibration must account for each link's potential energy explicitly, as the revised tilt-up builder does.

The shape distribution of hardware inertia remains an authored approximation. Rotor frame sections and turnstile tubes also retain explicitly limited construction assumptions: the 38 mm arm tubes use 1.5 mm walls, and the full-height column uses a 4 kg allocation per wing. Solid primitive collision envelopes do not certify those internal wall dimensions. A detailed rotor bill of materials remains appropriate future work.

Hinge/roller helpers choose `per_body_dynamics` by their joint's body name. A bypass lane uses its own carried mass, not all three panels. A shared rotor or vertical-lift body uses its complete moving assembly. Historical benchmark results remain historical; this change does not recompute or relabel their outcomes.

Representative moving masses from the initial panel/glazing correction, in kilograms. Later rebuilt overhead/hatch mechanisms add explicit hardware and supersede those families' intermediate totals. The subsequent flexible-strip reconstruction also replaces DB0037's intermediate body count and removes its fixed hanger from moving mass; see [the strip audit](strips.md).

| Door | Published | Corrected | Physical leaf bodies |
|---|---:|---:|---:|
| DB0004 bifold | 14.31 | 28.32 | 2 |
| DB0008 bypass | 37.21 | 111.62 | 3 |
| DB0010 storefront pair | 101.68 | 75.52 | 2 |
| DB0023 patio slider | 205.33 | 51.93 | 1 |
| DB0031 saloon | 11.54 | 23.08 | 2 |
| DB0037 strip curtain | 2.88 | 10.02 | 5 |
| DB0053 elevator | 33.37 | 66.74 | 2 |
| DB0065 accordion | 12.93 | 46.90 | 6 |
| DB0066 revolving | 91.25 | 268.65 | 1 rotor / 3 wings |
| DB0095 Dutch | 21.86 | 21.68 | 2 halves |
| DB0130 automatic slider | 100.61 | 93.03 | 2 |
| DB0148 sectional garage | 65.10 | 65.10 | 1 rigid assembly |
| DB0187 full-height turnstile | 16.77 | 38.90 | 1 rotor / 3 wings |
| DB0202 tripod | 7.54 | 7.54 | 1 rotor / 3 arms |

## Framed glass

The 54 affected models comprise nine swing pairs, 18 single sliders, 12 automatic sliders and 15 single swing doors. Each now contains separate hollow frame-wall primitives, true glass plies and perimeter seating gaskets. There is no full-depth slab hidden across the glazing. The same original construction profile supplies both geometry and its material bill of materials:

| Construction | Frame | Actual glazing |
|---|---|---|
| Medium-stile storefront | Aluminum; 88.9 mm stiles/top, 165.1 mm bottom, authored 2.4 mm walls | One 6 mm pane |
| Wide-stile storefront | Aluminum; 127 mm stiles/top, 165.1 mm bottom, authored 2.4 mm walls | 6 mm + 13 mm sealed gap + 6 mm |
| Patio slider | PVC; authored 127 mm stiles/bottom, 100 mm top, 3 mm walls | 4 mm + 11 mm sealed gap + 4 mm |

Frame depth remains independent of glass thickness. Glass mass uses 2,500 kg/m³, supported by [Pilkington ATS-129](https://www.pilkington.com/-/media/pilkington/site-content/usa/window-manufacturers/technical-bulletins/ats-129---properties-of-soda-lime-silica-float-glass.pdf). The storefront sightline/depth classes are informed by [Kawneer standard entrances](https://www.kawneer.com/products/doors-and-entrances/190-350-500-standard-entrances/) and [350/500 IR entrance dimensions](https://www.kawneer.com/products/doors-and-entrances/350-500-ir-entrances/). The profiles are original simplified construction, not OEM CAD or a reproduction of certified products. Patio profile dimensions, tube walls, glass-ply selection, gaskets and spacers are authored parameters; thermal, pressure, impact, extrusion reinforcement and seal-life performance are not certified.

Manufacturer capacities are per physical door: [Johnson 111SD](https://johnsonhardware.com/111sd-sliding-bypass-door-hardware) explicitly states a maximum weight per door. [VT Industries' door-weight guidance](https://vtonline.vtindustries.com/graphics/ProductUpdates_Tab.pdf) also explains that cutouts, hardware and reinforcement change nominal area-based weights. Neither source justifies dividing a panel's physical material weight by a neighboring panel count.

Three accordion flush pulls (DB0177, DB0292, DB0830) now use actual recessed cutouts, collision sidewalls and grips. DB0373's correct 12 mm frameless pane exposed a real conflict between its exterior cup back and interior hook lever; the exterior cup is positioned 160 mm below that sweep. These four cases pass their full native QA independently.

## Reproduction and evidence

```sh
python scripts/audit_mechanical_inventory.py --published assets \
  --out out/mechanical-foundations/mass-audit/inventory-final.json
python -m pytest tests/test_mass_scope.py tests/test_mechanical_inventory.py \
  tests/test_sliding_mechanics.py tests/test_sliding_tracks.py -q
python scripts/generate_dataset.py \
  --out out/mechanical-foundations/mass-glass-final \
  --formats mjcf,json --no-thumbs --workers 4
```

The 47 focused tests include all 1,000 IR assembly budgets, six independently compiled MuJoCo mass round trips, actual per-lane friction, duplicated-geometry mass-allocation rejection, all 54 glazing constructions, and a native ray inside an empty 11 mm insulating-glass gap. Source-bound inventory evidence is in `out/mechanical-foundations/mass-audit/inventory-final.json` (SHA256 `555223dc55dae7ef539f1343ef9cba3cfc92e86d1d16bcf651b6c8a4311a367e`). Its source snapshot did not change during the audit. The output folder also retains earlier reports so initial defects are not erased by later source changes.

Full QA records are kept separately from inventory screening. The intermediate all-1,000 pass signed off 983 models, with all 1,000 mass checks and all 54 framed-glass models passing. Sixteen failures belonged to concurrently evolving gate/Suffolk mechanisms; DB0373 was the seventeenth and is repaired as described above. The final generation's manifest and per-door `qa.json` records supersede intermediate counts for their exact exported revision. No appearance renders or benchmark reruns were used for this work.

The follow-up `mass-glass-final` attempt signed off 991/1,000, including all 54 framed-glass models and the four repaired pull cases. Its other nine outcomes were four fork/post clearance failures and five gate-Suffolk build errors from concurrently evolving source. All 995 models that built passed the mass gate. This receipt remains an intermediate integration run; its directory name does not imply that the other mechanism work was frozen.
