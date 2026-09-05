# Blender appearance review

This review covers the optional Blender Cycles appearance layer requested after the original catalogue inspection. It adds independently selectable wall, floor, door finish and lighting recipes while retaining the generated model as the source of articulation and physical geometry. The existing full, simple and minimal physics exports remain available.

![Six actual Blender looks, including three material combinations on the same barn door](looks.jpg)

## Reproduce and inspect

See the [appearance guide](../../BLENDER_APPEARANCE.md) for batch rendering and the [vision-state guide](../../BLENDER_VISION_STATE.md) for rendering recorded MuJoCo observations. The local catalogue serves saved renders from `out/appearance`; source code alone does not publish these generated files to the live website.

```sh
.venv/bin/python -m doorbench appearance fetch-textures
.venv/bin/python scripts/render_appearance_examples.py
.venv/bin/python -m doorbench appearance render --doors all --out out/appearance --quality preview --width 480 --height 480 --resume
.venv/bin/python scripts/appearance_review.py --variant 0
```

Use `--quality photo` for the 96-sample path and choose the output resolution. Preview and photo use the same geometry, materials and lighting; preview uses 16 samples. Generated PNGs, packed Blender scenes and downloaded texture maps stay outside git. The compact contact sheets preserve actual rendered pixels and their full-image checksums.

## Visual findings

The scanned timber, surface relief, reflective metals, transmissive glass, softer illumination and continuous rooms substantially improve the catalogue's previous flat appearance. Wall and floor choices are independent; the same barn door can appear in plaster/limestone, brick/concrete and painted-wall/timber settings without regenerating its joints. Grain follows the moving body, and individual wooden rails, braces and frame headers use their member direction. Wreath proxies gain deterministic leaves and twigs inside the original decorative envelope.

Personal review caught and corrected oversized brick texture units, incorrectly directed header grain, procedural weathered-timber striping and severely underexposed outdoor gates. Outdoor exposure was checked using actual renders of six gates under four lighting modes, plus a rendered neutral-surface regression. Fine chain-link detail still becomes subpixel when a low-resolution image includes the full travel rail; inspect it at higher resolution or with a closer camera.

## Scope of approval

This is an appearance implementation and visual screening, not a claim that every source asset now matches a photograph or correct mechanical construction. The [takeover audit](../takeover/REVIEW.md) still applies: backed glazing/louvers, simplified garage and rollup motion, incomplete support/hardware, and mass inconsistencies require source-model repairs. Better shading does not repair these defects. Outdoor contexts are intentionally sparse, finite texture scans repeat, and automatic material combinations are not individually designed architectural interiors. Broad rectangular light reflections can dominate glass and mirrors; some surfaces still read as studio renders. Low-detail garage, accordion and elevator assemblies remain particularly dependent on source-model improvements.

The renderer adds tagged visual-only room surfaces and bounded decorative detail. Finish overrides are cosmetic: they do not change mass, contact or friction. Vision output is offline display-referred RGB from a full-tier door snapshot. Additional robot/person geometry, real-time policy-loop rendering, depth and segmentation outputs are not implemented. Saved catalogue images are fixed poses; the interactive viewer continues to control the simulation representation.

## Verification record

All 1,000 doors across 30 families rendered successfully at 480×480 with the 16-sample preview setting. The primary agent personally screened every preview across [50 contact sheets](CONTACT_SHEETS.md), and the [per-door ledger](screening.json) records the actual image hashes. This was contact-sheet screening for appearance, framing and broad defects; it does not establish detailed mechanical accuracy at every articulation state.

The independent [diversity audit](diversity.json) verifies the complete preview catalogue against the archived render jobs: all 3,000 door source files, 2,371 hardware mesh references, 30 texture maps and nine renderer modules match. It covers all 49 wall/floor pairs and 193 of 196 wall/floor/light combinations. The source collection contains 73 slab-material labels; the appearance resolver uses 11 automatic door finishes and 34 total surface presets across these renders. The 1,000 jobs use nine scanned material sets; the walnut example adds the tenth.

The [Blender/MuJoCo integration record](geometry_camera_validation.json) checks 320 source geometry instances over six states in three door families, including raw mesh placement. Four camera calibrations agree to within 0.00014 pixels. The Python suite passes 500 tests, including 58 focused state, snapshot, cache and pipeline checks. The viewer passes 15 tests, TypeScript checking and its production build.

All 14 higher-quality examples completed at 960×960 and 96 samples and were personally inspected at full size after the final renderer change. The [example ledger](examples.json) records their final image and job hashes. These include three independently selected looks for the same barn door. All three [packed Blender scenes](packed_scenes.json) reopen with their required image maps packed and verified, all 44 source visual objects retained, and zero orphan objects.

The final [verification record](verification.json) confirms 1,014 saved images, zero failed renders, all 1,000 catalogue IDs, and 42,289 source visual geometry instances. All output artifacts and current source/renderer hashes match their jobs. The frozen renderer is commit `82b4d204d88f584544c7373b50405de9b843665d`; later changes in this review commit affect documentation and QA wording only. Blender 5.2.1 LTS used Metal locally. Preview rendering took a median 1.56 seconds per door on this machine; this is a batch-render measurement, not interactive RL throughput.
