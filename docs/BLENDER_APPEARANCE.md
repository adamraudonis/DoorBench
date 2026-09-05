# Blender appearance layer

DoorBench's optional Blender Cycles renderer gives every generated door an independently configurable appearance. The articulated model remains the source of geometry and body poses. A small recipe chooses wall, floor, door finish and lighting. The full, simple and minimal physics exports continue to work without Blender.

This implementation uses Blender 5.2.1, scanned and procedural PBR materials, real-scale texture coordinates, submillimeter edge rounding, soft architectural lighting and AgX display output. Ten curated [Poly Haven](https://polyhaven.com/license) CC0 texture sets provide actual albedo, OpenGL normal and roughness maps for timber, plaster, brick, concrete, stone and tile. Other presets use built-in procedural shaders. These surfaces are appearance references, not measured BRDFs for the particular simulated door. Blender documents its display transforms in the [color-management manual](https://docs.blender.org/manual/en/4.5/render/color_management.html).

## Render a door or the whole collection

Run these commands from the repository root with an existing generated dataset:

```sh
.venv/bin/python -m doorbench appearance catalog
.venv/bin/python -m doorbench appearance fetch-textures

.venv/bin/python -m doorbench appearance render \
  --doors db0079_sliding_single --out out/barn-door-study \
  --wall wall_white_plaster --floor floor_oak \
  --door-finish wood_walnut --lighting daylight \
  --quality photo --width 1200 --height 1200 --save-blend

.venv/bin/python -m doorbench appearance render \
  --doors all --out out/appearance --quality photo --resume

.venv/bin/python -m doorbench appearance render \
  --doors families --variants 4 --seed 12 --out out/appearance-variants

# Fourteen art-directed examples, including three looks for the same barn door.
.venv/bin/python scripts/render_appearance_examples.py
```

`families` selects a stable representative of all 30 families. `all` includes every manifest entry. A comma-separated list selects exact IDs and rejects unknown IDs. Defaults are 960×960 and 96 samples with adaptive sampling and denoising. `--quality preview` uses 16 samples for inexpensive previews. `--validate-only` builds scenes and checks source coverage without rendering pixels; successful build validation is not a visual approval.

`fetch-textures` is the only command that accesses the network. It downloads 30 raster maps at 2k resolution, checks provider checksums, and writes a local provenance manifest under `out/appearance-textures`. The renderer automatically uses that manifest when present. Pass `--textures /path/to/manifest.json` for another cache or `--procedural-only` for the built-in fallback. Missing explicitly requested maps fail before rendering. The downloader follows Poly Haven's [public API terms](https://github.com/Poly-Haven/Public-API/blob/master/ToS.md), records source and license URLs, and downloads no executable Blender files. Render jobs work offline after the maps have been fetched.

Blender is discovered using `--blender`, `DOORBENCH_BLENDER`, PATH, or the standard macOS installation. `--device auto` uses available Metal hardware on macOS and otherwise CPU; `--device CPU` gives an explicit fallback. The first Metal render may spend considerable time compiling kernels. Nothing starts a cloud instance or purchases rendering capacity.

## Interchangeable recipes

The catalog currently provides seven wall finishes, seven floor finishes, 27 door/optical/hardware surface presets and four lighting arrangements. Available IDs are listed by the `catalog` command. A resolved recipe is JSON:

```json
{
  "schema_version": "1.0",
  "seed": 12,
  "wall": "wall_limestone",
  "floor": "floor_terrazzo",
  "door_finish": "wood_oak",
  "lighting": "overcast",
  "render_device": "auto"
}
```

Wall, floor and light selection use independent stable hashes of door ID, appearance seed and slot. Changing a floor override does not resample the walls. `--variants` changes the appearance seed. Automatic door finishes preserve the construction's material meaning and painted source colors; scanned timber uses the chosen scan's albedo. Explicit finish overrides provide cosmetic variants. Glass, mirrors, seals, metal fittings and timber lattice retain their own semantic material assignments. A wooden finish override does not turn a glass pane into wood.

Textures use meters rather than normalized object size. Scan dimensions come from provider metadata. Door grain follows its rigid body; separate braces, rails, stiles and planks use their own member axes. Color and normal coordinates follow articulation. Wall and floor coordinates remain fixed. Shared hardware meshes have per-object material slots, so two identical handles can have different finishes without modifying shared mesh data.

The older brick scan's nominal provider scale produced visibly oversized units. Its appearance scale is explicitly calibrated to approximately 225×75 mm brickwork modules, using the visible rows/columns and [manufacturer dimension guidance](https://www.wienerberger.co.uk/content/dam/wienerberger/united-kingdom/marketing/documents-magazines/technical/brick-technical-guidance-sheets/UK_MKT_DOC_Brickwork%20Dimension%20Tables.pdf). The library retains the provider's original scale alongside the override and its uncertainty. This is an art calibration, not a corrected field measurement. White plaster uses neutralized scan color for a clean finish; limewash retains stronger surface variation.

Appearance overrides do not change mass, friction or collision geometry. To change the actual door construction, regenerate its specification and physical exports as well. Mesh infill shaders can portray perforations on the source surface; this does not create simulated wire geometry. Submillimeter bevels round primitive edges inside their original envelope. Tagged room context provides surfaces for shadows and reflections outside the source assembly; it is render-only geometry.

Decorative wreaths replace the rendered ring proxy with deterministic foliage and twigs within the same decorative envelope. Their original simulated proxy and rigid attachment remain unchanged. These details are separately tagged and counted in render metadata.

## Vision data and simulation state

The [state bridge](BLENDER_VISION_STATE.md) captures authoritative MuJoCo body transforms and an optional calibrated camera. The Blender importer composes each native body transform with the authored local geometry transform. It does not animate doors by independently guessing joint angles or releasing locks. Compiled MuJoCo mesh transforms include a recentering transform; applying them directly to the original OBJ would be wrong.

```python
from doorbench.appearance.state import capture_mujoco_state
from doorbench.appearance.pipeline import render_trajectory
from doorbench.appearance.textures import load_texture_library

# In a simulation loop, capture selected observation times after env.step(...).
snapshots = [capture_mujoco_state(env.m, env.d, door_id=door_id,
                                camera={"name": "robot_view", "resolution": [640, 480]})]
render_trajectory("assets", door_id, snapshots, "out/vision-episode",
                  seed=42, quality="photo", wall="wall_concrete",
                  floor="floor_terrazzo", lighting="warehouse",
                  texture_library=load_texture_library("out/appearance-textures/manifest.json"))
```

Use an actual camera name from the loaded model or an explicit camera pose/intrinsic matrix; unknown cameras fail. The recipe, room, lighting and default camera stay constant throughout `render_trajectory`. Context and default camera are laid out from the authored reference pose; only door bodies and any explicitly supplied observation camera change per frame. Cycles rendering occurs outside the physics step. This is an offline observation/replay path, not a claim of real-time RL throughput.

The renderer currently consumes the door's full model. A snapshot containing additional robot or person bodies is rejected because their visual meshes have not been supplied; it never silently delivers a supposedly complete observation with the robot missing. A minimal-tier state cannot supply the internal bodies of a full visual model. Supply the full-tier snapshot for detailed vision rendering. RGB output is display-referred PNG; depth, segmentation and motion-blur labels are not generated by this implementation.

## Outputs and provenance

Each door/variant directory contains `rgb.png` and `render.json`, plus `scene.blend` when requested. The blend file contains the actual mesh, node networks and packed image maps and can be opened and edited directly in Blender. No external texture search is required.

The batch root contains `index.json`, `jobs.json`, `worker_results.json` and `blender.log`. Records include source spec/model/XML hashes, hardware OBJ hashes, renderer code hashes, scan provenance/map hashes, state hash, resolved recipe, camera matrix, resolution, sample count, device and Blender version. Resume requires a matching job hash and checksummed output files. Changed geometry, shaders, maps, state or settings invalidate the cached job. Pending jobs check their inputs again before rendering. Cache hits and retained entries have narrower checks, and changing the Blender installation alone does not invalidate a job; see the [state guide's provenance limits](BLENDER_VISION_STATE.md). Incremental batches retain verified images for other doors and variants; failed requested slots are removed from the index so an earlier image is never presented as a new successful render.

The viewer's development server reads `out/appearance` at `/appearance/`. Once renders exist, the catalogue offers Blender images alongside simulation thumbnails; a door page displays its available material variants and optional Blender download. Reload an already-open page after a render batch finishes: the viewer fetches the index once per page load. Missing renders retain the simulation thumbnail. Only actually rendered images appear as Blender previews. For a static site, copy the generated appearance directory to `viewer/dist/appearance` after building; generated dataset/media publication remains a separate step.

Generated renders and blend scenes stay under ignored `out/`. Compact review evidence belongs under `docs/review/`. The appearance layer does not revise the previous [construction/physics audit](review/takeover/REVIEW.md): backed-over apertures, missing mechanism geometry and other source defects still require corrections. Render success establishes a working appearance option, not complete physical or photographic sign-off for every asset.
