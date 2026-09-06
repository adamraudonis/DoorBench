# Sliding mechanics review

This change addresses all 35 bypass doors and all 22 pocket doors, with a separate audit of all 168 horizontal sliders. Generated evidence stays under `out/mechanical-foundations/sliding/`; no generated door assets or appearance renders are committed.

## Construction references

- [Baldwin 0465 edge pull](https://www.baldwinhardware.com/p/0465-edge-pull?variant=0465-033) specifies a narrow 3.875 × 0.75 inch pocket-door fitting. The manufacturer's [2024 Reserve price book](https://images.baldwinhardware.com/is/content/Baldwin/Baldwin-Reserve-Price-Book-2024) describes the spring returning its lever flush.
- [Deltana EP475](https://deltana.net/catalog/knobs-pulls-and-plates/viewing/edge-pulls-solid-brass/258_edge-pull-4x-34) and its [dimensioned drawing](https://deltana.net/media/catalog/products/9/62/258/EP475.pdf) provide another approximately 4 × 0.75 inch edge-pull envelope.
- [Johnson 111SD installation/catalogue sheet](https://www.johnsonhardware.com/Content/Images/uploaded/Documentation/111SDcatalogpage.pdf) shows independent track extrusions, wheel hangers, guides and stops. Its independent-track topology supports deliberate movement of a selected bypass leaf.
- [Johnson 1166 installation sheet](https://johnsonhardware.com/Content/Images/uploaded/Documentation/1166CatalogPage.pdf) describes front/rear installation, bottom guides and a rear stop that prevents hiding the rear pull. This is relevant to the access limitation below.

The geometry is original MIT-licensed primitive construction, not a copy of manufacturer CAD. The 98 × 19 mm edge-pull envelope is informed by these references; lever-arm lengths, the 28 mm case depth, 0.95 rad travel, return spring and clearance choices are authored simulation parameters. They are not measured properties of those commercial products. Reference PDFs are stored only in the local evidence folder with their URLs and hashes; their illustrations are not redistributed as DoorBench assets.

## Pocket sequence and geometry

The old pocket limit stopped at about 95% of the leaf width, and its only pulls were flat plates on an uncut slab. A face pull can disappear inside the wall. The revised stop places the closing edge exactly at the pocket mouth: `travel = (leaf_width + opening_width) / 2`.

Each pocket door now has a leaf-mounted edge-pull case, axle and bored rocker hub. Pressing the upper pad rotates the lower finger paddle outward. Pulling that paddle extracts the slab; releasing it lets a torsion spring return the rocker. This is a spring-return rocker, not a push-push catch that remains deployed after release. The case and faceplate have real empty space, and the slab is split around the mortise in both visible and collision geometry. A boxed closing-jamb relief accepts a still-deployed paddle during seating. Hook-lock variants place the edge pull 120 mm below the face pull so the released hook cannot strike it.

The three glass pocket variants use 10 mm glass and an original metal edge fitting around a prepared notch, with separated pads bearing on intact glass above and below it. There are no wood screws threaded into the glass. Its wider case fits between the pocket skins. Back-to-back face cups cannot fit in 10 mm glass: those doors instead use a lined through-finger aperture accessible from both faces. This is a simplified custom fitting, without tempered-glass machining, clamp preload, breakage or strength certification.

All recessed sliding pulls now have an open interior and collision side walls. A one-sided pull on a thin mirrored bypass panel uses a through-cutout with a rear cup, which clears the adjacent lane. Louvered-door pulls are mounted within the solid stile. The separate ring-pull dispatch omission is repaired; ring topology is corrected in the shared mesh builder.

`model.json.meta.pocket_edge_pull` identifies:

| Field | Meaning |
|---|---|
| `body`, `joint` | Leaf-child rocker and positive deployment hinge |
| `leaf_body`, `leaf_joint` | The slab that receives the extraction force |
| `press_site`, `press_direction` | Upper pad and inward press direction along the slide axis |
| `grip_site`, `extract_direction` | Lower paddle and direction out of the pocket |
| `deploy_range`, `minimum_grasp_q` | `[0, 0.95]` rad and 0.70 rad minimum tested deployment |
| `recessed_leaf_q` | The actual fully recessed native slide limit |
| `face_grip_after_extract_m` | 0.14 m extraction before handing off to the face pull |
| `spring_return`, `glass_patch` | Explicit mechanism and fitting distinctions |

The intended sequence is: reach the exposed edge pad from the aperture, press, grasp the deployed lower paddle, extract, then use the face pull. The metadata is an interaction description; it does not grant access through a wall or override a lock.

## Bypass and suspension

There was no native equality coupling DB0008's leaves. The simultaneous opposite motions came from treating `secondary_joint` as another automatically driven leaf. The geometry now records `interaction_mode: independent_bypass`, `preferred_manual_leaf: leaf_0_slide` and per-leaf controls, including joint, direction, nominal stroke, face grips and separate mechanism grips. One-hand attempts should deliberately select one leaf; the controller/reference changes are separate from this geometry implementation.

The third leaf's operator edge previously used the opposite sign from its actual slide axis. That sign is corrected. The native joint, rail and guide computations now share the same per-leaf stroke. Three-leaf outer strokes previously exceeded the rail metadata's one-panel stroke; the middle leaf was incorrectly inspected over a bidirectional range it did not possess.

Pocket and top-hung bypass tracks now contain two trolleys, four 25 mm wheel proxies, running ledges, a central stem slot, leaf-mounted plates, axles, carriage contact blocks, bolted end stops and channel-to-header mounts. A continuous structural header supports the pocket-side track. The specification reserves 95 mm above the slab for this mechanism and enough frame depth for all bypass lanes. Rail coverage is tested against the actual trolley envelope, so a channel need not extend through the closing jamb merely to cover the slab's unsupported end overhang. Floor guide stations cover the complete stroke, including the pocket mouth; native measured jaw-to-leaf gap is 1 mm.

The wheels are rigid visual bearing proxies on the translating carriage. The prismatic constraint carries the door and the calibrated slide models rolling resistance; wheel spin, hanger compliance and bearing loads are not separately simulated. Carriages are collidable and actually bear against the end stops. This distinction avoids the artificial static friction of a non-spinning wheel collider while retaining physical stopping contacts.

**Rear-leaf access is state dependent.** The audit found all 35 front-leaf pulls exposed at both endpoints. All 42 rear-leaf pulls are exposed initially but become hidden at full stacking while foreground leaves remain closed. A planner must rearrange/select leaves using current visibility, or use a rear-stop configuration that preserves access. The current independent-lane construction does not model Johnson's moving rear-pull protection stops. It must not imply that an obscured rear pull can be grasped through another leaf.

## Verification and evidence

Run the focused regressions with:

```sh
python -m pytest tests/test_sliding_mechanics.py tests/test_sliding_tracks.py -q
```

The 24 focused tests cover all 57 pocket/bypass models. Each of the 77 bypass leaves is driven through its entire stroke while the other leaves remain unactuated and within 10 micrometres of their initial coordinates. Other tests use an actual colliding 6 mm fingertip to deploy DB0018's pull, apply forces at the compiled press/grip sites, exercise spring return, and disable the slide limit to prove the carriage end stop itself bears the load. Negative fixtures reject a filled mortise, missing spring, insufficient pocket travel, removed/moved guides and a displaced running ledge. A static block 70 mm into DB0008's stroke still exceeds the unchanged 20 N jam limit when endpoint braking is enabled; the unobstructed control passes.

Measured pocket geometry has zero edge-to-pocket-mouth error, an unobstructed 80 mm press approach ray, and 26.85 mm paddle projection at the minimum 0.70 rad grasp angle. The initial force probe on DB0018 used 12 N to press, then 25 N at the lower paddle; it extracted 301.5 mm in 0.8 seconds. These are mechanism checks, not a whole-hand grasp or human-motion certification.

The complete horizontal export/QA command is:

```sh
python scripts/generate_dataset.py \
  --out out/mechanical-foundations/sliding/fullqa-r2 \
  --families sliding_single,sliding_bypass,automatic_sliding,elevator,gate_sliding \
  --formats mjcf,json --no-thumbs --workers 4
```

The fresh `fullqa-r2` export passes full QA for **168/168** horizontal sliders: 100 single sliders, 35 bypass doors, 15 automatic sliders, 10 sliding gates and eight elevator doors. This includes the native full/simple/minimal tiers, geometric clearance, running clearance, the sliding mechanism/rail checks, mass, settling and applicable dynamic operation gates.

`fullqa-r2/manifest.json` and each door's `qa.json` are authoritative for that generated source revision. `audit.json` records the separate per-leaf access/rail inspection and hashes of the source snapshot, exports, QA records and shared hardware. Do not treat an earlier report as bound to later edits. The first full pass identified real pocket axle/jamb interference, which was repaired geometrically, and a jam-test false classification of deliberate end-stop reaction after the complete stroke. The revised jam push limits cruise speed and brakes before a geometrically verified end stop while continuing to measure every static/moving contact against the same force threshold. Both full QA and the physical-obstruction regression remain required.

## Final pocket-entry handoff

A face cup becomes covered by the pocket skin before the leaf reaches full
recession. The controller now has a separate `pocket_edge_pull.final_push_site`
on intact existing slab/frame edge material. `final_push_switch_q` is derived
from the actual cup rim's leading X bound and the pocket-mouth plane, with a
20 mm handoff margin; `face_cup_occlusion_q` records the geometric boundary.
`final_push_direction` follows the leaf's native slide axis. The edge contact
remains accessible through the final stroke, including the fully recessed pose.

The additional tests check all 22 pocket doors in both opening orientations,
cast native rays at 17 final-stroke positions per door, complete DB0018's final
stroke by force at the real edge site, and reject both an off-surface site and a
fabricated late handoff threshold. There are now 17 tests in
`tests/test_sliding_mechanics.py`. The controller's contact transition remains
separate from the older full-QA receipt above; it does not retroactively validate
an earlier replay that kept pushing a hidden face cup.
