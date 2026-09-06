# Hatch supports and hand access

All 18 horizontal hatches were inspected in source and compiled native geometry. The previous nine `prop_arm` variants had only a note suggesting an environment joint lock. Eight gas-strut variants had a single lid-mounted decorative cylinder and an unrelated torsional spring. Neither represented a frame-to-lid load path.

The rebuilt catalog has nine spring-engaged locking stays and eight separate gas struts across 12 hatches. Five hatches have both devices, mounted on opposite sides. Each assembly has a fixed curb bracket, pivot eyes, a moving hollow case, a sliding member, a lid clevis, native hinge/slide joints and a point constraint to the lid. A retaining collar bounds axial extension. Gas spring rods remain round and unperforated; they do not double as the pin-lock member.

The locking stay has an actual transverse slot. Its spring pushes a 9 mm pin through the slot at full extension. Away from that position, the solid sliding member physically blocks engagement. Closing requires supporting the lid, pulling the knob 16 mm to withdraw the pin, and then lowering the lid. Native contact carries the hold-open load. There is no environment freeze, artificial lock equality, or ignored rod/pin collision pair.

The gas spring is an original, orientation-independent simplified design: the authored force is the fully extended force, rising linearly by 10% at compression. It acts along the slider, so mounting geometry determines its changing moment arm. The former phantom hatch torsion spring is removed. These are mechanisms for simulation, not copies of rated commercial products or manufacturer force curves. Moving stay components retain their material-derived masses through `mechanism_mass_bodies`; legacy operator budgets cannot scale away their steel.

## Hand access and scope

Lifting rings now have two sides, a grip bar and an open finger aperture. Their cups cut both visible and collision slab geometry. A thin steel hatch uses a through-cut cup projecting onto the opposite face; it does not contain an impossible deep blind mortise. D pulls and their grip sites are constructed in the hatch's horizontal frame. Floor-hatch controls face upward, and ceiling-hatch controls face downward. The four previous ceiling `none` operators are explicitly normalized to lifting rings.

Ceiling hatches are approximately 2.4 m high. Their underside controls still require an elevation aid; the model does not establish floor-standing robot reach or traversal through an attic opening. The loft-side slide bolts on DB0389 and DB0598 are explicitly unavailable from below. DB0389 starts engaged and is therefore locked from the approached side. These distinctions are separate from whether the support mechanism operates in an unlocked test fixture.

| Door | Main hand contact | Gas strut | Locking stay | Access distinction |
|---|---|---|---|---|
| DB0017 | Ring, underside | — | — | Elevation aid |
| DB0121 | Ring, underside | Yes | — | Elevation aid |
| DB0208 | D pull, underside | — | — | Elevation aid |
| DB0241 | Ring, top | Yes | Yes | Top-side slide bolt |
| DB0357 | Ring, underside | Yes | — | Elevation aid |
| DB0360 | D pull, top | — | Yes | Floor hatch |
| DB0380 | Ring, top | Yes | — | Floor hatch |
| DB0389 | Ring, underside | Yes | Yes | Engaged loft-side bolt; unavailable below |
| DB0412 | Ring, top | — | — | Floor hatch |
| DB0442 | Ring, top | — | Yes | Floor hatch |
| DB0449 | Ring, top | — | — | Floor hatch |
| DB0452 | Ring, top | Yes | Yes | Floor hatch |
| DB0529 | Ring, top | — | — | Floor hatch |
| DB0559 | Ring, top | — | Yes | Top-side slide bolt |
| DB0598 | Ring, underside | — | Yes | Unengaged loft-side bolt; unavailable below |
| DB0834 | D pull, underside | Yes | — | Elevation aid |
| DB0976 | D pull, top | Yes | Yes | Floor hatch |
| DB0987 | Ring, underside | — | Yes | Elevation aid |

## Verification

Run `python -m pytest tests/test_hatch_mechanics.py tests/test_mass_scope.py tests/test_mechanical_inventory.py -q`.

The 29 hatch regressions cover all 18 hatches at 81 opening samples in full, simple and minimal native tiers; separate frame/lid constraints; real ring-to-slab clearance queried directly despite parent collision filtering; correctly mounted hand contacts; and all eight axial gas-force curves. All nine locking stays are lifted from closed by a finite virtual test motor, engage passively, hold against an additional closing load, and release through a force-driven knob. A missing-pin negative cannot hold the hatch, and a closed solid rod prevents premature engagement. The test motor is an assembly test and makes no human-strength claim.

The final source-bound export and full-QA receipt is `out/mechanical-foundations/hatches/final/receipt.json`. It retains every source/model/XML hash, ordinary locked-state QA, and separately labeled unlocked native support fixtures. Benchmark outcomes are not recomputed by this receipt. Ideal joints, rigid contacts and authored damping do not establish structural strength, gas-seal life, crush safety, or a complete human interaction. URDF requires loop/contact/spring support; USD is explicitly marked static interchange for these mechanisms.

## Primary references

The operating principle follows real locking lid stays, including [Sugatsune LSP supports](https://www.sugatsune.com/content/site-assets/Sugatsune_Resources_Catalogs/catalog-300-099-157-lid-supports.pdf), which lock at full opening and use a pulled knob to release. Its published product load ratings are not assigned to DoorBench's original geometry. [Stabilus installation guidance](https://www.stabilus.com/contact-and-support/service-gas-springs) calls for application-specific sizing, free pivoting without transverse loads, and rod-down mounting unless the spring is designed for other orientations. The DoorBench gas model explicitly declares the latter simplified assumption. No manufacturer CAD, textures or restricted assets were downloaded or redistributed.
