# Hardware realism review (task G3)

A close-up inspection of every distinct operator, latch and lock model in the dataset, and the geometry fixes that came
out of it.  Trigger: on `db0006_gate_swing` the heavy gate slide bolt was a bare rod with a knob floating next to the
rail, no barrel, no guide loops and a slot cut in the post instead of a keeper.  The standard applied to every mechanism
is the one a hardware catalogue would apply: every part must be mounted on something, bolts need housings / guides and
keepers, strikes need plates, hooks need keepers, readers and keypads sit on the outside face, surface hardware is not
buried in the slab, and sizes and heights match the real product.

## Method

`scripts/hardware_review.py` picks one representative door per model from `assets/manifest.json` (swing_single
preferred; for locks an *engaged* instance so the lock geometry is present) and renders it with the MuJoCo offscreen
renderer (800x600, `MjSpec` used to enlarge the offscreen buffer, a free camera aimed at the hardware's bounding box
~0.5 m away, collision-only proxies hidden):

| view | what it shows |
|---|---|
| `a` | closed door, robot (-y) face |
| `b` | closed door, far (+y) face |
| `c` | mechanism actuated: operator at full travel, bolts retracted / hooks lifted (couplings resolved exactly as the clearance gate does) |
| `d` | latches & locks: door opened 40 % with the mechanism at rest, looking at the latch edge (bolt extended, strike visible) |

Images: `docs/review/<kind>_<model>_{a,b,c,d}.jpg`, contact sheets `docs/review/sheet_{operators,latches,locks}.jpg`,
`docs/review/index.json` (model -> door), and before/after composites `docs/review/before_after_<kind>_<model>.jpg`
for the models whose geometry changed.  Every image was looked at; the verdicts below are from that inspection.

Verdicts: **ok** (looks like the product), **cosmetic** (recognisable but missing a plate / housing / detail),
**unrealistic** (a part floats, is on the wrong face, is buried in the slab, or the mechanism is not there at all),
**broken** (the user-visible geometry does not represent the mechanism), **n/a** (no hardware by design).

## Summary

| kind | models | ok | cosmetic | unrealistic | broken | n/a | fixed |
|---|---|---|---|---|---|---|---|
| operators | 59 | 31 | 11 | 12 | 3 | 2 | 15 unrealistic/broken + 4 cosmetic |
| latches | 26 | 9 | 8 | 5 | 1 | 3 | 6 unrealistic/broken + 7 cosmetic |
| locks | 27 | 6 | 8 | 8 | 2 | 3 | 10 unrealistic/broken + 6 cosmetic |
| **total** | **112** | **46** | **27** | **25** | **6** | **8** | **31 / 31 unrealistic + broken, 17 / 27 cosmetic** |

Every mechanism rated unrealistic or broken was rebuilt (per-family exceptions are listed at the end).  All mechanisms
remain functional in MuJoCo: after the changes the full regeneration signs off **998 / 1000** doors (unchanged from the
baseline; the two remaining doors, `db0788_swing_double` and `db0793_swing_single`, are the pre-existing plank-brace vs
hinge-jamb clearance failures at 80 deg, unrelated to hardware) and the kinematic clearance gate reports
**998 / 1000 clean** with the same two exceptions.

## Operators

| model | real product (reference) | what the geometry showed | verdict | fix |
|---|---|---|---|---|
| baby_gate_latch | Pressure-gate lift latch: plastic housing on the post, spring pin drops into a striker on the gate (Regalo / Munchkin) | pin + knob standing on a tiny block, bracket to the post, housing barely there | unrealistic | housing around the pin bracketed off the post (two brackets + post plate), knob on top, striker cup with ramp kept |
| barn_privacy_hook | Teardrop / hook privacy latch over an eye on the jamb | L-shaped bar over a small keeper post | cosmetic | - |
| bifold_knob | 30 mm bifold knob | knob on the leading panel | ok | - |
| cold_storage_handle | Kason 58 SafeGuard: lever on a latch body, strike on the frame, inside release | bare bent bar with a hub block, no latch body / strike | cosmetic | not fixed (see below) |
| cremone_bolt | Cremone / espagnolette: knob gearbox drives surface rods to shoot bolts in head and sill | knob only, no rods, nothing at head or sill | unrealistic | surface rods up and down beside the knob with 4 guide loops and a rod junction; top shoot bolt is an articulated body coupled to the knob, shooting into a y-offset pocket in the head |
| dog_lever | Watertight-door dog: lever on the face, wedge over a frame cleat | lever + wedge + cleat, 6 per door | ok | - |
| elevator_none | - | - | n/a | - |
| gate_latch_fork | Chain-link fork latch: strap + pivot on the gate frame, flat arm to a two-prong fork straddling the post; lift to open **and** to close | boxy fork hanging from a bracket on the post over the gate's top rail (reversed), big flat lift tab | unrealistic | rebuilt gate-mounted: strap plates and pivot pin on the latch stile, pivot eye, arm rising to two parallel prongs straddling the post, lift handle; the fork is now the operator joint (`leaf_fork_hinge`); padlock through the arm eye + lug when locked.  Not self-latching (like the product): QA's closer-return check is marked not applicable for `fork_gravity` |
| gate_latch_magnetic | D&D MagnaLatch Top Pull: vertical body on the post, pull knob on top, pin drops into the striker on the gate | pin + knob on a small block, a separate "body" mesh planted beside the post, unconnected | unrealistic | one housing around the pin, bracketed off the post with a post plate, knob on top; striker cup with ramp kept |
| handleset_thumb | Kwikset / Schlage handleset: grip with thumb press at the top of the plate, deadbolt cylinder above | two thumb pieces: the mesh's (at the plate *bottom* on right-hinged doors, `q_face` flips asymmetric meshes) and an articulated one floating above the deadbolt cylinder | unrealistic | single articulated thumb press at the plate top, deadbolt cylinder moved above it, mesh placed with an upright quaternion; thumb hinge axis no longer depends on handing |
| hasp_padlock | 4.5 in hasp & staple + padlock | flat plate on the leaf, staple box embedded in the jamb, padlock = a box, nothing articulated | broken | `add_hasp_assembly`: hinge plate / packing block, articulated strap (`leaf_hasp_hinge`), U-staple with base on the post or jamb face, padlock (shackle + body) hanging in the eye when locked; unlocked hasps are flipped open |
| hatch_ring | Recessed flush ring pull | ring on the loft side of ceiling hatches (unreachable) | unrealistic | ring and recess on the underside for ceiling hatches |
| hook_lock_slider | Patio slider handle with hook bolt | lever + flush pull; hook a bare nub at the stile edge | ok (hook: cosmetic) | hook faceplate on the stile, keeper plate over the jamb pocket |
| knob_childproof, knob_egg, knob_glass_antique, knob_porcelain, knob_round, knob_round_privacy | 54 mm knobs on 64 mm roses | as expected; privacy button now on the inside face for either handing | ok | - |
| knob_keypad_deadbolt | Schlage BE365 keypad deadbolt + passage knob | keypad on the outside face with 12 physical keys | ok | - |
| knob_rim_lock | Carpenter rim lock: surface case on the inside face, knob through it, keeper on the frame | plain knob, no case (the `rim_box` style param was unused) | unrealistic | surface rim case on the inside face, knob mounted on the case |
| lever_card_reader | Hotel RFID lever (Onity / Saflok): reader in the outside escutcheon | reader always on the robot face | unrealistic | reader on the outside face, upright |
| lever_curved, lever_euro_backplate, lever_keypad, lever_l_shape, lever_loose, lever_mortise_escutcheon, lever_return, lever_straight | 110-130 mm levers, 66-70 mm roses / escutcheons | as expected | ok | keyed sets now show a key cylinder outside and a turn button inside (see locks) |
| none | - | - | n/a | - |
| paddle_hospital_arm, paddle_push_pull | Glynn-Johnson push/pull paddle | plate-on-plate paddle | cosmetic | - |
| panic_crossbar | Von Duprin 88 crossbar: arms pivot in a latch case and an end case | bar + two bare arms | cosmetic | latch case at the strike edge and end case at the hinge edge |
| panic_touchbar_alarm | Chexit delayed-egress device with alarm module | plain touch bar | cosmetic | alarm module on the rail |
| panic_touchbar_mortise | Von Duprin 9975: mortise latch in the slab | touch bar + mortise bolt | ok | - |
| panic_touchbar_rim, panic_touchbar_rim_light, panic_touchbar_stiff | Von Duprin 99 rim device: Pullman latch in the surface case at the door edge, 299 strike on the stop | case stopped 30 mm short of the edge, latch bolt buried in the slab like a mortise latch | unrealistic | case runs to 12 mm from the edge; bolt body lives in the case (22 mm off the push face); jamb pocket with flat lip + ramp at that offset (`_strike_column` generalised to off-centre pockets, ramp always runs past the swing face), stop moulding cut to the case height |
| panic_touchbar_svr | Surface vertical rod device | rods + top latch into the head | ok | - |
| pull_bar_offset, pull_barn_iron, pull_d, pull_ladder_full, pull_lift_garage, pull_ring, push_plate | pulls / plates | as expected | ok | - |
| pull_finger_cup, pull_flush_recessed, shoji_finger_pull | recessed pulls | flat rectangles (no recess possible with convex primitives) | cosmetic | - |
| pull_t_handle_garage | Garage T-handle with keyed cylinder | one-sided straight lever, no cylinder | cosmetic | T bar centred on the spindle + key cylinder |
| push_button_screen | Wright Products push-button latch | button in a housing, inside pull | ok | - |
| slide_bolt_barrel | 4 in brass barrel bolt: plate, 2 guide loops, rod, knob, keeper on the post | bare rod + knob capsule, a thin plate under it, slot in the post | broken | `add_barrel_bolt`: mounting plate, 2 guide loops, rod with L-handle and ball, U keeper loop with base on the post face at the rod standoff (rod clears posts thicker than the leaf) |
| slide_bolt_heavy | 12 in heavy gate slide bolt (padlockable) | as above (the user report) | broken | as above with 3 guide loops; padlock lug + padlock when locked |
| stall_slide_latch | Bobrick partition slide latch: housing + flat bar + keeper | bar and knob floating on the door, keeper only | cosmetic | back plate + guide bracket on the door |
| thumb_latch_suffolk | Suffolk latch: thumb pad at the top of the grip plate, lifter through the door, bar level with it on the far face, keeper on the frame | vertical thumb plate (pressing it popped *out* on left-hinged doors), bar 10 cm below it, no lifter, bar tip in a bare slot | unrealistic | horizontal thumb pad on a pivot boss, lifter tang through the door under the bar, bar level with the pad on a pivot plate, keeper plate around the pocket |
| turnstile_arm | tripod / full-height rotor | as expected | ok | - |
| wheel_ship_hatch, wheel_vault | handwheels driving dogs / boltwork | as expected | ok | - |

## Latches

| model | real product (reference) | what the geometry showed | verdict | fix |
|---|---|---|---|---|
| deadlatch_grade1, tubular_residential, tubular_residential_70, mortise_latch, mortise_euro | Mortised latch with a faceplate on the edge, strike plate with a lip on the jamb | bolt capsule out of a bare edge; strike = pocket + a 20 mm lip block | cosmetic | flush faceplate on the leaf edge (lip geometry unchanged: it is the functional re-latch ramp) |
| dogs_6 | wedge dogs on cleats | as expected | ok | - |
| electric_bolt | electric drop bolt (solenoid over the leaf, keeper on the leaf) | as expected | ok | - |
| elevator_interlock | hoistway interlock inside the header | nothing modelled | n/a | - |
| fork_gravity | see gate_latch_fork | | unrealistic | fixed (operators) |
| gravity_bar | see thumb_latch_suffolk | | unrealistic | fixed (operators) |
| hook_slider | hook bolt from the stile faceplate into a jamb keeper | hook + bar keeper, no plates | cosmetic | faceplate + keeper plate |
| magnalatch | see gate_latch_magnetic | | unrealistic | fixed (operators) |
| magnetic_catch | bifold magnetic catch | nothing modelled | n/a | - |
| magnetic_gasket | cold-storage gasket | visual gasket | ok | - |
| multi_bolt_4, multi_bolt_8 | vault boltwork | bolts from the edge into frame pockets | ok | - |
| none | - | - | n/a | - |
| pet_flap_magnet | magnet strips | as expected | ok | - |
| rim_exit | see panic_touchbar_rim | | unrealistic | fixed (operators) |
| roller_latch, screen_pushbutton, teardrop | | as expected | ok | - |
| slide_bolt | barrel bolt (hatches, garden gates) | bare rod, keeper = two blocks / slot | unrealistic | hatch: barrel bolt on the top face into a keeper loop on the curb; gate: as slide_bolt_barrel |
| slide_bolt_heavy | cane / drop bolt on sliding gates | bare vertical rod with a stub handle | broken | `add_barrel_bolt` vertical: plate, 3 guides, bent-over top handle, floor socket kept |
| stall_slide | see stall_slide_latch | | cosmetic | fixed (operators) |
| vertical_rods | SVR device: top and bottom latches | top latch only, bottom rod ends free | cosmetic | not fixed |

## Locks

| model | real product (reference) | what the geometry showed | verdict | fix |
|---|---|---|---|---|
| card_reader | RFID reader in the outside trim or on the wall | wall reader for plain levers; on-lever reader on the wrong face | ok / see lever_card_reader | reader moved to the outside face |
| chain | Door chain: anchor plate on the door, slotted track on the jamb, chain between | six floating beads and a box sunk into the jamb | unrealistic | anchor plate, 6 chain links, slotted track on the jamb's inside face |
| child_lock_cover | knob cover | as expected | ok | - |
| deadbolt_single, deadbolt_double, thumbturn_only | A156.5 deadbolt: faceplate, strike plate | bolt out of a bare edge into a bare pocket | cosmetic | flush faceplate + 1 mm strike plate ring around the pocket |
| delayed_egress | maglock + delayed-egress device | maglock + armature (+ alarm module now) | ok | - |
| dogs | dogged watertight door | as expected | ok | - |
| electric_bolt | drop bolt with card reader (sliding gates, auto sliders) | nothing at all when the latch was not `electric_bolt` (joint range only) | broken | solenoid housing, drop bolt body, keeper blocks on the leaf (when the latch is not already the bolt) |
| electric_strike | HES 1006 strike with a wide faceplate | wall reader only, strike not visible | cosmetic | not fixed |
| garage_slide_lock | Garage side lock: plate, spring bar with knob, slot in the track | bare bar with no handle, keeper blocks | unrealistic | barrel bolt with knob and guides; keeper loop off the track flange; roll-up lock moved to the bottom bar (below the coil when raised), guide split follows |
| hook_lock | privacy hook with thumbturn | thumbturn + hook | cosmetic | faceplate + keeper plate |
| interlock, jam_stuck, none | - | - | n/a | - |
| keyed_cylinder | keyed lever / knob: cylinder in the outside hub, turn button inside | invisible | unrealistic | key cylinder + keyway on the outside hub / knob face, turn button inside |
| keypad_code_4, keypad_code_6 | Schlage FE595 / BE365 keypad | keypad only when the operator was a keypad model; invisible with plain levers / handlesets | unrealistic | keypad unit with 12 physical keys on the outside face for every keypad lock, above escutcheons / handleset cylinders |
| keypad_mechanical | Kaba Simplex 1000: 5-button column, key override | invisible | unrealistic | 5 round buttons in a column (bodies, `keypad_key_<n>_slide`) + key override cylinder |
| mag_lock | Securitron maglock | as expected | ok | - |
| multipoint | Multipoint lock: 3 lock points on a faceplate strip | single deadbolt | cosmetic | two extra lock points driven with the main bolt, faceplate strip, strike plates |
| night_latch | Yale 77 rim night latch: surface case with snib inside, cylinder outside | mortise deadbolt + thumbturn | unrealistic | surface case on the inside face carrying the turn, cylinder outside |
| padlock | padlock on a hasp | only drawn for the hasp operator (as a box); invisible for every other operator | broken | hasp/staple/padlock assembly on swing doors, gates, sliding gates and sliding doors; padlock lug on padlockable slide bolts; padlock through the fork-latch arm |
| privacy_button | privacy set: turn button inside | only on the privacy knob mesh; invisible on levers | cosmetic | turn button on the inside hub / knob face |
| slide_bolt | barrel bolt on the inside face | bare rod + knob, slot in the jamb | unrealistic | barrel bolt with plate, guides, knob; keeper plate over the jamb pocket (pairs: mounted inboard of the astragal) |
| swing_bar_guard | Ives swing bar guard | bar + stud | cosmetic | not fixed |
| vault_wheel | vault boltwork | as expected | ok | - |

## Before / after

Each composite is `before | after` from the same door and camera.

### Slide / barrel / drop bolts, hasps, padlocks

![](review/before_after_operator_slide_bolt_heavy.jpg)
![](review/before_after_operator_slide_bolt_barrel.jpg)
![](review/before_after_latch_slide_bolt_heavy.jpg)
![](review/before_after_lock_slide_bolt.jpg)
![](review/before_after_lock_garage_slide_lock.jpg)
![](review/before_after_operator_hasp_padlock.jpg)
![](review/before_after_lock_padlock.jpg)

### Gate latches

![](review/before_after_operator_gate_latch_fork.jpg)
![](review/before_after_latch_fork_gravity.jpg)
![](review/before_after_operator_gate_latch_magnetic.jpg)
![](review/before_after_latch_magnalatch.jpg)
![](review/before_after_operator_baby_gate_latch.jpg)

### Suffolk latch, handleset, cremone, rim lock, T-handle

![](review/before_after_operator_thumb_latch_suffolk.jpg)
![](review/before_after_latch_gravity_bar.jpg)
![](review/before_after_operator_handleset_thumb.jpg)
![](review/before_after_operator_cremone_bolt.jpg)
![](review/before_after_operator_knob_rim_lock.jpg)
![](review/before_after_lock_night_latch.jpg)
![](review/before_after_operator_pull_t_handle_garage.jpg)

### Exit devices

![](review/before_after_latch_rim_exit.jpg)
![](review/before_after_operator_panic_crossbar.jpg)
![](review/before_after_operator_panic_touchbar_alarm.jpg)

### Keypads, readers, cylinders, chains, multipoint

![](review/before_after_lock_keypad_code_4.jpg)
![](review/before_after_lock_keypad_code_6.jpg)
![](review/before_after_lock_keypad_mechanical.jpg)
![](review/before_after_operator_lever_card_reader.jpg)
![](review/before_after_lock_keyed_cylinder.jpg)
![](review/before_after_lock_privacy_button.jpg)
![](review/before_after_lock_chain.jpg)
![](review/before_after_lock_multipoint.jpg)
![](review/before_after_lock_deadbolt_single.jpg)
![](review/before_after_latch_tubular_residential.jpg)
![](review/before_after_lock_electric_bolt.jpg)

### Stall latch, hooks, hatch ring

![](review/before_after_operator_stall_slide_latch.jpg)
![](review/before_after_latch_stall_slide.jpg)
![](review/before_after_latch_hook_slider.jpg)
![](review/before_after_lock_hook_lock.jpg)
![](review/before_after_operator_hatch_ring.jpg)

## Shared builders added (`doorbench/geometry/common.py`)

* `add_barrel_bolt` - mounting plate, 2-3 guide loops, rod with L-handle; geometry authored engaged, `initial` moves it
  to the withdrawn state (`Model.bake_initial`).  Used by gate slide bolts, the auxiliary barrel bolt, the garage slide
  lock, hatch bolts, the Dutch joining bolt and the sliding-gate drop bolt.
* `add_keeper_loop` (collidable U keeper with base plate), `add_keeper_ring` / `add_strike_plate` (visual plates around
  pocket mouths), `add_guide_loop`, `obox` (box in an axis/normal frame).
* `add_hasp_assembly` + `add_padlock` - articulated hasp strap on a packing block, U-staple, padlock in the eye.
* `add_keypad` - electronic 12-key or Simplex 5-button unit with physical key bodies; `q_face_upright` for asymmetric
  face-mounted meshes.
* `add_rotary_operator` gained `cylinder_face`, `button_face`, `rim_case_face`; `add_deadbolt` gained `tt_standoff`,
  `couple_to`, faceplates; `_strike_column` handles off-centre pockets; `add_head` pockets take a `y` offset.

Joint names driven by `qa.py` (`leaf_slide_bolt_slide`, `leaf_aux_bolt_slide`, `hatch_bolt_slide`, `join_bolt_slide`,
`garage_slide_lock_slide`, `slide_latch_slide`, `leaf_pin_slide`, `leaf_thumb_hinge`, `leaf_latch_bolt_slide`) are
unchanged.  New joints: `<leaf>_hasp_hinge`, `<leaf>_cremone_top_bolt_slide`, `<leaf>_multipoint_{upper,lower}_slide`,
`<leaf>_keypad_key_<n>_slide` (mechanical keypads).  `leaf_fork_hinge` is now the operator joint of fork-latch gates.

QA changes: the closer-return check is not applicable to gravity fork latches (the gate is closed with the fork lifted,
like the real product - a wedge cam was tried and cannot lift the fork against the post's friction with a 4 N m gate
spring); a cremone's shoot bolts count as the door's latch (`has_holding`), so those doors are tested with
actuate-then-open instead of free-opens.

## Not fixed (and why)

* **Padlocks on garage / roll-up / tilt-up doors (5 doors) and hatches (3 doors)**: still only the locked joint; a
  padlock through the track or curb needs family-specific geometry and was left out of this pass.
* **`slide_bolt` lock on sliding doors (6 doors) and the pet-door locking panel (4 doors)**: the leaf / flap joint is
  locked but no bolt is drawn.
* **Electric strike**: the wall reader is there, but the strike's wide faceplate on the jamb is not drawn (the lip /
  ramp strike geometry occupies that spot and is functional).
* **Cold storage handle**: the SafeGuard latch body and frame strike are not modelled; the lever is bare.
* **Recessed pulls** (finger cup, flush pull, shoji hikite) are flat plates: recesses cannot be cut with convex
  primitives.
* **SVR bottom rod** has no floor strike; **swing bar guard** is a bar and a stud; **paddles** are plate-on-plate;
  **vault boltwork** has no bolt carrier / cover; **elevator interlock**, **bifold magnetic catch** and **turnstile
  credential locks** have no visible hardware.
* **Strike lips** on mortised latches remain 20 mm blocks (they are the functional re-latch ramps); only the faceplate
  was added.
* **Fork latch is not self-latching** (see above).

## Verification

```
rm -rf assets/doors assets/hardware assets/manifest.json
python scripts/generate_dataset.py --out assets --workers 8          # 1000 doors, 998 signed off (baseline 998)
python scripts/clearance_report.py --workers 8                        # 998/1000 clean (db0788, db0793 pre-existing)
python scripts/hardware_review.py --out docs/review                   # 108 models, 379 images + 3 sheets
python -m pytest tests -q                                             # 6 passed
```
