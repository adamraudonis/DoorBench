# Gate mechanisms

This is a source-generator repair and native mechanical review, started
2026-09-05. Generated review fixtures live only in
`out/mechanical-foundations/gates`; the published asset snapshot is preserved.
The inventory contains 40 swing gates and 10 baby gates. It includes six magnetic
top-pull latches, 12 fork latches, ten baby-gate lift latches, five Suffolk thumb
latches, six slide bolts, four mortise operators, four fixed rings and three
hasps. All 50 regenerated gate fixtures pass the final native QA run. The
review also repaired and checked the seven indoor Suffolk variants. Dedicated
mechanism tests cover 33 of the gates and all seven indoor Suffolk doors; the
remaining 17 gates have the existing generic native QA and mounting inventory.
Those scopes must not be presented as equivalent mechanical certification.

## Magnetic top-pull latch: six gates

The published DB0014 release knob was disconnected from the short pin: the pin's
top and knob's bottom were separated by 27 mm. Its nominal 1.5 m operator was
clamped to about 1 m; its native slide allowed 60 mm rather than the declared
30 mm. A downward spring and an approach ramp represented the supposed magnet.
The chain-link structural members also used the 0.3 mm infill-equivalent sheet
thickness. These were mechanical defects, not appearance preferences.

The replacement has a post-mounted hollow guide, two screwed brackets, a
continuous pin/rod/knob, an upward return spring, and a gate-mounted pocket with
a permanent-magnet pole. A connected bracket carries the pocket's holding load
to the actual latch stile. The knob is at the specified 1.5 m height. Thirty
millimetres of lift withdraw the pin; independent fixed pulls on both gate faces
apply the subsequent leaf-opening force. A 19 mm latch gap and a structural
41.275 mm chain-link frame replace the former 9 mm gap/thin-frame combination.
The source mass model treats structural tube and wire infill separately.

The topology follows D&D's
[MagnaLatch Top Pull product documentation](https://us.ddtech.com/products/magnalatch-top-pull),
[installation instructions](https://dam.ddtech.com/consumer/asset-pdf-viewerdrawings/he8wf08o0tohuu1z3fssywtguslaol)
and [dimensional drawing](https://dam.ddtech.com/consumer/asset-pdf-viewerdrawings/40ycikvht5uav3rkumyompuet).
The drawing permits a 13–38 mm gate gap, recommends 19 mm, and the instructions
show the separated post brackets, striker alignment and high release mounting.
The instruction assembly page and dimensional drawing were visually inspected.
The new geometry is an original, dimensioned approximation of that topology;
the undocumented internal pin dimensions and magnetic force are not measured
manufacturer data or a reproduction of certified pool-safety hardware.

The passive magnet is explicit. With pole displacement `d`, capture axes
`a=(18,18,50) mm`, and `s=sum((d/a)^2)`, it uses the compact conservative
potential `U=-E(1-s)^3` inside `s<1`, zero outside. Its negative gradient acts
equally and oppositely at the moving pin and striker poles. `E` is chosen for a
15 N peak axial force. The force smoothly reaches zero outside the capture
region; neither coordinates nor a latch weld are imposed. This is an engineering
force approximation, not a calibrated magnetic-field or pull-force model.

Native tests continuously settle, load the closed gate, lift with 22.2 N, open,
let go of the release while holding the gate, close at bounded speed and load
the recaptured pin. Removing the magnet is a meaningful negative test: the pin
returns upward and cannot recapture the keeper. Moving the knob, striker mount
or post bracket away fails an explicit attachment test.

## Baby-gate lift latch: ten gates

The old generic rod had the same disconnected-grip problem and 50 mm of native
travel despite a 20 mm specification. The replacement is a continuous pin,
shaft and grip with a hollow guide, post carrier, screws, stile-mounted keeper,
and independent pulls on both faces. It uses the declared 20 mm release travel,
12 mm engagement and a downward return spring. Two shallow approach ramps allow
the keeper to raise the released pin on either permitted approach direction.
The rod has a physically open guide and a clear path below it; no collision
exclusion conceals a solid housing or obstructing support.

All ten full/simple native fixtures pass the continuous hold/lift/open/release/
close/recapture cycle. Their worst contact penetration in that probe is
0.55 mm. Forty-one samples over each complete authored leaf range check the
released keeper and pulls against the fixed housing, including both directions
for double-acting examples. The dynamic cycle currently exercises positive
opening; the reverse-direction evidence is a geometric sweep, not a second
dynamic-cycle claim. This is a generic one-action spring lift latch. It does not
model or certify a named product's two-action child-resistant release.

## Fork latch: twelve gates

The former fork clamps missed actual gate structure in 11 fixtures. More
seriously, exact native geometry-distance queries found the fork bridge passing
28–36 mm through the continuous latch stile during release. Parent-child
collision filtering hid that obstruction from ordinary contact checks.

The rebuilt flat fork uses a plate fixed to the real stile, a connected carrier,
two clevis cheeks, an open pivot eye and a pin. The pivot is outside the leaf
edge, with a 51 mm latch-side operating gap. Its prongs surround the actual post
with 2 mm side clearance. The fork is lifted nearly vertical, 1.55 rad, before
moving the gate; it must be lifted again to close. Its separate lift grip stays
in the fork plane, and separate fixed gate pulls carry the opening force.
Short-gate grip heights are constrained by both the stile and the raised
fork/post-cap envelope. DB0566 retains its padlock and limited operator range.

The construction follows the clamp-and-fork arrangement described by
[Nationwide Industries](https://nationwideindustries.com/product/gate-fork-latch/)
and the separate frame-sized clamp/post-sized fork components in
[Hoover Fence's fabrication guide](https://www.hooverfence.com/chain-link-fence-overview).
The rectangular-post dimensions, clearances, friction and mount are original
engineering choices. A fork's gravity return is not self-latching: closing with
the fork already dropped remains an invalid operation.

Each fixture is checked at 81 release angles against its fixed carrier and
actual leaf support, followed by 81 leaf angles against every actual post
piece and cap. These queries include parent-child pairs. Native cycles use
bounded force-equivalent leaf torque, lift, open, gravity return, re-lift, close,
drop and re-hold. Counterexamples move the clamp or grip, shorten release travel
or restore an inboard pivot. They must fail. Bearing support is represented by
ideal joints; padlock immobilization uses an ideal joint limit, not simulated
key operation or shackle deformation.

## Suffolk thumb latch: twelve doors, including five gates

The former equality imposed 0.6 rad of bar lift for 0.3 rad of thumb rotation.
The visible short tang rose only about 8 mm, while the imposed bar motion rose
about 60 mm at that contact location. Two picket gates also lacked structural
support for the grip and bar pivot. Replacing only the mount would have left the
false transmission intact.

The replacement removes the equality. A continuous thumb lever passes through
a real slot in both backing plates and every underlying leaf collision box.
Its far end contacts and raises the gravity bar. Open bearings, pins, retaining
washer, screwed backing plates and a screwed keeper connect the hardware to
the actual stile/slab and post/jamb. Explicit geom masses are prorated when a
slot divides a slab. The thumb bearings sit outside the pad sweep; an explicit
parent-child distance check catches interference hidden by native contact
filtering. The shallow keeper and two approach ramps permit native
gravity recapture. Fixed pulls on both faces carry the opening force.

The bar is on the opening-side face; the thumb is on the opposite face. A
person on the thumb side presses the pad; a person on the bar side lifts the
bar directly. The selected operator metadata follows the accessible side.
Indoor variants use the actual offset wall/jamb surface, and the backing plate
leaves the frame stop's latch-edge strip clear.

This is original generic geometry informed by
[Snug Cottage's Suffolk installation instructions](https://snugcottagehardware.com/wp-content/uploads/2025/02/6500_Contemporary-Suffolk-Latch_Installation-Instructions_Setback-Mount_Street-Swing.pdf)
and [From The Anvil's mechanism description and dimensions](https://www.fromtheanvil.co.uk/products/cast-suffolk-latch).
The installation page's two hardware photos and instructions were visually
inspected. They establish the through-door lever, gravity bar resting on its
far end, screwed supports and adjustable keeper; they do not calibrate this
model's contact friction or certify its strength.

All twelve native mechanisms transmit thumb force through actual tang/bar
contacts. Disabling the tang collider prevents bar lift despite a fully pressed
thumb; adding an equality or obstructing the slot is rejected. The independent
engaged slide bolt on DB0693 remains blocking: that fixture verifies release
operation and continued locking, not traversal. DB0147's friction-only jam can
be overcome by the bounded 40 N-equivalent pull; the receipt must not describe
it as an immovable positive lock. Native dynamic probes also verify both the
thumb and direct-bar release methods, gravity return and ramp recapture on the
unblocked cases.

Inspection uses a separate copy of the original native model. Narrowly
conditioning the active joint and leaf pose allows the passive bar to settle
through contact; unreachable, unsettled or penetrating results fail inspection.
This is a geometry-sampling procedure, not a new benchmark actuator. The
all-geometry collision model remains separate for subsequent overlap queries.
Fork latched scans similarly stop at the first exact intended tine/post contact;
fully released travel is checked separately. Neither procedure allows the
inspection to force a locked mechanism through its load-bearing counterpart.

## Evidence and limits

The final `out/mechanical-foundations/gates/full-qa-gates-and-suffolk.json`
records **57/57 signed off**, including **50/50 gates**, all 12 fork latches and
all 12 Suffolk mechanisms. Each row binds its exact `spec.json`, `model.json`
and `door.xml` SHA-256. The receipt SHA-256 is
`06aecbcfa1f56461d499912e72981f8cbf30869e2467b7e3b64e94c9d98487ff`.
`source-receipt.json` confirms the recorded generator/QA sources did not change
during the run. This run generated full/simple/minimal MJCF, URDF and JSON in
the isolated review directory; it did not replace published assets.

| Dedicated mechanism | Fixtures | Maximum native probe penetration |
| --- | ---: | ---: |
| Magnetic top-pull | 6 | 0.465 mm |
| Baby spring lift | 10 | 0.547 mm |
| Gravity fork | 12 | 0.328 mm |
| Contact-driven Suffolk | 12 | 0.518 mm |

All listed probes remain below the unchanged 1 mm gate. The fork's minimum
fully released post clearance is 4.209 mm; its operator/carrier clearance is
1.000 mm. Suffolk tang/slot clearance is at least 5.999 mm, and thumb/bearing
clearance is 2.000 mm. Per-class `magnetic-qa.json`, `baby-qa.json`, `fork-qa.json`
and `suffolk-qa.json` retain the exact states, forces, attachment distances and
failures. `full-qa-all-gates.json` and `gate-inventory.json` contain the gate-only
view; `gate-inventory-before-repair.json` is explicitly historical.

The focused suite passes **26 tests**, including all full/simple positive
mechanism fixtures and counterexamples for absent magnetic force, disconnected
mounts/grips, obstructed slots, fake equality transmission, parent-child bearing
interference and missing intended fork arrest. `gate_hardware_qa.py` is integrated
into normal QA. The other 17 gates have no new dedicated dynamic mechanism test
in this work; their generic native checks are retained without an expanded claim.

The magnetic force requires the declared runtime helper. Plain MJCF, URDF and
USD geometry plus an upward spring do not reproduce that magnetic interaction.
The native receipts are not an Isaac/PhysX parity certificate. They establish
specific attachment, release-clearance and bounded load/operation behavior,
not material strength, fatigue, hardware safety certification, articulated
human manipulation or whole-scene mechanical completeness.

Manufacturer PDFs retained for provenance:

* Installation PDF, SHA-256
  `7885eef74af8958a5f9eee02e7d459d2edf2c649ae0a6757d82e6082726c2fe8`.
* Dimension PDF, SHA-256
  `80852186084696c2be75d384a60d98b44fb125f09b8750a769d1883e4bfefbe2`.
* Suffolk installation PDF, SHA-256
  `aada276b6ab25ee80fba92549c2fcbb937dafcda6e59d4ee420b8a20fe5ff0ae`.
