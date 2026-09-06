# Independent inside rotary controls

Status: the 97-door source rebuild and three-door/all-tier pilot pass; the complete 291-case native receipt is in progress. This document does not yet claim the whole class is signed off.

The former shared spindle gave both faces one joint. Its locked range depended on credential availability: some locked exterior trims began free, while other locked trims also prevented inside egress. The replacement gives each face a separate retained shaft and full-travel operator joint. An actual spring-return pin arrests only the exterior cam lug. Having a usable credential does not change the installed pin position, either operator range, or interior access.

The scope is 94 `swing_single` and three `automatic_swing` doors with installed both-side lever/knob trim and privacy, keyed, keypad or card-entry locking. DB0264 alone receives a post-RNG descriptor correction to a both-side keypad-entry lever; its lock state and credential availability remain unchanged. The unrelated special functions, panic devices, childproof covers, deadbolts and auxiliary bolts are outside this replacement. Exterior placement derives from both approach side and `robot_outside`; a reversed-approach regression checks that changing the observer does not swap the physical exterior.

## Mechanical implementation and source support

[Schlage's ND catalogue](https://commercial.schlage.com/content/dam/allegion-us-2/web-files/schlage/information-documents/Schlage_ND_Series_Catalog_106501.pdf), function table on printed page 17, distinguishes locked exterior trim from free interior egress for ordinary entry/storeroom functions. The [ND cut sheet](https://commercial.schlage.com/content/dam/allegion-us-2/web-files/schlage/information-documents/Schlage_ND_Series_Cut_Sheet_113158.pdf) describes electrically controlled exterior trim with free inside operation. These support the independent-input topology, not the dimensions or internals of this original generic chassis. Some real special functions can lock both faces; this repair does not generalize the egress rule to those functions.

The two 12 mm stub shafts end 1 mm apart in prepared leaf stock. Fixed bored roses, supported catch guides and a reaction bridge provide connected mounting geometry. A keyed exterior lug meets the real locking pin. Front and rear plates carry the catch collar at the nominal 0/8 mm endpoints; its broader joint range is only a safety bound. Core trim geometry and its material-derived inertia remain in every tier. The old operator mass allowance is replaced once by the authored moving/fixed chassis BOM; removed stock is separately recorded.

Two independent one-sided tendon constraints represent the concealed latch cams: either input withdraws the complete calibrated latch stroke; simultaneous operation does not add the two strokes or backdrive the opposite handle. The internal cam contact profile remains ideal. Bearings use native joint constraints; the model does not certify stress, manufacturing tolerances or OEM construction.

The catch has an 8 N bounded ideal pull actuator, a real return spring, authored viscous resistance and real endpoint contact. This is an explicit controller input, not simulated key insertion, a tumbler mechanism, reader electronics or privacy-button cancellation logic. Runtime authorization and observed withdrawal must be separate from credential availability. The manufacturer reference establishes the egress function, not every product-specific privacy/entry reset behavior.

## Contact input and evidence contract

Levers receive a single tangential force at an actual surface site. Knobs receive two equal/opposite tangential forces at two actual surface sites. The knob cap is **22.2 N per site, 44.4 N total absolute force per knob**, with nearly zero net resultant and measured spindle torque. This differs from the old 22.2 N single-point test. No free joint torque or spring reduction makes the knob pass; the initial DB0111 single-point failure is retained in `out/mechanical-foundations/rotary-locksets/db111-one-cycle.json`.

`rotary_locksets[].input_model` and `input_sites` bind that input model to the actual native input joints and site names. `operator_force_cap_N` is per surface point. `inside_egress_inputs` identifies the real interior joint. Catch metadata binds its joint, pin, lug, collar, guides, stop geometry, 8 mm stroke and 7.5 mm observed-release threshold. Immutable compile/apply helpers only add bounded native force; they never change ranges, poses or permissions.

The component gate uses two continuous native cycles: locked exterior load and named pin reaction; independent inside operation and return; measured catch withdrawal and rear-stop load; exterior operation; simultaneous inputs; release and relatch. Removing only the actual pin collision must remove the exterior arrest. The negative does not enlarge/move compiled geometry, so it does not depend on stale broadphase bounds. Every phase checks finite state, at most 1 mm penetration, no catch safety-limit load, warning counters and global MuJoCo warning messages.

The fixture deliberately exercises a seated catch even when the source starts unlocked, using its own spring. Its private callback isolates the catch and does not apply unrelated closer pinion/track fields. It is a component proof, not a complete source-door service, credential or humanoid benchmark. Whole-door QA must still run with the integrated runtime fields.

Explicit exterior operator clearance sweeps first require a private native 8 N release snapshot with observed withdrawal, named collar/stop reaction and no warnings. Other source-state/inside sweeps retain the installed catch state. This conditioning does not waive any solid clearance.

## Receipts and remaining limits

- Pilot: DB0111, DB0166 and DB0264, two cycles in full/simple/minimal, plus pin-removal negatives; 110 focused tests passed before the final additional surface-frame assertions.
- Full corpus: `out/mechanical-foundations/rotary-locksets/final-source/receipt.json` (pending at this writing), with per-tier XML/model/spec hashes, proof JSON, warning logs and immutable-input checks.
- Full native assembly mass matches its reconciled declared mass. Reduced tiers retain identical mass/inertia for every new rotary input/catch body. Existing optional keypad-key, pet-flap, mail-slot and closer-riser tier omissions are recorded separately; their removal is not hidden by claiming full aggregate tier mass equality.
- The full native source has the ideal cam constraints and force helper. A successful URDF/USD serialization alone does not establish equivalent cam, catch-input or complete lock behavior in those engines.
- No published assets or recordings are regenerated by this component task. No strength, security rating, full enclosure seal, physical hand-friction capacity or natural human manipulation claim follows from these tests.
