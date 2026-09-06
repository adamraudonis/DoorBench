# Dutch joining bolts and inactive paired leaves

The Dutch correction covers DB0095, DB0118, DB0204, DB0333, DB0391, DB0423, DB0460, DB0626, DB0700, DB0847, DB0906 and DB0974. Both leaf hinges retain full independent travel. A real sliding shaft through a keeper on the lower leaf transfers opening load when joined; no equality or restricted hinge range binds the leaves together.

The original generic fittings use the installation class in the [Ives 054 instructions](https://allegion.ca/content/dam/allegion-us-2/web-files/ives/installation-documents/Ives_054_Dutch_Door_Bolt_Installation_Instructions_107772.pdf): two prepared guides spaced 44.45 mm apart, positioned 60 mm from the door edge. The 110 mm stainless shaft is 12 mm across, with 30 mm withdrawal. The keeper has 0.75 mm radial clearance and is physically attached to the lower-leaf face; the fixed guides have open bores and contact-active stock. The exposed knob has an actual surface grasp site. This is original simplified geometry, not an OEM replica or strength rating. The guide joints and frictional position retention are idealized.

`meta.dutch_joining_bolt` records the actual joint/site and both leaf bodies/hinges, interior face, approach accessibility, initial engagement, keeper/guide names and dimensions. `join_bolt_slide` and `join_bolt_grip` remain stable identifiers. Withdrawal of at least 25 mm clears the keeper. The assembly stays on the interior face, so outside approaches do not acquire access to it. Reinsertion requires both leaves to be seated; a command to lower the rod alone is not proof of joining.

The upper ball catch is retained in every native tier. Each Dutch slab now receives its existing per-panel material mass during construction. The former `phys=None` call left nearly massless slab proxy geometry and allowed steel hardware to dominate the body inertia after total-mass calibration. The corrected native moments lie within broad material-derived slab bounds in every tier; decorative and hinge simplification can still change their detailed mass distribution.

`doorbench.paired_mechanics_qa.run_dutch_join_qa(model, metadata)` uses a fresh native state and two complete service cycles. It seats both leaves, inserts the rod with force at its physical knob, loads the upper leaf against the lower, withdraws the rod, opens the upper leaf alone, closes and rejoins, then opens the lower while the upper follows passively. The ordinary lower latch is held withdrawn to isolate this hardware. Finger force is capped at 20 N and leaf fixture torque at 40 Nm. The closing controller brakes toward a 0.20 rad/s maximum speed target before seating; it does not drive an abrupt position step into the stop. No intermediate native pose is assigned during these cycles. All contacts count toward the unchanged 1 mm penetration gate, and any native warning fails the proof.

All 12 doors passed all three tiers: 72 complete cycles. Maximum measured penetration was 0.107 mm and maximum relative leaf angle under joined loading was 0.001157 rad. Actual keeper contacts carried load in both tested directions. Native negative tests remove the keeper, fill a guide, restrict the lower hinge or move the grip off its knob; all fail. Every door also passed the existing full generic geometry/running-clearance check. These checks establish the recorded service behavior, not robot access, strength, one-hand usability or benchmark success.

The source-bound receipt is `out/mechanical-foundations/paired-dutch/receipt.json`, SHA256 `ec23d4a0d4f4f4c1465d275f49347fe8c9f33ac0463d5f2b2d8955e34eca44d9`; `clearance.json` records the separate generic checks. Source tests are `tests/test_dutch_joining_bolts.py`. Earlier prototype receipts preserve their failures and are superseded by this compact hardware and braked native service fixture.

The separate inventory identified 28 inactive paired leaves: 20 declared flush-bolt and eight cane-bolt installations. Their physical holding/release reconstruction is a following task; this Dutch receipt does not certify the previous inactive-leaf range restriction.

## Primary task and personal inspection follow-up

The subsequent source adds supported upper-leaf pulls on both faces: a 14 mm bar,
160 mm length and 31 mm finger gap, attached through two stems and fixing pads.
DB0095 and DB0906 replace incomplete handlesets with complete knob sets. The peek
case opens only the upper leaf. Unjoined passage cases open the upper leaf before
the lower leaf; joined cases keep the joining bolt inserted.

All 12 pass fresh whole-door QA and native primary task recordings under
`out/mechanical-foundations/paired-dutch/native-access-v2`. Every saved native
frame matches the recorded source. The parent reviewer personally inspected all
12 in 36 initial, operating and opened views. This is a bounded review of the
mechanism poses and selected contacts, not a continuous embodied reach, balance
or grasp certificate. The earlier component receipt above predates these
additional pulls and must not be presented as a hash of the newer complete door.
