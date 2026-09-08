# Declared Shadow contact profiles

`distal-pad-v1` remains the default. Its surface predicate and force/opposition
thresholds are unchanged. `volar-phalange-v1` is an explicit option for a new
experiment, frozen in [this protocol](../configs/dexterous/grasp-profiles/volar-phalange-v1.json).
Historical operation-v2-004 remains failed under its original distal protocol.
Its separate retrospective review motivated a new test; it is not qualified
training data.

The new profile accepts inner proximal and middle surfaces of the four fingers
in addition to their distal pads. It retains a **distal thumb** on the opposing
side. This is an opposed volar-phalange contact contract, not a claim of palm
contact or a textbook power grasp. The [GRASP taxonomy](https://www.eng.yale.edu/grablab/pubs/Feix_THMS2016.pdf)
distinguishes contact surfaces and opposition as well as hand pose. Our selected
surface contract is an engineering specialization, not a result prescribed by
that taxonomy. The [manufacturer's finger description](https://shadow-robot-company-dexterous-hand.readthedocs-hosted.com/en/latest/user_guide/md_finger.html)
defines the separate finger links and unilateral J1≤J2 constraint; the local
surface extents are calibrated to the frozen robot asset.

Both engines use the same surface predicate: local Y below −1 mm, outward normal
with more than 0.5 along local −Y, and local Z from 2 mm to the link's surface
extent. Extents are 45 mm for proximal, 25 mm for middle and 40 mm for distal.
The four-finger proximal and middle lengths agree with the actual XML child-joint
offsets; the distal band retains the existing calibration. The 2 mm joint-origin
exclusion and explicit segment names reject proximal joint and knuckle loads.
Thumb middle/proximal surfaces remain excluded.

The cylinder-side, axial-clearance, radial-normal, all-five load and opposition
thresholds stay unchanged. So do joint limits, loopbacks, collision tolerances,
native motor caps and the prohibition on runtime pose writes or assistance.
A broader surface set does not permit a wrong-side or dorsal contact to disappear
behind a finger's average contact position. Task-specific acquisition and final
hold durations must still be declared by the task protocol.

Use `profile='volar-phalange-v1'` in `shadow_lever_pad_grasp`,
`native_grasp_sample`, `audited_native_step`, `shadow_physx_pad_grasp`, or the
`PhysXShadowPadAudit` constructor. Unknown profiles raise an error. Each result
declares `grasp_profile`. When the broader profile is selected, every contact
also preserves `distal_pad_qualified`, and `distal_pad_grasp` contains the
unchanged original aggregate score. Keep both in the saved 500 Hz evidence and
record the selected profile before the next run.

Regression coverage includes positive middle/proximal contact cases, unchanged
distal boundary behavior, native/PhysX surface parity, and rejection of dorsal,
wrong-side, lever-endcap, off-segment, joint-origin, knuckle, lateral, unverified
thumb-segment and extra bad patches. Forty-one verifier tests pass. A fresh
physical result is still required before this profile can qualify an episode
for training.
