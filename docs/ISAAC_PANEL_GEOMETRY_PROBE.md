# Detached panel geometry points

`IsaacPanelGeometryProbe(context, candidate)` validates the existing distinct
Isaac candidate against a freshly admitted released source, then creates its
own private MuJoCo geometry calculator. It never takes a dynamics step or reads
or writes a live Isaac plant. It is not connected to a controller.

The existing dense audit checks a single declared leaf-lag slice. This helper
evaluates an explicit reference aperture against all three measured mechanism
coordinates. The reference aperture determines the requested left-palm pose;
the measured leaf, operator and latch remain in the private collision scene.
Thus either lead or lag remains visible in the collision calculations.

Coordinates retain the panel convention: six root displacement/relative
rotation-vector coordinates and 25 named joint angles. An optional complete
`robot_joint_targets` mapping supplies every original scalar robot joint,
including the released fingers, and must match the 25 coordinates exactly.
All supplied joints affect FK, joint bounds, COM and collision checks. Omitting
the mapping explicitly holds remaining joints at measured source values;
this default must not be mistaken for a continuous 69-joint handoff.

The original static panel limits are unchanged: hand/foot position 0.1 mm,
orientation 0.001 rad, root translation 30 mm, root rotation 0.05 rad, COM shift
15 mm, torso tilt 4 degrees, RH/environment clearance 40 mm, elbow/environment
clearance 3 mm, and unchanged joint-limit and penetration rules. Collision
coordinates retain any source joint-limit excess plus the original 1e-6
increase allowance; tiny measured PhysX excursions are never clipped to zero.
Larger mechanism excursions reject before calculation. Collision
checks include original active and receiving-only shapes. Signed plane bounds
retain shapes below a plane before exact distance calculation.

Call `verify_inputs()` before and after a batch and save `probe.hashes` in any
future batch receipt. Rehashing source archives inside each geometry point
would be wasteful; point results do not claim that files stayed unchanged over
an independently running batch. Constructor validation and explicit batch
checks are separate from each point's numerical result.

A passing point grants no runtime stage, mechanism domain, rate envelope,
handoff continuity, palm load, tracking or physical qualification. A future
domain audit must sample its declared domain, preserve all failures, bind the
inputs before and after sampling, and state the limits of a sampled proof.
A future bridge must additionally bound all 69 target derivatives and retain
the live support and controller histories. Neither is implemented by this
helper. Synthetic tests compare the helper to the existing dense geometry,
exercise both signs of lag, include an extra finger outside the panel subset,
and preserve below-plane/nonfinite failures; they are not H1 performance tests.
