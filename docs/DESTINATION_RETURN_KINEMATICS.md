# Bind return planning to actual destination poses

The opt-in `admit_destination_return_kinematics` helper checks copied robot/door coordinates against actual handle, palm, foot and torso body poses before an unstepped planner can use them. Its explicit float32 profile requires position agreement within 2 micrometres and rotation agreement within 2 microradians. It retains measured and reconstructed poses separately, records quaternion normalization, rejects stale/incomplete measurements, and changes no active simulator. The original native return planner retains its 1e-9 check.

An archived Isaac sample from `own-imu-grasp-003` passes for **72 mapped bodies** at 0.002 seconds: maximum position error 0.837 micrometres and rotation error 1.843 microradians. The legacy producer wrote its ambiguously named `initial-body-poses.json` after the first physics step; the exact recorded root and first sensor/physics epochs were matched before comparison. Future captures explicitly include that timestamp and convention. [Evidence summary](evidence/isaac-return-kinematics-001.json).

This validates one measured kinematic mapping, not a release-state plan, collision model or physical motion. Three USD wrapper bodies have no native counterpart and remain explicitly listed. Seven focused tests include stale poses, incomplete state, missing/duplicate bodies, invalid quaternions and a rejected millimetre displacement; stepping entry points are disabled in the positive test. Full Isaac release still needs a plan built at the actual reached state, a dense geometric screen and uninterrupted physical qualification.

The destination admission now feeds `iter_destination_whole_body_return`, which
uses the same 41-node solver as the strict native entry point. Each node carries
the measured-state admission receipt. A recorded Isaac first-step fixture produces
exactly the same 41 solutions as the native solver given its admitted private
state, with physics stepping disabled (4.52 seconds on CPU). Sixteen focused
admission/planner tests pass. This verifies the adapter and unchanged solver;
it does **not** qualify a release pose, destination collision model or physical
return motion. Those remain required before using a generated path in Isaac.
