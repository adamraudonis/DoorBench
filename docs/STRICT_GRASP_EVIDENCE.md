# Strict evidence for the Shadow hand lever grasp

A loaded finger is not necessarily a correctly placed finger. The historical centroid-only score accepted a dorsal index-finger collision, a middle-finger tip collision, and a dorsal thumb press in recent failed trials. None of those trials established a successful acquisition; the new audit explains why their intermediate contact feedback was misleading.

The opt-in evaluator is `doorbench.dexterous.grasp_verification`. It is currently specific to the unmodified H1/Shadow hand and a straight cylindrical lever. This is a deliberately narrow contract for the chosen canonical reference, not a universal definition of natural grasping. Legitimate power grasps can load volar middle or proximal finger surfaces; those require an explicit mechanism-specific surface contract. Dorsal thumb/finger contacts remain invalid for the intended palmar grasp. A different robot must supply its own anatomical pad definition.

## Chosen canonical straight-lever contract

- Each of the four fingers and the thumb must load its **distal volar pad** by at least 0.2 N. For this Shadow asset the volar face is local −Y; the contact must lie beyond −1 mm in Y, between 2 and 40 mm along the distal link, with an outward normal having a −Y component greater than 0.5. Knuckles, dorsal faces, and tip-only contacts cannot substitute.
- The contact must be on the cylindrical side of the lever: at least 1 mm from the cylinder/endcap boundary, with hand-to-lever normal alignment greater than 0.8 against the inward radial direction.
- All four finger directions must agree pairwise within 60°, and the thumb must oppose **each finger** by more than 120°. Every loaded patch is checked; a digit cannot conceal a load on the wrong side behind its force-weighted average. At most 5% of a digit's load may fall outside the qualified region.
- These conditions must hold continuously for the declared hold interval, default 0.5 s, with evidence at **every physics tick**, including time zero and the final tick. Sparse 50 Hz traces cannot establish a 500 Hz safety claim.
- For acquisition, the leaf and lever start within 1 mrad of their closed/rest positions, and the hand starts without scene or self contact. Initialized diagnostics must explicitly disable only the contact-free-start requirement and keep their initialized scope.
- Every tick must satisfy the existing development bounds: native motor force limits, ≤20 mrad soft joint-limit excursion, ≤3 mm nonfoot penetration, torso tilt <12°, root height >0.7 m, finite states, and no numerical warnings or external applied assistance. These are the current simulation tolerances, not hardware safety or a proof of biological naturalness.

The verifier does not certify appearance or natural motion from scalar contacts. A release candidate also needs zoomed multi-angle hand inspection through reach, seating, lever operation and release; whole-body inspection; velocity/acceleration/contact-load traces for chatter and impacts; and a complete physical approach/opening/traversal result. Keep those outcomes separate from the strict grasp flag. Do not derive a naturalness claim from this flag.

## Native integration

Capture the initial sample after the declared reset, then use the audited wrapper in place of each `sim.plant.step()` call. It applies exactly the same single native step; it adds no force or constraint.

```python
from doorbench.dexterous.grasp_verification import (
    native_grasp_sample, audited_native_step, audit_grasp_steps,
)

kwargs = dict(handle_joint="leaf_handle_hinge")
rows = [native_grasp_sample(sim, "leaf_handle_lever_col_n", **kwargs)]
for _ in range(round(horizon / sim.m.opt.timestep)):
    # Compute and assign bounded robot motor controls here.
    rows.append(audited_native_step(sim, "leaf_handle_lever_col_n", **kwargs))
report = audit_grasp_steps(
    rows, physics_dt=sim.m.opt.timestep, expected_duration=horizon,
    required_hold=0.5,
)
```

`DoorEnv.step()` clears `qfrc_applied` and `xfrc_applied` after advancing physics. Consequently, post-step diagnostics reading those buffers alone do **not** prove absence of assistance. The wrapper captures body wrenches before stepping and reads `plant.last_applied_qfrc` afterward. A post-step sample without the pre-step body-wrench record is rejected. This still requires review of the plant/controller source and a preserved model contract: force buffers cannot reveal silent mass, geometry, friction, limit, support-constraint, or state-write changes. Do not accept an uninstrumented constant `runtime_pose_writes=0` as independent evidence.

Preserve original model/asset hashes, native motor force caps and gains, constraint inventory, control units, and the executed controller source. The explicit motor adapter may convert position-servo commands into the equivalent original servo law with native caps; that conversion is not permission to change strength. `scalar_transmission_matrix` includes joint and fixed-tendon motor gearing and sums repeated coefficients. Spatial tendons, ball/free transmissions, and incomplete ordering fail explicitly. The current Shadow tendon gears are one, so the previous missing tendon-gear multiplier did not cause the current grasp failure; it would affect a different geared robot.

## Audited examples, 2026-09-08

These are re-evaluations of selected stored native states, not retrospective proofs of every 2 ms in an older rollout.

| State | Strict anatomical result |
|---|---|
| Canonical initialized 6 N grip, middle and final states | All five distal volar pads pass |
| Standing-height initialized grip, middle and final states | All five distal volar pads pass |
| Failed acquisition-013 final | RF/LF volar; FF dorsal; MF tip; thumb unloaded |
| Thumb-first-001 final | Thumb force 4.46 N is dorsal, therefore invalid; all fingers unloaded |
| Thumb-first-002 final | MF/RF volar; FF/LF/thumb unloaded |

The canonical loaded thumb is near the handle end but remains on its cylindrical side: axial position 49.58–50.28 mm on a 53 mm half-length cylinder, leaving 2.716 mm to the cap transition. Its hand outward normal matches the inward lever radial normal by 0.9999995; its volar −Y component is about 0.867. In contrast, thumb-first-001 contacts local Y≈+7.20 mm with outward +Y≈0.826.

Full local evidence is `/tmp/doorbench-standing-grasp/strict-pad-review.json`, with the immutable summary and inspection script in `handoffs/diagnostics/2026-09-08-standing-grasp/`. The historical reports remain preserved; their positive intermediate thumb-contact labels must not be treated as anatomical success.
