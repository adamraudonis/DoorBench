# PhysX fingertip-pad evidence

`doorbench.dexterous.isaac_pad_audit` checks actual contact points and surface
normals in every contacted Shadow link's frame. It reuses the native
`grasp_verification.pad_opposition` rule. Four fingers need loaded distal volar
pads on one cylindrical side; the thumb needs its pad on the opposing side.
Dorsal, middle-link, knuckle, lateral, endcap, and extra misplaced digit patches
cannot pass through an averaged centroid.

This is a privileged evaluator. Its body identities, world transforms and exact
lever geometry never enter the finite actor sensor packet. The existing
initialized controller remains a privileged teacher.

## Integration

After the existing hand body/contact views are created:

```python
from doorbench.dexterous.isaac_pad_audit import PhysXShadowPadAudit
pad_evaluator = PhysXShadowPadAudit(hand_bodies, hand_contacts,
                                  handle_filter_index=0, side='rh')
```

After **each** `sim.step`, `robot.update(dt)`, and `door.update(dt)`, compute the
lever center/axis from the actual updated handle body pose and authored local
capsule center/axis, then capture:

```python
pad = pad_evaluator.read(
    physics_dt=dt, time_s=(step + 1) * dt,
    center=center, axis=axis, half_length=grip_half, radius=grip_radius)
physics_row['pad_grasp'] = pad
```

Also capture a t=0 row for complete-step acceptance. The reader obtains current
`RigidBodyView.get_transforms()` itself: those poses are **xyz, xyzw**, whereas
Isaac Lab `body_state_w` stores **xyz, wxyz**. Contact-row and transform-row paths
must match exactly. All buffers are copied before another backend getter can
reuse their storage. The reader cross-checks the sum of normal patch forces
against PhysX's independent pair-force matrix; disagreements fail closed.

Write the bounded per-frame dictionaries to a gzip JSONL evidence file. Include
`pad_grasp.valid_pad_grasp` in the same every-2-ms final-hold gate as the native
acquisition audit. A 50-Hz diagnostic trace cannot establish a 500-Hz hold.

The current native contact filter selects the entire handle rigid body. It does
not identify child colliders. Patches outside the lever's cylindrical surface
remain unqualified rather than being discarded. This is stricter than the
native lever-collider-only filter and is explicit in each result's scope.

## Verification status

CPU tests cover global rotation/translation invariance, correct xyzw transforms,
normal sign, all five volar pads, dorsal/middle-link/endcap rejection, extra bad
patches, missing transforms, row-order mismatch, and backend buffer reuse.
Seventeen combined pad/planning tests passed on 2026-09-08. **This module has
not yet been exercised live on the corrected v2 hand in Isaac.** The earlier
separate sensor fixture measured force-on-sensor signs, static/dynamic support
and pair-force agreement; that evidence does not certify the new anatomical
grader or a grasp.

The native [PhysX tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)
defines the contact force/point/normal buffers and per-pair counts and offsets.
The tested force-on-hand convention is negated to obtain the hand's outward
surface normal. Asset-local anatomy follows `grasp_verification.py`; a different
robot needs its own anatomical contract.
