# Native contact and state timing

`NativeTransitionRecorder` archives the real MuJoCo dynamics solution before refreshing body poses. This is privileged teacher/evaluator infrastructure, not an actor observation interface.

```python
from doorbench.dexterous.native_transition_audit import NativeTransitionRecorder
recorder = NativeTransitionRecorder(sim, "leaf_handle_lever_col_n",
                                    handle_joint="leaf_handle_hinge")
recorder.before_step()
sim.plant.step()                 # exactly one physical timestep
row, raw = recorder.after_step() # before any mj_forward or other refresh
```

For Euler/implicit stepping, `mj_step` leaves contact frames and body transforms at the preceding dynamics configuration while integrating `qpos`/`qvel` to the new endpoint. `after_step` verifies those transforms against the saved pre-state, then copies every contact wrench, contact frame, distance and contacted-body transform before any refresh. The row's anatomical contact classification and load evidence use that solution. Its `pre_integration_state` names the matching joint/root state. The separate endpoint fields check newly integrated joint limits, passive loopback limits, root height and tilt.

Only `mj_kinematics` refreshes the endpoint's body/site poses. No extra force solve is performed. RK4 is rejected because its intermediate force epoch needs a different recorder. A later `mj_forward` cannot alter the detached archived solution, but its recomputed contact forces must never be presented as the preceding interval's physical forces.

The next teacher call receives current endpoint poses and joints, plus `recorder.hand_forces` and `recorder.left_surface` from the completed interval `[recorder.contact_time_s, recorder.contact_interval_end_s]`. Those loads are explicitly interval observations; they are not instantaneous endpoint measurements. The initialized reset observation uses a zero-duration interval and is not a physical transition. The full-opening teacher requires the interval argument separately from `pose_time_s`.

`raw` preserves actual pre/post states, controls, original motor forces and all contact data for independent reevaluation. `row` separates `contact_geometry_time_s`, `contact_interval_start_s`, `contact_interval_end_s`, `measurement_pose_time_s`, and `joint_state_time_s`. IDs, exact world geometry and object relationships belong only in this privileged archive.

## Evidence status

The original full-opening primitive mixed old body measurements and newly integrated joints. Follow-up trials named `full-opening-teacher-coherent-001` through `010` used an additional `mj_forward`; their refreshed loads are counterfactual endpoint solves, not the actual preceding transition. All are retained as diagnostics and do not qualify the new synchronized force-only teacher.

`full-opening-teacher-actual-001` exercised the corrected recorder into operation, with geometry-epoch assertions intact, but local storage filled before completion. It is an incomplete attempt, not a pass. Full physical qualification is pending a fresh complete trial. Tests use real MuJoCo contact dynamics to verify force signs, matching pre-state geometry, immutable archived buffers after a counterfactual solve, and rejection of unsupported/incorrect step ordering.

## Lossless raw archives

`NativeTransitionArchive` stores complete float64 arrays in compressed NPZ chunks with ragged contact/body offsets. It preserves every contact, including unloaded/speculative pairs, and uses neither quantization nor pickle. Chunk hashes and counts are checked on reading; interrupted streams remain explicitly incomplete. This avoids repeated JSON text overhead while bounding writer memory.

```python
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
archive = NativeTransitionArchive("out/run/raw-transitions")
archive.write(raw)  # each recorder result, every physics tick
archive.close(complete=True)  # only after the full trial finishes
for transition in NativeTransitionArchive.read("out/run/raw-transitions"):
    pass  # independent evaluator
```
