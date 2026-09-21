# Exact live witness for an actual released Isaac source

`LiveIsaacPanelPrefixWitness` is a detached prerequisite checker, not a panel controller. Nothing calls it from the active producer yet. It never writes a robot state, steps physics, replays a command, or converts a native plan into an Isaac proof.

Construction independently admits the completed actual released source through `admit_isaac_panel_source`, then freshly admits the saved continuation-observation audit. The supplied source endpoint SHA, source epoch, original 61-motor contract, physical coordinate order, sensor capture contract and backend motor-readback contract must agree. Source admission alone cannot authorize entry.

At every completed 2 ms interval, supply the actual newly recorded physics row and the actual `pack_standing_continuation` record. The witness compares all original core fields plus the measured leaf pose, backend submitted/reconstructed efforts, pre-step joint velocity, continuation body poses, foot loads and legacy root diagnostic. These NPZ values use exact dtype, shape and bytes. There is no numerical tolerance or interpolation.

The continuation JSON comparison canonicalizes dictionary key order only. Array order, sparse pair order, occupied slot identities, force/point/normal values, scalar JSON representations and summary decisions remain exact. Normal and friction inventories stay separate. A permutation of equivalent raw patches is **not** a passing ordered measurement comparison. Historical JSON contains scalar values, not the original tensor dtype; the receipt explicitly avoids claiming that dtype was compared. Independent contact accounting and any unordered multiset analysis are different evidence and cannot override an exact physical-prefix failure.

The comparator uses fixed blocks of 128 rows from each compressed NPY member plus one bounded JSON record. It does not retain an episode-sized array or record list. Existing upstream released-source admission may temporarily load larger source arrays; its one-time memory use is not claimed to be constant. Readers are closed on failure, completion or explicit `close()`.

All streams must end exactly at the admitted epoch. Missing, malformed, duplicated, reordered, extra, nonfinite, truncated or changed evidence rejects the witness. `require_stage_entry(T)` rechecks all hashes and authorizes exactly once after every row matched. An early, late, repeated or failed entry is sticky; its failed receipt stays available. Even a passing receipt grants no panel geometry, load, opening or walking success.

Proposed call shape (integration remains separate):

```python
witness = LiveIsaacPanelPrefixWitness(
    released_run,
    robot=robot_xml, door_xml=door_xml, door_usd=door_usd,
    expected_source_state_sha256=admitted_source_state_sha256,
    stage_start_s=source_terminal_time,
    runtime_configuration=actual_live_configuration,
    runtime_motor_contract=actual_live_motor_contract,
    runtime_continuation_contract=actual_live_sensor_contract,
    runtime_motor_readback_contract=actual_live_readback_contract,
    continuation_audit=independent_continuation_audit,
)
# Immediately after each actual completed interval, before a new-stage command:
witness.observe(actual_recorded_physics, continuation_observation=actual_observation)
# Only once all records match, at the exact source epoch:
receipt = witness.require_stage_entry(source_terminal_time)
```

The final recorded command belongs to the interval ending at T and was submitted at T−0.002. A later panel controller must inherit that actual returned command and the live withdrawal/stance/6 N support/filter state. The witness neither reconstructs nor resets those objects.

Extra read-only observer queries can change GPU synchronization or expose reduction nondeterminism. Transfer004/005 matched physical arrays through 42.260 s, then a 2.38e-7 N reduced-force difference preceded command divergence. That does not identify a unique cause, but it means exact reruns are not guaranteed. Keep the exact comparison. If it fails, preserve failure and consider a separately implemented pause at a qualified measured phase in the same live episode: freeze the physics clock, plan and audit copies, verify the plant/controller state stayed unchanged, then continue the retained objects without reset or playback. Such a pause requires its own distinct live-source proof and is not implemented here.
