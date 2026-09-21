# Detached panel bridge inputs

`isaac_panel_bridge_seed.prepare_isaac_panel_bridge_seed(source, *, continuation_audit, panel_candidate, robot, door_xml, door_usd)` prepares numeric inputs for a future continuous bridge. It freshly reuses the existing actual released-Isaac source admission, panel candidate integrity validator, continuation observation admission, and three-reference audit. It does not authorize or execute a stage. No qualifying actual withdrawal is assumed by its synthetic tests.

The returned `doorbench.isaac-panel-bridge-seed.v1` preserves:

- `previous_reference`: the exact accepted 75 nominal values and stored coordinate velocities at command time **T−0.002**, including their original root chart and exact 69 short joint names. These values are distinct from `measured_terminal` at **T**.
- `panel_first_target_in_old_chart`: the first panel position reference converted into that old root chart, with only the original 25 panel joints replaced. The remaining **44 previous nominal joint values** are copied exactly. They are never replaced with measured fingers. This vector therefore differs from the static candidate and inherits none of its geometry checks.
- `predecessor_controller`: the recorded stance targets/biases/primal warm start, RH goals and corrections, LH IK/support/filter values, finger release/preload state, and actual enhanced tracking contract. These are observations, not a controller restoration recipe.
- `support`: the unchanged source target and the actual active target separately. For `release-unload-restore-v1`, the last command can request `5.999999720839327` N while its source target remains exactly `6` N. Existing profile accounting establishes the recorded event clock; this adapter neither rounds nor reevaluates the load profile.
- `preceding_submitted_motor_handoff`: the original 61 submitted command values, never a delivered-torque claim. The source endpoint state hash and the original withdrawal-entry state hash are kept separate because they describe different epochs.
- `held_left_goal`, discrepancy vectors, all three references, and input hashes. Every bound file is checked again before return.

`panel_coordinates_in_withdrawal_chart(seed, coordinates31)` is a pure position mapping. Its root arithmetic is:

```
world_position = panel_origin_position + panel_translation
world_rotation = Exp(panel_rotvec) * panel_origin_rotation
old_translation = world_position - withdrawal_origin_position
old_rotvec = Log(world_rotation * withdrawal_origin_rotation.T)
```

It preserves the other 44 previous nominal joints. The stored old-chart velocity remains unchanged in the seed. A rotation-vector derivative is not a world angular velocity or the model's body angular qvel. No future panel derivative, missing acceleration, limiter state, or new target at T is invented.

The LH goal is held by a 100 Hz IK loop, so its final `last_update_s` can precede all three tail commands. The helper reads the exact leaf pose at that saved epoch from the admitted physics archive, with bounded memory and complete stream checks. Reconstruction is restricted to the original minimal Isaac withdrawal mode: all three rows must have completed LH progress, the fixed captured material path, constant panel-local orientation, and an offset unchanged between identical IK epochs. Other runtime options or an unavailable/stale/misaligned IK epoch are rejected. The reconstructed goal includes the recorded leaf-normal offset once.

`left_path_goal_before_offset(leaf_pose, world_position, world_rotation, normal_offset_m)` expresses a desired world touch goal in a new leaf frame and subtracts the existing local +Y offset once. A later consumer that retains its normal correction would add it back once. The seed includes this calculation at the measured terminal leaf pose, but does not change the current controller's path or target.

The remaining engineering work is an explicit continuous bridge and settling phase, including measured-leaf geometry, original motion limits and stance/support continuity. A source may satisfy the withdrawal's 1 mm pose limits without satisfying the panel's 0.1 mm limits. This module does not tighten those bounds or claim a feasible transition. Every output retains zero authorized stages, zero physics/playback/state writes, and false runtime-export, restoration, bridge-feasibility and physical-task authority.
