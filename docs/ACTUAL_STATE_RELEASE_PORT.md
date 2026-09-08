# Remaining actual-state release port

The current whole-body return and ungrip controllers are native development
components. Their exact attained-state guards deliberately reject a copied
native path when an Isaac rollout reaches a different body/hand state. The
body-frame arm correction and scalar palm-load feedback do not remove this
portability requirement.

| Component | Current admission | Remaining portable work |
|---|---|---|
| `WholeBodyLeverReturn.begin` | Exact screened root and body joints within1e−5 | Generate and independently screen a new return from the current supported grasp. |
| `WholeBodyMeasuredUngrip._begin_ungrip` | Exact newly attained root,69 joints, leaf and operator | Plan again after the physical lever reaches rest; do not assume the predicted return endpoint occurred. |
| `ScreenedWholeBodyPanel.begin` | Exact attained complete released state and aperture | Generate the flat-palm/body path from the new released state and validate actual imported collision geometry. |
| `PostOpeningTeacher.force` | Current measurements and independent actual prefix audit | Its runtime stow screen already exists; compose after qualified full opening without resetting the episode. |

The next planner extraction should separate the solvers in
`screen_whole_body_return.py`, `screen_whole_body_ungrip.py` and
`screen_whole_body_panel.py` from their archived-native input readers. Each
solver should accept an immutable numeric attained-state packet and an owned,
unstepped model; no active simulator handle or state-writing callback should
enter it. The packet must contain current robot root13, complete scalar q/dq,
door joint q/dq, actual body-origin and palm touch-site poses, actual foot
frames, current commanded targets/velocities, and separately timestamped
preceding-interval hand/palm contact evidence. A state hash and actual clock
must identify each returned screen. Force buffers must be copied before reuse.

For every new cluster/import:

1. Verify the path-independent source-design identity and original gains,
   masses, transmissions, joint/loopback limits and61 motor caps. Preserve the
   strict compiled-runtime receipt; do not replace it with source identity.
2. Validate authored FK against actual imported body/site poses. Re-screen
   collisions on the destination geometry; an old Mac screen or changed XML
   path string alone cannot provide this evidence.
3. Freeze the current supported-grasp state for an unstepped return solve.
   Keep attained foot frames as planning objectives, original caps and upright
   gates. Retain any infeasible plan and its residuals; never reset into it.
4. Execute the return through existing original motors. Measure the actual
   resting-lever state, then solve/screen the withdrawal from that new state.
   Do not reuse the predicted endpoint or relax the1e−5 admission guards.
5. Execute withdrawal with actual loaded RH patch checks and all scene shapes
   included in clearance. Only complete, measured safe release may start the
   freshly screened panel path.
6. Admit the existing post-opening controller only after the entire actual
   opening prefix qualifies. Preserve global time, preceding force interval,
   attained feet, last actually delivered motor effort and cumulative failures.

This is an explicit remaining work package, not a claim that the native paths
already work in Isaac or with another robot. Live PhysX release and full
opening/traversal must be reported separately from native evidence. Current
experiments and frozen failures are in [the opening evidence log](BIMANUAL_OPENING_SCREEN.md).
