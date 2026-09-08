# Continuous walking and full-opening development

The best native run reaches a 1.200824 rad aperture while upright, after real
walking, contact-free preparation, opposed acquisition and lever operation.
It **fails 1 of 29 checks**: right-hand withdrawal loads invalid finger surfaces.
It does not traverse and is not a qualified complete task.

| Native trial | Outcome | Diagnosis |
|---|---|---|
| [001](../results/dexterous/2026-09-08/native-walking-opening-001.json) | Failed, 85 s | The attained body heading differs by 13.119°. Original LH targets become poorly conditioned; fingers contact the panel without palm support and joint stops are exceeded. |
| [002](../results/dexterous/2026-09-08/native-walking-opening-002.json) | 28/29, 61.166 s; 1.200824 rad, 8.310 N final palm load | Actual-state LH targets work. Withdrawal produces 1,200 invalid RH contacts over 53.934–57.070 s. |
| [003](../results/dexterous/2026-09-08/native-walking-opening-003-recovered.json) | Failed; 1.200427 rad at 48.644 s without palm support | Unrestricted following removes aperture stiffness before LH support. Final report serialization also failed; recovered diagnostics and complete numeric prefix remain separate. |
| [004](../results/dexterous/2026-09-08/native-walking-opening-004.json) | Failed, 85 s; final leaf 0.086126 rad | Restricting following to sustained support and a bounded aperture still disrupts transfer. This option stays disabled by default. |

All attempts use the original capped motors, free base, eight passive hand
loopbacks and unchanged active model parameters. Actual `mj_step` forces are
archived with their preceding geometry; endpoint poses use kinematics-only
refresh. No runtime root pose resets, external support or direct door commands
are used. Reports do not replace the lossless raw transition archives.

The root independently inspected grasp and withdrawal close-ups at 30.622,
54.022 and 57.062 seconds. In the last view the lever end bears on the little
finger during withdrawal. The release controller is being corrected before
claiming success. The renderer now checks exact recorded robot/door input hashes
and accepts `--at` for specific recorded times.

## Actual-state planning and Isaac

The [landed LH planner](LANDED_LEFT_CONTACT_PLANNER.md) reproduces trial002's
61 targets and audits 601 nominal and 601 Cartesian/IK states. For the actual
Isaac pre-LH state, the first adaptation
[fails early finger contact](../results/dexterous/2026-09-08/isaac-left-plan-001.json).
A separately declared 3 mm panel-normal clearance envelope preserves both
endpoints and [passes the dense audit](../results/dexterous/2026-09-08/isaac-left-clearance-003.json).
The same calculation also passes in the destination Linux runtime. This is
sampled geometry, not physical contact qualification.

[Isaac full-opening002](../results/dexterous/2026-09-08/isaac-full-opening-v2-002.json)
acquires and operates the handle, but fails transfer and is gracefully stopped
at 37.2 s. Isaac003 fails before physics because launch metadata created the
runner's required-new output directory. Isaac004 fixes launch ordering and tests
the actual-state LH targets with the explicit plain-v1 8 N panel controller;
measured-leaf following is disabled. It begins on September 8, 2026 at
14:51:10 UTC. Its source, inputs and destination planning evidence are frozen.
No result is claimed before its final report and independent review.

## Scene and continuation boundaries

These native walking runs use door XML `5b12a0a19c8b…`. The live Isaac native
counterpart is `db3c68cd54e8…`; their textual difference is a degenerate inertia
principal-frame representation. The older initialized passage and palm-contact
convergence experiments use `d3b367b62eed…`, which has different mechanism
geometry. Their internal replay checks do not establish cross-scene equivalence.

The portable passage controller now builds a fresh stow route from actual
measured state. The current door's first route is rejected for collisions, and
its first physical sequential-route attempt hits one stance-solver iteration
limit. Both failures are retained. Joining approach, opening and passage still
requires one uninterrupted physical qualification, then a separate Isaac run.
