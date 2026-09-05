# First complete constrained-motion baseline

The September 5, 2026 run attempted all 1,000 doors with a fixed adult rig and an independent validator. It produced **61 traversal references across eight door families and 51 locked-door checks**. Another 404 candidates failed validation and 484 cases remain unresolved. No attempt failed because its worker crashed or timed out in this final run.

| Result | Doors |
|---|---:|
| Accepted: open and traverse | 58 |
| Accepted: unlock and traverse | 3 |
| Accepted: locked-door check, without traversal | 51 |
| Rejected candidate | 404 |
| Unresolved | 484 |

These are **prescribed-door kinematic references**. Acceptance checks sampled rigid-body geometry, permitted contact, planted feet, joint and derivative bounds, and the declared actor route. It does not certify that the actor causes the door to move, understands its lock, maintains dynamic balance, or meets the original benchmark clock. “Unlock and traverse” names the source task; it is not a certificate of humanoid unlocking.

The accepted clips span 11 families overall. All use at most one active hand. Nine show meaningful motion of multiple leaves or flaps; three are uncoupled double doors whose source commands both leaves simultaneously. Those cases need a mechanism-aware, potentially bimanual schedule before they can represent causal operation. Among the 61 traversal clips, 53 exceed their original task budgets; median duration is 75.81 seconds and maximum duration is 117.93 seconds.

## Visual review

The primary agent personally inspected Blender phase samples for all 112 accepted clips, covering the approach, operation, traversal when present, and final pose. The 19 overview sheets contain five selected samples per clip; the underlying storyboards retain additional exact sampled poses. The actor and native door transforms were checked against the source arrays during rendering.

Posture is more upright than the initial pilots. Remaining concerns include slow, segmented progression, bent knees, flat-footed walking, spherical hands without an oriented grasp, and incomplete manual sequences for multiple moving parts. Open leaves obscure parts of several trajectories in the fixed camera view. Context walls are hidden for inspection, so the sheets cannot assess the complete room appearance.

This is **phase-sample inspection, not full-motion visual approval**. No clip receives a personal visual pass from these sheets. Motion Lab provides separate local review notes and decisions bound to each exact clip version.

## Reproducibility

The final local corpus is `out/reference-planned-corpus-v1`; Blender storyboards are in `out/planned-motion-storyboards`. The final claim audit rechecked all 1,000 result/source bindings and closed artifact inventories (10,686 unique bound files). It independently recomputed native-pose interpolation and actor aperture/goal evidence for all 112 accepted clips. It verified the saved independent collision reports and their hashes; it did not repeat the complete collision computation during packaging.

| Binding | SHA-256 |
|---|---|
| Corpus index | `67bed7b37c45f96bfc9a014a4ea649d8764a10d98025ca648c6e08bd1f1949b0` |
| Native recording index | `d9f5c60f6f76992e7a1a7efba0163bab63defe8e7bef0400cfd58aea0fbfe204` |
| Dataset manifest | `97699b43489a06b64f90d605ef87aec156e4ccc97fb9bbf0db61dac01c05ab34` |
| Generator inventory | `12d8e3d5991e0c328240386e242f120193621911de726cb1d5e5372887aa121f` |

See [the planner and validation commands](../PLANNED_REFERENCE_MOTIONS.md), [release preparation and downloads](planned-reference-release.md), and [the detailed limits of the evidence](planned-reference-scope.md). The earlier `v2026.09.05` native dataset and its oracle outcomes remain a separate release.
