# Second constrained-motion corpus: local preview

The complete second run adds **47 accepted traversals without losing any previous acceptance**. It contains **108 traversals and 51 locked-door checks**, with 357 rejected candidates and 484 unresolved cases across the same 1,000 doors. There were no worker failures. This revision is prepared locally as `planned-v2026.09.05.2`; it has **not been uploaded to Hugging Face**. The public Motion Lab still serves [the first supplement](planned-reference-release.md).

| Result | First run | Second run |
|---|---:|---:|
| Accepted: open and traverse | 58 | 102 |
| Accepted: unlock and traverse | 3 | 6 |
| Accepted: locked-door check, without traversal | 51 | 51 |
| Rejected candidate | 404 | 357 |
| Unresolved | 484 | 484 |

“Accepted” means the existing independent sampled kinematic and actor-route checks passed. Door motion is still prescribed from the source recording. Source labels such as “unlock” and “recognize” do not certify humanoid unlocking or recognition. This is progress toward useful reference motion; it does not establish natural, causal humanoid operation for every door.

## What changed

The planner now supports a guarded wait-and-traverse schedule for powered sliding doors, an explicit manual hasp-to-pull transfer, and checks the interpolated actor path during solver steps when native motion is held and neither hand is gripping. The adult rig, native dataset, collision/contact tolerances and independent validator remain unchanged.

The accepted set contains 12 powered schedules, two manual transfers and 145 baseline contact schedules. Powered cases require zero recorded generalized effort, no actor hand contact and no hand/door collision exemptions. Actual sensor or button triggering remains unverified. Both manual cases use an already-unengaged padlock/hasp; their small residual hasp effort is an explicit unverified mechanical-hold assumption.

Accepted traversals now cover nine families; all accepted references cover 12 of the 30 families. The exact comparison retained all 112 previous acceptances and found 47 rejected-to-accepted transitions. Changes to individual timings are preserved in the comparison, rather than substituting old accepted files into the new run.

## Timing, contact and visual review

**86 of the 108 traversals exceed their original scenario time budget.** Median traversal duration is 75.29 seconds; the range is 22.62–117.88 seconds. These retimed trajectories must not be reported as success under the original benchmark clock.

Every accepted clip has at most one active hand. Eighteen contain meaningful simultaneous motion of multiple door leaves or flaps, and 100 move rotational operator/lock hardware. Oriented grasps, articulated fingers, contact forces and causal control remain unfinished. All six accepted unlock scenarios retain source badge API events. Also, 157 of the 159 clips stop before the original native recording ends; later closure is outside these references.

The primary agent personally inspected five selected exact Blender poses for **all 159 accepted clips**, across 28 overview sheets. The underlying storyboards include additional phase samples and verify actor/native transforms. Diagnostic brown/gold materials make mechanisms easier to see, while contextual walls and transparent glass are hidden for inspection.

The review records bent knees, flat-footed walking, long segmented timing, occlusion by open leaves and missing grasp detail. All 159 local review notes are marked **needs work**, tied to the exact prepared browser clip checksums. This is phase-sample inspection, not full-motion visual approval or a photorealistic room review. See the [local preview command](../PLANNED_REFERENCE_MOTIONS.md#review-in-motion-lab) and the separate [stepping experiment](planned-reference-style.md).

## Reproduce and inspect

The completed corpus is `out/reference-planned-corpus-v2`; the prepared release is `out/planned-release/planned-v2026.09.05.2`. [Preparation and publication](planned-reference-release.md) are separate operations. Hugging Face updates are batched at most once per day; local experiments and source/website updates can continue between dataset releases.

```sh
.venv/bin/python scripts/compare_planned_reference_corpora.py \
  --before out/reference-planned-corpus-v1 \
  --after out/reference-planned-corpus-v2 \
  --out out/reference-comparison/v1-v2.json
```

The final claim audit rechecked all 1,000 result identities and closed artifact inventories, covering 10,689 unique bound files. It independently recomputed exact native-pose interpolation and actor aperture/goal evidence for all accepted clips. Saved collision reports and unchanged settings were verified by hash; this final packaging audit did not repeat the complete collision computation.

| Binding | SHA-256 |
|---|---|
| Corpus index | `3228ae28673d76ada4b7e6ce9d9fa487352c5687de50301a9be005b55e44a7a6` |
| Generator inventory | `954bf2e7e077866e0f9db8090f656c88cb3d5695722c54a995a042729a658c54` |
| Native recording index | `d9f5c60f6f76992e7a1a7efba0163bab63defe8e7bef0400cfd58aea0fbfe204` |
| Dataset manifest | `97699b43489a06b64f90d605ef87aec156e4ccc97fb9bbf0db61dac01c05ab34` |
| Prepared browser index | `017e112b89bcf9caae0d0622180ce87659f3e0ef4ad7308387eaca22767b02fe` |
| Prepared release | `5b44a25fbe28ec4b3b41627ec6a84c7544ee74580798a88203a53f18b3cfd561` |

Local evidence is retained under `out/reference-release-claims-v2`, `out/reference-comparison`, `out/planned-motion-storyboards-v2` and `out/planned-v2-local-qa`. Import `out/planned-motion-storyboards-v2/phase-review-notes.json` into Motion Lab to read the phase-review notes. These generated artifacts are not committed to Git. The separate [paddle correction](paddle-mechanics.md) changes future native geometry and was not applied to the frozen motion inputs.
