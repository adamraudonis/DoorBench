# Reference-motion and research UI review

All 1000 doors have native MuJoCo recordings for their primary core scenario at seed 0, plus a separate procedural articulated figure. Native outcomes: **879 success, 118 fail, 3 damaged**. These single-scenario recordings are separate from the historical multi-scenario benchmark scores.

## Numeric and source checks

The corpus validator checks every door ID, every source/artifact hash, compressed/plain clip equivalence, finite native/actor arrays, strict timelines, native/web agreement, fixed metric limb lengths and nonnegative ankle height. Eight native pose samples per door are independently recomputed from qpos to detect stale world transforms. Every door passes these integrity checks.

The recorder observer is tested against unobserved benchmark runs across swing, bifold, saloon, revolving and tripod families. A strict fallback-family test compares live qpos, qvel, qacc, warm start, controls, applied forces, body poses and time before/after every observer call. World poses are refreshed on private data; reset-time force limits are passed from the runner.

**531 recordings contain at least one reference hand-target error above 8 cm.** These remain explicit in the index, clips, native arrays and viewer. The remaining recordings are not certified contact-feasible either.

## Personal visual screening

The coordinating agent personally inspected 64 playback views across 32 doors spanning all 30 families, followed by camera/layout corrections and a final representative pass. The review included small pet doors, overhead and floor hatches, large gates, sliding rails, bifold, revolving, turnstile, automatic/elevator, ship dog mechanisms and keypad hardware.

The screening found limitations that remain visible: the kinematic figure may overlap tilt-up panels, strip curtains and hatch geometry; overhead targets can exceed reach; a fixed-size humanoid cannot traverse small pet openings; failed automatic/elevator/accordion attempts remain failed. These references show task sequence and recorded door motion, not collision-free humanoid execution.

An actual Blender Cycles barn-door image was inspected. Reopening the packed animated scene confirmed 33 animated objects and 12 packed images, with zero measured position error at three native door and three actor samples. The exporter remains separate from the frozen appearance renderer.

## Website and review workflow

Clean-browser checks cover Catalogue, Door types, Results, About and Review at 1440 and 390 pixels: no missing images, horizontal overflow or JavaScript errors. Playback checks cover brown/gold materials, exact restoration of source materials, play/pause, manual reset, rapid door switching and mobile layout. The mobile renderer reserves space for controls so the figure is not hidden behind its timeline.

The human review workspace has validated import preview/merge, actual export/import between isolated browser profiles, local persistence, keyboard safety, undo and explicit appearance/construction/mechanism assessments. Queue scrolling does not move the outer page. Human assessments remain separate from automated QA.

See [reference format and Blender use](../../REFERENCE_MOTIONS.md), [human review](../../HUMAN_REVIEW.md), [existing construction audit](../takeover/REVIEW.md) and [all-door Blender appearance screening](../blender/REVIEW.md). Numeric/browser receipts and contact sheets are retained under ignored `out/reference-review/` and `out/ui-review/`; generated datasets and packed scenes are distributed outside Git.

## Published release verification

On September 5, 2026, the research release was merged with the concurrently completed Isaac parity tooling at master `6c1c98435867ca72d630bbd079d70511e8a44210`. The combined source passed 679 Python tests (one skipped), 29 viewer tests, typechecking and the production build. The catalogue retains the new parity filter and badges.

[Pages run 33951862652](https://github.com/adamraudonis/DoorBench/actions/runs/33951862652) succeeded. The live deployment, dataset, appearance and reference index hashes match the pinned bundle; a compressed clip and its decoded JSON match the recorded source hashes. Browser checks verified reference playback, pause, brown/gold contrast, mobile layout, the human-review workspace and historical result loading. Blender image/metadata and packed Blender/USDC downloads also passed. All 122 shared USDC dependencies resolve in the restored bundle; the complete site is approximately 916 MB.

The public, ungated [Hugging Face release v2026.09.05](https://huggingface.co/datasets/adamraudonis/DoorBench/tree/v2026.09.05) resolves to `6e17f0f588bf81fec0f04b2a329b471488164366`. Its published standalone downloader was executed anonymously, verified 17,246 simulation asset files, preserved the complete 22,310-file inventory and successfully loaded/stepped a downloaded door in MuJoCo. Source and archive provenance remain fixed to the frozen generation inputs documented in the release; merging later simulator tooling does not silently replace the published data.
