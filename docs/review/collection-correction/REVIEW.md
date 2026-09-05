# Baby-gate headroom and supplementary pet collection

The September 5 correction removes the overhead wall from all ten baby gates and separates the fifteen standalone pet doors from the robotics benchmark. The website combines a checksum-pinned base release with a small, independently verified correction archive. All 1,000 assets remain downloadable; the standard collection contains 985 doors across 29 families.

## Baby gates

All ten were rebuilt from their exact published authored specifications. The change removes `wall_header` and preserves moving geometry and derived physical parameters. Current master's existing wall-position constant also moves the two side walls by 13 mm relative to the older published release; that difference is retained explicitly. Shared hardware is byte-identical.

Each corrected gate passes full QA, penetration and running-clearance checks, and a new compiled-geometry test of the passage above the gate. That test includes visible non-colliding geometry and does not depend on the header's name. All three MJCF tiers were checked. All ten gates were rendered in Blender/Cycles with their existing material/room recipes; the coordinating agent personally inspected every render. An independent Blender check also examined source geometry, added details and room context in all ten packed scenes.

The reviewed doors are db0176, db0332, db0336, db0483, db0505, db0661, db0675, db0698, db0844 and db0853. Their old native/reference clips retain their original bytes and source hashes, but the corrected catalogue disables their playback because they describe the prior geometry. Packed Blender scenes remain local; the compact website correction contains images and accurate image-only metadata.

This is a headroom correction and bounded geometry review, not child-safety or whole-dataset certification.

## Pet-door scope

`pet_door` is supplementary regardless of stale scenario metadata. It is excluded from default selections, explicit benchmark requests, scenario creation, `DoorEnv`, reference generators and Isaac task selections. Raw simulator asset validation and downloads remain available. Fourteen ordinary doors with a pet-flap insert retain their standard-door eligibility.

The separate pet catalogue offers the 15 models and individual downloads without evaluation, baseline or reference controls. Deep links cannot turn these controls back on. Historical raw result JSON is unchanged; displayed tables recompute the eligible subset from its recorded episodes and record the original file checksum. The three historical core scores become 849/985, 150/985 and 42/985. This filtering is not a new policy run or a result against corrected geometry.

## Outstanding findings from broader checks

The full current-source regeneration signs off 989/1,000 doors. Its eleven failures are pre-existing paddle support-neck/pin overlaps: db0039, db0074, db0116, db0158, db0347, db0536, db0615, db0648, db0660, db0884 and db0973. An independent rebuild from parent `3b3acd3f21f161d8ccb6e14c5d9534889973ea1b` produced byte-identical model JSON, all three MJCF tiers and hardware, with identical failure pairs and depths. The reported 4 mm interfaces require a separate mechanism review; no collision exception was added here.

The older website assets also do not pass every newer geometric check: applying the current checker to the complete correction candidate yields 972/1,000 penetration passes and 803/1,000 running-clearance passes, including coupling-definition differences in the older exports. Those unrelated models are unchanged by this correction. Their saved signoffs remain historical. The newly checked baby-gate subset passes 10/10 in both categories.

Detailed generated evidence stays in ignored `out/baby-gate-review/`, `out/collection-release/` and `out/pet-collection-browser/` in the publication worktree. The public correction manifest records the source commit and exact changed-file inventory.
