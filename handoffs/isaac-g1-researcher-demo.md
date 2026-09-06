# Isaac G1 researcher demo — September 6, 2026

The researcher walkthrough and original Unitree policy adapter are published on
master at `7d36d7921ff9400ed767c24476cabe071592e9ba`. The guide pins the complete
tested mechanical/source snapshot `85b4a81fe` on `codex/mechanical-master-integration`.
The nine preceding mechanical source commits remain under review; publishing the
guide did not publish those geometry changes to master.

## Completed

- `docs/ISAAC_G1_DEMO.md`: installation, pinned checkpoint, asset preparation,
  actual native results, and a researcher `module:factory` policy interface.
- Original G1Policy was extracted unchanged (AST-identical) from the MuJoCo demo.
- Native Isaac Sim 5.1.0 + Isaac Lab v2.3.2 on an L40S: **3/4 selected cases pass**,
  zero simulator errors in the final suite. Open doorway and automatic slider:
  8.902 s each. Saloon: 10.348 s, one passive leaf physically pushed open. Closed
  latched door: robot falls at 10.588 s, leaf opens only about 0.10 degrees.
- This uses canonical seven-joint USD and a root-based upright traversal metric.
  It is not a full benchmark, full-mechanism certification, or human reference.
- Receipts, generator inventory, and package versions:
  `docs/review/isaac-g1/2026-09-06/`.
- Full local logs, 50 Hz traces, native inputs, and diagnostics:
  `/tmp/doorbench-master-integration/out/isaac-g1-demo/`.
- All 162 generator source files matched mechanical revision `2b61dee71` on the
  pod. Native input, runner, checkpoint, robot layer, and trace hashes verified.
- Thirteen contract/runtime checks pass; viewer typecheck and build pass.
- Owned RunPod `doorbench-g1-researcher-demo` was deleted after downloading and
  verifying evidence; its absence was confirmed via the pod inventory.

## Website deployment remains blocked

The About page link is committed on master, but Pages run **34046023058** failed
its published-dataset validation: **110 pass, 12 fail**. The preceding run
**34011154271** at the old master commit failed the same checks. Do not describe
the site link as deployed, and do not bypass the checks to publish it.

Failures concern individual ship/blast operators, wheel linkages, operator-return
profiles/labels, and keypad metadata in the historical site snapshot. The tests
expect newer model metadata than the archived deployment contains. Reconcile
published assets, their source revision and validation scope as part of the
mechanical publication work. Native G1 results are separate and complete.

The publication checkout is `/tmp/doorbench-g1-isaac-publish`; captured previous
and current Pages failure logs are in its `out/g1-publish/`. The mechanical
integration branch has merged the new master changes so future work retains the
guide and About link.
