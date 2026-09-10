# Reproduce the standing transfer experiment

This is a **privileged H1 with dual Shadow Hands development test**, not a learned policy or a complete opening/traversal benchmark. Isaac042 qualifies a standing grasp, lever operation and held partial opening. Its transfer route passes geometry checks. Physical transfer043 is pending; see [the execution ledger](DEXTEROUS_EXPERIMENTS.md) and [dispatch](evidence/isaac043-transfer-dispatch.json).

## Prepare the destination

Use [the one-click environment launcher](ISAAC_ONE_CLICK.md), including its existing-SSH-node option for another cluster. Preserve the source manifest, readiness receipt, original robot XML/USD, door XML/USD, motor contract and input grasp reference. Run the commands below from the destination source checkout with `PYTHONPATH=.`. Use the asset environment for geometry/audits and the Isaac environment for physics; do not mix their USD installations.

All path arguments below are placeholders for files on **that destination**. Every output path must be new. Existing files are retained as evidence. On RunPod, use the owned allocation journal and verified teardown guards. On another cluster, reserve scheduler time and collect outputs before the job ends. The native prerequisite, Isaac simulation, contact audits and collection all need time within that limit.

## Obtain and qualify the measured starting state

First reproduce a successful standalone36-second operation using `scripts/isaac/run_standing_operation.py`, without `--standing-transfer-route`. The full043 invocation is preserved in [its dispatch receipt](evidence/isaac043-transfer-dispatch.json); remove that option to run the baseline. Supply a known grasp reference and the destination readiness receipt. The coordinator re-screens the reference, requires a native grasp/contact pass and then runs actual Isaac physics.

A successful run directory contains `coordinator-result.json`, `isaac-independent-audit.json` and `trial/operation-report.json`. Do not substitute a manually edited pose, a geometry-only result or a failed episode.

```sh
ASSET_PY=/your/work/asset-venv/bin/python
RUN=/your/completed-operation-run
MEASURED=/your/new-output/measured-state.json

PYTHONPATH=. "$ASSET_PY" scripts/dexterous/extract_isaac_attained_state.py \
  --run "$RUN/trial" --time-s 36 --output "$MEASURED"
PYTHONPATH=. "$ASSET_PY" scripts/dexterous/audit_qualified_isaac_grasp.py \
  --run "$RUN" --extracted "$MEASURED" \
  --output /your/new-output/source-admission.json
```

The second command checks complete runtime/contact reports, source-file hashes and the terminal hold, then re-extracts the state and body poses from the original recording. An altered extracted state is rejected. This is source qualification; it does not certify the next motion.

## Plan on the original destination model

```sh
PYTHONPATH=. "$ASSET_PY" scripts/dexterous/rebase_standing_transfer.py \
  --isaac-source "$RUN" --isaac-extracted "$MEASURED" \
  --robot /your/original/h1-shadow-loopback-v2.xml \
  --door /your/original/door.xml --door-usd /your/original/door.usda \
  --template /your/prior-transfer.json --pose-weight 400 \
  --output /your/new-output/transfer-plan
```

The template provides named joint/root posture preferences only. Its old clearance result and mechanism state are not reused. The destination's recorded feet, torso, palms and handle must agree with its original-model kinematics before planning. A successful plan produces `transfer.json` plus a101-node report and an independent1001-sample interpolation/collision audit. The source042 candidate required pose weight400; this is an optimizer weight, not a relaxed acceptance threshold.

Paths and hashes bind the plan to its source. Do not rewrite XML paths or hashes to make a route load on another machine. Prepare the destination and replan there. Keep the template as a reusable posture suggestion, not as a transferable physical proof.

## Run the physical comparison

Run `scripts/isaac/run_standing_operation.py` with the same baseline settings, adding:

```text
--standing-transfer-route /your/new-output/transfer-plan/transfer.json
--isaac-timeout-seconds 6000
```

The coordinator performs its unchanged36-second native **grasp** prerequisite. It then requests50 seconds in Isaac: acquisition and partial opening, followed by the transfer at36 seconds. The native prerequisite does not qualify the Isaac-specific transfer route. Runtime start-state checks reject a grasp that differs materially from the screened endpoint.

Only original capped motor commands actuate the robot. Per-step panel-force measurements are written in bounded chunks and exported to `trial/standing-transfer-steps.json.gz`. Qualification retains every existing physics and handle-contact check and adds complete transfer timing, stance-solver checks and at least2N of actual left-palm support throughout the final half-second. Finger force alone cannot substitute for palm support.

The coordinator runs `scripts/dexterous/audit_isaac_standing_transfer.py` to independently reduce that force stream. This audit checks the recorded per-body panel force vectors; it does not replace the separate physics and raw handle-contact audits. Any missing/incomplete stream, failed runtime report or failed audit leaves the experiment failed.

Attach [the evidence collector](GPU_EVIDENCE_COLLECTION.md) to the coordinator's `run.pid` and terminal `coordinator-result.json`, and register the run in Run Center before leaving it unattended. Require final-byte verification before removing remote evidence. Review actual wide footage and hand close-ups as well as numerical results.

## A different robot

The current controller and planner still contain H1/Shadow joint names, motor mappings, palm sites and foot assumptions. Another robot requires an explicit adapter and fresh model/import, kinematic, motor, contact and balance validation. A passed H1 route is not evidence for that robot. Vision/tactile policy learning and complete approach/open/traverse evaluation remain subsequent milestones in [the approved plan](DEXTEROUS_NEXT_STEPS.md).
