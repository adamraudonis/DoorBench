# Local GPU Run Center

Open **http://127.0.0.1:5183** on the owner's computer. This read-only dashboard is separate from the public catalogue. It refreshes every five seconds and supports several registered local or SSH runs.

Before starting a new catalogue GPU run, start the dashboard and register the run's output directory. Keep it connected throughout the run so the owner can see progress without asking for updates.

```bash
python3 scripts/gpu_dashboard/server.py \
  --config out/gpu-dashboard/runs.json --port 5183
```

In another terminal, register the dedicated pod. Replace the example values with that pod's actual connection information; use its SSH key, not a RunPod API token:

```bash
python3 scripts/gpu_dashboard/server.py --register-only \
  --config out/gpu-dashboard/runs.json \
  --id g1-next --name 'G1 · next catalogue run' \
  --ssh-host root@POD_IP --ssh-port SSH_PORT --ssh-key ~/.ssh/runpod_doorbench \
  --results /opt/g1-catalogue/results-next
```

The running dashboard reloads this registry automatically. A missing output directory is shown as waiting/unavailable until the run creates it. The monitor only executes its read-only Python collector and `nvidia-smi` through SSH. SSH must already be configured and the host key verified; it never disables host-key checks or asks the browser for credentials. The server binds only to `127.0.0.1`.

Run `scripts/isaaclab/run_g1_catalogue.py` as usual on the pod. Its new `progress.json` heartbeat reports preparation, the current batch and door IDs, isolated retries, auditing, hero recording, completion or interrupted shutdown. The collector also reads `GRID_PROGRESS` from native logs. A stale heartbeat is shown as **not reporting**, never assumed to mean the GPU has been terminated. Older runners still work through ledger/log activity, with less precise stall detection.

The overview shows completed coverage, approximate ETA, observed doors/minute, elapsed evaluation time, native errors, isolated retries, and a receipt timeline. GPU utilization, VRAM and temperature come from live `nvidia-smi` samples; historical utilization is not invented. Door outcomes can be searched, filtered and exported to CSV with review links. The log tab shows the last 100 lines of the active native log.

Raw goals stay separate from audited opening crossings. The latter appear only if the audit's source-ledger checksum matches the displayed results. Errors stay in the denominator. Current-batch and retry activity is not double-counted as finalized progress. ETA is a rough estimate based on the observed average and can change with door complexity or retries; it excludes unknown future setup and hero work.

To retain a completed run after teardown, copy its results locally and re-register the **same ID** without SSH options. Optionally supply a local confirmed teardown receipt:

```bash
python3 scripts/gpu_dashboard/server.py --register-only \
  --config out/gpu-dashboard/runs.json --id g1-next --name 'G1 · completed run' \
  --results out/saved-run --teardown out/teardown-receipt.json
```

Teardown is an independent action. A completed benchmark or disconnected monitor does not stop billing. The monitor never starts, stops or deletes a GPU. Keep the dedicated teardown timer in place and verify pod deletion separately.

The September 6 run is loaded from its preserved local evidence. No GPU was provisioned to build this dashboard. Future native runs should use the current runner for heartbeat support; the historical experiment remains reproducible with its frozen source revision.
