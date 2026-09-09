# Persistent experiment archives

DoorBench's raw physics evidence outgrew the development Mac. Closed archive
files can now move to a private RunPod network volume after byte verification.
This is evidence storage, not a dataset release or a Hugging Face update.

Current volume: `lgek0t3fuh`, named `doorbench-evidence-archive`, **100 GB** in
`US-CA-2`. Standard storage is **$7/month**, billed while the volume exists.
It survives compute-pod deletion. See [RunPod storage documentation](https://docs.runpod.io/storage/network-volumes).

The temporary CPU transfer pod `j7a7rhmf5hwbnm` costs **$0.06/hour** and has
independent local and remote teardown guards with a two-hour deadline. These
guards delete only that pod, never the archive volume. Its operational journal
is `~/.runpod/doorbench_archive_pod.json`; the volume journal is separate at
`~/.runpod/doorbench_archive_volume.json`. Do not publish credential/guard configs.

## Offload a closed archive

```sh
python scripts/isaac/offload_closed_archive.py \
  --source /absolute/path/to/closed-run.tar.gz \
  --destination /workspace/archive/closed-run.tar.gz \
  --host root@POD_IP --port SSH_PORT --key ~/.ssh/runpod_doorbench \
  --volume-id lgek0t3fuh --region US-CA-2 \
  --receipt-directory /absolute/path/to/persistent/remote-archives \
  --evict-local
```

The helper checks the actual mounted network-volume identity, hashes the closed
source, transfers it, independently hashes the remote file, and verifies the
source again. It refuses to overwrite different remote evidence. Only then can
`--evict-local` remove the local file. Omit that flag to retain both copies.

Retrieval receipts are retained beside the former file and in the explicit
persistent receipt directory. The development machine's index is
`~/Desktop/Projects/DoorBench-runs/remote-archives/`. A receipt records the source
path, persistent volume, remote path, byte count and SHA-256. No credentials are
included. Existing experiment manifests remain unchanged.

## Recover on another machine

Attach this volume to a new Secure Cloud pod in `US-CA-2`, arm a compute teardown
guard, and copy the receipt's destination back to its original source path (or
an explicitly recorded new path). Verify the downloaded SHA-256 and byte count
against the receipt before extracting or using it. Container-local `/root` and
`/tmp` files are not persistent; all offloaded evidence is under
`/workspace/archive/` on the named network volume.

The S3 interface can also access the volume without a running pod, but it needs
separate S3 credentials. The current transfer path uses SSH and existing RunPod
authorization. [S3 access instructions](https://docs.runpod.io/storage/s3-api).

## September 9 archive status

The temporary CPU transfer pod was terminated after verification. The 100 GB
network volume remains provisioned. Three closed archives and complete failed
standing-transfer001–008 directories are stored under `/workspace/archive/`.
Twelve index files were verified on the volume before teardown. Reattachment to
a new pod has not yet been tested.

Large raw recordings in local transfer001–008 directories were replaced with
`remote-artifacts.json` retrieval receipts after remote and local SHA-256 checks.
Reports, manifests and source archives remain local. Restore the listed raw files
before rerunning an independent contact audit; a local report alone is insufficient.
