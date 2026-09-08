# Run Center camera previews

The selected Isaac pipeline shows its latest whole-robot and hand close-up
captures below the progress summary. **Pause images** stops image requests while
run status keeps refreshing. Switching runs clears the old views. Captures are
diagnostic snapshots, not evidence that the task passed.

Run the existing server against the existing local registry. Restart the server
after installing this change; no registry migration is needed. A run's registered
`results` directory may be a local archive or an SSH directory. The renderer must
write `frame-NNNNN.png` and `hand-frame-NNNNN.png` directly there. Missing cameras
show placeholders; no extra rendering or GPU commands are issued.

Only the selected run is requested through `/api/previews?run=<registered-id>`.
The endpoint accepts no file paths or view parameters. The server keeps SSH
credentials and registered directories private, retains the existing exact Host
allowlist, and returns only fixed-view PNG bytes, names, hashes and timestamps.

The collector caches reads for ten seconds, deduplicates in-flight requests and
uses at most two independent preview workers with a twelve-second SSH timeout.
It considers at most 10,000 directory entries and the newest three candidates per
view, rejects symlinks and non-regular files, and validates complete PNG chunks
and checksums (4 MiB / 4096 pixels per dimension maximum). A partially written
latest image falls back to an older complete capture. Connection failures retain
the previous capture with a stale notice. The browser requests every five
seconds while visible and resumes without replaying another run's images.

Validation: `python -m pytest -q tests/test_gpu_dashboard.py tests/test_gpu_previews.py`.
The tests cover exact file selection, partial writes, symlinks, corrupt/oversized
images, SSH quoting, selected-run caching, stale fallback, registration identity,
HTTP Host rejection, arbitrary-path rejection and private-field exclusion.
Browser inspection against an actual archived Isaac reach verified both camera
views, pause/resume and clearing images when selecting a run without captures.
