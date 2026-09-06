# Review doors

Open **[Review](https://adamraudonis.github.io/DoorBench/#/review)**. Each door’s saved reference starts playing automatically when its source checks pass. Missing or stale motion is identified rather than replayed against different geometry.

1. Watch the mechanism and orbit the model. Choose **Blender appearance** to inspect its saved render.
2. Choose **Good · next**, or type a comment describing the problem.
3. Press **Tab** for the next door or **Shift+Tab** for the previous one. These shortcuts also work from the comment field. The queue wraps through every door.
4. Open **My feedback** to see your assessments, comments and links to the doors. **Download links and comments** produces a text file to send back; **Download JSON backup** preserves the structured records.

Comments save as you type. Good records acceptance and advances; it keeps your existing comment. Typing a substantive comment flags the door for follow-up. **Full inspector** opens the individual door separately with joint controls, mechanism visibility and the opening procedure.

Saved feedback belongs to this browser, site address and dataset fingerprint. A different localhost port or the public site has a separate store. No feedback is uploaded automatically. Back up before clearing browser data or changing datasets. If localStorage fails, the screen reports the error and retains the current notes for download. Other tabs’ newer records are merged; avoid editing the same door simultaneously in two tabs.

Existing `doorbench-human-review/1` records are preserved. Their three detailed ratings and issue tags remain in JSON backups even though the simplified screen shows only Good or Needs review. The fingerprint covers manifest metadata, not every model file; older acceptance is not certification of revised geometry. Visual review also does not certify force behavior or simulator parity.

For local use, serve the dataset, `appearance/` and `reference-motions/` alongside the viewer, then run `bun run dev` in `viewer`. See [local catalogue setup](DATASET_RELEASE.md). Run `bun test src/reviewState.test.ts src/mechanismInspection.test.ts`, `bun run typecheck` and `bun run build` for focused checks.
