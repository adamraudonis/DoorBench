# Human review workspace

Open **Review** in the viewer (`#/review`). The workspace keeps a queue for the complete loaded manifest, embeds the interactive door inspector, and records your judgement separately from automated QA.

For a local session, start the viewer from the repository root with `cd viewer` followed by `bun run dev`, then open the local address printed by Vite and choose Review. The viewer needs the generated dataset served at its configured `assets` path. See the project setup instructions if those files are absent.

## A quick review pass

1. Select a door. Search matches its ID, family, use, hardware, tags and your notes. Family, status and issue-tag filters combine.
2. Orbit the assembly from both sides. Inspect proportions, materials, repeated details, supports and hardware attachments. Use the appearance images where available; they are a visual supplement to the simulation geometry.
3. Operate the door and each relevant mechanism through the full range. Examine intermediate positions, endpoints, latch/strike alignment, hinges, guides, tracks, closer arms, collisions and missing supports. Use the inspector's diagnostic visibility controls and simulation reference when available.
4. Rate **Appearance**, **Physical construction** and **Mechanism** as Pass, Issue or Unsure. Leave Not rated for an aspect you have not examined. Add issue tags and a note identifying the part, pose, observed problem and expected behaviour.
5. Choose **Accept & next** only after all three aspects meet your visual review standard. It records three Pass ratings and advances within the current queue. Existing flags, issue tags and Issue/Unsure ratings block quick acceptance until explicitly resolved. **Flag for follow-up** keeps the door open and focuses its notes.

Use the **Unreviewed** filter to work through untouched doors, **In progress** to finish partial/unsure assessments, and **Flagged** to revisit defects. The current door stays open when an edit causes it to fall outside the filters; Previous/Next follows the remaining queue. The queue does not wrap at its ends. The full inspector link opens separately so the review workspace stays available.

Keyboard shortcuts work when focus is outside interactive controls:

| Key | Action |
| --- | --- |
| Right arrow, N or J | Next matching door |
| Left arrow, P or K | Previous matching door |
| A | Accept all three aspects and advance |
| F | Flag and focus notes |

Shortcuts pause inside text fields, selects, buttons, links, editable content and timeline sliders. Modified, repeated and composing keystrokes are ignored. **Undo last edit** restores the last changed assessment, including an accidental acceptance or clear operation.

## What the counters mean

- **Unreviewed:** no rating, substantive note, flag or issue tag.
- **In progress:** a partial assessment, notes, or an Unsure rating, without a defect flag.
- **Accepted:** all three human ratings Pass, with no manual flag or issue tag.
- **Flagged:** a manual flag, issue tag, or Issue rating. This can still be partially assessed.
- **Fully rated:** all three aspects have a nonempty rating, including Issue or Unsure. A manual flag alone does not count.

The counters cover the loaded dataset, while the queue count reflects its filters. Automated QA, benchmark success and prior review reports do not prepopulate your human ratings. Visual acceptance does not establish contact/force behaviour, structural strength, complete mechanism realism or MuJoCo/PhysX parity. Reference playback only supplies evidence for its recorded sequence and simulator.

## Saving, backup and import

Edits and the selected door save automatically in this browser's localStorage. Storage is specific to the site address and browser profile: changing localhost port, switching to the hosted site, using another browser, or clearing browser data does not carry the notes across. Storage failures are shown in the workspace; the current tab retains its edits so **Export JSON** can back them up.

Progress is scoped by a fingerprint of the loaded manifest, including its door metadata. Object-key order and generation timing are ignored. Changes to manifest contents get a separate review store. This is **not a checksum of model/mesh files**: geometry changed without changing the manifest cannot automatically invalidate an earlier assessment. Export and begin a separate review when the underlying models change independently.

Export JSON regularly and before changing dataset or browser. The versioned `doorbench-human-review/1` document contains the dataset fingerprint, exact door ID list, export timestamp, and each stored assessment's ratings, manual flag, issue tags, notes and edit timestamp. Empty assessments retained after a clear operation carry a new timestamp so an older import does not resurrect them.

**Import JSON** validates the entire document before offering a merge preview. It rejects incompatible schema/dataset, unknown or duplicate doors, invalid ratings/tags/types/timestamps and extra fields. Notes are limited to 10,000 characters per door and files to 64 MB. Import merges the newest timestamp per door; local records win ties and unrelated local doors remain intact. It does not combine individual fields inside a conflicting door assessment. Review the preview and choose Apply import or Cancel. Export a backup first if you need to retain both versions; import is not part of single-edit undo.

Newer edits from another tab are merged when saving and when its storage event arrives. This is a local review tool, not a multiuser database; avoid simultaneous edits to the same door in multiple tabs. No review is uploaded, no dataset file is edited, and the workflow does not change automated QA results.

If existing saved JSON is corrupt, it is left intact and never replaced by an empty document. New notes remain in the current tab when saving fails; export them before recovering the old storage manually.

## Focused checks

Run `bun test src/reviewState.test.ts` from `viewer` for dataset identity, rating/counter rules, combined filters, strict import validation, newest-record merge rules, persistence errors and keyboard focus safety. Run `bun run typecheck` and `bun run build` for integration checks.
