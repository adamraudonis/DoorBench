# Visual common-sense review of every door (task G8)

Read `handoffs/README.md` first. Branch to resume: `worktree-agent-a3c65620742f28a27` (2 real commits + a
snapshot: `doorbench/review/vision.py` (sheet rendering, rubric prompt, Claude API transport, cost estimate),
`scripts/vision_review.py` CLI, `tests/test_vision_review.py`, draft `docs/VISION_REVIEW.md`,
`docs/review/vision/`). The last agent was adding an "open-low" row and close-up angles to the sheets.

## Why (owner's words)

"For https://adamraudonis.github.io/DoorBench/#/door/db0079_sliding_single the rail does not extend long enough
so one of the wheels will fall off it. Is it possible for you to have a photo taken of each model and then have
that image be sent to a common sense agent that is prompted to catch things like that? As a human it is ultra
obvious to see."

## Goal

1. `scripts/vision_review.py`: for each door render a labelled review sheet (closed / open / mid-travel, 3
   viewpoints, hardware + mechanism close-ups; the label states what the spec says should be there: hinge count,
   travel, hardware names), send it to a vision LLM with the rubric, parse a strict JSON verdict
   `{door_id, ok, findings: [{category, severity, part, description, where}]}`. Categories: floating part, part
   through another, rail/track/guide too short for the travel, roller/hanger leaving its guide, missing hardware
   the spec implies, duplicate/extra parts, wrong scale, hardware on the wrong face, mechanism that cannot work,
   implausible proportions, "obviously wrong". Transport: Anthropic Claude API (`anthropic` SDK, image content
   blocks; key from `ANTHROPIC_API_KEY`), `--model`, `--max-cost-usd` guard with a pre-run estimate, retries,
   resumable, `--dry-run` (sheets + prompts only), `--from-verdicts` (rebuild the report). Output
   `docs/review/vision/<door>.{json,jpg}` and `docs/VISION_REVIEW.md` (how to run, cost for 1000 doors, counts by
   category x family, blockers first with images).
2. Run it. If you have an API key, run all 1000 doors (estimate the cost first; a few tens of dollars with a
   mid-size vision model). If not, run `--dry-run` and review a seeded sample of ~120 sheets (4 per family, plus
   db0079_sliding_single and db0024_swing_single as calibration cases: the rubric must catch the short rail and
   the floating stop) yourself if you can see images; write the same JSON verdicts with `"reviewer": "manual"`.
3. Triage every finding: geometry bug (which file/function; fix it if small and not owned by another brief,
   re-render to confirm), rendering artefact, or false positive. Put the geometry bugs in a "handoff" section of
   `docs/VISION_REVIEW.md` grouped by area (tracks, stops, closers, dogs, hinge counts, ...).

## Files

`doorbench/review/vision.py`, `scripts/vision_review.py`, `tests/test_vision_review.py` (sheet rendering for 3
doors, schema, mocked API round trip, report), `docs/VISION_REVIEW.md`, `docs/review/vision/` (JPEG, small).

## Done when

The CLI works end to end in `--dry-run` and with a mocked client in tests; `docs/VISION_REVIEW.md` starts with
"how to run"; the sample (or full) review is in the repo with findings triaged; the two calibration cases are
caught.
