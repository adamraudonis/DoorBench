# Sliding rails long enough; rollers stay on the track (db0079 and every sliding family) — task G9d

Read `handoffs/README.md` first. Start from `master` on a new branch.

## Why (owner's words)

"For https://adamraudonis.github.io/DoorBench/#/door/db0079_sliding_single the rail does not extend long enough
so one of the wheels will fall off it."

## Current state

`db0079_sliding_single`: kinematics `slide_horizontal`, travel 1.067 m, roller `barn_hanger`, track
`surface_flat_track`, stop `track_end`, opens toward the right. Track / hanger / guide geometry lives in
`doorbench/geometry/other.py` (search `track`, `hanger`, `roller`, `bypass`, `pocket`). The track is shorter
than leaf width + travel, so at full travel one hanger is past the end of the rail.

## Goal

For EVERY sliding family and roller type (barn hangers, top-hung patio/shoji, bypass `yk` stacked tracks, pocket
doors, cell/industrial, automatic sliding, bottom-rolling, and garage/sectional tracks if they share code):

1. Track length >= leaf width + travel + hanger offsets + end-stop length; end stops / bumpers at the real ends;
   floor guides positioned correctly; pocket cavities long enough for the leaf.
2. Deterministic QA check `rollers_on_track`: sweep the slide joint; every roller / hanger / guide geom stays in
   contact with (within 3 mm of) the track geometry at every sample; the leaf reaches the end stop and never
   leaves the rail. Wire it into `signed_off`.
3. Fix all failures; keep the clearance gate green (longer tracks must not hit casings, walls or the other leaf).

## Done when

Regenerate -> 1000 signed off; clearance 1000/1000; `rollers_on_track` 1000/1000; tests (add a test that fails on
a deliberately shortened track); `docs/media/track_db0079_{before,after}.jpg` rendered with MuJoCo at full travel.
