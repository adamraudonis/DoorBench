# Sliding track repair — takeover review

The 168 horizontal sliding doors now size and position their rails from the complete nominal unlocked leaf sweep. Single rails shift toward the opening side instead of remaining centered on the doorway; bypass doors have one rail per actual depth lane, including the middle leaf's two-direction travel. Pocket doors have a full-length header rail. Barn hangers now place each wheel tread on the rail top, with bent straps, through axles and fixing bolts joining the wheels to the leaf; wall spacers bridge to the wall face, rail-mounted bumpers meet the terminal wheels, and floor-mounted guide forks overlap the leaf at both travel endpoints. Patio rollers contact the bottom rail at its actual top height.

## Evidence

- Targeted regeneration: **168/168 signed off**, **168/168 clearance clean** (full/simple/minimal MJCF and JSON; no USD/URDF regeneration in this worker).
- New deterministic track gate: **168/168 pass**, with 25 samples per nominal unlocked slide range. Actual tread contact checks cover **48 doors / 96 wheels**; the other **120 doors explicitly report rail-only coverage** because their moving suspension hardware is not yet modeled.
- Seven dedicated regression tests pass in `tests/test_sliding_tracks.py`: layout coverage, deliberately shortened rail, vertically floating wheel, short track masked by an engaged lock, missing support metadata, an end stop displaced away from the wheel plane, and explicit rail-only reporting.
- `db0079_sliding_single`: rail length 2.234 m, now centered at x = 0.5335 m. Its original rail length was already sufficient (2.234 m), but its center x = 0 made the opening-side wheel leave it. The repair shifts the rail, seats the treads in all three dimensions, and adds end stops. Maximum measured full-sweep tread gap and leaf overhang: 0 m.
- Parent owns all-1000 regeneration, integrated QA sign-off, the complete test suite, and viewer verification. No assets were hand-edited or committed here.

## Personally inspected renders

I inspected the after image directly: both wheels remain above the full-travel bar, the right bumper meets the terminal wheel, straps reach the door, and the fork is mounted at the lower trailing edge. Before/after share the same camera and full-open joint position (1.067 m):

- `docs/review/track_db0079_before.jpg`
- `docs/review/track_db0079_after.jpg`

## Remaining model limitations

This is not complete G9d sign-off for every suspension mechanism. The 120 rail-only doors still need modeled top-hung carriages, glide/groove geometry, and true cantilever gate support. The current cantilever spec is still rendered with a ground rail. New rail span coverage cannot certify those absent mechanisms. Barn end-stop contact is checked; other horizontal track types do not yet have separately modeled terminal bumpers. The separately implemented garage/sectional tracks are unchanged and are outside this new horizontal gate. The new helper does not replace the benchmark's constrained slide joint with explicit wheel rotation/contact dynamics.

## Integration

`doorbench.sliding_track_qa.run_sliding_track_qa(m, model_meta)` returns pass/fail plus explicit geometry coverage, missing geometry errors, worst tread gap, rail overhang, and end-stop failure evidence. It uses exported MuJoCo geometry, including non-colliding visual wheels; it sweeps the nominal range even when a lock narrows the active joint range. The parent integrates this into `run_qa` under `sliding_track_support`.

Changes are uncommitted in the shared checkout by parent instruction. No branch changes, commits, or pushes were made.

## Independent review follow-up: preserve requested bypass guides

A final independent review caught a scope regression in the first repair: restricting guide creation to barn tracks removed guide geometry from **14 bypass doors whose specs still request `floor_guide`**. The old single fork was not correctly placed for their individual lanes. This has been corrected with floor-mounted guide pairs for each requested lane: **34 stations across 14 bypass doors**, including two stations for each bidirectional middle leaf so at least one fork remains engaged throughout travel. Each fork has explicit feet touching the floor, 1 mm nominal face clearance, and jaws extending 8 mm above the panel's actual lower edge. The 26 barn doors retain their existing guide stations, bringing guided-door coverage to **40 doors / 60 stations**.

The gate now requires declared guides to exist, verifies that each jaw meets a static floor-mounted foot, and measures actual MuJoCo geom distances from both sides of the fork to the panel throughout 25 nominal travel samples. Multiple stations are checked as a set: at least one must straddle and engage its assigned leaf at every sample. The bound is 3 mm for guide engagement/mounting; continuous collision freedom remains the separate clearance gate's scope.

After this follow-up, **all 168 horizontal sliders were regenerated in a temporary directory with full MJCF, URDF, USD and JSON exports: 168 signed off, 168 clearance clean, and 168 track-gate passes**. The dedicated tests now report **11 passed**, including all 14 requested-guide doors plus negative fixtures for omitted guide metadata, a bidirectional panel losing one of its two stations, and a floating guide foot. An independent direct `mj_geomDistance` sweep confirms the current **48 wheel-bearing doors** have a maximum absolute wheel–rail surface distance below **0.0005 mm**, rather than relying only on the track gate's axis-aligned formulas.

Personally inspected render evidence: [three-lane guide arrangement](bypass_guides/db0008_middle_left.jpg), [fork and floor feet detail](bypass_guides/db0008_station_detail.jpg). Counts for the temporary export are in [bypass_guides/verification.json](bypass_guides/verification.json). Shared assets were not regenerated by this reviewer. The suspension, groove, cantilever, roller-rotation and export-dynamics limitations above still apply; guide forks add lateral restraint geometry and do not supply missing top-hung suspension.
