# Detached panel planning from an actual Isaac withdrawal

This pipeline prepares a left-palm opening segment from a **completed, independently qualified actual Isaac right-hand withdrawal**. It does not run physics, command the door, export a live route, or qualify full opening or passage. No successful actual Isaac withdrawal is assumed by the implementation or its synthetic tests.

The new files leave `screen_whole_body_panel.py`, `audit_whole_body_panel_screen.py`, and all existing native constructors unchanged. Their numerical fitting objectives, original dense geometry checks and cubic derivative bounds are reused in a distinct actual-Isaac source path.

## API and artifacts

`isaac_panel_source.admit_isaac_panel_source(source, robot=..., door_xml=..., door_usd=...)` independently admits a released endpoint. It requires finalized local process/launch evidence, the original withdrawal qualification and live-prefix chain, complete actual archive/raw evidence, final support and clearance holds, actual motor inputs, original assets and complete body-coordinate admission. It intentionally does not require a final RH grasp after intentional release.

`isaac_panel_planning.admit_isaac_panel_context(...)` wraps that detached admission in an immutable context. Its `qpos`, `qvel` and `admission` properties return copies. `scene()` owns a fresh unstepped geometry model; it never receives an active plant.

`generate_panel_candidate(context, preferences=None, target_aperture_rad=1.62, solver_method='constrained', flatten_palm=False)` returns `doorbench.isaac-panel-candidate.v1`. The first pose is the exact normalized source endpoint, not the optimizer's preferred replacement. The original finger/mechanism coordinates remain fixed except for the prospective leaf angle and the explicit named whole-body references. Source state, epoch, full admission and source/helper hashes are bound into the candidate.

`numeric_panel_preferences(old_document)` reads only a bounded allowlist of scalar fitting preferences: knot/evaluation counts, radius/height shift, root fitting extents, interior joint/pose margins, objective weights, flattening interval, elbow margin/ramp and palm twist. Old state, coordinates, source paths, targets, qualification and relaxed proof modes are discarded. Values outside the current bounds reject rather than silently clamp. Target aperture, solver mode and palm flattening are explicit new-run options.

`isaac_panel_geometry_audit.audit_panel_candidate(candidate_path, ...)` re-admits the actual source, checks its exact agreement with the candidate, and returns `(receipt, geometry_plan, traces)`. The schemas are `doorbench.isaac-panel-geometry-audit.v1` and `doorbench.isaac-panel-geometry-plan.v1`; neither is a legacy native plan accepted by `StandingPanelReference`. The audit rejects hidden finger/mechanism changes, altered first coordinates, changed assets/helpers, old native receipts and authority claims. Input hashes are checked again before returning.

All artifacts retain `authorized_stages=0`, `physics_steps=0`, `source_sample_playback=0`, `active_state_writes=0`, `physical_admission=false` and `runtime_route_exported=false`. A successful independent screen sets only `geometric_admission=true`.

## Original geometric limits retained

The independent audit requires at least 2,001 samples and retains the strict upright profile: hand/foot position error ≤0.1 mm, hand/foot rotation error ≤1 mrad, root translation ≤3 cm, root rotation increment ≤0.05 rad, COM XY displacement ≤1.5 cm and torso tilt ≤4°. It retains the original joint-limit-increase and forbidden-penetration checks, all-RH/environment clearance ≥4 cm and elbow/environment clearance ≥3 mm. Both collision-initiating and receiving-only authored shapes are included (`contype or conaffinity`); nonfinite geometry/distances cannot pass.

Reference limits remain 1.2 rad/s and 3 rad/s² for joints, 0.02 m/s root translation and 0.03 rad/s root rotation. The measured-aperture derivative envelope uses exact cubic derivative extrema and the original maximum aperture speed/acceleration of 0.149 rad/s and 0.08 rad/s². Lower prospective limits may be declared; a fit that violates these bounds stays failed, with per-sample constraint/clearance failures retained.

The optional static leaf lag is one declared slice, not proof of an entire lag envelope or of actual tracking. The exact source is included without a lag. Initial measured velocities are preserved as evidence, but this static solver does not prove a dynamically continuous handoff, support force, friction, motor tracking, collision response or successful opening.

## CLI

Use a new output directory for every candidate and audit. The following are command templates; `QUALIFIED_WITHDRAWAL` must be a completed newly qualified actual run, not an earlier passing window from a failed run.

```powershell
python scripts/dexterous/plan_local_isaac_panel.py --source QUALIFIED_WITHDRAWAL --robot ORIGINAL_ROBOT_XML --door-xml ORIGINAL_DOOR_XML --door-usd ACTUAL_IMPORTED_DOOR_USD --target-aperture-rad 1.62 --output NEW_CANDIDATE_DIRECTORY
python scripts/dexterous/audit_local_isaac_panel.py --candidate NEW_CANDIDATE_DIRECTORY/candidate.json --duration-s 40 --samples 2001 --output NEW_AUDIT_DIRECTORY
```

The planner optionally accepts `--numeric-preferences OLD_NUMERIC_PREFERENCE_DOCUMENT`, `--solver-method least-squares`, and `--flatten-palm`. The default constrained solver includes original interior pose, COM/root, elbow and exact RH clearance constraints. An unconverged fit remains a candidate; only the independent geometry audit can evaluate its geometry. No solver status alone grants success.

The audit writes `report.json`, `geometry-plan.json` and `target-traces.npz`, retaining failed results and returning exit code 1 for a failed screen. `--aperture-speed-limit-rad-s`, `--aperture-acceleration-limit-rad-s2`, `--actual-leaf-lag-rad` and `--lag-start-angle-rad` declare bounded prospective audit conditions.

## Aperture and segment scope

The default 1.62 rad target is a prospective goal, not an assumed reachable aperture. Existing passage guards require **at least 1.57 rad actual aperture**, while the post-opening controller's 1.2 rad entry minimum alone is insufficient for passage. The candidate and receipt report both the requested target and `target_reaches_passage_aperture`.

A smaller target is permitted as an explicit separately screened segment; its receipt truthfully reports that it does not reach the passage threshold. A candidate's endpoint never becomes a qualified physical source. Chaining later actual panel segments will require their own physically qualified endpoint and explicit source-kind adapter, rather than relabeling a geometric endpoint or the current withdrawal-only admission.

Future live integration must inherit the actual preceding 6 N LH feedback state, stance references and returned motor command; validate target/velocity continuity; follow measured aperture inside an independently admitted domain; and capture actual forces and contacts. Those pieces are deliberately not exported here. The current CPU tests use synthetic source fixtures/private FK and include passing and failing geometry, source corruption, original rate failures and receiving-only collision shapes; they are not robot performance results.
