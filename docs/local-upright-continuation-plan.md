# Fresh local upright continuation

This is an integration plan, not a full-task qualification. It uses the original
H1 / corrected Shadow loopback geometry and original capped motors. No stage
may reset the active state, replace measured poses, replay archived dynamics,
apply root forces, drive the door directly, or inherit an old untracked route's
success. The current initialized standing opening still needs release, a
1.2 rad aperture, stow, walking through the opening and a quiet final stop.

## Current source boundary

* `out/local-native-volar-001`: fresh 36 s source, physical operation and
  independent selected-volar/whole-handle audits pass. Original distal-only
  grasp score remains false. This qualifies the partial opening only.
* `out/local-native-transfer-001` and `002`: failed physical transfer. Increasing
  the total LH target from 4 N to 6 N loaded the little-finger knuckle/middle,
  while actual palm force remained zero. Neither is a continuation source.
* `out/local-planning/qualified-face-on-transfer-001/transfer.json`: fresh
  prospective transfer from the qualified native source, 1,001 original dense
  geometry checks pass. SHA256:
  `c3aa13b945e533c4e59dc892cd06ccfcc8d0df3d5163537f4c9d8589dac80e71`.
  Its transfer003 physical trial now qualifies actual palm support. It fixes the root, both
  feet, RH, torso and all finger coordinates. The endpoint preference rotates
  one third toward the panel normal and raises the numerical height preference
  by 0.10 m; the resulting endpoint is 0.043 m higher than the previous route.
  Palm capture gap is 5.190 mm and it leads every other LH collider by 5.804 mm.
* `out/local-native-transfer-003`: all 21 runtime checks, independent selected
  contact/whole-handle/transfer audits and all 25,000 continuity intervals pass.
  Final-half-second palm support stays above 3.28695 N; handle rests at
  -0.000282 rad and leaf holds 0.095701 rad. The RH transient lasts 0.174 s.
  A separate one-second motor handoff in transfer004 also passes but increases
  total interrupted-grasp intervals from 107 to 124; retain transfer003 as the
  release-planning source until a better handoff is demonstrated.
* `out/local-isaac-operation-002`: completed 24 s, fails sustained grasp. Its
  exact actual state passes the existing 2 micrometre / 2 microradian kinematic
  admission, but this does not repair its physical qualification.

## Exact measured-state entry into Isaac planning

`scripts/dexterous/plan_local_isaac_transfer.py` replaces the historical cloud
archive layout dependency with an explicitly local report/audit loader. It
uses the existing original qualification validator, exact terminal extraction,
hash binding, coordinate convention and six-body kinematic admission. It
does not create a fake cloud coordinator result. Any existing route supplies
only numerical LH targets; its root, legs, mechanism state and success claims
are discarded.

The required local files are the completed `trial/operation-report.json`,
`configuration.json`, `motor-contract.json`, `provenance.json`,
`acquisition-physics.npz`, `acquisition-pad-steps.json.gz`, and the independently
generated `independent-contact-audit.json`. All contact intervals and the final
half-second hold must pass. A source containing a transfer additionally needs
the original independent transfer audit and force stream. No previous
machine's tangent038 reference, transfer template or cloud receipt is needed.

For a future qualified local PhysX source:

```powershell
& C:/Users/adamr/Documents/Codex/doorbench-env/Scripts/python.exe `
  scripts/dexterous/plan_local_isaac_transfer.py `
  --isaac-source out/NEW-QUALIFIED-ISAAC-SOURCE `
  --robot out/local-ready/h1-shadow-loopback-v2.xml `
  --door assets/doors/db0055_swing_single `
  --door-usd assets/doors/db0055_swing_single/door.usda `
  --preferences out/local-planning/qualified-face-on-transfer-001/transfer.json `
  --output out/NEW-ISAAC-TRANSFER-PLAN
```

The planner emits a runtime route only after source admission and the unchanged
1,001-sample geometry audit pass. Start the continuation at that actual source
epoch in the same episode. A native 36 s timestamp is not an Isaac 24 s state.
The current failed-source refusal is retained under
`out/local-planning/isaac-standing-transfer-001/failure.json`.

## Shortest continuous sequence

| Stage | Reusable implementation | Required measured transition |
| --- | --- | --- |
| RH partial opening and LH receiving | `StandingTransferTeacher` and fresh route | Selected opposed grasp plus actual palm-only panel support at least 2 N for the original half-second hold; all physics/whole-handle checks pass. Total LH force cannot substitute. |
| Lever rest | `StandingReturnTeacher`, only if the lever is not already at rest | Keep RH grasp and LH palm support while returning the operator. If actual operator is within 0.05 rad and bolt within 0.001 m already, preserve that actual rest instead of creating a needless return motion. |
| RH release and axial withdrawal | `StandingWithdrawalTeacher`, fresh `plan_direct_standing_release.py` route | Preserve qualified opposed grasp through the declared release start; smoothly unload finger preload; measure every hand-body contact and original hub clearance until RH environment clearance is at least 0.04 m. |
| LH opening to 1.2 rad | `StandingPanelReference`, `AttainedPanelSchedule`, bounded panel-force controller | A fresh route begins at the actually attained released-hand state. Progress follows measured aperture, with actual palm support and quiet supported feet. |
| LH unload and arm stow | `PostOpeningTeacher` with `sequential-v2` as the initial prospective choice | Same actual root/joints/velocities, body poses, last delivered 61 motor forces, measured outward panel normal and immediately preceding contact interval; aperture at least 1.2 rad, RH contact-free, feet at least 10 N, root speed at most 0.1 m/s. |
| Walk through and stop | Existing `StowRiseController` / `PassageWaypoints` / pinned H1 checkpoint | Motor-only same-episode continuation, aperture and body-clearance guards, crossed-body criterion and quiet final stop. |

The optional rest step is a software-composition issue: `StandingWithdrawalTeacher`
currently accepts an object exposing `transfer`, `operation`, `acquisition`,
`started`, `return_started`, and `force`. The original `StandingReturnTeacher`
also requires a 41-node / 401-sample audited return route. A new explicit
`RestingTransferBridge` can delegate the existing transfer controller and expose
those references while reporting a distinct measured-rest event. It must leave
`return_started` unset and must not forge a completed return route or return
milestone. A preceding source audit and the live half-second actual palm/volar
grasp/rest gate are still required. This bridge and native runner wiring are
now implemented and tested; see `LOCAL_MEASURED_REST_RELEASE.md`. The first
continuous 66 s release trial repeats transfer003's first 50 s exactly and
clears the RH by at least 0.161984 m. It remains a failed release: the LH palm
loses support after 60.958 s, and 2,315 loaded RH patches cross original
anatomical or lever-end boundaries. A revised coupled palm/body path and RH
release clearance are required before this becomes a qualified continuation.

## Route generation after a qualified receiving endpoint

`plan_direct_standing_release.py` already builds a fresh measured release with
bounded radial separation, optional free-end axial slide and subsequent lift.
It uses the source's actual terminal coordinates and original collision model.
The new `release_source_admission.py` adapter binds the actual local contact
report, original source hashes and the final 251 samples. It does not fabricate
historical source files. The selected-volar path now captures actual dominant
loaded palmar material points, including middle phalanges, and preserves
whole-finger and hub clearance checking.

`audit_standing_ungrip.py` now accepts the explicit source-bound grasp profile
and retains the original distal counters. The first new route passed all
2,001 geometric samples; that did not predict the loaded-contact success of
release001. `audit_standing_return_route.py` still needs corresponding profile
plumbing if a future source actually requires a lever-return motion. No existing
result is relabeled and no anatomy, force, penetration or hold threshold changes.

The dense withdrawal audit retains 2,001 samples, exact source start, 1 mm /
0.01 rad fixed hand/foot bounds, 4 degree torso bound, 3 mm nonfoot penetration
bound, 2 rad/s joint-reference bound, and at least 4 cm final RH environment
clearance. The default route durations are numerical preferences and can be
slowed for a smoother motion without relaxing these limits.

After actual physical release, `screen_whole_body_panel.py` can build a fresh
upright path directly from the recorded state; it does not require the old
successful crouched route. Use its explicit 4 degree torso limit. The
independent `audit_whole_body_panel_screen.py` produces a source-bound
`whole-body-panel-plan.v1`: at least 2,001 samples, exact first state,
0.1 mm / 0.001 rad fixed hand/foot bounds, RH clearance at least 4 cm, elbow
clearance at least 3 mm, root displacement at most 3 cm, COM horizontal shift
at most 1.5 cm, joint reference speed/acceleration at most 1.2 rad/s and
3 rad/s², root speed at most 0.02 m/s, root angular rate at most 0.03 rad/s,
and aperture speed/acceleration at most 0.149 rad/s and 0.08 rad/s².
Failure to reach 1.2 rad under those constraints must trigger a new measured
stance/palm plan; it is not permission to splice an archived endpoint.

## Runtime composition and missing artifacts

`PostOpeningTeacher` already constructs its own new stow path from the attained
state. The walking-reset file is a posture goal only. It accepts no active
plant and returns only the original capped motor forces. Live input needs
actual leaf, handle, both palm and ankle body poses, complete hand-body force
rows, full root and named joint velocities and the last applied motor vector.
The current Isaac standing archive includes torso instead of leaf in its
six-body witness; include the synchronized actual leaf pose for a later
post-opening offline handoff receipt, rather than reconstructing it and calling
that an actual body measurement.

`ContinuousDoorTeacher` assumes a genuine walking/approach/preparation prefix,
closed door and initially free hands. It should not receive the currently
initialized near-handle episode and claim that its approach milestones passed.
Use an explicit initialized-standing composition until the actual walking
approach has been connected and separately qualified. The final uninterrupted
opening-to-passage handoff can still reuse its same-timestep motor-delivery and
contact-clock contracts. `probe_post_opening.py` resets from a source snapshot;
that runner is diagnostic and cannot prove this continuous final episode.

Available locally: current source geometry/motors, screened standing reference,
selected volar profile, corrected transfer planner, exact Isaac state adapter,
H1 reset and pinned walking checkpoint. Still to generate and physically
qualify: prospective material-point release route,
upright full-aperture panel route, actual stow/passage, and the same-episode
runner integration. Historical untracked templates are not required for those
new plans, and their successful physical behavior is not assumed.
