# Detached Isaac coupled release geometry

The Isaac coupled adapter initializes the existing unstepped measured-angle
evaluator from a freshly qualified **actual PhysX source**. It does not call
the native environment constructor, fabricate a native manifest, replay source
samples, or grant stage permission. The existing native path is unchanged.

## Inputs and source contract

`IsaacCoupledReleaseGeometry(envelope_path, source_config=config_path)` requires
an explicit Isaac `doorbench.standing-withdrawal.v1` source configuration accepted
by `load_withdrawal_source_context`. This configuration contains the actual
source, exact episode epoch and normalized endpoint, original robot/door assets,
prospective grasp profile, fresh Isaac RH candidate and its distinct passing
2,001-sample dense geometry audit. The actual recorded motor contract is read
from `trial/motor-contract.json` and independently bound in full.

Keep this source configuration separate from the coupled map. The source loader
continues to reject native coupled/moving-leaf proofs for Isaac. The opt-in
[actual-Isaac runtime adapter](ISAAC_WITHDRAWAL_RUNTIME.md) can consume the map
only through a separate outer runtime document, its distinct passing envelope
audit, and a completed live source-prefix witness. The detached map itself
continues to grant no stage or physical qualification.

The new map schema is `doorbench.isaac-coupled-release-envelope.v1`.
`plan_local_isaac_coupled_release.py` creates it from the freshly admitted Isaac
context. It solves both palm poses and fixed feet using the existing coupled
least-squares arithmetic, with the new actual-source RH path supplying numerical
preferences. No native coordinates or historical door trajectory are accepted.
The generated map retains:

- Exact source admission, context hash, state hash, normalized qpos, epoch,
  profile, audited duration and motor-contract fingerprint.
- Absolute source-config, RH-candidate, RH-dense-audit and actual physics-NPZ
  identities, plus every source/candidate/audit input hash.
- The fixed `JOINT_NAMES` order declared in the adapter. Tensor coordinates are
  root translation delta, root rotation vector, then those scalar joints.
- At least 81 progress nodes over the audited duration and 17 aperture nodes
  over `[0.08, 0.4]` radians, with maximum gaps of duration/80 and 0.02 radians.
  Include the actual source aperture as a node. Its first raw coordinate must
  equal `[0,0,0,0,0,0,*actual_joint_coordinates]` exactly.
- A finite source-containing operator interval within `[-0.05,0.05]` radians,
  actual operator reference, and optional explicit progress/aperture upper
  boundary. The measured latch domain remains `[-0.001,0.001]` metres.
- An explicit handle-follow release clock below the original eight-second RH
  route endpoint. The inherited geometric evaluator and arm correction are
  unchanged; the RH route must still end at eight seconds.
- Zero physics steps, source playback, active state writes and authorized
  stages; physical admission and runtime-route export remain false.

The planner's separate preference file can contain only bounded scalar choices:
`time_nodes` (81..401), `max_nfev` (1..500),
`follow_handle_through_route_seconds` (0..7.5), `operator_margin_rad`
(0.001..0.05), `initial_aperture_headroom_rad` (0..0.05), and
`final_aperture_upper_rad` (0.1..0.4). Defaults are 81, 180, 3.5, 0.01,
0.015 and 0.4, respectively. Any old report contributes only these fields;
coordinates, qualifications, asset paths and trajectories are ignored.

The operator domain is the actual operator coordinate plus/minus the chosen
margin, clipped to the original resting bounds. The aperture upper boundary
rises linearly from actual aperture plus the chosen headroom to the selected
final upper bound. This is a prospective domain for measured-state admission;
it does not command or predict a door trajectory. The full interpolation grid
is solved, including nodes outside that narrower domain. Every solver result
records convergence and residuals, without treating convergence as admission.

The adapter derives initial feet, left palm coordinates in the leaf frame, and
center of mass from the exact actual source. It never accepts those values from
an older native candidate. The first raw tensor node is checked independently
before the evaluator's exact-source special case can overwrite anything.

## Complete preparation sequence

The following commands are usable only after the specified actual source has
passed its original physical/contact/transfer gates. Each stage rechecks its
inputs and retains failures; none promotes a failed source. These example paths
are placeholders and the output must be fresh. Actual transfer-002 and
transfer-003 failed qualification and cannot supply this pipeline.
The optional prospective `--standing-transfer-stop-on-rest` mode can select a
new transfer's first fully qualified resting window before its deadline; see
[runtime endpoint rules](ISAAC_WITHDRAWAL_RUNTIME.md). Its recorded actual
terminal epoch is the source epoch for every planning stage. An intermediate
window found retrospectively in a failed run is not a qualified source.

```powershell
$source = 'out/QUALIFIED_ISAAC_TRANSFER'
$robot = 'out/local-ready/h1-shadow-loopback-v2.xml'
$doorXml = 'assets/doors/db0055_swing_single/door.xml'
$doorUsd = 'assets/doors/db0055_swing_single/door.usda'
$plan = 'out/local-planning/isaac-release-next'

python scripts/dexterous/plan_local_isaac_release.py `
  --isaac-source $source --robot $robot --door-xml $doorXml --door-usd $doorUsd `
  --output "$plan/rh" --grasp-profile volar-phalange-v1

python scripts/dexterous/audit_local_isaac_release.py `
  --candidate "$plan/rh/candidate.json" --isaac-source $source `
  --robot $robot --door-xml $doorXml --door-usd $doorUsd `
  --output "$plan/rh-audit.json" --duration 16 --grasp-profile volar-phalange-v1

python scripts/dexterous/prepare_local_isaac_withdrawal_source.py `
  --isaac-source $source --candidate "$plan/rh/candidate.json" `
  --dense-audit "$plan/rh-audit.json" --robot $robot `
  --door-xml $doorXml --door-usd $doorUsd --output "$plan/withdrawal-source.json" `
  --grasp-profile volar-phalange-v1

python scripts/dexterous/plan_local_isaac_coupled_release.py `
  --source-config "$plan/withdrawal-source.json" --output "$plan/map"

python scripts/dexterous/audit_local_isaac_coupled_release.py `
  --envelope "$plan/map/envelope.json" --source-config "$plan/withdrawal-source.json" `
  --output "$plan/map-audit.json"
```

The source-configuration command obtains state/epoch/assets from fresh admission
and requires the full distinct RH audit. It exports detached input data only.
The coupled generator captures its source files and emits `envelope.json` only
after completing finite solves and rechecking source hashes. Incomplete numerical
convergence remains visible in the unadmitted candidate for the independent
audit; no passing audit is manufactured.

## Independent static audit

Run only after a new source-bound map exists:

```powershell
python scripts/dexterous/audit_local_isaac_coupled_release.py `
  --envelope <new-Isaac-envelope.json> `
  --source-config <Isaac-withdrawal-source.json> `
  --output <fresh-independent-envelope-audit.json>
```

The distinct receipt schema is
`doorbench.isaac-coupled-release-envelope-audit.v1`. The audit samples independent
dense progress/aperture points, actual map knots and cell centers, operator
extrema, and both latch extrema. It includes the exact actual source state.
`--coarse` is diagnostic only and can never pass.

Original limits remain: 1 mm feet/palm position error, 0.01 rad orientation
error, 4 degree torso tilt, 30 mm root translation, 15 mm horizontal COM change,
1 mm palm-panel gap change, 3 mm nonfoot penetration, and 0.02 rad original joint
and tendon tolerance. The selected anatomical profile and unchanged distal
counter-score share the original 1 mm axial margin and strict 0.8 normal
alignment threshold. After handle following ends, all RH/handle geometry must
have at least literal 4 mm clearance. Every sampled final mechanism state must
also retain the original 40 mm RH/environment clearance.

Passing geometry does not qualify motor loads, contact forces, dynamic balance,
mechanism speeds or geometry after rate limiting. The opt-in runtime adapter
requires the exact live source-prefix witness, constraint-preserving motion
limits and per-step post-limiter geometry checks. An actual trial must then pass
the independent physical contact gates.
The receipt explicitly leaves those qualifications and stage authorization
false/zero. No successful map or physical release is claimed by these additions.

## Validation

The new test files contain synthetic binding cases and unstepped toy-model
geometry/solver tests, clearly separate from robot trials. They cover source
and audit failure, source capture, exact first coordinates, solver bounds,
all-node coverage, nonfinite solutions and original geometric checks.
A read-only admission check rejected actual `local-isaac-transfer-002` with
`Complete passing physical operation report required`; tracked evidence hashes
were unchanged. Its failure is preserved and it was not used to generate a map.
