# Mechanical foundations review

Started 2026-09-05. **Incomplete and not yet deployed.** Earlier sign-off and
benchmark results do not certify the rebuilt models.

The owner found obvious mechanism and action errors in eight examples. Earlier
checks concentrated on loading, selected poses and movement under generalized
force. They missed impossible assemblies, unavailable controls and inappropriate
operation sequences. Known tilt-up clearance defects also remained behind the
old quality claim. That was a review failure.

This pass checks actual parts, load paths, stock preparation, accessible inputs,
release and return sequences, independently moving leaves and usable passage.
Native positive trials are paired with removed-part, blocked-input or restored-
defect counterexamples. Source and native-state hashes bind the evidence.
**Appearance renders are not regenerated.**

## Component reviews

- [Garage, sectional, folding and rolling mechanisms](garage-folding.md)
- [Sliding and pocket mechanisms](sliding.md)
- [Gate releases and supports](gates.md)
- [Closer mounts, pinion springs and hydraulic valves](closers.md)
- [Hatch stays and release pins](hatches.md)
- [Flexible PVC sheets and clamps](strips.md)
- [Freely rotating childproof knob covers](knob-covers.md)
- [Prepared lock stock and spindle bores](locks-and-stock.md)
- [Turnstile catches and drop arms](turnstiles.md)
- [Marine dog assemblies](ship-vault.md)
- [Vault boltwork and load paths](vaults.md)
- [Dutch joining bolts and inactive paired leaves](paired-dutch.md)
- [Automatic activation stations](automatic-pads.md)
- [Elevator guides, physical interlocks and sequencing](elevators.md)
- [Material and assembly mass](mass-and-framed-glass.md)
- [Native baseline recordings](native-baseline.md)
- [Other findings and limitations](remaining-classes.md)

## Evidence and remaining work

The first inventory covered all 1000 published models. Fresh native inventories
also compile all 1000 source models and compare authored/native site transforms
and closed-aperture passage. These bounded checks do not establish complete
mechanical correctness or embodied reach. Access-side triage remains open.

The eight owner-reported primary tasks passed a native pilot. Sixteen operating
and opened frames were personally inspected; that review caught a pocket-door
cup entering the wall before the hand released it. The corrected extraction
sequence changes to the exposed leading edge before cup occlusion. All seven
multipoint recordings were also personally inspected at their actual maximum
opening. Five open; two correctly remain locked in recognition tasks.

Current component proofs cover supported tilt-ups, all18 sectional panel paths,
30 bifold doors,22 pocket-edge pulls,57 gate assemblies,233 closer mounts on196
doors,18 hatch supports,8 PVC curtains,20 turnstile catches,6 indexed drop-arm
mechanisms,10 marine dog assemblies,12 Dutch joining bolts and8 stationary
level-car elevator assemblies. See each linked review for exact scope and
remaining failures. Component counts are not whole-door certification counts.

The12 Dutch primary tasks now pass fresh whole-door QA and native recordings.
The peek variant opens the upper leaf only; unjoined passage variants open the
upper leaf before the lower leaf. Joined pairs retain their joining bolt. Upper
leaves have actual supported pulls. Two incomplete handleset configurations
were replaced with through-spindle knob sets. Every saved native frame matches
its source model; all 12 were personally inspected in 36 initial, operating and opened native-state views. This does not certify continuous human reach or grasp.

The six chain-hoist doors require actual keeper withdrawal, chain operation,
load transfer and hands-free retention before traversal. All six source-native
keeper-v5 initializers now reach the open state and remain hands-free with the
keeper seated. Peak applied chain force is at most 120 N and peak penetration
is below 0.278 mm. This is an initialization proof; final task recordings and
the keeper-versus-upper-stop load split still require review. A formerly accepted final state
without a keeper falls1.862m in five seconds with zero hand force. The single
inside chain cannot close the door after its operator walks outside, so these
variants use an inside-starting close-only task.

A whole-door batch of 859 models outside the principal active rebuilds returned
819 passes and 40 needing attention. It exposed thin-panel cartridge interference,
additive panic-bar/trim coupling and unavailable inside service controls. The
16 repaired cartridge doors and both independent panic/trim doors now pass fresh
whole-door QA. Separate native tests verify the supported spindle bores and both
latch inputs alone and together. Eight inside thumbturn/bolt cases and four inside
panic-device service cases also pass fresh QA. Conventional rotary locksets remain
under repair; credential availability must not silently pre-unlock the exterior.

All 14 rebuilt vaults pass whole-door QA and primary native opening tasks. The
saved native frames match the exact source models. Independent component checks
cover 42 tier variants, 84 full service cycles and removed-rod/bolt controls.
All 28 closing tasks also pass with consistent initial boltwork, actual closing-stop contact and both carriers returned before latch success. All 14 have been personally inspected in 42 diagnostic views; all individual inputs and the corrected head/sill frame fit have also been personally inspected. These are mechanism inspections, not embodied human validation.

The later sectional source passed all 18 whole-door QA checks before master
integration. Native primary results include five openings and four locked
recognition attempts; nine operating cases still fail. The integration exposes
additional attachment defects, and six glazed variants incorrectly repeat the
window row on every section. These remain open work.

Security-chain components and inactive paired-leaf components have native cycle
proofs, but changed source assemblies require renewed integration checks. Ship
holdbacks, complete operation review and publication remain unfinished. Degraded
counterbalance and self-latching failures remain visible in the results.
The [native warning gate](native-warnings.md) now rejects global solver messages,
including warnings that do not increment MuJoCo's per-data warning counters.

The scripted hand remains an **oracle mechanism controller**. It uses bounded
authored-site forces or contact-gated generalized efforts and a synthetic base.
It is not a human motion reference, a physically embodied hand, or a validated
humanoid policy. Contact-dependent browser previews require recorded native
states; independent joint sliders cannot prove that a real catch has released.

### QA evidence in the master integration

Generation QA now names its task check `task_declarations_consistent`. This read-only check catches contradictory scenario names, approach-side release declarations and coordinate bounds. It does not release constraints, prescribe all mechanisms to their upper joint limit, or label a feasible-looking pose as task completion. Articulated lift progress is measured in its declared track distance or bottom height rather than compared to a barrel angle. Native task completion remains separate, source-bound evidence. The old `run_task_achievable` function is retained only for archived synthetic regression fixtures.

The attachment graph accepts separately mounted fixed-child components only when **each** disconnected island reaches actual parent stock within 2 mm, using surface distance rather than overlapping bounding boxes. Jointed carriages still need their own connected support. Synthetic missing-stock, floating-mount and jointed-carriage defects remain failures. The integrated pocket channel omits a redundant generic header brace that filled its trolley-stem slot; its own five roof mounts remain present.

### Integration checkpoint, 2026-09-06

The eight owner-reported doors pass the fresh `owner-eight-v4` automated QA
batch. This is not a new human-reference or full-task certification. The garage
carriage now has a connected steel crossmember with supports inside the track
flanges; fixed folding-door knobs have actual seated necks and mounting roses.
All 56 garage/folding tests pass. Actual garage slide bolts retain their full
physical stroke; approach-side permissions are separate from joint limits.

Paired doors now have a 4 mm hinge-jamb gap with matching mass dimensions.
Ordinary glazing accounts for the vertical-rod floor-strike undercut; all 1000
IR models build and all 11 glazing tests pass. Elevator replacement hooks no
longer coexist with obsolete interlock welds; all 35 elevator tests pass.

Wheel-driven marine doors declare four dogs and receive the four-rod assembly,
without independent dog levers or retention springs intersecting the rods.
All 88 native linkage and holdback regression tests pass on this source,
including full-travel inspection, repeated service cycles and removed-part
counterexamples. Runtime scripted holdback release remains separate open work.
Hatch rings now have an actual cross-pin and bored cup-mounted bearing ears;
the stay-release pin reaches its knob. All 47 hatch tests pass, including direct
surface-distance checks through ring travel and native hold/release tests.

A fresh 1000-door QA batch is in progress. It has already found remaining sign
and pull mounts, lock-stock defects, paired closer-return failures and a roll-up
inspection/clearance problem. These findings prevent a new blanket quality claim.
Generated evidence stays under `out/integration`; regenerated assets are not
committed. No appearance renders or deployment were performed in this checkpoint.
