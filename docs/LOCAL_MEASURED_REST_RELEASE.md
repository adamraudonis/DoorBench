# Local measured-rest release experiment

This is a privileged native CPU control experiment. It does not establish Isaac
equivalence, sensor-only control, a learned policy, or passage through the door.
The continuation starts from the qualified local transfer recipe and runs the
entire acquisition, opening, transfer and release in one motor-controlled
episode. It never installs a recorded state in the running plant.

## Qualified receiving source

`out/local-native-transfer-003` passed all 21 original physical checks, the
selected-profile actual contact audit, complete-handle audit and independent
500 Hz state/force continuity audit. Its final half-second actual palm load was
at least 3.287 N, against the original 2 N gate. The mechanism had already
spring-returned: maximum final operator magnitude was 0.000282 rad and latch
magnitude 0.00002152 m. No commanded-return milestone is inferred.

The separate one-second handoff trial, `out/local-native-transfer-004`, also
passed. It reduced the longest right-hand invalid-grasp transient from 174 to
164 ms, but increased invalid intervals from 107 to 124 and lowered final palm
margin. The original transfer003 remains the selected continuation source.

## Explicit bridge and admission

Native CLI flag `--standing-measured-rest` requires a transfer route, withdrawal
route and complete transition recording. It excludes a standing-return route.
The controller still delegates to the existing capped transfer motors until
the exact source epoch. Before withdrawal, it requires 251 consecutive actual
500 Hz observations with selected-profile opposed grip, left **palm-only** load
at least 2 N, operator magnitude at most 0.05 rad, latch magnitude at most
0.001 m, and leaf angle between 0.075 and 0.10 rad. Missing intervals reset the
window. Finger loads cannot substitute for palm contact.

Source admission requires passed physical, contact, whole-handle and transfer
reports with unchanged input hashes. It checks the prospective grasp-profile
declaration, complete actual archive, and exact equality of terminal trajectory
coordinates with the last archived post-step state. The live withdrawal also
retains the original exact attained root/joint/door-state entry check. Reports
use `measured_rest_transfer`; `return_started` remains null.
Once withdrawal starts, its recorded rest readiness is frozen entry evidence;
it does not claim ongoing grip or rest after intentional release.

## First prospective release route

`out/local-planning/measured-rest-release-001` uses the actual dominant loaded
material patch for each digit under the source-bound `volar-phalange-v1`
contract. The numerical route separates those points radially by 18 mm, slides
toward the free lever end, then lifts the hand. Every finger collider still
enters collision screening.

The independent audit passed all original 2,001 samples, with no physics
integration: maximum hand/foot position error 9.61 micrometres, orientation error
48.8 microradians, torso tilt 2.418 degrees, joint reference speed 0.621 rad/s,
and final hand clearance 51.52 mm. The unchanged distal-only counter-score is
retained and reports 412 samples with invalid distal anatomy. This route is
admitted only under its prospectively selected volar contract.

The configuration is `out/local-planning/measured-rest-release-001/withdrawal.json`.
The new physical experiment is `out/local-native-release-001`, scheduled for
66 simulation seconds, with withdrawal beginning at the source's actual 50 s
endpoint. Its physical outcome and independent audits must be read separately;
passing unstepped geometry is not physical release success.

## First physical result: failed, retained

Release001 completed all 66 seconds. Its first 100 raw chunks (0–50 s) are
byte-identical to the qualified source. Independent reduction verified all
33,000 intervals, position/velocity continuity, original motor caps, and zero
external assistance. Withdrawal began at 50 s and intentional release at
53.338 s after the original opposed-grip and palm gates passed.

The result passed 26 of 28 runtime checks. It failed final palm support and
permitted right-hand contact surfaces. The complete-handle audit passed with
zero extra loaded patches. Independent contact classification matched exactly,
but found 2,315 invalid loaded lever patches. Final right-hand clearance was
at least 161.98 mm; the leaf reached 0.35533 rad. Neither clearance nor opening
overrides the two failures.

The left palm first briefly lost contact at 58.114 s, recovered intermittently,
and had its last contact at 60.958 s. From 60.960 to 66 s the palm was unloaded;
left finger contacts carried the remaining force. The target position and
normal already follow the measured rotating leaf, and right-hand motor
tracking does not overwrite left-arm motors. By the endpoint, target error
grew to 17.5 mm and the normal offset reached its original 8 mm cap. An
unstepped multistart solve at the actual endpoint reproduced this error at the
original wrist bounds. A more face-on target worsened that solve. This is
evidence for replanning the reachable palm/body path through the opening arc,
rather than increasing stiffness or the support-force limit.

The runtime snapshots, independent audits and diagnostic reductions remain in
`out/local-native-release-001`. No physical release milestone is qualified.

All default controller behavior remains available without the new flag.
The original motor limits, contact anatomy thresholds, 2 N palm gate,
half-second qualification windows and 4 cm final hand-clearance gate remain.
