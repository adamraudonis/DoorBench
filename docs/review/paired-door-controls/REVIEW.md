# Paired-door controls: independent review

**Reviewed 2026-09-05: no remaining required fix identified within the paired-control scope.** The review used all 76 published `swing_double` assets in `out/collection-release/candidate/assets`. No native asset or recording was modified.

The controls provide a **kinematic mechanism preview**. They do not certify robot accessibility, human manipulation, complete collision clearance, or dynamics. Dutch, folding and sliding-door behavior remains separate.

## Resulting behavior

Each free leaf uses its own authored joint range, axis and operator descendants. Operator actuation retains the existing tendon/equality follower updates. Partial pairs open fully before toggling closed. Independent leaf sliders do not mirror the other leaf and respect the same lock and astragal dependencies.

Inspection sliders accept continuous coordinates, including exact authored endpoints. A fixed 0.001 step could round DB0792's 1.5708 rad thumbturn limit down to 1.57, leaving its coupled deadbolt short of full withdrawal. The mechanical guard remains strict.

Secured inactive leaves retain their 0–0.001 rad ranges. Active lock welds remain enforced. Independent thrown deadbolts must be withdrawn; fixed inaccessible bolts remain secured. Manually throwing an auxiliary barrel bolt also blocks its leaf until withdrawal. A French door's free A leaf remains operable while inactive B stays bolted closed.

For an astragal attached to B, A opens first and B closes first. The slider guard requires the dependent endpoint within 1e-6 rad: a broad “nearly closed” tolerance is insufficient. With DB0871 at A=0 and B=0.015 rad, native contact geometry reports 14.08 mm slab/astragal penetration.

## Published inventory

| Pattern | Count | Interpretation |
| --- | ---: | --- |
| Both active, ordinary same-device pair | 38 | Independent leaves and mechanisms |
| Both active, double egress | 10 | Same world-Z hinge-axis sign; do not negate B's coordinate |
| Inactive B, flush bolts | 20 | Preserve its restricted range |
| Inactive B, cane bolt | 8 | Preserve its restricted range |

The 48 active pairs have matching numeric ranges in this snapshot: 90° (33), 100° (10), 105° (1), 110° (4). Controls still use each joint's own limits. Global `meta.u/v/hinge_x` describes the last built leaf; individual body/joint transforms are authoritative.

Nine active pairs contain engaged maglock/delayed-egress welds. DB0396 and DB0841 weld both leaves; seven weld A only. The current representation is an active `weld` between a leaf body and `world`. Six French pairs have fixed inaccessible deadbolt/multipoint locks; DB0792 has an engaged, retractable independent deadbolt. Four barn pairs have auxiliary barrel bolts, initially withdrawn: DB0127, DB0279, DB0704 and DB0788.

Four active pairs have B-mounted overlapping astragals: DB0183, DB0261, DB0494 and DB0871. The first three weld A closed, consequently preventing B's opening too. At authored initial states, the final helper moves **103 free leaves across 64 pairs** and leaves all leaves stationary in **12 secured/dependent pairs**, consistent with this inventory.

## DB0019 native evidence

Both hinges span 0–π/2 rad with Z axes −1/+1. Each panic bar travels 16 mm and drives a separate top latch through the one-sided fixed tendon `latch_q >= (8/7) * bar_q`; the latch range is 0–19 mm. There is no leaf-to-leaf coupling.

Its null `meta.operator_joint` is intentional approach-side metadata: an exterior robot cannot press the interior panic bars, and the exterior trim is a fixed pull. Twelve pairs share this null-pointer-with-real-panic-operators pattern. See `doorbench/geometry/hinged.py:621`. The preview demonstrates the inside devices without changing benchmark accessibility metadata.

A private native MuJoCo probe used unchanged XML and a three-second bounded hinge push. With neither bar pressed, both hinges ended at 0.001920 rad; pressing only A gave A=0.891037/B=0.001920; pressing both gave A=B=0.891037 rad. No MuJoCo warnings occurred. This establishes the bounded native mechanism behavior, not a benchmark result or validated manipulation trajectory. The reported one-leaf UI behavior does not require a physical asset change.

DB0019 SHA-256:

| File | SHA-256 |
| --- | --- |
| spec.json | fd6be0fc448d9b87eb5e9ef8ff75dbc04e38ebbd3007b70640d0b153c3a7737d |
| model.json | 43d3bb6171c760b77ae8b3ee7f24df8b2cc4cb5af012563a52498bab0de8d920 |
| door.xml | d15a2026fdd4283728c31a0781da66995e44c57f3ecb7196230922b6598d990c |

Candidate and tracked DB0019 model/XML bytes match.

## Verification and limits

Independent run: **13 focused tests passed, 657 assertions; TypeScript checking passed.** Regression fixtures include DB0019's missing default operator, partially open/asymmetric coordinates, double-egress axes, inactive B, active welds, DB0832's free French A leaf, DB0534's fixed deadbolt, DB0792's explicit thumbturn, DB0714's already-retracted bolt, DB0127's thrown/withdrawn barrel bolt, and astragal button/slider ordering.

Ordering also has bounded native geometry evidence: cross-leaf contact queries sampled all four active overlapping-astragal pairs over 0–1.4 rad in 0.01 rad increments, with both latches retracted. A fully opening before B produced no cross-leaf penetration. Equal-angle simultaneous opening produced 5.34–5.96 mm penetration; B first produced 22.52–22.93 mm. These checks cover cross-leaf contacts only. They are not continuous or full-scene collision certification; arbitrary inspection poses and dynamically feasible manipulation remain outside this review.

Reviewed source SHA-256 (before commit):

- `viewer/src/doorLogic.ts`: `667760b6ef89ba6e83c0e7ab9ef9e9cd795b865a2af3190d972fc8476b82a6ba`
- `viewer/src/doorLogic.test.ts`: `5177bc6ace9775f93a8724aab889e424617b10be4f4d53a29ef4b446b56f1b16`
