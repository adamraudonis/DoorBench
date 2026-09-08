# Tactile thumb relief: bounded native comparison

This opt-in component keeps the first23s of the original sensor-balanced
thumb-flexion trial. It then changes only five thumb joint goals. The palm,
four other fingers, arm press route, balance estimator, original force caps,
couplings and all original physical/contact/tracking gates remain unchanged.
The high-level reaching and pressing route is scripted for this door instance.
This is not a learned or vision-based policy.

The preceding scripted thumb endpoint failed because its geometrically interior
pose was not enforced by the available pressure-controlled motor directions.
THJ2 crossed its authored stop despite zero virtual-normal request. This new
comparison uses the position-effort directions that the existing controller
retains: THJ5/4/3 and the THJ2/1 direction `K^-1 t`, where `t` is orthogonal to
the original pressure-moment vector. Therefore `p.T K qdot=0` for the paired
goal adjustment. It neither invents independent unavailable finger torques nor
removes the existing THJ1/THJ2 normal-pressure control.

The own thumb tactile grid supplies a finite force resultant. Its channel order
is z,x,y; positive into-pad force indicates a retreat direction in the own-palm
frame. This resultant includes measured shear and is **not** a known object
surface normal. The unchanged local palmar projection target is3N, compared
with2N per other finger. Their difference deliberately leaves a net pressing
load to be balanced through wrist/handle reactions.

At23s the component captures the five incoming thumb goals and actual
robot-local pad pose. Subsequent high-level thumb changes are ignored within
this profile; the captured goals plus this bounded controller own the thumb
targets. Other named goals are forwarded unchanged. Only numeric encoder
positions, calibrated local tactile bins and a2ms local clock are consumed.
The separate robot-only FK data object is never stepped. No world root, door
geometry, contact IDs, simulator handles or offline contact centroids enter it.

The frozen [profile](../configs/dexterous/sensor-thumb-normal-admittance-v1.json)
declares:

- local-force filtering20ms and admittance0.15mm/(N·s), capped at0.5mm/s;
- maximum reference displacement2mm in the own-palm frame;
- target changes≤80mrad, speed≤0.06rad/s and acceleration≤0.2rad/s²;
- THJ2 interior velocity priority toward35mrad authored-stop margin, other
  thumb joints toward10mrad margins;
- a terminal controller guard if actual encoder THJ2 margin falls below15mrad,
  before the existing physical stop-tolerance gate would permit penetration;
- own-thumb frame rotation limited to0.12rad, with original loaded distal
  anatomy and lever-end clearance evaluated independently at every interval.

The velocity solve first respects hard command/bound/rate envelopes. An active
interior request is limited by its already-declared acceleration bounds and
records any remaining recovery-velocity deficit. The2mm admittance accumulator
clips without integrating beyond the bound and pauses force integration on
fresh unloaded samples. Freshness errors still reject the entire episode.
Goal changes enter before the existing normal-effort projector and original
61-motor cap/history owner; returned motor forces are never edited afterward.

The detached screen
`out/continuous/sensor-thumb-admittance-plan-002/report.json` passes five groups
of checks. At the actual23s source pose the retained position Jacobian has full
translation rank; singular values are0.08994,0.07083 and0.00472m/rad. A maximum
1mrad positive relief perturbation changes minimum signed gap from−101.606µm
to−95.247µm; its negative control increases overlap to−107.952µm. All overlapping
patches retain the original allowed surfaces.

Two detached1001-sample kinematic paths with fixed synthetic3N/6N local loads
preserve all other joints, fingers and palm, keep thumb opposition, and leave
at least4.745mm axial clearance for the actual source contact material point.
That offline point is never sent to the controller. The6N case geometrically
separates after relieving pressure, as a kinematic screen may; this is not a
claim of physically sustained contact or predicted load. The actual comparison
must still pass the original opposed-pad interval gate without exemptions.

The finite physical probe takes the original thumb-flexion inputs plus:

```sh
--thumb-admittance-protocol configs/dexterous/sensor-thumb-normal-admittance-v1.json \
--thumb-admittance-screen out/continuous/sensor-thumb-admittance-plan-002/report.json
```

This profile is mutually exclusive with the failed frozen thumb endpoint
coordinator. Every physical attempt uses a fresh output directory and retains
its exact source, command/sensor history and actual-step contact evidence.
