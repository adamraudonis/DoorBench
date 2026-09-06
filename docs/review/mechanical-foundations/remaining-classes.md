# Remaining native classes: bounded mechanical review

The initial isolated audit generated all 90 current fixtures across revolving,
tripod/full-height turnstiles, Dutch, saloon, strip curtain, elevator and automatic
sliding families. It preserved the published inputs and generated only under
`out/mechanical-foundations/remaining`. `inventory.json` and `receipt.json` bind
the initial per-door exports and source inventory. The initial generic QA passed
84/90; all six failures were strip curtains. Passing generic QA did not establish
that the hardware was connected or that contact sites lay on real surfaces.

## Rotating hardware repaired

All nine revolving variants with authored push bars had a 38 mm air gap between
the bar and the nearest leaf material. These are DB0066, 0108, 0260, 0575, 0777,
0779, 0822, 0932 and 0958. They now use a radial bar connected by two mounting
legs to the central shaft and the outer stile. All 15 revolving doors, including
the six bare-glass variants, have an explicit exposed tangential push surface.
The manufacturer's [Tourniket product documentation](https://www.boonedam.com/products/revolving-doors/tourniket)
establishes the manual revolving-door class; this original support geometry is
not an OEM mounting drawing or a strength certification.

All ten tripod variants had an eccentric X-oriented rotating shaft perpendicular
to the actual inclined hinge axis. The shaft center orbited the joint instead
of remaining coaxial. The repair preserves the existing axis and arm sweep: a
coaxial journal now passes through a hollow stationary bearing housing, connected
by a cantilever to the actual cabinet back. The 0.5 mm radial assembly clearance
is explicitly tested through a full revolution. Internal bearing contact and
axial retention remain ideal joint behavior. [Boon Edam's component description](https://www.boonedam.com/en-us/products/tripod-turnstiles)
identifies the rotating hub, supporting housing and separate credential lock;
it does not validate this generic bracket's dimensions or load rating.

All ten full-height variants put their push sites 59.57 mm away from the nearest
arm, with an upward normal that produced zero moment about the vertical rotor
axis. Their sites now lie on a real arm retained in every tier; the surface
normal is tangential and pressing inward produces the intended positive moment.
Tripod contact sites were also corrected from the arm center to its surface.

`rotor-final-qa.json` records **35/35 dedicated checks and 35/35 full native QA**.
Five focused tests cover all full/simple/minimal surfaces, geometry-counterexample
rejection, preserved credential-locked ranges and native 40 N surface pushes on
every unlocked rotor. No coordinates are reset during those force trials. This
is a bounded hardware/force test, not a humanoid grasp or traversal certificate.

## Current status after the component rebuilds

The initial findings above are historical evidence, not the current repair status.

- **Turnstiles:** all 20 now have physical credential catches and six have indexed
  drop arms. Native tier, direction, credential and repeated reset tests are in
  [turnstiles.md](turnstiles.md). Arbitrary mid-rotation power failure, impact loads
  and structural strength are not certified.
- **Strip curtains:** all eight now use flexible sheets with supported clamps and
  actual push surfaces. Native bounded-force tests and remaining modeling limits
  are in [strips.md](strips.md).
- **Closers:** 233 mounts on 196 doors are rebuilt, with actual pinion springs,
  supported arms and separately checked hold/delay behavior. See
  [closers.md](closers.md). DB0508's degraded self-latching failure remains visible.
- **Dutch doors:** all 12 pass fresh whole-door QA and primary native tasks, with
  supported upper pulls and explicit joined, sequential and upper-only operation.
  All 12 were personally inspected in 36 diagnostic views. See
  [paired-dutch.md](paired-dutch.md). This is not embodied two-hand certification.
- **Saloon doors:** all 12 generic native checks pass. The upward push-site normals
  were corrected to the outward leaf-face normal. This does not independently
  certify simultaneous two-leaf human contact.
- **Elevators:** all eight have stationary level cars, physical door interlocks,
  drives and causal call/obstruction/presence logic. See [elevators.md](elevators.md)
  for measured controller and mount tests. This does not model car travel or
  certify an elevator safety system. Automatic sliding sensor operation still
  requires review beyond a generic opening check.
- **Security accessories:** chain links, mounted eyes, slotted keepers and free
  release heads now replace range-limited rigid decorations. Four chains remain
  in final repeated insertion/release testing; both swing guards pass fresh
  whole-door QA. The old range-only accessory results are invalid for this repair.
- **Locks and service faces:** thin-panel stock, both independent panic/trim
  inputs and retained inside thumbturn/bolt controls have fresh tests. Independent
  conventional rotary locksets are still under repair. A usable credential is
  permission to operate a release, not a reason to start with the lock released.
- **Jammed conditions:** twelve `jam_stuck` cases represent breakaway friction,
  not an immovable lock. Bounded forces may open them. A mechanical latch, a true
  bolt and the authored task must be distinguished from that friction.

The current gate also rejects global native solver warnings that do not appear
in `MjData.warning`; see [native-warnings.md](native-warnings.md). Neither the
initial generic passes nor the component proofs constitute an all-catalog
mechanical sign-off. The [review index](README.md) tracks outstanding families.
