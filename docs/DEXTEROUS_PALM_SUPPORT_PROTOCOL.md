# Palm support: measured contact chatter and a proposed protocol

**Status: proposal, not an active benchmark rule.** The original native
`full-opening-teacher-actual-008` remains **failed, 20/22 gates**. Its final
aperture is 1.200613 rad; it fails the frozen requirement that palm and total
left-hand load each remain at least 2 N at every 2 ms sample. No earlier run is
rescored by this document.

The following investigation separates mechanical loss of support from short
contact impulses. It is an initialized numerical experiment on Door55 and the
versioned H1/dual-Shadow v2 plant. It is not an Isaac result, learned policy,
walking result, or complete new opening qualification.

## What was measured

The last 0.5 s contains 250 physical intervals. Eighteen intervals have zero
palm load, each lasting 2 ms. The largest matching palm-to-leaf separation is
17.55 micrometres. Mean palm load is 9.617 N; even the least-loaded rolling
10 ms window averages 5.848 N. Little-finger knuckle/middle contacts contribute
another 1.378 N on average, reported separately; they cannot stand in for palm
support. Two intervals lose all left-hand load.

The raw dynamics contacts, their forces, and pre-integration body frames were
captured immediately after each `mj_step`, before any forward recomputation.
All 250 independently reconstructed pre-state geometry samples match the
archive. Endpoint joint/root checks use the integrated state independently.
The old 251-point boundary-inclusive score remains untouched; an extra state
boundary is not integrated as an extra force interval.

## Controlled timestep experiment

The original 18,049 controls were replayed from the exact reset to recover the
complete held integration state, including solver warmstarts. Every prefix
qpos/qvel value matched bitwise. From that snapshot the same controls were held
over their original 2 ms intervals, with physics steps of 2, 1, and 0.5 ms.
Only numerical timestep changed: mass, geometry, friction, passive hand
constraints, original motor limits, and controller commands did not.

| Physics step | Mean palm load | Largest positive gap | Longest zero-load gap | Minimum 10 ms mean | Minimum 20 ms mean | Final leaf |
|---|---:|---:|---:|---:|---:|---:|
| 2 ms | 9.6173 N | 17.55 µm | 2 ms | 5.848 N | 6.365 N | 1.200612703 rad |
| 1 ms | 9.6176 N | 6.07 µm | 2 ms | 6.314 N | 6.739 N | 1.200606753 rad |
| 0.5 ms | 9.6330 N | 4.98 µm | 1.5 ms | 6.421 N | 6.624 N | 1.200592146 rad |

All eight inherited mechanical/anatomy checks pass in all three initialized
diagnostics. An independent read of the raw 2 ms replay proves bitwise equality
of all 250 final qpos, qvel, and control samples. Final aperture varies by only
20.6 microradians across timesteps; mean palm force varies by about 0.16%.
Microscopic unloading persists at finer steps, so this is not proof that a
different sampling rate removes the phenomenon. It is evidence of stable
integrated support and macroscopic motion despite contact chatter.

MuJoCo documents soft constraint impedance and stiffness/damping, and the
importance of warmstarts for exact trajectory reproduction. Its `refsafe`
rule uses `max(solref[0], 2*timestep)` for positive-format constraints. In this
experiment the palm/leaf contact time constant is 5 ms and corrected passive
loopback time constant is 4 ms; decreasing timestep from 2 ms does not change
those effective settings. These facts motivate the diagnostic, not an automatic
acceptance threshold. Sources: [solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters),
[simulation state](https://mujoco.readthedocs.io/en/latest/programming/simulation.html),
[refsafe](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-flag-refsafe).

## Proposed separate profile: `palm-impulse-support-v1`

Predeclare the profile and controller/configuration hashes before a **fresh**
full trial. Always report the original pointwise score alongside it. Proposed
acceptance combines all the following over the final complete 0.5 s:

| Measurement | Proposed limit | Purpose |
|---|---:|---|
| Maximum positive signed palm-to-leaf gap | 0.1 mm | Reject visible separation even if a delayed force sample exists |
| Longest consecutive interval below 2 N palm normal load | 5 ms | Bound actual unsupported time, independently of step count |
| Every rolling 10 ms palm normal impulse | At least 0.02 N·s | Require at least 2 N mean support in each short window |
| Every rolling 20 ms palm normal impulse | At least 0.04 N·s | Reject sparse isolated impacts carrying a high global mean |

These are proposed engineering tolerances, not published human biomechanics
constants. The 0.1 mm gap is a separate contact-resolution budget, much smaller
than the unchanged 3 mm forbidden-penetration bound. The 5 ms duration is tied
to this declared contact relaxation scale; it must not silently increase with
timestep or a different robot. A model with different compliance requires a
new profile review and validation.

The endpoint must still reach the declared aperture. Keep all existing
finite-state, original strength, joint-stop, passive finger-loopback,
forbidden-contact, upright, anatomical grasp, causal operation, right-release,
and no-helper-force/no-runtime-reset requirements. No time averaging may hide
a violation of those gates. A new support protocol must not hide a mechanically
invalid right-hand patch or an incomplete release.

Use only the actual palm's normal contact impulses on the leaf. Do not include
fingers, forearm, friction force magnitude, solver contact count, or commanded
force. Record contact locations and palm orientation so a dorsum/edge-only
shortcut can be rejected by the separately declared contact-surface audit;
the present convergence measurement alone does not establish that anatomical
surface audit. Contacts can exist in a force-free margin/gap buffer, so contact
presence is not support. [MuJoCo contact definition](https://mujoco.readthedocs.io/en/stable/XMLreference.html#contact-pair)

Use physical interval boundaries and integrate impulse once. Every rolling
window starts at a physics boundary; the timestep must resolve 10/20 ms exactly
and must be no larger than 2 ms for this proposed profile. Missing intervals,
nonfinite samples, unknown force timing, stale unmatched geometry, or contact
buffer overflow invalidate the evidence. In Isaac, use actual PhysX impulse
buffers and measured timestamps. Any native geometry mirror needs a verified
import/frame error budget before its submillimetre gap can certify this gate.

## Validation required before adoption

1. Freeze the support profile, plant identity, controller, and initial-state
   distribution. Keep the existing failed runs and reports immutable.
2. Run fresh complete acquisition → operation → palm transfer → opening
   trials, with all mechanical and anatomy gates unchanged. Include nearby
   initial-state variations rather than only one optimized posture.
3. Repeat exact-state/control convergence at another held state. Compare
   aperture, joint trajectories, signed gap, and integrated impulse; disclose
   continued contact chatter and numerical tolerances.
4. Run physical negative controls using the original plant/motor limits:
   withdraw the palm and allow coasting; leave only finger contact; create a
   longer unsupported interval. They must fail the new profile. Synthetic
   metric unit tests exercise bookkeeping only and do not replace these tests.
5. Independently verify raw archives and run an Isaac repeat with native
   contact buffers. Report engine differences without changing thresholds
   after seeing the result.

## Reproduction and evidence

Run from the repository with the MuJoCo 3.12 environment and original local
robot, Door55, and motor-contract inputs:

```sh
python scripts/dexterous/analyze_panel_contact.py \
  --run out/bimanual-native/full-opening-teacher-actual-008 \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --door assets/doors/db0055_swing_single/door.xml
python scripts/dexterous/probe_panel_contact_convergence.py \
  --source-run out/bimanual-native/full-opening-teacher-actual-008 \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --door assets/doors/db0055_swing_single/door.xml \
  --motors /path/to/h1-import.motors.json --output out/contact-convergence-new
python scripts/dexterous/audit_panel_contact_replay.py \
  out/contact-convergence-new --output out/contact-convergence-new/independent-audit.json
```

The compact [evidence receipt](../results/dexterous/2026-09-08/palm-contact-convergence.json)
records run names, hashes and scope. Raw evidence includes the complete source
trial, the held `mjSTATE_INTEGRATION`, original controls, every replay interval,
sources frozen at execution, and the independent equality audit. Trial
`contact-convergence-001` failed before stepping and is retained. Frozen trial
002 omitted its in-report tail-error field because of a floating-point equality
test on inferred timestep; its independent audit proves zero error. The tool
now rounds the inferred source timestep for future runs; the frozen report is
not rewritten.
