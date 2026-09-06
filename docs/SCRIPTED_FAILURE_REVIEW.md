# Scripted baseline failure review

Reviewed September 6, 2026. The archived run is dated **2026-09-04 22:15:55 UTC**,
source `a0daad2f2083a67341b3c7456ee1f0b41e5e6ebb`. It predates the current mechanical
rebuild and does not certify the new fixtures.

After excluding the separately downloadable pet collection, there are **455 failed
episodes on 136 doors**, out of 4,182 episodes on 985 doors. Of those failures,
20 were marked damaged and all 20 recorded a slam. The other 435 did not satisfy
the task criteria. These are outcome labels, not automatically diagnosed causes.

| Unmet criterion | Failed episodes (overlapping) |
|---|---:|
| `traversed` | 319 |
| `opened` | 283 |
| `closed_behind` | 98 |
| `unlock` | 60 |
| `!opened` | 33 |
| `!damage` | 20 |
| `!slam` | 20 |
| `closed` | 18 |
| `latched` | 12 |
| `latched_behind` | 10 |

Examples from retained event traces:

- **DB0009, open then close, seed 1:** opened and traversed, then slammed while
  closing. Recorded closing impact was 9.1 against a threshold of 4.0. This is
  a controller/task failure; removing it would reward unsafe closing behavior.
- **DB0192, open then close:** opened and traversed but did not close and latch
  behind. The return/re-latching sequence needs diagnosis, not score filtering.
- **DB0736, locked recognition:** opened when the task required remaining locked.
  This requires a lock-holding/model check as well as controller review.
- **DB0891, unlock and traverse:** opened and traversed without the required
  unlock event. It is not a valid success merely because the robot passed through.
- **Accordion doors:** 72 failed episodes in the archived run. Later engineering
  notes identify impossible fold coupling/range combinations in earlier models.
  Validate the corrected source and rerun; old results must not be relabeled.

## Exclusion decision

**Do not exclude a test because the scripted policy fails it.** Keep timeouts,
missed handles, slams and failed relatching in the denominator when the fixture
and task are valid. A weak baseline is informative.

Quarantine a specific fixture/scenario revision only with evidence of an invalid
export, impossible mechanism, inaccessible required control, contradictory task,
or unsupported simulator physics. Record the door ID, source hash, reason and
counterexample. Report the quarantined count alongside the evaluated count;
never silently remove failed episodes. Restore the test after repair and validation.

The current integration already preserves failing QA and native warnings instead
of signing off all doors. Its frozen 1,000-door QA inventory and limitations are
in [the handoff review](../handoffs/handoff-reconciliation-2026-09-06.md).
The new G1 diagnostic is separate from these scripted-hand results: its uniform
closed-start traversal task is not the assigned core benchmark suite.
