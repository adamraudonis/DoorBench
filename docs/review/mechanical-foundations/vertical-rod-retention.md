# Vertical-rod latch retention — unresolved

The 2026-09-06 master integration has 31 doors declaring vertical-rod latches.
DB0019 provides a reproducible native failure: after the closer starts at 60°
and runs for 12 seconds, the active leaf stops at 0.03270 rad. Its top latch
remains 14.99 mm retracted against the head strike; the bottom latch has extended
and bears against the floor-strike wall at approximately 44.5 N. Source-bound
probe output is under `out/integration/paired-closer/db0019.json`.

The existing independent spring-return top and bottom sliders omit a necessary
retention sequence. Allegion describes the top-latch holdback keeping the bottom
latch retracted while the door is open, and identifies failure of this feature
as a cause of dragging bottom rods:
[manufacturer explanation](https://kc.allegion.com/kb/article/3347a-bottom-latch-dragging/).
The [electrical-options installation booklet](https://us.allegion.com/content/dam/allegion-us-2/web-files/von-duprin-/installation-documents/Von_Duprin_Electrical_Options_Booklet_Installation_Instructions_107111.pdf)
also distinguishes the retracted open-door latch and extended release trigger
from the closed, deadlocked state. This supports the missing function; it does
not validate the current generated geometry as a manufacturer reproduction.

Required repair: model an actual supported retention part, its frame-contact
release trigger, and a mechanically connected bottom-rod retractor. Preserve
real independent leaves, accessible push bars, spring loads and physical force
limits. Releasing the push bar after opening must leave both latches withdrawn;
closing-frame contact must release retention so both latch points seat.

Acceptance must include repeated native opening/closing with no per-step pose
writes, no angle-triggered latch freeze, and no increased closing force to ram
past the defect. Removed retention and blocked release-trigger cases must fail
for the corresponding mechanical reason. Review both sides and mirrored leaves.
Do not report the 31 affected source models as mechanically certified pending
that repair and fresh source-bound tests.
