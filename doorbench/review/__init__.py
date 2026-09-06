"""Vision review: photograph every door and ask a vision model the question a person would ask.

The deterministic gates in ``doorbench/qa.py`` answer *measurable* questions - does anything
interpenetrate, does every part touch what holds it, does the leaf move under a push.  Every one of
them was written after a human looked at a picture and said "that is obviously wrong".  This package
is the systematic version of that human step:

``sheet``    renders one labelled review sheet per door (three poses x three viewpoints plus three
             close-ups), captioned with what the spec SAYS should be there, so completeness can be judged.
``prompt``   the rubric and finding categories the reviewer is asked to apply.
``verdict``  the strict JSON verdict schema, its validation and its normalisation.
``api``      the Anthropic transport (per-door messages or the 50 %-cheaper Batches API), cost
             estimation, retries and resumability.
``report``   docs/VISION_REVIEW.md from the verdicts on disk.
"""
from __future__ import annotations

__all__ = ["sheet", "prompt", "verdict", "api", "report"]
