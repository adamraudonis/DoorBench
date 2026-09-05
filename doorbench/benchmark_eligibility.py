"""Benchmark scope, independent of simulator and stale asset metadata.

Standalone pet doors remain downloadable supplementary assets. A pet-flap
insert in a standard-size slab does not change the host door's eligibility.
"""
from collections.abc import Mapping

POLICY_VERSION = "doorbench.benchmark-eligibility.v1"
EXCLUSION_REASON = ("Standalone pet doors are downloadable supplementary assets and "
                    "excluded from robot/human benchmark evaluation.")

class BenchmarkExcludedError(ValueError):
    """The requested asset belongs to a supplementary collection."""


def is_benchmark_eligible(spec_or_family) -> bool:
    family = spec_or_family.get("family") if isinstance(spec_or_family, Mapping) else spec_or_family
    return family != "pet_door"


def benchmark_eligibility(spec_or_family) -> dict:
    eligible = is_benchmark_eligible(spec_or_family)
    return {"eligible": eligible, "collection": "standard" if eligible else "supplementary_pet_doors",
            "reason_code": None if eligible else "standalone_pet_door",
            "reason": None if eligible else EXCLUSION_REASON}


def require_benchmark_eligible(spec_or_family, *, operation="benchmark evaluation") -> None:
    if not is_benchmark_eligible(spec_or_family):
        name = spec_or_family.get("id", "pet_door") if isinstance(spec_or_family, Mapping) else spec_or_family
        raise BenchmarkExcludedError(f"{name}: excluded from {operation}. {EXCLUSION_REASON}")


def collection_counts(doors) -> dict:
    rows = list(doors)
    standard = sum(is_benchmark_eligible(d) for d in rows)
    return {"n_assets_total": len(rows), "n_doors_eligible": standard,
            "n_doors_supplementary": len(rows) - standard, "eligibility_policy": POLICY_VERSION}
