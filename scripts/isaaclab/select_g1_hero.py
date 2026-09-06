#!/usr/bin/env python3
"""Choose sixteen distinct audited native successes for illustration, not scoring."""

import argparse
import hashlib
import json
from pathlib import Path


def select(audit, assets, count=16):
    candidates = []
    for row in audit["per_door"]:
        if not (
            row["traversal_success"]
            and row["vertical_traversal_applicable"]
            and row["native_spatial_elements_supported"]
        ):
            continue
        spec = json.loads((assets / "doors" / row["door_id"] / "spec.json").read_text())
        # Compact cells are reserved for appropriately sized assemblies.
        if spec["opening"]["width"] <= 5.0 and spec["opening"]["height"] <= 3.3:
            candidates.append(row)
    chosen, families = [], set()
    for row in candidates:
        if row["family"] not in families:
            chosen.append(row["door_id"])
            families.add(row["family"])
    chosen = (
        chosen + [r["door_id"] for r in candidates if r["door_id"] not in chosen]
    )[:count]
    if len(set(chosen)) != count:
        raise ValueError(
            f"Need {count} distinct audited successes; found {len(chosen)}"
        )
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ids = select(json.loads(a.audit.read_text()), a.assets)
    a.out.write_text(
        json.dumps(
            {
                "scope": "Sixteen selected successful vertical fixtures rerun together for illustration; not a random performance sample",
                "audit_sha256": hashlib.sha256(a.audit.read_bytes()).hexdigest(),
                "ids": ids,
            },
            indent=2,
        )
        + "\n"
    )
    print(" ".join(ids))


if __name__ == "__main__":
    main()
