#!/usr/bin/env python3
"""Draw the README's historical scripted-hand overview from verified raw episodes.

Run from any directory; --check compares exact deterministic SVG bytes without
writing. Only the Python standard library is required. No benchmark is executed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
POLICY = "doorbench.benchmark-eligibility.v1"
SOURCE_FILE = "scripted_hand.json"
COLORS = {"all": "#397450", "some": "#d2a33e", "none": "#b46750"}


@dataclass(frozen=True)
class DoorOutcome:
    door_id: str
    family: str
    successes: int
    episodes: int

    @property
    def outcome(self) -> str:
        return "all" if self.successes == self.episodes else "some" if self.successes else "none"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verified_outcomes(index_path: Path, results_path: Path) -> tuple[list[DoorOutcome], dict]:
    """Recompute the eligible core subset; reject stale or inconsistent inputs."""
    index_bytes, raw_bytes = index_path.read_bytes(), results_path.read_bytes()
    index, raw = json.loads(index_bytes), json.loads(raw_bytes)
    require(index.get("schema_version") == "1.1", "Expected results index schema 1.1")
    require(index.get("eligibility_policy") == POLICY, "Index lacks the standard-door eligibility policy")
    rows = [r for r in index["results"] if r.get("file") == SOURCE_FILE
            and r.get("policy") == "scripted_hand" and "core" in r.get("suites", {})]
    require(len(rows) == 1, "Expected exactly one scripted_hand.json core result in the index")
    row, subset = rows[0], rows[0].get("historical_subset", {})
    require(subset.get("policy") == POLICY and subset.get("source_file") == SOURCE_FILE,
            "Historical subset source/policy is missing or inconsistent")
    raw_hash = sha256(raw_bytes).hexdigest()
    require(raw_hash == subset.get("source_sha256"),
            "Raw scripted_hand.json SHA-256 does not match index historical_subset.source_sha256")
    require(raw.get("policy", {}).get("name") == "scripted_hand", "Raw policy is not scripted_hand")
    require(raw["run"].get("suite") == row.get("suite") == "core", "Expected a core-suite source run")
    require(raw["run"]["date"][:10] == row["date"], "Source and index run dates differ")
    require(raw["benchmark"].get("commit") == row.get("commit"), "Source and index run commits differ")
    seeds = row["seeds"]
    require(isinstance(seeds, list) and len(seeds) == 3
            and all(type(s) is int for s in seeds) and len(set(seeds)) == 3
            and raw["run"].get("seeds") == seeds, "Expected the same three distinct seeds in source and index")
    scenario_names = {s["name"] for s in raw["run"]["scenarios"] if s["suite"] == "core"}
    require(scenario_names == set(row["suites"]["core"]["scenarios"]), "Core scenario inventories differ")
    families, identities = {}, set()
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    coverage: dict[tuple[str, str], set[int]] = defaultdict(set)
    excluded, excluded_episodes = set(), 0
    for episode in raw["episodes"]:
        require(episode.get("suite") == "core", "Source contains an episode outside its declared core suite")
        door, family, scenario, seed = (episode[k] for k in ("door_id", "family", "scenario", "seed"))
        require(isinstance(door, str) and re.fullmatch(r"db\d{4}_[a-z][a-z0-9_]*", door) is not None,
                f"Malformed door ID: {door!r}")
        require(door.split("_", 1)[1] == family, f"Door ID/family mismatch for {door}")
        require(families.setdefault(door, family) == family, f"Inconsistent family for {door}")
        require(scenario in scenario_names and type(seed) is int and seed in seeds,
                f"Unexpected scenario or seed for {door}")
        identity = (door, scenario, seed)
        require(identity not in identities, f"Duplicate episode: {identity}")
        identities.add(identity)
        require(type(episode.get("success")) is bool, f"Non-boolean success field for {identity}")
        if family == "pet_door":
            excluded.add(door)
            excluded_episodes += 1
            continue
        counts[door][0] += int(episode["success"])
        counts[door][1] += 1
        coverage[(door, scenario)].add(seed)
    require(bool(counts), "No eligible core doors found")
    for (door, scenario), seen in coverage.items():
        require(seen == set(seeds), f"Incomplete seed coverage for {door} / {scenario}: {sorted(seen)}")
    suite = row["suites"]["core"]
    require(dict(counts) == suite["doors"], "Recomputed per-door episodes/successes differ from the index")
    doors = [DoorOutcome(door, families[door], *counts[door]) for door in sorted(counts)]
    totals = Counter(d.outcome for d in doors)
    episodes, successes = sum(d.episodes for d in doors), sum(d.successes for d in doors)
    expected = {"n_doors": len(doors), "n_doors_suite": len(doors), "doors_solved": totals["all"],
                "doors_solved_any": totals["all"] + totals["some"], "n_episodes": episodes, "n_success": successes}
    for key, value in expected.items():
        require(suite.get(key) == value, f"Recomputed {key}={value} differs from indexed {suite.get(key)!r}")
    require(suite.get("complete") is True, "The indexed core run is incomplete")
    require(index.get("n_doors_total") == len(doors), "Index eligible-door total differs")
    require(sorted(excluded) == sorted(subset.get("excluded_door_ids", [])), "Excluded pet-door IDs differ")
    for key, value in {"original_n_doors": len(families), "original_n_episodes": len(raw["episodes"]),
                       "excluded_n_doors": len(excluded), "excluded_n_episodes": excluded_episodes,
                       "retained_n_doors": len(doors), "retained_n_episodes": episodes}.items():
        require(subset.get(key) == value, f"Historical subset {key} differs from source episodes")
    return doors, {
        "schema": "doorbench.scripted-baseline-overview.v1",
        "source_sha256": {"results/index.json": sha256(index_bytes).hexdigest(),
                          "results/scripted_hand.json": raw_hash},
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "run": {"policy": "scripted_hand", "suite": "core", "date": row["date"], "seeds": seeds,
                "commit": row["commit"], "simulator": row["simulator"], "tier": row["tier"]},
        "eligibility_policy": POLICY, "excluded_pet_doors": len(excluded),
        "counts": {"doors": len(doors), "all": totals["all"], "some": totals["some"], "none": totals["none"],
                   "episodes": episodes, "successful_episodes": successes},
    }


def render_svg(doors: list[DoorOutcome], metadata: dict) -> bytes:
    n = metadata["counts"]
    columns, cell, pitch_x, pitch_y = 50, 13, 17, 16
    rows = (len(doors) + columns - 1) // columns
    height = 300 + rows * pitch_y
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{height}" viewBox="0 0 1080 {height}" role="img" aria-labelledby="title desc">',
           f'<title id="title">Scripted-hand core baseline: {n["all"]} of {n["doors"]} eligible doors solved</title>',
           '<desc id="desc">One square per standard door, sorted by ID. Green means every assigned scenario succeeded on all three seeds; amber means some episodes succeeded; rust means none succeeded. Standalone pet doors are excluded. Historical source run; this is not a new evaluation of current geometry.</desc>',
           f'<metadata>{escape(json.dumps(metadata, sort_keys=True, separators=(",", ":")))}</metadata>',
           '<rect width="1080" height="100%" rx="18" fill="#f7f9f5"/>',
           '<style>text{font-family:Inter,Arial,Helvetica,sans-serif;fill:#253b2e}.muted{fill:#647369}.eyebrow{font-size:11px;font-weight:700;letter-spacing:1.5px}.range{font-family:ui-monospace,Menlo,monospace;font-size:10px;fill:#647369}.count{font-variant-numeric:tabular-nums}</style>',
           '<text x="40" y="37" class="eyebrow">SCRIPTED-HAND BASELINE · CORE SUITE</text>',
           f'<text x="40" y="104" font-size="56" font-weight="700" class="count">{n["all"]}<tspan fill="#748174" font-size="38"> / {n["doors"]}</tspan></text>',
           '<text x="40" y="135" font-size="16">Doors solved across every scenario and all 3 seeds</text>',
           f'<text x="828" y="71" font-size="26" font-weight="700" class="count">{n["successful_episodes"]:,}<tspan font-size="18" fill="#748174"> / {n["episodes"]:,}</tspan></text>',
           '<text x="828" y="95" font-size="12" class="muted">successful episodes</text>',
           f'<text x="828" y="129" font-size="12" class="muted">Seeds {", ".join(str(s) for s in metadata["run"]["seeds"])}</text>']
    for i, (status, label) in enumerate((("all", "All episodes succeeded"), ("some", "Some episodes succeeded"), ("none", "No episodes succeeded"))):
        y = 58 + i * 29
        svg.extend([f'<rect x="507" y="{y}" width="13" height="13" rx="2" fill="{COLORS[status]}"/>',
                    f'<text x="531" y="{y + 11}" font-size="13"><tspan font-weight="700" class="count">{n[status]}</tspan>  {label}</text>'])
    svg.extend(['<path d="M40 164H1040" stroke="#dce4d8"/>',
                '<text x="40" y="191" class="eyebrow muted">DOOR IDS</text>',
                '<text x="170" y="191" font-size="12" class="muted">One square = one eligible door · sorted by ID</text>',
                '<g id="door-outcomes">'])
    for row in range(rows):
        row_doors = doors[row * columns:(row + 1) * columns]
        y = 205 + row * pitch_y
        first, last = (d.door_id.split("_", 1)[0][2:] for d in (row_doors[0], row_doors[-1]))
        svg.append(f'<text x="40" y="{y + 10}" class="range">{first}–{last}</text>')
        for col, door in enumerate(row_doors):
            x = 170 + col * pitch_x
            tooltip = f'{door.door_id}: {door.successes}/{door.episodes} successful core episodes; {door.outcome}. All assigned scenarios across seeds {metadata["run"]["seeds"]}.'
            svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" fill="{COLORS[door.outcome]}" data-door-id="{escape(door.door_id, quote=True)}" data-outcome="{door.outcome}"><title>{escape(tooltip)}</title></rect>')
    foot_y = 223 + rows * pitch_y
    svg.extend(['</g>', f'<path d="M40 {foot_y}H1040" stroke="#dce4d8"/>',
                f'<text x="40" y="{foot_y + 26}" font-size="12"><tspan font-weight="700">Historical run · {escape(metadata["run"]["date"])}</tspan><tspan class="muted"> · Recomputed eligible-door subset; no benchmark rerun.</tspan></text>',
                f'<text x="40" y="{foot_y + 47}" font-size="12" class="muted">Predates current geometry repairs. {metadata["excluded_pet_doors"]} supplementary pet doors excluded. Click the image for per-door results.</text>',
                '</svg>'])
    return ("\n".join(svg) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "results/index.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results/scripted_hand.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/review/scripted-baseline/overview.svg")
    parser.add_argument("--check", action="store_true", help="Verify exact output bytes without writing")
    args = parser.parse_args(argv)
    try:
        doors, metadata = verified_outcomes(args.index, args.results)
        output = render_svg(doors, metadata)
        if args.check:
            require(args.out.is_file() and args.out.read_bytes() == output,
                    f"Overview is missing or stale: regenerate {args.out}")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(output)
        n = metadata["counts"]
        print(f'{"Verified" if args.check else "Wrote"} {args.out}: {n["doors"]} doors; '
              f'{n["all"]} all / {n["some"]} some / {n["none"]} none; '
              f'{n["successful_episodes"]}/{n["episodes"]} successful episodes')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Overview validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
