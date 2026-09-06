"""docs/VISION_REVIEW.md from the verdicts on disk.

Everything mechanical - the counts, the category-by-family table, the blocker/major gallery, the
comparison against qa.json - is generated, so re-running after a fix updates the numbers.  The two
sections that require judgement (the triage of each finding, and the handoff list) are written by
hand and spliced in from a markdown file, because "is this a real geometry bug, a rendering artefact
or a false positive" is not a question the counts can answer.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .prompt import CATEGORIES, SEVERITIES
from .verdict import counts

SEVERITY_ORDER = {"blocker": 0, "major": 1, "minor": 2}


def load_verdicts(vdir: str) -> List[dict]:
    out = []
    for fn in sorted(os.listdir(vdir)) if os.path.isdir(vdir) else []:
        if not fn.endswith(".json") or fn.endswith(".sheet.json") or fn == "index.json":
            continue
        with open(os.path.join(vdir, fn)) as f:
            try:
                v = json.load(f)
            except json.JSONDecodeError:
                continue
        if isinstance(v, dict) and "findings" in v:
            out.append(v)
    return out


def qa_checks(assets: str, door_id: str) -> dict:
    p = os.path.join(assets, "doors", door_id, "qa.json")
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        return json.load(f).get("checks", {})


def _table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_(none)_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def category_by_family(verdicts: List[dict]) -> str:
    grid: Dict[str, Counter] = defaultdict(Counter)
    for v in verdicts:
        fam = v.get("family") or "?"
        for f in v["findings"]:
            grid[fam][f["category"]] += 1
    cats = [c for c in CATEGORIES if any(g[c] for g in grid.values())]
    if not cats:
        return "No findings in any family.\n"
    headers = ["family", "doors"] + cats + ["total"]
    seen = Counter(v.get("family") or "?" for v in verdicts)
    rows = []
    for fam in sorted(grid, key=lambda f: (-sum(grid[f].values()), f)):
        row = [fam, str(seen[fam])] + [str(grid[fam][c] or "") for c in cats]
        row.append(str(sum(grid[fam].values())))
        rows.append(row)
    tot = Counter()
    for g in grid.values():
        tot.update(g)
    rows.append(["**all**", str(len(verdicts))] + [f"**{tot[c]}**" for c in cats]
                + [f"**{sum(tot.values())}**"])
    clean_fams = sorted(f for f in seen if f not in grid)
    body = _table(headers, rows)
    if clean_fams:
        body += f"\nFamilies with no findings at all: {', '.join(clean_fams)}.\n"
    return body


def gallery(verdicts: List[dict], severities=("blocker", "major"), rel: str = "review/vision") -> str:
    """One entry per door, with its sheet, listing that door's findings at these severities."""
    rows = []
    for v in verdicts:
        sel = [f for f in v["findings"] if f["severity"] in severities]
        if sel:
            rows.append((min(SEVERITY_ORDER[f["severity"]] for f in sel), v["door_id"], v, sel))
    rows.sort()
    if not rows:
        return f"_No {' or '.join(severities)} findings._\n"
    out = []
    for _, _, v, sel in rows:
        sheet = v.get("sheet") or f"{v['door_id']}.jpg"
        out.append(f"#### `{v['door_id']}` ({v.get('family', '?')})\n")
        if v.get("summary"):
            out.append(f"_{v['summary']}_\n")
        for f in sorted(sel, key=lambda f: SEVERITY_ORDER[f["severity"]]):
            conf = f" _(confidence {f['confidence']:.2f})_" if f.get("confidence") is not None else ""
            out.append(f"* **{f['severity'].upper()} / {f['category']}** - {f['part']}: "
                       f"{f['description']} (seen in {f['where']}){conf}")
        out.append(f"\n![{v['door_id']}]({rel}/{sheet})\n")
    return "\n".join(out)


def minor_table(verdicts: List[dict]) -> str:
    rows = []
    for v in sorted(verdicts, key=lambda v: v["door_id"]):
        for f in v["findings"]:
            if f["severity"] == "minor":
                rows.append([f"`{v['door_id']}`", v.get("family", "?"), f["category"],
                             f["part"], f["description"].replace("|", "/"), f["where"]])
    return _table(["door", "family", "category", "part", "description", "where"], rows)


def gates_comparison(verdicts: List[dict], assets: str) -> str:
    """What the deterministic gates said about the same doors the reviewer flagged."""
    flagged = [v for v in verdicts if not v.get("ok")]
    rows = []
    all_pass = 0
    for v in sorted(flagged, key=lambda v: v["door_id"]):
        checks = qa_checks(assets, v["door_id"])
        failed = [k for k, ok in checks.items() if not ok]
        if not failed:
            all_pass += 1
        sev = min((SEVERITY_ORDER[f["severity"]] for f in v["findings"]), default=2)
        rows.append([f"`{v['door_id']}`", v.get("family", "?"),
                     ["blocker", "major", "minor"][sev],
                     str(len(v["findings"])),
                     ", ".join(sorted({f["category"] for f in v["findings"]})),
                     ", ".join(failed) if failed else "**all gates pass**"])
    head = (f"{len(flagged)} of {len(verdicts)} reviewed doors carry at least one finding; "
            f"**{all_pass} of those {len(flagged)} pass every deterministic gate in `qa.json`** "
            f"(clearance, running_clearance, attachment, no_jam, sliding_track_support, "
            f"linkage_feasibility, mass, settle, hold, free_opens, actuate_opens, latch_returns, "
            f"relatch, closer_returns, locked_holds, operator_returns, operator_holds, "
            f"keypad_code_works, all_latches_release, rod_points_hold, USD validation, Isaac parity).\n\n")
    return head + _table(["door", "family", "worst", "n", "categories", "qa.json checks failing"], rows)


def render(verdicts: List[dict], assets: str, run: Optional[dict] = None,
           triage_md: str = "", how_to_run: str = "", rel: str = "review/vision") -> str:
    c = counts(verdicts)
    run = run or {}
    est = run.get("cost_estimate_1000") or {}
    fam_seen = sorted({v.get("family") or "?" for v in verdicts})
    lines = [
        "# Vision review",
        "",
        "Deterministic gates measure; they cannot see. Every gate in `doorbench/qa.py` exists because a",
        "person looked at a picture and said \"that is obviously wrong\" - a door stop floating in mid-air,",
        "a closer arm ending in space, a barn rail too short for its own door. This is the systematic",
        "version of that step: photograph every door from every angle that matters, caption the picture",
        "with what the specification says should be there, and ask a vision model the question a person",
        "would ask.",
        "",
        "---",
        "",
        "## How to run it",
        "",
        how_to_run.strip() or "_(see `scripts/vision_review.py --help`)_",
        "",
        "---",
        "",
        "## Sample and method",
        "",
        f"* **{c['n_doors']} doors reviewed**, covering {len(fam_seen)} families"
        + (f" ({run['selection']})" if run.get("selection") else "") + ".",
        f"* **{c['n_clean']} clean**, {c['n_doors'] - c['n_clean']} with at least one finding, "
        f"{c['n_findings']} findings in total.",
        "* Sheet: 12 panels - the door closed, at 50 % of its travel and fully open, from three",
        "  viewpoints each (near/robot side, far side, hinge- or track-side), plus a hardware close-up",
        "  on each face and a mechanism close-up at full open. One camera per column, so the three rows",
        "  are the same shot at three points in the travel.",
        "* Poses are kinematic: joint equalities and tendon couplings are resolved exactly as the",
        "  clearance gate resolves them, and closed loops are solved numerically, with the residual",
        "  printed on the sheet.",
        f"* Reviewer(s): {', '.join(sorted({v.get('reviewer', '?') for v in verdicts}))}.",
        "",
        "### Findings by severity",
        "",
        _table(["severity", "count"], [[s, str(c["by_severity"].get(s, 0))] for s in SEVERITIES]),
        "### Findings by category",
        "",
        _table(["category", "count", "what it means"],
               [[k, str(c["by_category"].get(k, 0)), CATEGORIES[k]]
                for k in CATEGORIES if c["by_category"].get(k)]),
        "### Findings by category and family",
        "",
        category_by_family(verdicts),
        "---",
        "",
        "## What the deterministic gates would not have caught",
        "",
        gates_comparison(verdicts, assets),
        "---",
        "",
        "## Blockers and major findings",
        "",
        gallery(verdicts, ("blocker", "major"), rel=rel),
        "---",
        "",
    ]
    if triage_md.strip():
        lines += [triage_md.strip(), ""]
    if est:
        lines += [
            "---",
            "",
            "## Estimated cost for all 1000 doors",
            "",
            _table(["path", "model", "input tokens", "output tokens (est)", "estimated USD"],
                   [[k, v.get("model", "?"), f"{v['image_tokens'] + v['text_tokens']:,}",
                     f"{v['est_output_tokens']:,}", f"${v['est_cost_usd']:.2f}"]
                    for k, v in est.items()]),
        ]
    lines += [
        "---",
        "",
        "## Minor findings",
        "",
        minor_table(verdicts),
    ]
    return "\n".join(lines).rstrip() + "\n"
