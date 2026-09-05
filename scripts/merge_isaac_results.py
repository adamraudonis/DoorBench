#!/usr/bin/env python
"""Publish the Isaac parity verdicts per door: qa.json["isaac_parity"] and a badge field in assets/manifest.json.

    PYTHONPATH=$PWD python scripts/merge_isaac_results.py                       # reads results/parity/summary.json
    PYTHONPATH=$PWD python scripts/merge_isaac_results.py --summary out/summary.json --assets /path/to/assets
    PYTHONPATH=$PWD python scripts/merge_isaac_results.py --recompute            # rebuild the verdicts from results/parity/*.json first
    PYTHONPATH=$PWD python scripts/merge_isaac_results.py --check                # exit 1 if qa.json / manifest are out of date (CI)

Per door, qa.json gets (nothing else in the file is touched - in particular not `checks` or `signed_off`):

    "isaac_parity": {"version", "date", "commit", "ok", "grade", "status", "kinds": {"full": {...}, "rl": {...}},
                     "classes": [...], "primary_class", "likely_root_cause", "engines"}

and the manifest entry gets `isaac_parity: "ok" | "fail" | "untested"` (+ a dataset-level `isaac_parity` block) so the
viewer can show the badge and filter on it.  Idempotent: re-running with the same summary rewrites nothing (the date
is the summary's date, not today's), and doors without a verdict are marked `untested` without losing anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from doorbench.parity import results as R  # noqa: E402

VERSION = "1"


def qa_block(verdict: dict | None, summary: dict) -> dict:
    """The qa.json["isaac_parity"] block for one door (an `untested` block when there is no verdict)."""
    base = {"version": VERSION, "date": summary.get("date") or (summary.get("generated") or "")[:10], "commit": summary.get("commit"), "engines": summary.get("engines") or {}}
    if not verdict:
        return base | {"ok": None, "grade": None, "status": "untested", "kinds": {k: {"status": "untested"} for k in R.KINDS}, "classes": [], "primary_class": "UNTESTED", "likely_root_cause": "-"}
    kinds = {}
    for k in R.KINDS:
        kv = (verdict.get("kinds") or {}).get(k) or {"status": "untested"}
        kinds[k] = {"status": kv.get("status", "untested"), "grade": kv.get("grade"), "ok": kv.get("ok"), "phases": kv.get("phases", {}), "classes": kv.get("classes", []),
                    "details": kv.get("details", [])[:6], "metrics": kv.get("metrics", {})}
        if kv.get("errors"):
            kinds[k]["errors"] = kv["errors"][:3]
    return base | {"ok": verdict.get("ok"), "grade": verdict.get("grade"), "status": verdict.get("status"), "kinds": kinds, "classes": verdict.get("classes", []),
                   "primary_class": verdict.get("primary_class"), "likely_root_cause": verdict.get("likely_root_cause", "-")}


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def merge(summary: dict, assets_dir: str, check: bool = False, verbose: bool = True) -> dict:
    """Write qa.json / manifest for every door in the manifest (or in the summary when there is no manifest). Returns counters."""
    doors: dict[str, dict] = summary.get("doors") or {}
    man_path = os.path.join(assets_dir, "manifest.json")
    manifest = _load_json(man_path) if os.path.exists(man_path) else None
    ids = [d["id"] for d in manifest["doors"] if d.get("id")] if manifest else sorted(doors)
    stats = {"qa_written": 0, "qa_unchanged": 0, "qa_missing": 0, "manifest_changed": 0, "ok": 0, "fail": 0, "untested": 0, "stale": [], "stale_inputs": []}
    for did in ids:
        verdict = doors.get(did)
        if (verdict or {}).get("status") == "stale":
            # the Isaac run and the MuJoCo reference were not the same door: no verdict may be published for it
            stats["stale_inputs"].append(did)
        block = qa_block(verdict, summary)
        status = R.manifest_status(verdict)
        stats[status] += 1
        qa_path = os.path.join(assets_dir, "doors", did, "qa.json")
        if os.path.exists(qa_path):
            qa = _load_json(qa_path)
            signed_before = qa.get("signed_off")
            if _same(qa.get("isaac_parity"), block):
                stats["qa_unchanged"] += 1
            else:
                stats["qa_written"] += 1
                stats["stale"].append(did)
                if not check:
                    qa["isaac_parity"] = block
                    assert qa.get("signed_off") == signed_before
                    with open(qa_path, "w") as f:
                        json.dump(qa, f, indent=1)
        else:
            stats["qa_missing"] += 1
    if manifest:
        changed = False
        for d in manifest["doors"]:
            verdict = doors.get(d.get("id"))
            status = R.manifest_status(verdict)
            grade = verdict.get("grade") if verdict else None
            if d.get("isaac_parity") != status or d.get("isaac_parity_grade") != grade:
                d["isaac_parity"] = status
                d["isaac_parity_grade"] = grade
                changed = True
        top = {"version": VERSION, "date": summary.get("date") or (summary.get("generated") or "")[:10], "commit": summary.get("commit"),
               "n_ok": stats["ok"], "n_fail": stats["fail"], "n_untested": stats["untested"]}
        if not _same(manifest.get("isaac_parity"), top):
            manifest["isaac_parity"] = top
            changed = True
        if changed:
            stats["manifest_changed"] = 1
            if not check:
                with open(man_path, "w") as f:
                    json.dump(manifest, f)
    if verbose:
        verb = "would write" if check else "wrote"
        print(f"[merge-isaac] {verb} isaac_parity into {stats['qa_written']} qa.json ({stats['qa_unchanged']} unchanged, {stats['qa_missing']} missing); "
              f"manifest {'changed' if stats['manifest_changed'] else 'unchanged'}; doors ok {stats['ok']} fail {stats['fail']} untested {stats['untested']}")
        if stats["stale_inputs"]:
            print(f"[merge-isaac] WARNING: {len(stats['stale_inputs'])} doors have a STALE verdict (the Isaac run and the MuJoCo reference were produced from "
                  f"different protocol inputs) and are published as `untested`, not as ok or fail: {', '.join(stats['stale_inputs'][:6])}"
                  f"{' ...' if len(stats['stale_inputs']) > 6 else ''}. Re-run the Isaac side against the current dataset to restore them.")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the Isaac parity verdicts into qa.json (isaac_parity) and the manifest badge field.")
    ap.add_argument("--summary", default=os.path.join(ROOT, "results", "parity", "summary.json"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "parity"), help="used with --recompute (and when summary.json is missing)")
    ap.add_argument("--assets", default=os.environ.get("DOORBENCH_ASSETS", os.path.join(ROOT, "assets")))
    ap.add_argument("--recompute", action="store_true", help="rebuild the verdicts from the result files instead of reading summary.json")
    ap.add_argument("--check", action="store_true", help="report what would change and exit 1 if anything is stale")
    a = ap.parse_args(argv)
    if a.recompute or not os.path.exists(a.summary):
        from doorbench.parity.report import build_report
        if not a.recompute:
            print(f"[merge-isaac] {a.summary} missing: recomputing from {a.results}")
        summary, _md = build_report(a.results, a.assets if os.path.isdir(a.assets) else None, None, plots=False, root=ROOT)
    else:
        summary = _load_json(a.summary)
    if not os.path.isdir(os.path.join(a.assets, "doors")):
        print(f"[merge-isaac] no dataset at {a.assets}")
        return 2
    stats = merge(summary, a.assets, check=a.check)
    if a.check and (stats["qa_written"] or stats["manifest_changed"]):
        print(f"[merge-isaac] stale: {', '.join(stats['stale'][:10])}{' ...' if len(stats['stale']) > 10 else ''}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
