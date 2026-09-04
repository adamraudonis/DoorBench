#!/usr/bin/env python
"""Visual common-sense review agent (task G8): photograph every door and ask a vision model what is obviously wrong.

For each door a labelled review sheet is rendered with MuJoCo's offscreen renderer (closed / open / mid-travel from the
robot side, the far side and above, plus a hardware close-up and a mechanism close-up, headed by the spec facts), sent to
the Claude API with a deficiency rubric, and the strict-JSON verdict is stored next to the sheet:

    docs/review/vision/<door>.jpg            the sheet
    docs/review/vision/<door>.json           {door_id, ok, summary, findings: [{category, severity, part, description, where}], reviewer, model, usage}
    docs/review/vision/<door>.prompt.json    (--dry-run) the request that would be sent, image replaced by its path
    docs/VISION_REVIEW.md                    the compiled report (counts by category x family, blockers first, gate comparison)

Usage:
  scripts/vision_review.py [--assets assets] [--out docs/review/vision] [--report docs/VISION_REVIEW.md]
                           [--doors id,id] [--families f,f] [--limit N] [--sample N] [--per-family K] [--seed S]
                           [--model claude-opus-5] [--effort high] [--max-cost-usd X] [--batch] [--force]
                           [--dry-run] [--from-verdicts] [--cell 400x300] [--supersample 2] [--price in,out]

The API key is read from ANTHROPIC_API_KEY (the SDK also accepts an `ant auth login` profile).  Without a key use
--dry-run (renders + prompts) or --from-verdicts (report only).  Doors that already have a verdict on disk are skipped
unless --force, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from doorbench.review import vision as V  # noqa: E402


def parse_cell(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--out", default="docs/review/vision", help="sheets + verdicts directory")
    ap.add_argument("--report", default="docs/VISION_REVIEW.md")
    ap.add_argument("--doors", default="", help="comma-separated door ids (always included)")
    ap.add_argument("--families", default="", help="comma-separated families to restrict to")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="N seeded random doors")
    ap.add_argument("--per-family", type=int, default=0, help="K seeded doors of every family")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--model", default=V.DEFAULT_MODEL)
    ap.add_argument("--effort", default=V.DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-cost-usd", type=float, default=25.0, help="abort before the first request if the estimate exceeds this")
    ap.add_argument("--price", default="", help="override price for --model as 'input,output' USD per 1M tokens")
    ap.add_argument("--batch", action="store_true", help="use the Message Batches API (50 %% price, async, up to 24 h)")
    ap.add_argument("--force", action="store_true", help="re-review doors that already have a verdict")
    ap.add_argument("--dry-run", action="store_true", help="render sheets + write prompts, no API call")
    ap.add_argument("--from-verdicts", action="store_true", help="only rebuild the report from the verdicts on disk")
    ap.add_argument("--no-render", action="store_true", help="reuse existing sheets (render only missing ones)")
    ap.add_argument("--cell", default="400x300", help="panel size in px; the sheet is 4 cells wide")
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--reviewer-notes", default="", help="markdown file with intro / triage / handoff sections for the report ({intro, triage, handoff} JSON)")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    man = json.load(open(os.path.join(a.assets, "manifest.json")))
    prices = dict(V.MODEL_PRICES)
    if a.price:
        pi, po = (float(x) for x in a.price.split(","))
        prices[a.model] = (pi, po)

    notes = {}
    if a.reviewer_notes and os.path.exists(a.reviewer_notes):
        notes = json.load(open(a.reviewer_notes))

    def build_report(cost_estimates=None):
        verdicts = V.load_verdicts(a.out)
        V.write_report(verdicts, a.report, a.assets, a.out, manifest=man, cost_estimates=cost_estimates,
                       intro=notes.get("intro", ""), handoff=notes.get("handoff", ""), triage_notes=notes.get("triage", ""))
        n_bad = sum(1 for v in verdicts if not v["ok"])
        print(f"report: {a.report} ({len(verdicts)} verdicts, {n_bad} with blocker/major findings)")
        return verdicts

    if a.from_verdicts:
        build_report()
        return 0

    rows = V.select_doors(man, ids=[d for d in a.doors.split(",") if d], families=[f for f in a.families.split(",") if f], limit=a.limit or None,
                          sample=a.sample or None, per_family=a.per_family or None, seed=a.seed)
    print(f"{len(rows)} doors selected")
    # ---- render sheets
    sheets = []
    t0 = time.time()
    for i, row in enumerate(rows):
        door_dir = os.path.join(a.assets, "doors", row["id"])
        sheet_path = os.path.join(a.out, f"{row['id']}.jpg")
        info_path = os.path.join(a.out, f"{row['id']}.sheet.json")
        if a.no_render and os.path.exists(sheet_path) and os.path.exists(info_path):
            info = json.load(open(info_path))
        else:
            try:
                info = V.render_sheet(door_dir, sheet_path, cell=parse_cell(a.cell), supersample=a.supersample)
            except Exception as e:
                print(f"  {row['id']}: render error {type(e).__name__}: {e}", flush=True)
                continue
            with open(info_path, "w") as f:
                json.dump(info, f)
        sheets.append((row, sheet_path, info))
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  rendered {i + 1}/{len(rows)} ({time.time() - t0:.0f} s)", flush=True)
    # ---- cost estimate (all selected doors, and only the ones still to review)
    todo = [(row, p, info) for row, p, info in sheets if a.force or not os.path.exists(os.path.join(a.out, f"{row['id']}.json"))]
    est_all = V.estimate_cost([info for _, _, info in sheets], a.model, batch=a.batch, prices=prices)
    est_todo = V.estimate_cost([info for _, _, info in todo], a.model, batch=a.batch, prices=prices) if todo else dict(est_all, n_doors=0, usd=0.0)
    per_door = est_all["usd_per_door"]
    print(f"cost estimate ({a.model}{', batch' if a.batch else ''}): {len(sheets)} doors -> ${est_all['usd']:.2f} (${per_door:.4f}/door; "
          f"{est_all['image_tokens']:,} image + {est_all['text_tokens']:,} text input tokens, {est_all['output_tokens']:,} output budget); "
          f"all 1000 doors: about ${per_door * 1000:.2f}; still to review: {len(todo)} doors -> ${est_todo['usd']:.2f}")
    if a.dry_run:
        for row, p, info in sheets:
            req = V.build_request(p, info["facts"], info, model=a.model, effort=a.effort, with_image=False)
            req["_image"] = os.path.abspath(p)
            req["_estimate_usd"] = per_door
            with open(os.path.join(a.out, f"{row['id']}.prompt.json"), "w") as f:
                json.dump(req, f, indent=1)
        print(f"dry run: {len(sheets)} sheets + prompts in {a.out}; no API call made")
        build_report(cost_estimates=[est_all])
        return 0
    if not todo:
        print("nothing to review (all selected doors have verdicts; use --force to redo)")
        build_report(cost_estimates=[est_all])
        return 0
    if est_todo["usd"] > a.max_cost_usd:
        print(f"ABORT: estimate ${est_todo['usd']:.2f} exceeds --max-cost-usd {a.max_cost_usd:.2f}", file=sys.stderr)
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        print("note: ANTHROPIC_API_KEY is not set; the SDK will try an `ant auth login` profile", file=sys.stderr)
    client = V.make_client()
    usages = []
    n_ok = n_bad = n_err = 0
    if a.batch:
        items = [{"door_id": row["id"], "sheet": p, "facts": info["facts"], "sheet_info": info} for row, p, info in todo]
        results = V.run_batch(client, items, model=a.model, effort=a.effort)
        for did, (verdict, err) in results.items():
            if verdict is None:
                n_err += 1
                print(f"  {did}: ERROR {err}")
                continue
            with open(os.path.join(a.out, f"{did}.json"), "w") as f:
                json.dump(verdict, f, indent=1)
            usages.append(verdict.get("usage", {}))
            n_ok += verdict["ok"]
            n_bad += not verdict["ok"]
    else:
        for k, (row, p, info) in enumerate(todo):
            try:
                verdict, usage = V.review_door(client, p, info["facts"], info, model=a.model, effort=a.effort)
            except Exception as e:
                n_err += 1
                print(f"  {row['id']}: ERROR {type(e).__name__}: {e}", flush=True)
                continue
            with open(os.path.join(a.out, f"{row['id']}.json"), "w") as f:
                json.dump(verdict, f, indent=1)
            usages.append(usage)
            n_ok += verdict["ok"]
            n_bad += not verdict["ok"]
            spent = V.cost_from_usage(usages, a.model, prices=prices)
            worst = max((f["severity"] for f in verdict["findings"]), key=lambda s: -V.SEVERITIES.index(s), default="ok")
            print(f"  [{k + 1}/{len(todo)}] {row['id']}: {'ok' if verdict['ok'] else worst} ({len(verdict['findings'])} findings), spent ${spent:.3f}", flush=True)
            if spent > a.max_cost_usd:
                print(f"STOP: measured spend ${spent:.2f} exceeds --max-cost-usd; rerun to resume", file=sys.stderr)
                break
    spent = V.cost_from_usage(usages, a.model, batch=a.batch, prices=prices)
    print(f"done: {n_ok} ok, {n_bad} with findings, {n_err} errors; measured cost ${spent:.2f}")
    build_report(cost_estimates=[est_all])
    return 0


if __name__ == "__main__":
    sys.exit(main())
