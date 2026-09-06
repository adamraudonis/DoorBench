#!/usr/bin/env python
"""Photograph every door and ask a vision model whether it looks obviously wrong.

The deterministic gates in `doorbench/qa.py` answer measurable questions.  Every one of them was
written after a person looked at a picture and said "that is obviously wrong" - a floating door stop,
a closer arm ending in mid-air, a barn rail too short for the door's own travel.  This script is the
systematic version of that step: it renders a labelled review sheet per door (three poses x three
viewpoints plus three close-ups, captioned with what the spec SAYS should be there) and sends it to a
vision model with a rubric, parsing a strict JSON verdict.

Outputs
    docs/review/vision/<door>.jpg      the review sheet
    docs/review/vision/<door>.json     the verdict (or the sheet record, with --dry-run)
    docs/review/vision/<door>.prompt.txt  the exact prompt (with --dry-run)
    docs/review/vision/index.json      per-door sheet records + the run's cost estimate
    docs/VISION_REVIEW.md              the report

Examples
    # render every sheet and write the prompts, without calling the API (no key needed)
    python scripts/vision_review.py --sample 120 --dry-run

    # live, one request per door, with a hard spend guard
    ANTHROPIC_API_KEY=sk-... python scripts/vision_review.py --sample 120 --max-cost-usd 20

    # the whole dataset through the Batches API (half price, up to 24 h)
    ANTHROPIC_API_KEY=sk-... python scripts/vision_review.py --batch --max-cost-usd 60

    # rebuild the report from verdicts already on disk
    python scripts/vision_review.py --from-verdicts
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doorbench.review import api, report, sheet          # noqa: E402
from doorbench.review.prompt import prompt_for           # noqa: E402
from doorbench.review.verdict import VerdictError        # noqa: E402

HOW_TO_RUN = """
```bash
# 1. dry run - renders every sheet and writes the exact prompt per door; no API key needed
python scripts/vision_review.py --sample 120 --dry-run

# 2. live, one request per door, with a hard spend guard (aborts before the first call if the
#    estimate exceeds the cap)
ANTHROPIC_API_KEY=sk-... python scripts/vision_review.py --sample 120 --max-cost-usd 20

# 3. the whole dataset through the Batches API - half price, results within 24 h
ANTHROPIC_API_KEY=sk-... python scripts/vision_review.py --batch --max-cost-usd 60

# 4. rebuild this report from the verdicts already on disk (no rendering, no API)
python scripts/vision_review.py --from-verdicts
```

Selection: `--doors a,b,c`, `--families swing_single,rollup`, `--limit N`, `--sample N` (seeded by
`--seed`, stratified so every family appears), `--force` to re-review doors that already have a
verdict.  Everything else resumes: a door with a verdict on disk is skipped.

Cost controls: `--model` (default `claude-opus-5`), `--effort`, `--no-thinking`, `--batch`,
`--max-cost-usd`, `--est-output-tokens`, `--price-in` / `--price-out` if the published prices move.
The estimate is computed from the actual rendered sheets - their real pixel dimensions and their real
prompt text - not from a nominal size, and it is printed before anything is sent.
"""


# -------------------------------------------------------------------------------------------------
# selection
# -------------------------------------------------------------------------------------------------
def select(manifest: dict, doors: str, families: str, limit: int, sample: int, seed: int,
           force_include: str = "") -> list:
    rows = [d for d in manifest["doors"] if not d.get("error")]
    forced = [x for x in force_include.split(",") if x]
    if doors:
        want = set(doors.split(","))
        rows = [d for d in rows if d["id"] in want]
    if families:
        want = set(families.split(","))
        rows = [d for d in rows if d["family"] in want]
    if sample:
        by_fam = {}
        for d in rows:
            by_fam.setdefault(d["family"], []).append(d)
        rng = random.Random(seed)
        per = max(1, sample // max(1, len(by_fam)))
        picked = []
        for fam in sorted(by_fam):
            pool = sorted(by_fam[fam], key=lambda d: d["index"])
            picked += rng.sample(pool, min(per, len(pool)))
        # top up to the requested size from whatever is left, deterministically
        rest = sorted((d for d in rows if d not in picked), key=lambda d: d["index"])
        while len(picked) < sample and rest:
            picked.append(rest.pop(rng.randrange(len(rest))))
        rows = sorted(picked, key=lambda d: d["index"])
    if forced:
        have = {d["id"] for d in rows}
        rows += [d for d in manifest["doors"] if d["id"] in forced and d["id"] not in have]
        rows = sorted(rows, key=lambda d: d["index"])
    if limit:
        rows = rows[:limit]
    return rows


# -------------------------------------------------------------------------------------------------
# rendering (parallel)
# -------------------------------------------------------------------------------------------------
def _render_job(job):
    assets, out, door_id, width, quality = job
    try:
        rec = sheet.render_sheet(os.path.join(assets, "doors", door_id),
                                 os.path.join(out, f"{door_id}.jpg"), width=width, quality=quality)
        return rec
    except Exception as e:                                   # noqa: BLE001
        return {"door_id": door_id, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-900:]}


def render_all(assets: str, out: str, rows: list, width: int, quality: int, workers: int) -> list:
    jobs = [(os.path.abspath(assets), os.path.abspath(out), d["id"], width, quality) for d in rows]
    recs = []
    if workers > 1:
        with Pool(workers, maxtasksperchild=40) as p:
            for i, rec in enumerate(p.imap_unordered(_render_job, jobs), 1):
                recs.append(rec)
                if i % 25 == 0 or i == len(jobs):
                    print(f"  rendered {i}/{len(jobs)}", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            recs.append(_render_job(j))
            if i % 25 == 0 or i == len(jobs):
                print(f"  rendered {i}/{len(jobs)}", flush=True)
    bad = [r for r in recs if r.get("error")]
    for r in bad:
        print(f"  !! {r['door_id']}: {r['error']}", flush=True)
    return sorted((r for r in recs if not r.get("error")), key=lambda r: r["door_id"])


# -------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--out", default="docs/review/vision", help="where sheets and verdicts go")
    ap.add_argument("--report", default="docs/VISION_REVIEW.md")
    ap.add_argument("--triage", default="docs/review/vision/triage.md",
                    help="hand-written triage + handoff markdown spliced into the report")
    # selection
    ap.add_argument("--doors", default="")
    ap.add_argument("--families", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="seeded, family-stratified sample of N doors")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--force-include", default="", help="door ids always added to the sample")
    ap.add_argument("--force", action="store_true", help="re-review doors that already have a verdict")
    # rendering
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--quality", type=int, default=78)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-render", action="store_true", help="reuse the sheets already on disk")
    # model
    ap.add_argument("--model", default=api.DEFAULT_MODEL)
    ap.add_argument("--effort", default=api.DEFAULT_EFFORT,
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-tokens", type=int, default=api.DEFAULT_MAX_TOKENS)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--batch", action="store_true", help="use the Batches API (50 % cheaper, up to 24 h)")
    ap.add_argument("--poll-s", type=float, default=30.0)
    # cost
    ap.add_argument("--max-cost-usd", type=float, default=0.0,
                    help="abort before the first API call if the estimate exceeds this")
    ap.add_argument("--est-output-tokens", type=int, default=api.DEFAULT_EST_OUTPUT_TOKENS)
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    # modes
    ap.add_argument("--dry-run", action="store_true", help="render sheets and write prompts; no API call")
    ap.add_argument("--from-verdicts", action="store_true", help="rebuild the report from disk only")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    triage_md = Path(a.triage).read_text() if a.triage and os.path.isfile(a.triage) else ""

    if a.from_verdicts:
        verdicts = report.load_verdicts(out)
        run = {}
        idx = os.path.join(out, "index.json")
        if os.path.isfile(idx):
            run = json.load(open(idx)).get("run", {})
        md = report.render(verdicts, a.assets, run=run, triage_md=triage_md, how_to_run=HOW_TO_RUN)
        Path(a.report).write_text(md)
        print(f"{len(verdicts)} verdicts -> {a.report}")
        return 0

    manifest = json.load(open(os.path.join(a.assets, "manifest.json")))
    rows = select(manifest, a.doors, a.families, a.limit, a.sample, a.seed, a.force_include)
    if not rows:
        print("no doors selected", file=sys.stderr)
        return 2
    done = {p[:-5] for p in os.listdir(out)
            if p.endswith(".json") and not p.endswith(".sheet.json") and p != "index.json"}
    todo = rows if a.force else [d for d in rows if d["id"] not in done]
    print(f"{len(rows)} doors selected across {len({d['family'] for d in rows})} families; "
          f"{len(rows) - len(todo)} already have a verdict; {len(todo)} to do", flush=True)

    t0 = time.time()
    if a.no_render:
        recs = []
        for d in todo:
            p = os.path.join(out, f"{d['id']}.sheet.json")
            if os.path.isfile(p):
                recs.append(json.load(open(p)))
        print(f"reusing {len(recs)} sheets from disk", flush=True)
    else:
        print(f"rendering {len(todo)} sheets with {a.workers} workers ...", flush=True)
        recs = render_all(a.assets, out, todo, a.width, a.quality, a.workers)
        for r in recs:
            Path(os.path.join(out, f"{r['door_id']}.sheet.json")).write_text(json.dumps(r, indent=1) + "\n")
        print(f"  {len(recs)} sheets in {time.time() - t0:.0f}s", flush=True)

    # --- cost estimate, from the sheets that were actually rendered -------------------------------
    est = api.estimate_cost(recs, a.model, a.batch, a.est_output_tokens, a.price_in, a.price_out)
    per_door = est["est_cost_usd_per_door"]
    est_1000 = {
        "single request per door": dict(api.estimate_cost(recs, a.model, False, a.est_output_tokens,
                                                          a.price_in, a.price_out),
                                        **{"scaled_to": 1000}),
        "Batches API (50 %)": dict(api.estimate_cost(recs, a.model, True, a.est_output_tokens,
                                                     a.price_in, a.price_out),
                                   **{"scaled_to": 1000}),
    }
    for k, v in est_1000.items():                  # scale the per-door figures up to the whole dataset
        n = max(1, v["n_doors"])
        for f in ("image_tokens", "text_tokens", "system_tokens_billed", "est_output_tokens"):
            v[f] = int(v[f] / n * 1000)
        v["est_cost_usd"] = round(v["est_cost_usd"] / n * 1000, 2)
        v["n_doors"] = 1000
    print(json.dumps(est, indent=1), flush=True)
    print(f"extrapolated to all 1000 doors: "
          f"${est_1000['single request per door']['est_cost_usd']:.2f} single / "
          f"${est_1000['Batches API (50 %)']['est_cost_usd']:.2f} batched", flush=True)

    run_meta = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selection": (f"seeded sample of {a.sample} (seed {a.seed})" if a.sample else
                      (f"--doors {a.doors}" if a.doors else
                       (f"--families {a.families}" if a.families else "all doors"))),
        "model": a.model, "effort": a.effort, "batch": bool(a.batch), "dry_run": bool(a.dry_run),
        "cost_estimate_this_run": est, "cost_estimate_1000": est_1000,
        "manifest_generated": manifest.get("generated"),
    }
    Path(os.path.join(out, "index.json")).write_text(
        json.dumps({"run": run_meta, "sheets": recs}, indent=1) + "\n")

    if a.dry_run:
        for r in recs:
            p = prompt_for(r)
            Path(os.path.join(out, f"{r['door_id']}.prompt.txt")).write_text(
                "=== SYSTEM ===\n" + p["system"] + "\n\n=== USER ===\n" + p["user_text"] + "\n")
        print(f"dry run: {len(recs)} sheets + prompts written to {out}; no API call made", flush=True)
        verdicts = report.load_verdicts(out)
        Path(a.report).write_text(report.render(verdicts, a.assets, run=run_meta,
                                                triage_md=triage_md, how_to_run=HOW_TO_RUN))
        return 0

    if a.max_cost_usd and est["est_cost_usd"] > a.max_cost_usd:
        print(f"ABORT: estimated ${est['est_cost_usd']:.2f} exceeds --max-cost-usd "
              f"${a.max_cost_usd:.2f} ({len(recs)} doors at ${per_door:.4f} each). "
              f"Raise the cap, use --batch, use --sample, or lower --effort.", file=sys.stderr)
        return 3

    client = api.make_client()
    verdicts, errors, usages = [], [], []
    if a.batch:
        jobs = [(r, os.path.join(out, f"{r['door_id']}.jpg")) for r in recs]
        bid = api.submit_batch(client, jobs, a.model, a.max_tokens, a.effort, not a.no_thinking)
        print(f"batch {bid} submitted with {len(jobs)} requests; polling every {a.poll_s:.0f}s",
              flush=True)
        verdicts, errors = api.collect_batch(client, bid, {r["door_id"]: r for r in recs},
                                             a.model, poll_s=a.poll_s)
        usages = [v.get("usage", {}) for v in verdicts]
    else:
        for i, r in enumerate(recs, 1):
            try:
                v = api.review_door(client, r, os.path.join(out, f"{r['door_id']}.jpg"), a.model,
                                    a.max_tokens, a.effort, not a.no_thinking)
                verdicts.append(v)
                usages.append(v.get("usage", {}))
            except Exception as e:                           # noqa: BLE001
                errors.append({"door_id": r["door_id"], "error": f"{type(e).__name__}: {e}"})
                print(f"  !! {r['door_id']}: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"  [{i}/{len(recs)}] {r['door_id']}: "
                  f"{'clean' if v['ok'] else str(len(v['findings'])) + ' findings'}", flush=True)

    for v in verdicts:
        Path(os.path.join(out, f"{v['door_id']}.json")).write_text(json.dumps(v, indent=1) + "\n")
    spent = api.actual_cost(usages, a.model, a.batch, a.price_in, a.price_out)
    run_meta["actual_cost"] = spent
    run_meta["errors"] = errors
    Path(os.path.join(out, "index.json")).write_text(
        json.dumps({"run": run_meta, "sheets": recs}, indent=1) + "\n")
    print(f"{len(verdicts)} verdicts, {len(errors)} errors, ${spent['cost_usd']:.2f} spent", flush=True)

    all_verdicts = report.load_verdicts(out)
    Path(a.report).write_text(report.render(all_verdicts, a.assets, run=run_meta,
                                            triage_md=triage_md, how_to_run=HOW_TO_RUN))
    print(f"report -> {a.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
