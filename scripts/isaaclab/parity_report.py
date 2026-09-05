#!/usr/bin/env python
"""Isaac parity report: join the MuJoCo and Isaac results per door and USD kind, apply the tolerances, classify every
door (parity / discrepancy class codes), aggregate by family / hardware / kinematics / USD kind and write

    docs/ISAAC_PARITY.md            headline (N / 1000 doors at parity per kind), tables by class and family, top offenders
                                    with their metrics and small curve plots (docs/media/parity/*.png), likely root causes
    results/parity/summary.json     the same, machine-readable, plus the per-door verdicts consumed by
                                    scripts/merge_isaac_results.py (qa.json isaac_parity + manifest badge)

    PYTHONPATH=$PWD python scripts/isaaclab/parity_report.py                     # results/parity/*.json -> docs + summary
    PYTHONPATH=$PWD python scripts/isaaclab/parity_report.py --results out/par --no-plots --top 40

Inputs (written by the runners in doorbench/parity/protocol.py): results/parity/mujoco.json (reference),
results/parity/isaac_full.json, results/parity/isaac_rl.json; optional sensitivity reruns isaac_<kind>_<variant>.json.
Runs without Isaac Sim, MuJoCo or a GPU (pure Python + Pillow).
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from doorbench.parity import results as R  # noqa: E402
from doorbench.parity.report import build_report, rel, write_outputs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Join MuJoCo and Isaac parity results, classify, render docs/ISAAC_PARITY.md + summary.json")
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "parity"), help="directory with mujoco.json / isaac_<kind>.json")
    ap.add_argument("--assets", default=os.environ.get("DOORBENCH_ASSETS", os.path.join(ROOT, "assets")), help="dataset root (manifest.json, doors/<id>/spec.json, qa.json); optional")
    ap.add_argument("--docs", default=os.path.join(ROOT, "docs", "ISAAC_PARITY.md"))
    ap.add_argument("--media", default=os.path.join(ROOT, "docs", "media", "parity"), help="PNG plots of the top offenders")
    ap.add_argument("--summary", default=None, help="summary.json path (default <results>/summary.json)")
    ap.add_argument("--top", type=int, default=20, help="number of offenders to detail / plot")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    summary_path = a.summary or os.path.join(a.results, "summary.json")
    assets = a.assets if a.assets and os.path.isdir(a.assets) else None
    files = R.collect_result_files(a.results)
    if not files["mujoco"] and not files["isaac"]:
        print(f"[parity-report] no result files in {a.results} (expected mujoco.json, isaac_full.json, isaac_rl.json); writing an empty report")
    summary, md = build_report(a.results, assets, None if a.no_plots else a.media, top_n=a.top, plots=not a.no_plots, root=ROOT)
    write_outputs(summary, md, summary_path, a.docs)
    c = summary["counts"]
    n = c["n_doors_total"]
    for k in R.KINDS:
        ck = c[k]
        print(f"[parity-report] {k:4s}: tested {ck['tested']}/{n}  A {ck['A']}  B {ck['B']}  C {ck['C']}  X {ck['X']}  untested {ck['untested']}")
    d = c["door"]
    print(f"[parity-report] doors: ok {d['ok']}  fail {d['fail']}  untested {d['untested']}")
    for code, e in list(summary["by_class"].items())[:12]:
        print(f"  {code:24s} full x{e['full']:<4d} rl x{e['rl']:<4d} e.g. {', '.join(e['examples'][:3])}")
    print(f"[parity-report] wrote {rel(a.docs)} and {rel(summary_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
