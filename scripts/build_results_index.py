#!/usr/bin/env python
"""Build results/index.json (the leaderboard index the site reads) and the leaderboard table in results/README.md
from every result file under results/.

    python scripts/build_results_index.py            # validates, writes results/index.json, updates results/README.md
    python scripts/build_results_index.py --check    # exit 1 if index.json / README.md are out of date (CI)

index.json holds, per result file, the run metadata, the aggregate numbers, the per-family / per-task /
per-difficulty / per-lock-state breakdowns and a compact per-door outcome map for the `default` scenario
(`doors: {door_id: [n_success, n_episodes]}`), so the catalogue can show a badge per door without loading the
multi-megabyte result files.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
RESERVED = {"schema.json", "index.json"}
START, END = "<!-- leaderboard:start -->", "<!-- leaderboard:end -->"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_result import validate_file  # noqa: E402

FAMILY_ORDER = None


def _slim_group(g: dict) -> dict:
    return {k: g.get(k) for k in ("n_doors", "n_episodes", "n_success", "success_rate", "doors_solved", "doors_solved_any", "damage_rate", "mean_time_to_pass_s")}


def summarize(path: str) -> dict:
    with open(path) as f:
        doc = json.load(f)
    run, pol, agg, bench = doc["run"], doc["policy"], doc["aggregate"], doc["benchmark"]
    default_eps = [e for e in doc["episodes"] if e.get("scenario") == "default"] or doc["episodes"]
    doors = {}
    for e in default_eps:
        s = doors.setdefault(e["door_id"], [0, 0])
        s[0] += int(bool(e.get("success")))
        s[1] += 1
    scen_names = [s["name"] for s in run["scenarios"]]
    head = agg["by_scenario"].get("default", agg) if "default" in scen_names else agg
    return {
        "file": os.path.basename(path), "policy": pol["name"], "description": pol.get("description", ""), "embodiment": pol.get("embodiment", "hand_base"),
        "policy_class": pol.get("class"), "extra": pol.get("extra", {}),
        "simulator": run["simulator"], "simulator_version": run.get("simulator_version"), "tier": run["tier"], "date": run["date"][:10], "label": run.get("label", ""),
        "commit": bench.get("commit"), "dataset_version": bench.get("dataset_version"), "n_doors_total": bench.get("n_doors_total"),
        "n_doors": run["n_doors"], "seeds": run["seeds"], "scenarios": scen_names, "randomize": run.get("randomize"), "time_budget_s": run.get("time_budget_s"),
        "wall_time_s": run.get("wall_time_s"), "mean_wall_s": agg.get("mean_wall_s"), "host": (run.get("host") or {}).get("platform"),
        "leaderboard": "default" in scen_names and run["n_doors"] >= (bench.get("n_doors_total") or run["n_doors"]),
        "doors_solved": head["doors_solved"], "doors_solved_any": head["doors_solved_any"], "n_episodes": head["n_episodes"], "n_success": head["n_success"],
        "success_rate": head["success_rate"], "damage_rate": head["damage_rate"], "mean_time_to_pass_s": head.get("mean_time_to_pass_s"), "median_time_to_pass_s": head.get("median_time_to_pass_s"),
        "outcomes": head.get("outcomes", {}),
        "by_family": {k: _slim_group(v) for k, v in agg["by_family"].items()},
        "by_task": {k: _slim_group(v) for k, v in agg["by_task"].items()},
        "by_difficulty": {k: _slim_group(v) for k, v in agg["by_difficulty"].items()},
        "by_lock_state": {k: _slim_group(v) for k, v in agg.get("by_lock_state", {}).items()},
        "by_scenario": {k: _slim_group(v) for k, v in agg["by_scenario"].items()},
        "doors": doors,
    }


def leaderboard_md(index: dict) -> str:
    rows = sorted(index["results"], key=lambda r: (-int(bool(r["leaderboard"])), -r["doors_solved"], -r["success_rate"]))
    total = index.get("n_doors_total") or 1000
    lines = ["| policy | embodiment | simulator | tier | doors | seeds | solved (all seeds) | episode success | damage | median time-to-traverse | date | commit |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tt = f"{r['median_time_to_pass_s']:.1f} s" if r.get("median_time_to_pass_s") is not None else "-"
        lines.append(f"| [{r['policy']}]({r['file']}) | {r['embodiment']} | {r['simulator']} {r.get('simulator_version') or ''} | {r['tier']} | {r['n_doors']} | {len(r['seeds'])} | "
                     f"**{r['doors_solved']} / {total}** | {100 * r['success_rate']:.1f} % | {100 * r['damage_rate']:.1f} % | {tt} | {r['date']} | {(r.get('commit') or '')[:8]} |")
    fams = sorted({f for r in rows for f in r["by_family"]})
    lines += ["", "Per family, doors solved on every seed (default scenario, `n` doors per family):", "",
              "| family | n | " + " | ".join(r["policy"] for r in rows) + " |", "|---|---|" + "---|" * len(rows)]
    for f in fams:
        n = max((r["by_family"].get(f, {}).get("n_doors") or 0) for r in rows)
        lines.append(f"| {f} | {n} | " + " | ".join(str(r["by_family"].get(f, {}).get("doors_solved", 0)) for r in rows) + " |")
    tasks = sorted({t for r in rows for t in r["by_task"]})
    lines += ["", "Per task (doors solved on every seed / doors with that task):", "",
              "| task | n | " + " | ".join(r["policy"] for r in rows) + " |", "|---|---|" + "---|" * len(rows)]
    for t in tasks:
        n = max((r["by_task"].get(t, {}).get("n_doors") or 0) for r in rows)
        lines.append(f"| {t} | {n} | " + " | ".join(str(r["by_task"].get(t, {}).get("doors_solved", 0)) for r in rows) + " |")
    locks = ["unlocked", "locked_releasable", "locked_no_release"]
    lines += ["", "Per lock state:", "", "| lock state | n | " + " | ".join(r["policy"] for r in rows) + " |", "|---|---|" + "---|" * len(rows)]
    for k in locks:
        n = max((r["by_lock_state"].get(k, {}).get("n_doors") or 0) for r in rows)
        if n:
            lines.append(f"| {k} | {n} | " + " | ".join(str(r["by_lock_state"].get(k, {}).get("doors_solved", 0)) for r in rows) + " |")
    lines.append("")
    lines.append(f"_Generated by `scripts/build_results_index.py` on {index['generated'][:10]} from {len(rows)} result file(s)._")
    return "\n".join(lines)


WHAT = {
    "scripted_hand": "the per-family oracle heuristic of `scripts/demo_mujoco.py` (reads joint names, lock parts and keypad codes from the spec; DoorEnv hand + synthetic base)",
    "g1_locomotion": "Unitree G1 (MuJoCo Menagerie) + pretrained unitree_rl_gym locomotion policy, walks toward the goal, arms parked",
    "random": "uniform random torques within the hand limits on every reachable joint + a random-walk base",
}
ROOT_START, ROOT_END = "<!-- baseline-results:start -->", "<!-- baseline-results:end -->"


def root_readme_block(index: dict, manifest: dict | None) -> str:
    """The headline + per-family tables of the root README 'Baseline results' section."""
    rows = sorted([r for r in index["results"] if r["leaderboard"]], key=lambda r: -r["doors_solved"])
    total = index.get("n_doors_total") or 1000
    out = ["| policy | what it is | doors solved (all seeds) | episode success | damage | median time-to-traverse | wall time |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        tt = f"{r['median_time_to_pass_s']:.1f} s" if r.get("median_time_to_pass_s") is not None else "-"
        wall = f"{r['wall_time_s'] / 60:.1f} min" if r.get("wall_time_s") else "-"
        out.append(f"| `{r['policy']}` | {WHAT.get(r['policy'], r['description'])} | **{r['doors_solved']} / {total}** | {100 * r['success_rate']:.1f} % | {100 * r['damage_rate']:.1f} % | {tt} | {wall} |")
    fam_n = {}
    if manifest:
        for d in manifest["doors"]:
            fam_n[d["family"]] = fam_n.get(d["family"], 0) + 1
    else:
        for r in rows:
            for f, g in r["by_family"].items():
                fam_n[f] = max(fam_n.get(f, 0), g.get("n_doors") or 0)
    fams = sorted(fam_n, key=lambda f: (-fam_n[f], f))
    out += ["", "Doors solved per family (of the family's door count):", "", "| family | doors | " + " | ".join(f"`{r['policy']}`" for r in rows) + " |", "|---|---|" + "---|" * len(rows)]
    for f in fams:
        out.append(f"| {f} | {fam_n[f]} | " + " | ".join(str(r["by_family"].get(f, {}).get("doors_solved", 0)) for r in rows) + " |")
    out.append("")
    return "\n".join(out)


def update_root_readme(path: str, block: str) -> str | None:
    """Rewrite the block between the markers in the root README (None when the markers are absent)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cur = f.read()
    if ROOT_START not in cur or ROOT_END not in cur:
        return None
    pre, rest = cur.split(ROOT_START, 1)
    _, post = rest.split(ROOT_END, 1)
    return pre + ROOT_START + "\n" + block + "\n" + ROOT_END + post


def update_readme(path: str, table: str) -> str:
    header = ("# DoorBench benchmark results\n\n"
              "Every file here is one full run of a policy over the DoorBench doors, written by `doorbench benchmark run` "
              "and validated by `scripts/validate_result.py` against `schema.json`.  `index.json` is generated from them "
              "(`scripts/build_results_index.py`) and feeds the [Results page](https://adamraudonis.github.io/DoorBench/#/results) "
              "of the site.  \"Solved\" means success on **every** seed evaluated for that door in the `default` scenario "
              "(each door's own task).  To add your own run, read [docs/SUBMITTING.md](../docs/SUBMITTING.md).\n\n")
    if os.path.exists(path):
        with open(path) as f:
            cur = f.read()
    else:
        cur = header + START + "\n" + END + "\n"
    if START in cur and END in cur:
        pre, rest = cur.split(START, 1)
        _, post = rest.split(END, 1)
        new = pre + START + "\n" + table + "\n" + END + post
    else:
        new = cur.rstrip() + "\n\n" + START + "\n" + table + "\n" + END + "\n"
    return new


def build(check: bool = False) -> int:
    files = [p for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))) if os.path.basename(p) not in RESERVED]
    with open(os.path.join(RESULTS, "schema.json")) as f:
        schema = json.load(f)
    manifest = None
    mp = os.path.join(ROOT, "assets", "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f:
            manifest = json.load(f)
    bad = 0
    results = []
    for p in files:
        errs = validate_file(p, schema, manifest, submission=False)
        if errs:
            bad += 1
            print(f"FAIL {p}: {errs[:5]}")
            continue
        results.append(summarize(p))
    if bad:
        return 1
    n_total = max([r.get("n_doors_total") or 0 for r in results] + [len(manifest["doors"]) if manifest else 0]) or 1000
    prev_generated = None
    ip = os.path.join(RESULTS, "index.json")
    if os.path.exists(ip):
        try:
            with open(ip) as f:
                prev_generated = json.load(f).get("generated")
        except Exception:
            pass
    index = {"schema_version": "1.0", "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "n_doors_total": n_total, "results": results}
    table = leaderboard_md(index)
    rp = os.path.join(RESULTS, "README.md")
    new_readme = update_readme(rp, table)
    root_rp = os.path.join(ROOT, "README.md")
    new_root = update_root_readme(root_rp, root_readme_block(index, manifest))
    if check:
        stale = []
        if os.path.exists(ip):
            with open(ip) as f:
                old = json.load(f)
            old.pop("generated", None)
            cur = dict(index)
            cur.pop("generated", None)
            if old != cur:
                stale.append("results/index.json")
        else:
            stale.append("results/index.json")
        if os.path.exists(rp):
            with open(rp) as f:
                old_readme = f.read()
            strip = lambda s: "\n".join(ln for ln in s.splitlines() if not ln.startswith("_Generated by"))
            if strip(old_readme) != strip(new_readme):
                stale.append("results/README.md")
        else:
            stale.append("results/README.md")
        if new_root is not None:
            with open(root_rp) as f:
                if f.read() != new_root:
                    stale.append("README.md (baseline-results block)")
        if stale:
            print("out of date: " + ", ".join(stale) + " (run scripts/build_results_index.py)")
            return 1
        print(f"index up to date ({len(results)} result files)")
        return 0
    if prev_generated and os.path.exists(ip):
        with open(ip) as f:
            old = json.load(f)
        old.pop("generated", None)
        cur = dict(index)
        cur.pop("generated", None)
        if old == cur:
            index["generated"] = prev_generated      # nothing changed: keep the timestamp stable
            table = leaderboard_md(index)
            new_readme = update_readme(rp, table)
    with open(ip, "w") as f:
        json.dump(index, f, separators=(",", ":"))
        f.write("\n")
    with open(rp, "w") as f:
        f.write(new_readme)
    if new_root is not None:
        with open(root_rp, "w") as f:
            f.write(new_root)
    print(f"wrote {ip} ({os.path.getsize(ip) / 1e3:.0f} kB, {len(results)} results), {rp}" + (f" and the baseline-results block of {root_rp}" if new_root is not None else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify index.json and README.md are current; exit 1 otherwise")
    a = ap.parse_args(argv)
    return build(check=a.check)


if __name__ == "__main__":
    sys.exit(main())
