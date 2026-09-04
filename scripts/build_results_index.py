#!/usr/bin/env python
"""Build results/index.json (the leaderboard index the site reads) and the leaderboard tables in results/README.md and
the root README from every result file under results/.

    python scripts/build_results_index.py            # validates, writes results/index.json, updates results/README.md + README.md
    python scripts/build_results_index.py --check    # exit 1 if index.json / README.md are out of date (CI)

index.json holds, per result file, the run metadata and one block per suite present in the file (`suites.core`,
`suites.human`): the aggregate numbers, the per-family / per-scenario / per-difficulty / per-lock-state breakdowns
and a compact per-door outcome map (`doors: {door_id: [n_success, n_episodes]}`) so the catalogue can show a badge
per door without loading the multi-megabyte result files.  Core and human numbers are never mixed: the headline
"N / 1000 doors" is the core suite; the human suite (advanced, opt-in) gets its own table.
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
ROOT_START, ROOT_END = "<!-- baseline-results:start -->", "<!-- baseline-results:end -->"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_result import CORE_SCENARIOS, HUMAN_SCENARIOS, SUITE_OF, SUITES, manifest_scenarios, validate_file  # noqa: E402

SCENARIO_ORDER = list(CORE_SCENARIOS) + list(HUMAN_SCENARIOS)
LOCKS = ["unlocked", "locked_releasable", "locked_no_release"]


def _slim_group(g: dict) -> dict:
    return {k: g.get(k) for k in ("n_doors", "n_episodes", "n_success", "success_rate", "doors_solved", "doors_solved_any", "damage_rate", "mean_time_to_pass_s", "mean_return", "human_collision_rate")}


def suite_block(suite: str, tab: dict, eps: list[dict], n_doors_suite: int) -> dict:
    doors = {}
    for e in eps:
        s = doors.setdefault(e["door_id"], [0, 0])
        s[0] += int(bool(e.get("success")))
        s[1] += 1
    return {
        "suite": suite, "scenarios": tab.get("scenarios", []), "n_doors": tab["n_doors"], "n_doors_suite": n_doors_suite,
        "complete": tab["n_doors"] >= n_doors_suite,
        "doors_solved": tab["doors_solved"], "doors_solved_any": tab["doors_solved_any"], "n_episodes": tab["n_episodes"], "n_success": tab["n_success"],
        "success_rate": tab["success_rate"], "damage_rate": tab["damage_rate"], "human_collision_rate": tab.get("human_collision_rate"), "mean_return": tab.get("mean_return"),
        "mean_time_to_pass_s": tab.get("mean_time_to_pass_s"), "median_time_to_pass_s": tab.get("median_time_to_pass_s"),
        "outcomes": tab.get("outcomes", {}), "n_errors": tab.get("n_errors", 0), "mean_wall_s": tab.get("mean_wall_s"),
        "by_family": {k: _slim_group(v) for k, v in tab["by_family"].items()},
        "by_scenario": {k: _slim_group(v) for k, v in tab["by_scenario"].items()},
        "by_difficulty": {k: _slim_group(v) for k, v in tab["by_difficulty"].items()},
        "by_lock_state": {k: _slim_group(v) for k, v in tab.get("by_lock_state", {}).items()},
        "doors": doors,
    }


def summarize(path: str, n_total: int, n_human: int) -> dict:
    with open(path) as f:
        doc = json.load(f)
    run, pol, agg, bench = doc["run"], doc["policy"], doc["aggregate"], doc["benchmark"]
    suites = {}
    for suite in SUITES:
        if suite in agg:
            eps = [e for e in doc["episodes"] if e.get("suite") == suite and e.get("outcome") != "error"]
            suites[suite] = suite_block(suite, agg[suite], eps, n_total if suite == "core" else n_human)
    return {
        "file": os.path.basename(path), "policy": pol["name"], "description": pol.get("description", ""), "embodiment": pol.get("embodiment", "hand_base"),
        "policy_class": pol.get("class"), "extra": pol.get("extra", {}),
        "simulator": run["simulator"], "simulator_version": run.get("simulator_version"), "tier": run["tier"], "date": run["date"][:10], "label": run.get("label", ""),
        "commit": bench.get("commit"), "dataset_version": bench.get("dataset_version"), "n_doors_total": bench.get("n_doors_total"),
        "suite": run.get("suite", "core"), "scenario_filter": run.get("scenario_filter", "all"), "n_doors": run["n_doors"], "seeds": run["seeds"],
        "scenarios": [s["name"] for s in run["scenarios"]], "randomize": run.get("randomize"), "time_budget_s": run.get("time_budget_s"),
        "wall_time_s": run.get("wall_time_s"), "host": (run.get("host") or {}).get("platform"), "door_selection": run.get("door_selection", "all"),
        # leaderboard = a complete core run (every door, every listed core scenario, the scenario's own budget)
        "leaderboard": "core" in suites and suites["core"]["complete"] and run.get("scenario_filter", "all") == "all" and not isinstance(run.get("time_budget_s"), (int, float)),
        "suites": suites,
    }


def _rows(index: dict, suite: str, complete_only: bool = False) -> list[dict]:
    rows = [r for r in index["results"] if suite in r["suites"] and (not complete_only or r["suites"][suite]["complete"])]
    return sorted(rows, key=lambda r: (-int(bool(r["suites"][suite]["complete"])), -r["suites"][suite]["doors_solved"], -r["suites"][suite]["success_rate"]))


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f} %"


def _secs(x):
    return "-" if x is None else f"{x:.1f} s"


def _scenario_table(rows: list[dict], suite: str, scenarios: list[str]) -> list[str]:
    lines = ["| scenario | doors | " + " | ".join(f"`{r['policy']}`" for r in rows) + " |", "|---|---|" + "---|" * len(rows)]
    for s in scenarios:
        if not any(s in r["suites"][suite]["by_scenario"] for r in rows):
            continue
        n = max((r["suites"][suite]["by_scenario"].get(s, {}).get("n_doors") or 0) for r in rows)
        lines.append(f"| {s} | {n} | " + " | ".join((lambda g: f"{g['doors_solved']} ({_pct(g['success_rate'])})" if g else "-")(r["suites"][suite]["by_scenario"].get(s)) for r in rows) + " |")
    return lines


def leaderboard_md(index: dict) -> str:
    total, n_human = index.get("n_doors_total") or 1000, index.get("n_doors_human") or 0
    core = _rows(index, "core")
    lines = ["### Core suite (default: no simulated person)", "",
             "| policy | embodiment | simulator | tier | doors | seeds | solved (every scenario, every seed) | episode success | damage | median time-to-traverse | date | commit |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in core:
        c = r["suites"]["core"]
        note = "" if c["complete"] else f" (subset: {r.get('door_selection', '')})"
        lines.append(f"| [{r['policy']}]({r['file']}){note} | {r['embodiment']} | {r['simulator']} {r.get('simulator_version') or ''} | {r['tier']} | {c['n_doors']} | {len(r['seeds'])} | "
                     f"**{c['doors_solved']} / {c['n_doors_suite'] if c['complete'] else c['n_doors']}** | {_pct(c['success_rate'])} | {_pct(c['damage_rate'])} | {_secs(c['median_time_to_pass_s'])} | {r['date']} | {(r.get('commit') or '')[:8]} |")
    fams = sorted({f for r in core for f in r["suites"]["core"]["by_family"]})
    lines += ["", "Per family, doors solved (core suite, `n` doors per family):", "",
              "| family | n | " + " | ".join(f"`{r['policy']}`" for r in core) + " |", "|---|---|" + "---|" * len(core)]
    for f in fams:
        n = max((r["suites"]["core"]["by_family"].get(f, {}).get("n_doors") or 0) for r in core)
        lines.append(f"| {f} | {n} | " + " | ".join(str(r["suites"]["core"]["by_family"].get(f, {}).get("doors_solved", 0)) for r in core) + " |")
    lines += ["", "Per scenario (doors solved on every seed / doors listing the scenario; episode success in brackets):", ""] + _scenario_table(core, "core", list(CORE_SCENARIOS))
    lines += ["", "Per lock state:", "", "| lock state | n | " + " | ".join(f"`{r['policy']}`" for r in core) + " |", "|---|---|" + "---|" * len(core)]
    for k in LOCKS:
        n = max((r["suites"]["core"]["by_lock_state"].get(k, {}).get("n_doors") or 0) for r in core) if core else 0
        if n:
            lines.append(f"| {k} | {n} | " + " | ".join(str(r["suites"]["core"]["by_lock_state"].get(k, {}).get("doors_solved", 0)) for r in core) + " |")
    human = _rows(index, "human")
    lines += ["", "### Human suite (advanced, opt-in: `--suite human`)", "",
              f"A kinematic person is simulated; {n_human} doors list one of `hold_open_for_human`, `wait_for_human`, `knock_and_wait`.  "
              "These numbers are reported separately and never enter the core number above.", ""]
    if human:
        lines += ["| policy | embodiment | simulator | tier | doors | seeds | solved | episode success | human collisions | damage | date | commit |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in human:
            h = r["suites"]["human"]
            lines.append(f"| [{r['policy']}]({r['file']}) | {r['embodiment']} | {r['simulator']} {r.get('simulator_version') or ''} | {r['tier']} | {h['n_doors']} | {len(r['seeds'])} | "
                         f"**{h['doors_solved']} / {h['n_doors_suite'] if h['complete'] else h['n_doors']}** | {_pct(h['success_rate'])} | {_pct(h['human_collision_rate'])} | {_pct(h['damage_rate'])} | {r['date']} | {(r.get('commit') or '')[:8]} |")
        lines += [""] + _scenario_table(human, "human", list(HUMAN_SCENARIOS))
    else:
        lines.append("_No human-suite run yet._")
    lines.append("")
    lines.append(f"_Generated by `scripts/build_results_index.py` on {index['generated'][:10]} from {len(index['results'])} result file(s)._")
    return "\n".join(lines)


WHAT = {
    "scripted_hand": "the per-family oracle heuristic of `scripts/demo_mujoco.py` (reads joint names, lock parts and keypad codes from the spec; DoorEnv hand + synthetic base)",
    "g1_locomotion": "Unitree G1 (MuJoCo Menagerie) + pretrained unitree_rl_gym locomotion policy, walks toward the goal, arms parked",
    "random": "uniform random torques within the hand limits on every reachable joint + a random-walk base",
}


def root_readme_block(index: dict, manifest: dict | None) -> str:
    """The root README 'Baseline results' tables: core headline + per family + per scenario, then the human suite."""
    total, n_human = index.get("n_doors_total") or 1000, index.get("n_doors_human") or 0
    rows = _rows(index, "core")
    out = ["**Core suite** (the default: every door, every core scenario it lists, no simulated person):", "",
           "| policy | what it is | doors solved (every scenario, every seed) | episode success | damage | median time-to-traverse | wall time |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["suites"]["core"]
        wall = f"{r['wall_time_s'] / 60:.1f} min" if r.get("wall_time_s") else "-"
        head = f"**{c['doors_solved']} / {total}**" if c["complete"] else f"**{c['doors_solved']} / {c['n_doors']}** (subset `{r.get('door_selection', '')}`)"
        out.append(f"| `{r['policy']}` | {WHAT.get(r['policy'], r['description'])} | {head} | {_pct(c['success_rate'])} | {_pct(c['damage_rate'])} | {_secs(c['median_time_to_pass_s'])} | {wall} |")
    fam_n = {}
    if manifest:
        for d in manifest["doors"]:
            fam_n[d["family"]] = fam_n.get(d["family"], 0) + 1
    else:
        for r in rows:
            for f, g in r["suites"]["core"]["by_family"].items():
                fam_n[f] = max(fam_n.get(f, 0), g.get("n_doors") or 0)
    fams = sorted(fam_n, key=lambda f: (-fam_n[f], f))
    out += ["", "Doors solved per family (core suite; of the family's door count):", "", "| family | doors | " + " | ".join(f"`{r['policy']}`" for r in rows) + " |", "|---|---|" + "---|" * len(rows)]
    for f in fams:
        out.append(f"| {f} | {fam_n[f]} | " + " | ".join(str(r["suites"]["core"]["by_family"].get(f, {}).get("doors_solved", 0)) for r in rows) + " |")
    out += ["", "Per core scenario (doors solved on every seed / doors listing it; episode success in brackets):", ""] + _scenario_table(rows, "core", list(CORE_SCENARIOS))
    hrows = _rows(index, "human")
    out += ["", f"**Human suite** (advanced, opt-in `--suite human`: a simulated person; {n_human} doors list `hold_open_for_human`, `wait_for_human` or `knock_and_wait`; reported separately, never part of the core number):", ""]
    if hrows:
        out += ["| policy | doors | doors solved | episode success | human collisions | damage |", "|---|---|---|---|---|---|"]
        for r in hrows:
            h = r["suites"]["human"]
            out.append(f"| `{r['policy']}` | {h['n_doors']} | **{h['doors_solved']} / {h['n_doors_suite'] if h['complete'] else h['n_doors']}** | {_pct(h['success_rate'])} | {_pct(h['human_collision_rate'])} | {_pct(h['damage_rate'])} |")
        out += [""] + _scenario_table(hrows, "human", list(HUMAN_SCENARIOS))
    else:
        out.append("_No human-suite run yet._")
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
              "Every file here is one run of a policy over the DoorBench doors, written by `doorbench benchmark run` "
              "and validated by `scripts/validate_result.py` against `schema.json`.  `index.json` is generated from them "
              "(`scripts/build_results_index.py`) and feeds the [Results page](https://adamraudonis.github.io/DoorBench/#/results) "
              "of the site.  A door is **solved** when the policy succeeded on **every** scenario the door lists in the suite, on "
              "**every** seed.  The **core** suite (no simulated person) is the default and the headline number; the **human** "
              "suite is advanced and opt-in, reported in its own table.  To add your own run, read [docs/SUBMITTING.md](../docs/SUBMITTING.md).\n\n")
    if os.path.exists(path):
        with open(path) as f:
            cur = f.read()
        if START in cur:
            cur = header + cur[cur.index(START):]
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
    n_total = (len(manifest["doors"]) if manifest else 0)
    listed = manifest_scenarios(manifest) or {}
    n_human = sum(1 for ss in listed.values() if any(SUITE_OF.get(s) == "human" for s in ss))
    bad = 0
    results = []
    for p in files:
        errs = validate_file(p, schema, manifest, submission=False)
        if errs:
            bad += 1
            print(f"FAIL {p}: {errs[:5]}")
            continue
        results.append(None)
    if bad:
        return 1
    n_total = max([n_total] + [json.load(open(p))["benchmark"].get("n_doors_total") or 0 for p in files]) or 1000
    results = [summarize(p, n_total, n_human) for p in files]
    prev_generated = None
    ip = os.path.join(RESULTS, "index.json")
    if os.path.exists(ip):
        try:
            with open(ip) as f:
                prev_generated = json.load(f).get("generated")
        except Exception:
            pass
    index = {"schema_version": "1.1", "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "n_doors_total": n_total, "n_doors_human": n_human,
             "suites": {"core": list(CORE_SCENARIOS), "human": list(HUMAN_SCENARIOS)}, "results": results}
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
