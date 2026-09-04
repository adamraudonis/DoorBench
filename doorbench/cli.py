"""DoorBench command line.

  doorbench list [--family swing_single] [--task ...]
  doorbench show <door_id>                  print spec + physics summary
  doorbench build <door_id> --out DIR       export one door (all formats)
  doorbench qa <door_id>                    run the sign-off checks on an exported door
  doorbench view <door_id>                  open in the MuJoCo viewer
  doorbench stats                           dataset statistics from assets/manifest.json
  doorbench benchmark run --policy scripted_hand --doors all --seeds 3 --workers 8 --out results/scripted_hand.json   # core suite
  doorbench benchmark run --policy scripted_hand --suite human --out results/scripted_hand_human.json                  # opt-in human suite
  doorbench benchmark run --policy my_pkg.policies:MyPolicy --doors family:swing_single --dry-run
  doorbench benchmark list-scenarios | list-policies
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    from .benchmark import runner as R
    ap = argparse.ArgumentParser(prog="doorbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--family"); p.add_argument("--task"); p.add_argument("--assets", default="assets")
    p = sub.add_parser("show"); p.add_argument("door_id"); p.add_argument("--assets", default="assets")
    p = sub.add_parser("build"); p.add_argument("door_id"); p.add_argument("--out", default="out/build"); p.add_argument("--formats", default="mjcf,urdf,usd,json")
    p = sub.add_parser("qa"); p.add_argument("door_id"); p.add_argument("--assets", default="assets")
    p = sub.add_parser("view"); p.add_argument("door_id"); p.add_argument("--assets", default="assets"); p.add_argument("--tier", default="full")
    p = sub.add_parser("stats"); p.add_argument("--assets", default="assets")
    b = sub.add_parser("benchmark", help="evaluate a policy over doors x scenarios x seeds (doorbench.benchmark.runner)")
    bsub = b.add_subparsers(dest="bcmd", required=True)
    r = bsub.add_parser("run", help="run a policy; writes a result JSON (results/schema.json)")
    r.add_argument("--policy", required=True, help="random | scripted_hand | g1_locomotion | module.path:Class | path/to/file.py:Class")
    r.add_argument("--doors", default="all", help="all | family:<f,..> | difficulty:<n,..> | scenario:<s,..> | lock:locked|unlocked | first:<n> | every:<n>[:<offset>] | sample:<n>[:<seed>] | ids:<a,b> | <a,b> | @file")
    r.add_argument("--seeds", default="3", help="number of seeds (0..n-1) or an explicit list 0,1,2; seed 0 = nominal door, >= 1 randomised")
    r.add_argument("--suite", default="core", choices=list(R.SUITE_CHOICES), help="core (default: no simulated person, all 1000 doors) | human (advanced, opt-in: the 79 doors with a human scenario) | all (both, kept in separate tables)")
    r.add_argument("--scenarios", default=None, help="narrow to these scenario types (comma list; must belong to --suite), 'primary' = each door's primary scenario only; default: every scenario the door lists in the suite")
    r.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    r.add_argument("--tier", default="full", choices=["full", "simple", "minimal"])
    r.add_argument("--assets", default="assets")
    r.add_argument("--budget", type=float, default=None, help="sim-time budget per episode (s); default: the scenario's own time_budget_s")
    r.add_argument("--wall-timeout", type=float, default=120.0, help="wall-clock limit per episode (s)")
    r.add_argument("--control-dt", type=float, default=None, help="override the policy's control period (s)")
    r.add_argument("--no-randomize", action="store_true", help="nominal physics for every seed")
    r.add_argument("--label", default="", help="free-text label stored in the result (e.g. team / hardware)")
    r.add_argument("--out", default=None, help="result JSON path (default results/<policy>.json)")
    r.add_argument("--dry-run", action="store_true", help="print the door list and exit")
    bsub.add_parser("list-scenarios", help="print the scenarios")
    bsub.add_parser("list-policies", help="print the built-in baseline policies")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        man = json.load(open(os.path.join(a.assets, "manifest.json")))
        for d in man["doors"]:
            if a.family and d["family"] != a.family:
                continue
            if a.task and d.get("task") != a.task:
                continue
            print(f"{d['id']:28s} {d['family']:20s} {d['mass_kg']:7.1f} kg  {d['operator']:24s} lock={d['lock']:18s} {'locked' if d['lock_engaged'] else '      '} {d.get('task','')}")
    elif a.cmd == "show":
        spec = json.load(open(os.path.join(a.assets, "doors", a.door_id, "spec.json")))
        phys = spec.pop("physics", {})
        print(json.dumps({k: v for k, v in spec.items() if k not in ("tags",)}, indent=1))
        print("--- physics summary ---")
        print(json.dumps({k: phys[k] for k in ("mass", "hinge", "closer", "latch", "lock", "compliance") if k in phys}, indent=1))
    elif a.cmd == "build":
        from .spec import generate_all
        from .build import export_door
        specs = {s["id"]: s for s in generate_all()}
        s = specs[a.door_id]
        out = export_door(s, os.path.join(a.out, "doors"), os.path.join(a.out, "hardware"), formats=tuple(a.formats.split(",")))
        print(json.dumps(out, indent=1, default=str))
    elif a.cmd == "qa":
        from .qa import run_qa
        from .spec import generate_all
        d = os.path.join(a.assets, "doors", a.door_id)
        spec = json.load(open(os.path.join(d, "spec.json")))
        meta = json.load(open(os.path.join(d, "model.json")))["meta"]
        files = {"mjcf": {t: os.path.join(d, f) for t, f in (("full", "door.xml"), ("simple", "door_simple.xml"), ("minimal", "door_minimal.xml")) if os.path.exists(os.path.join(d, f))},
                 "urdf": {"full": os.path.join(d, "door.urdf")}, "usd": os.path.join(d, "door.usda")}
        qa = run_qa(spec, d, meta, files, spec["physics"])
        print(json.dumps(qa, indent=1))
    elif a.cmd == "view":
        import subprocess
        f = {"full": "scene.xml", "simple": "door_simple.xml", "minimal": "door_minimal.xml"}[a.tier]
        subprocess.call([sys.executable, "-m", "mujoco.viewer", "--mjcf", os.path.join(a.assets, "doors", a.door_id, f)])
    elif a.cmd == "benchmark":
        from .benchmark.scenarios import SCENARIO_DESCRIPTIONS, SCENARIO_SUITE, SCENARIO_TYPES
        from .benchmark.policy import BASELINES, load_policy_class, policy_meta
        if a.bcmd == "list-scenarios":
            for n in SCENARIO_TYPES:
                print(f"{n:22s} suite={SCENARIO_SUITE[n]:6s} {SCENARIO_DESCRIPTIONS[n]}")
            print("\nThe core suite is the default (no simulated person); the human suite is opt-in: --suite human | all.")
        elif a.bcmd == "list-policies":
            for k, v in BASELINES.items():
                try:
                    meta = policy_meta(load_policy_class(v))
                    print(f"{k:16s} {v:64s} [{meta['embodiment']}] {meta['description']}")
                except Exception as e:
                    print(f"{k:16s} {v:64s} (not loadable: {e})")
        elif a.bcmd == "run":
            seeds = int(a.seeds) if a.seeds.isdigit() else [int(x) for x in a.seeds.split(",") if x.strip()]
            n_seeds = seeds if isinstance(seeds, int) else len(seeds)
            if a.dry_run:
                R.dry_run(a.doors, a.suite, a.scenarios, n_seeds, assets=a.assets)
                return
            out = a.out or os.path.join("results", f"{R.load_policy_class(a.policy).name}{'' if a.suite == 'core' else '_' + a.suite}.json")
            R.run_benchmark(a.policy, doors=a.doors, seeds=seeds, suite=a.suite, scenarios=a.scenarios, workers=a.workers, tier=a.tier, assets=a.assets, time_budget_s=a.budget,
                            wall_timeout_s=a.wall_timeout, randomize=not a.no_randomize, control_dt=a.control_dt, label=a.label, out=out)
    elif a.cmd == "stats":
        import collections
        man = json.load(open(os.path.join(a.assets, "manifest.json")))
        ds = [d for d in man["doors"] if not d.get("error")]
        print(f"{len(ds)} doors, {sum(1 for d in ds if d['signed_off'])} signed off")
        for key in ("family", "operator", "lock", "closer", "condition", "task", "difficulty"):
            c = collections.Counter(d[key] for d in ds)
            print(f"\n{key}:")
            for k, n in c.most_common():
                print(f"  {str(k):28s} {n}")


if __name__ == "__main__":
    main()
