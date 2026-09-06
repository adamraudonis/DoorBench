#!/usr/bin/env python
"""Validate DoorBench benchmark result files (results/*.json) against results/schema.json plus the submission rules.

    python scripts/validate_result.py results/scripted_hand.json [more.json ...]
    python scripts/validate_result.py --submission results/myteam_mypolicy.json     # leaderboard rules (all doors, >= 3 seeds, commit)
    python scripts/validate_result.py --all                                          # every result file in results/

Exit status 0 when every file is valid.  Uses `jsonschema` when it is installed; otherwise a small built-in checker
covering the subset of JSON Schema the file uses (type, required, properties, additionalProperties, enum, minimum,
maximum, exclusiveMinimum, minLength, maxLength, minProperties, pattern, items, prefixItems, minItems, maxItems,
$ref to $defs).

Suites: every scenario belongs to the `core` suite (no simulated person) or the `human` suite (advanced, opt-in).
A result file may hold both, but never mixed: each episode's `suite` must match its scenario, `aggregate` holds one
table per suite whose counts must equal the episodes of that suite, and a table listing a scenario of the other
suite is rejected.  The leaderboard (--submission) needs a core table over all doors on every scenario each door
lists, >= 3 seeds; a human table is optional (and may come in its own file, `results/<team>_<policy>_human.json`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCHEMA = os.path.join(ROOT, "results", "schema.json")
MANIFEST = os.path.join(ROOT, "assets", "manifest.json")
RESERVED = {"schema.json", "index.json"}
sys.path.insert(0, ROOT)
from doorbench.benchmark_eligibility import is_benchmark_eligible, POLICY_VERSION

SUBMISSION_MIN_SEEDS = 3
# mirrors doorbench.benchmark.scenarios.SCENARIO_SUITE (kept inline so CI can validate without the package's dependencies)
CORE_SCENARIOS = ("open_only", "open_and_traverse", "open_then_close", "close_only", "unlock_and_traverse", "locked_recognize")
HUMAN_SCENARIOS = ("hold_open_for_human", "wait_for_human", "knock_and_wait")
SUITE_OF = {**{n: "core" for n in CORE_SCENARIOS}, **{n: "human" for n in HUMAN_SCENARIOS}}
SUITES = ("core", "human")


# ----------------------------------------------------------------------------------------------- minimal checker
class MiniValidator:
    TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float), "null": type(None)}

    def __init__(self, schema: dict):
        self.schema = schema
        self.errors: list[str] = []

    def _type_ok(self, v, t):
        if t == "integer":
            return isinstance(v, int) and not isinstance(v, bool)
        if t == "number":
            return isinstance(v, (int, float)) and not isinstance(v, bool)
        if t == "boolean":
            return isinstance(v, bool)
        return isinstance(v, self.TYPES[t])

    def _resolve(self, s):
        if "$ref" in s:
            ref = s["$ref"]
            node = self.schema
            for part in ref.lstrip("#/").split("/"):
                node = node[part]
            return node
        return s

    def check(self, v, s, path="$"):
        s = self._resolve(s)
        if len(self.errors) > 200:
            return
        t = s.get("type")
        if t is not None:
            ts = t if isinstance(t, list) else [t]
            if not any(self._type_ok(v, x) for x in ts):
                self.errors.append(f"{path}: expected {t}, got {type(v).__name__}")
                return
        if "enum" in s and v not in s["enum"]:
            self.errors.append(f"{path}: {v!r} not in {s['enum']}")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if "minimum" in s and v < s["minimum"]:
                self.errors.append(f"{path}: {v} < minimum {s['minimum']}")
            if "maximum" in s and v > s["maximum"]:
                self.errors.append(f"{path}: {v} > maximum {s['maximum']}")
            if "exclusiveMinimum" in s and v <= s["exclusiveMinimum"]:
                self.errors.append(f"{path}: {v} <= exclusiveMinimum {s['exclusiveMinimum']}")
        if isinstance(v, str):
            if "minLength" in s and len(v) < s["minLength"]:
                self.errors.append(f"{path}: shorter than {s['minLength']}")
            if "maxLength" in s and len(v) > s["maxLength"]:
                self.errors.append(f"{path}: longer than {s['maxLength']}")
            if "pattern" in s and not re.search(s["pattern"], v):
                self.errors.append(f"{path}: {v!r} does not match {s['pattern']}")
        if isinstance(v, dict):
            for k in s.get("required", []):
                if k not in v:
                    self.errors.append(f"{path}: missing required key {k!r}")
            if "minProperties" in s and len(v) < s["minProperties"]:
                self.errors.append(f"{path}: fewer than {s['minProperties']} keys")
            props = s.get("properties", {})
            for k, sub in props.items():
                if k in v:
                    self.check(v[k], sub, f"{path}.{k}")
            ap = s.get("additionalProperties")
            if isinstance(ap, dict):
                for k, val in v.items():
                    if k not in props:
                        self.check(val, ap, f"{path}.{k}")
            elif ap is False:
                for k in v:
                    if k not in props:
                        self.errors.append(f"{path}: unexpected key {k!r}")
        if isinstance(v, list):
            if "minItems" in s and len(v) < s["minItems"]:
                self.errors.append(f"{path}: fewer than {s['minItems']} items")
            if "maxItems" in s and len(v) > s["maxItems"]:
                self.errors.append(f"{path}: more than {s['maxItems']} items")
            pi = s.get("prefixItems")
            for i, item in enumerate(v):
                if pi and i < len(pi):
                    self.check(item, pi[i], f"{path}[{i}]")
                elif "items" in s:
                    self.check(item, s["items"], f"{path}[{i}]")


def schema_errors(doc: dict, schema: dict) -> list[str]:
    try:
        import jsonschema
        v = jsonschema.Draft202012Validator(schema)
        return [f"{'/'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}" for e in sorted(v.iter_errors(doc), key=lambda e: list(e.absolute_path))][:200]
    except ImportError:
        mv = MiniValidator(schema)
        mv.check(doc, schema)
        return mv.errors


# ----------------------------------------------------------------------------------------------- semantic checks
def manifest_scenarios(manifest: dict | None, *, eligible_only: bool = True) -> dict[str, list[str]] | None:
    """door id -> the scenarios it lists (manifest benchmark summary), or None without a manifest."""
    if not manifest:
        return None
    out = {}
    for d in manifest["doors"]:
        if eligible_only and not is_benchmark_eligible(d):
            continue
        b = d.get("benchmark") or {}
        out[d["id"]] = list(b.get("scenarios") or [b.get("primary") or "open_and_traverse"])
    return out


def _table_errors(suite: str, tab: dict, eps: list[dict]) -> list[str]:
    """The aggregate table of `suite` must be computed from exactly the episodes of that suite (never mixed)."""
    errs = []
    p = f"aggregate.{suite}"
    if tab.get("suite") != suite:
        errs.append(f"{p}.suite={tab.get('suite')!r} but the table is keyed {suite!r} (mixed / mislabelled table)")
    other = [s for s in tab.get("scenarios", []) if SUITE_OF.get(s) != suite]
    if other:
        errs.append(f"{p}.scenarios lists {other} which belong to the other suite (mixed table)")
    other = [s for s in tab.get("by_scenario", {}) if SUITE_OF.get(s) != suite]
    if other:
        errs.append(f"{p}.by_scenario has {other} which belong to the other suite (mixed table)")
    good = [e for e in eps if e.get("outcome") != "error"] or eps
    n_err = sum(1 for e in eps if e.get("outcome") == "error")
    if tab.get("n_episodes") != len(good):
        errs.append(f"{p}.n_episodes={tab.get('n_episodes')} but {len(good)} non-error {suite} episodes")
    if tab.get("n_errors", n_err) != n_err:
        errs.append(f"{p}.n_errors={tab.get('n_errors')} but {n_err} error episodes")
    ns = sum(1 for e in good if e.get("success"))
    if tab.get("n_success") != ns:
        errs.append(f"{p}.n_success={tab.get('n_success')} but {ns} successful {suite} episodes")
    doors = {e.get("door_id") for e in good}
    if tab.get("n_doors") != len(doors):
        errs.append(f"{p}.n_doors={tab.get('n_doors')} but {len(doors)} distinct doors in the {suite} episodes")
    by_door = {}
    for e in good:
        by_door.setdefault(e.get("door_id"), []).append(bool(e.get("success")))
    solved = sum(1 for v in by_door.values() if all(v))
    if tab.get("doors_solved") != solved:
        errs.append(f"{p}.doors_solved={tab.get('doors_solved')} but {solved} doors succeeded on every {suite} episode")
    scen = {e.get("scenario") for e in eps}
    if set(tab.get("scenarios", [])) != scen:
        errs.append(f"{p}.scenarios={tab.get('scenarios')} but the {suite} episodes cover {sorted(scen)}")
    for s, g in tab.get("by_scenario", {}).items():
        n = sum(1 for e in good if e.get("scenario") == s)
        if g.get("n_episodes") != n:
            errs.append(f"{p}.by_scenario.{s}.n_episodes={g.get('n_episodes')} but {n} episodes")
    return errs


def semantic_errors(doc: dict, manifest: dict | None, submission: bool, path: str) -> list[str]:
    errs = []
    eps = doc.get("episodes", [])
    agg = doc.get("aggregate", {})
    run = doc.get("run", {})
    ids = {d["id"] for d in manifest["doors"]} if manifest else None
    fam = {d["id"]: d["family"] for d in manifest["doors"]} if manifest else None
    current_policy = doc.get("benchmark", {}).get("eligibility_policy")
    enforce_eligibility = submission or current_policy is not None
    eligible_ids = {d["id"] for d in manifest["doors"] if is_benchmark_eligible(d)} if manifest else None
    listed = manifest_scenarios(manifest, eligible_only=enforce_eligibility)
    if current_policy is not None and current_policy != POLICY_VERSION:
        errs.append(f"unknown benchmark eligibility policy {current_policy!r}")
    if current_policy is not None and eligible_ids is not None and doc.get("benchmark", {}).get("n_doors_total") != len(eligible_ids):
        errs.append(f"benchmark.n_doors_total must count {len(eligible_ids)} eligible doors")
    run_suite = run.get("suite")
    run_scen = {s.get("name") for s in run.get("scenarios", [])}
    for s in run.get("scenarios", []):
        if s.get("suite") != SUITE_OF.get(s.get("name")):
            errs.append(f"run.scenarios: {s.get('name')} is a {SUITE_OF.get(s.get('name'))} scenario, not {s.get('suite')}")
        if run_suite in SUITES and SUITE_OF.get(s.get("name")) != run_suite:
            errs.append(f"run.suite={run_suite} but run.scenarios lists the {SUITE_OF.get(s.get('name'))} scenario {s.get('name')}")
    seen = set()
    per_suite = {s: [] for s in SUITES}
    pairs = {}
    for i, e in enumerate(eps):
        if enforce_eligibility and (not is_benchmark_eligible(e) or (ids is not None and e.get("door_id") in ids - eligible_ids)):
            errs.append(f"episodes[{i}]: supplementary pet door excluded from benchmark evaluation")
        key = (e.get("door_id"), e.get("scenario"), e.get("seed"))
        if key in seen:
            errs.append(f"episodes[{i}]: duplicate door/scenario/seed {key}")
        seen.add(key)
        want = SUITE_OF.get(e.get("scenario"))
        if e.get("suite") != want:
            errs.append(f"episodes[{i}]: scenario {e.get('scenario')} belongs to the {want} suite, not {e.get('suite')}")
        elif run_suite in SUITES and want != run_suite:
            errs.append(f"episodes[{i}]: {want} episode in a run.suite={run_suite} file")
        if want in per_suite:
            per_suite[want].append(e)
        pairs.setdefault((e.get("door_id"), e.get("scenario")), set()).add(e.get("seed"))
        if ids is not None and e.get("door_id") not in ids:
            errs.append(f"episodes[{i}]: unknown door id {e.get('door_id')}")
        if fam is not None and e.get("door_id") in fam and e.get("family") != fam[e["door_id"]]:
            errs.append(f"episodes[{i}]: family {e.get('family')} does not match the manifest ({fam[e['door_id']]})")
        historical_pet = not enforce_eligibility and (e.get("family") == "pet_door" or (fam is not None and fam.get(e.get("door_id")) == "pet_door"))
        if not historical_pet and listed is not None and e.get("door_id") in listed and e.get("scenario") not in listed[e["door_id"]]:
            errs.append(f"episodes[{i}]: {e.get('door_id')} does not list the scenario {e.get('scenario')} (it lists {listed[e['door_id']]})")
        if e.get("success") and e.get("outcome") != "success":
            errs.append(f"episodes[{i}]: success=true but outcome={e.get('outcome')}")
        if e.get("success") and e.get("damage"):
            errs.append(f"episodes[{i}]: success with damage")
        if e.get("seed") not in run.get("seeds", []):
            errs.append(f"episodes[{i}]: seed {e.get('seed')} not in run.seeds")
        if e.get("scenario") not in run_scen:
            errs.append(f"episodes[{i}]: scenario {e.get('scenario')} not in run.scenarios")
        if len(errs) > 60:
            errs.append("... (further episode problems not listed)")
            break
    seeds = set(run.get("seeds", []))
    short = [k for k, v in pairs.items() if v != seeds]
    if short:
        errs.append(f"{len(short)} door/scenario pair(s) were not evaluated on every seed {sorted(seeds)} (e.g. {short[0]})")
    doors = {e.get("door_id") for e in eps}
    if run.get("n_doors") != len(doors):
        errs.append(f"run.n_doors={run.get('n_doors')} but {len(doors)} distinct doors in episodes")
    # ---- one table per suite, never mixed
    for suite in SUITES:
        have = suite in agg
        need = bool(per_suite[suite])
        if have and not need:
            errs.append(f"aggregate.{suite} present but there are no {suite} episodes")
        elif need and not have:
            errs.append(f"{len(per_suite[suite])} {suite} episodes but no aggregate.{suite} table")
        elif have:
            errs += _table_errors(suite, agg[suite], per_suite[suite])
    for k in agg:
        if k not in SUITES:
            errs.append(f"aggregate.{k}: tables are keyed by suite (core | human) only; no mixed / 'all' table")
    if submission:
        total = len(eligible_ids) if eligible_ids is not None else (doc.get("benchmark", {}).get("n_doors_total") or 985)
        core = agg.get("core")
        human = agg.get("human")
        if core is None and human is None:
            errs.append("submission has no core or human table")
        if core is not None:
            core_doors = {e.get("door_id") for e in per_suite["core"]}
            if eligible_ids is not None and core_doors != eligible_ids:
                errs.append(f"core suite submission must cover all {len(eligible_ids)} benchmark-eligible doors (has {len(core_doors)})")
            elif len(core_doors) < total:
                errs.append(f"core suite submission must cover all {total} doors (has {len(core_doors)})")
            if listed is not None:
                missing = [(d, s) for d in core_doors for s in listed.get(d, []) if SUITE_OF[s] == "core" and (d, s) not in pairs]
                if missing:
                    errs.append(f"core suite submission must evaluate every core scenario each door lists; {len(missing)} missing (e.g. {missing[0]}) - run without --scenarios")
        if human is not None and listed is not None:
            hdoors = {d for d, ss in listed.items() if any(SUITE_OF[s] == "human" for s in ss)}
            got = {e.get("door_id") for e in per_suite["human"]}
            if got != hdoors:
                errs.append(f"human suite submission must cover all {len(hdoors)} doors that list a human scenario (has {len(got)})")
            missing = [(d, s) for d in got for s in listed.get(d, []) if SUITE_OF[s] == "human" and (d, s) not in pairs]
            if missing:
                errs.append(f"human suite submission must evaluate every human scenario each door lists; {len(missing)} missing (e.g. {missing[0]})")
        if len(run.get("seeds", [])) < SUBMISSION_MIN_SEEDS:
            errs.append(f"submission needs >= {SUBMISSION_MIN_SEEDS} seeds (has {len(run.get('seeds', []))})")
        if not doc.get("benchmark", {}).get("commit"):
            errs.append("submission must record benchmark.commit (the DoorBench commit hash of the assets)")
        if not run.get("simulator_version"):
            errs.append("submission must record run.simulator_version")
        if isinstance(run.get("time_budget_s"), (int, float)):
            errs.append("submission must use each scenario's own time budget (no --budget override)")
        base = os.path.basename(path)
        pname = doc.get("policy", {}).get("name", "")
        bare = base in (f"{pname}.json", f"{pname}_human.json")          # the shipped baselines use the bare policy name
        if not bare and not re.match(r"^[a-z0-9]+_[a-z0-9_.-]+\.json$", base):
            errs.append(f"submission file name should be results/<team>_<policy>.json (got {base})")
        if pname and pname not in base:
            errs.append(f"policy.name {pname!r} should appear in the file name {base}")
        if core is None and human is not None and not base.endswith("_human.json"):
            errs.append("a human-suite-only submission should be named results/<team>_<policy>_human.json")
    return errs


def validate_file(path: str, schema: dict, manifest: dict | None, submission: bool) -> list[str]:
    try:
        with open(path) as f:
            doc = json.load(f)
    except Exception as e:
        return [f"not valid JSON: {e}"]
    errs = schema_errors(doc, schema)
    if not errs:
        errs = semantic_errors(doc, manifest, submission, path)
    return errs


def summary_line(path: str) -> str:
    with open(path) as f:
        d = json.load(f)
    parts = []
    for suite, ag in d["aggregate"].items():
        parts.append(f"{suite}: {ag['doors_solved']}/{ag['n_doors']} doors, {ag['n_episodes']} episodes, success {ag['success_rate'] * 100:.1f} %")
    return f"ok   {path}: {d['policy']['name']} | " + " | ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true", help="validate every results/*.json (except schema.json / index.json)")
    ap.add_argument("--submission", action="store_true", help="also enforce the leaderboard submission rules")
    ap.add_argument("--schema", default=SCHEMA)
    ap.add_argument("--manifest", default=MANIFEST, help="assets/manifest.json for door-id / scenario checks ('' to skip)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    files = list(a.files)
    if a.all:
        files += [p for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))) if os.path.basename(p) not in RESERVED]
    if not files:
        ap.error("no files given")
    with open(a.schema) as f:
        schema = json.load(f)
    manifest = None
    if a.manifest and os.path.exists(a.manifest):
        with open(a.manifest) as f:
            manifest = json.load(f)
    bad = 0
    for p in files:
        if os.path.basename(p) in RESERVED:
            continue
        errs = validate_file(p, schema, manifest, a.submission)
        if errs:
            bad += 1
            print(f"FAIL {p}: {len(errs)} problem(s)")
            for e in errs[:40]:
                print(f"   - {e}")
        elif not a.quiet:
            print(summary_line(p))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
