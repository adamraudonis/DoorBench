#!/usr/bin/env python
"""Validate DoorBench benchmark result files (results/*.json) against results/schema.json plus the submission rules.

    python scripts/validate_result.py results/scripted_hand.json [more.json ...]
    python scripts/validate_result.py --submission results/myteam_mypolicy.json     # leaderboard rules (all doors, >= 3 seeds, commit)
    python scripts/validate_result.py --all                                          # every result file in results/

Exit status 0 when every file is valid.  Uses `jsonschema` when it is installed; otherwise a small built-in checker
covering the subset of JSON Schema the file uses (type, required, properties, additionalProperties, enum, minimum,
maximum, exclusiveMinimum, minLength, maxLength, pattern, items, prefixItems, minItems, maxItems, $ref to $defs).
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
SUBMISSION_MIN_SEEDS = 3


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
def semantic_errors(doc: dict, manifest: dict | None, submission: bool, path: str) -> list[str]:
    errs = []
    eps = doc.get("episodes", [])
    agg = doc.get("aggregate", {})
    run = doc.get("run", {})
    ids = {d["id"] for d in manifest["doors"]} if manifest else None
    fam = {d["id"]: d["family"] for d in manifest["doors"]} if manifest else None
    seen = set()
    n_success = 0
    for i, e in enumerate(eps):
        key = (e.get("door_id"), e.get("scenario"), e.get("seed"))
        if key in seen:
            errs.append(f"episodes[{i}]: duplicate door/scenario/seed {key}")
        seen.add(key)
        if ids is not None and e.get("door_id") not in ids:
            errs.append(f"episodes[{i}]: unknown door id {e.get('door_id')}")
        if fam is not None and e.get("door_id") in fam and e.get("family") != fam[e["door_id"]]:
            errs.append(f"episodes[{i}]: family {e.get('family')} does not match the manifest ({fam[e['door_id']]})")
        if e.get("success") and e.get("outcome") != "success":
            errs.append(f"episodes[{i}]: success=true but outcome={e.get('outcome')}")
        if e.get("success") and e.get("damage"):
            errs.append(f"episodes[{i}]: success with damage")
        if e.get("seed") not in run.get("seeds", []):
            errs.append(f"episodes[{i}]: seed {e.get('seed')} not in run.seeds")
        if e.get("scenario") not in {s.get("name") for s in run.get("scenarios", [])}:
            errs.append(f"episodes[{i}]: scenario {e.get('scenario')} not in run.scenarios")
        n_success += bool(e.get("success"))
    doors = {e.get("door_id") for e in eps}
    if agg.get("n_episodes") != len(eps):
        errs.append(f"aggregate.n_episodes={agg.get('n_episodes')} but {len(eps)} episodes")
    if agg.get("n_success") != n_success:
        errs.append(f"aggregate.n_success={agg.get('n_success')} but {n_success} successful episodes")
    if agg.get("n_doors") != len(doors):
        errs.append(f"aggregate.n_doors={agg.get('n_doors')} but {len(doors)} distinct doors")
    if run.get("n_doors") != len(doors):
        errs.append(f"run.n_doors={run.get('n_doors')} but {len(doors)} distinct doors in episodes")
    by_door = {}
    for e in eps:
        by_door.setdefault(e.get("door_id"), []).append(bool(e.get("success")))
    solved = sum(1 for v in by_door.values() if all(v))
    if agg.get("doors_solved") != solved:
        errs.append(f"aggregate.doors_solved={agg.get('doors_solved')} but {solved} doors succeeded on every episode")
    expected = len(doors) * len(run.get("seeds", [])) * len(run.get("scenarios", []))
    if expected and len(eps) != expected:
        errs.append(f"{len(eps)} episodes but {len(doors)} doors x {len(run.get('seeds', []))} seeds x {len(run.get('scenarios', []))} scenarios = {expected}")
    if submission:
        total = doc.get("benchmark", {}).get("n_doors_total") or (len(manifest["doors"]) if manifest else 1000)
        if ids is not None and doors != ids:
            errs.append(f"submission must cover all {len(ids)} doors (has {len(doors)})")
        elif len(doors) < total:
            errs.append(f"submission must cover all {total} doors (has {len(doors)})")
        if len(run.get("seeds", [])) < SUBMISSION_MIN_SEEDS:
            errs.append(f"submission needs >= {SUBMISSION_MIN_SEEDS} seeds (has {len(run.get('seeds', []))})")
        if not doc.get("benchmark", {}).get("commit"):
            errs.append("submission must record benchmark.commit (the DoorBench commit hash of the assets)")
        if not run.get("simulator_version"):
            errs.append("submission must record run.simulator_version")
        if "default" not in {s.get("name") for s in run.get("scenarios", [])}:
            errs.append("submission must include the 'default' scenario")
        base = os.path.basename(path)
        if not re.match(r"^[a-z0-9]+_[a-z0-9_.-]+\.json$", base):
            errs.append(f"submission file name should be results/<team>_<policy>.json (got {base})")
        pname = doc.get("policy", {}).get("name", "")
        if pname and not base.startswith(pname) and pname not in base:
            errs.append(f"policy.name {pname!r} should appear in the file name {base}")
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true", help="validate every results/*.json (except schema.json / index.json)")
    ap.add_argument("--submission", action="store_true", help="also enforce the leaderboard submission rules")
    ap.add_argument("--schema", default=SCHEMA)
    ap.add_argument("--manifest", default=MANIFEST, help="assets/manifest.json for door-id checks ('' to skip)")
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
            with open(p) as f:
                d = json.load(f)
            ag = d["aggregate"]
            print(f"ok   {p}: {d['policy']['name']} {ag['doors_solved']}/{ag['n_doors']} doors, {ag['n_episodes']} episodes, success {ag['success_rate'] * 100:.1f} %")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
