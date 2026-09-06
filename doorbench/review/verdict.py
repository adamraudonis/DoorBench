"""The verdict schema: strict enough that a malformed answer is a failure, not a silent pass.

A vision reviewer that returns prose, or a finding with a category nobody defined, must not be able to
turn into "0 findings, door clean".  ``parse_verdict`` therefore raises on anything it cannot map onto
the schema, and the caller records the failure per door.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .prompt import CATEGORIES, SCHEMA_VERSION, SEVERITIES

REQUIRED_FINDING_KEYS = ("category", "severity", "part", "description", "where")


class VerdictError(ValueError):
    """The model's answer is not a verdict."""


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def extract_json(text: str) -> dict:
    """The first complete JSON object in ``text``.

    Models are asked for bare JSON and usually give it.  A markdown fence or a sentence of preamble is
    tolerated - a hard failure there would throw away a perfectly good verdict - but anything that is
    not a JSON object is an error.
    """
    if not isinstance(text, str) or not text.strip():
        raise VerdictError("empty response")
    stripped = _FENCE.sub("", text).strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(stripped[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
        start = stripped.find("{", start + 1)
    raise VerdictError("no JSON object in response")


def _clean_finding(f: Any, idx: int) -> Dict[str, Any]:
    if not isinstance(f, dict):
        raise VerdictError(f"finding {idx} is not an object")
    missing = [k for k in REQUIRED_FINDING_KEYS if k not in f]
    if missing:
        raise VerdictError(f"finding {idx} is missing {', '.join(missing)}")
    cat = str(f["category"]).strip()
    if cat not in CATEGORIES:
        raise VerdictError(f"finding {idx} has unknown category {cat!r}")
    sev = str(f["severity"]).strip().lower()
    if sev not in SEVERITIES:
        raise VerdictError(f"finding {idx} has unknown severity {sev!r}")
    conf = f.get("confidence")
    try:
        conf = None if conf is None else max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = None
    return {"category": cat, "severity": sev, "part": str(f["part"])[:200],
            "description": str(f["description"])[:1200], "where": str(f["where"])[:200],
            "confidence": conf}


def parse_verdict(text: str, door_id: str, reviewer: str, model: str = "",
                  extra: Dict[str, Any] | None = None) -> dict:
    """Validate and normalise one verdict.  Raises ``VerdictError`` on anything malformed.

    ``ok`` is recomputed from the findings rather than trusted: a model that returns ``ok: true``
    alongside three blockers has contradicted itself, and the findings are the evidence.
    """
    obj = extract_json(text)
    if "findings" not in obj:
        raise VerdictError("verdict has no 'findings' key")
    raw = obj["findings"]
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise VerdictError("'findings' is not a list")
    findings = [_clean_finding(f, i) for i, f in enumerate(raw)]
    got_id = str(obj.get("door_id") or door_id)
    if got_id != door_id:
        raise VerdictError(f"verdict is for {got_id!r}, expected {door_id!r}")
    out = {
        "schema_version": SCHEMA_VERSION,
        "door_id": door_id,
        "ok": not findings,
        "summary": str(obj.get("summary") or "")[:600],
        "findings": findings,
        "reviewer": reviewer,
        "model": model,
    }
    if obj.get("ok") is not None and bool(obj["ok"]) != out["ok"]:
        out["ok_claimed_by_model"] = bool(obj["ok"])
    out.update(extra or {})
    return out


def counts(verdicts: List[dict]) -> Dict[str, Any]:
    by_cat: Dict[str, int] = {}
    by_sev: Dict[str, int] = {s: 0 for s in SEVERITIES}
    n_find = 0
    for v in verdicts:
        for f in v.get("findings", []):
            n_find += 1
            by_cat[f["category"]] = by_cat.get(f["category"], 0) + 1
            by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
    return {"n_doors": len(verdicts), "n_clean": sum(1 for v in verdicts if v.get("ok")),
            "n_findings": n_find, "by_category": by_cat, "by_severity": by_sev}
