"""Dataset index helpers (pure Python, no Isaac imports): locate the DoorBench assets, list doors, curate subsets.

The extension needs to know where ``assets/doors/<id>/door_rl.usda`` lives.  Resolution order:
  1. ``DOORBENCH_ASSETS`` environment variable (path to the ``assets`` directory)
  2. ``<repo>/assets`` relative to this file (editable install inside the DoorBench checkout)
  3. ``~/DoorBench/assets``
"""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache

from doorbench.benchmark_eligibility import is_benchmark_eligible, require_benchmark_eligible

_HERE = os.path.dirname(os.path.abspath(__file__))

# operator kinds a simple end-effector can work without grasping (press / push / pull-through)
EASY_OPERATOR_PREFIXES = ("lever", "push_plate", "pull", "panic_touchbar", "panic_crossbar", "paddle", "none", "shoji_finger_pull", "push_button_screen")
EASY_FAMILIES = ("swing_single", "automatic_swing", "automatic_sliding", "saloon", "swing_double", "sliding_single", "pivot", "stall", "gate_swing", "strip_curtain")
EASY_TASKS = ("open_and_traverse", "open_only", "push_through", "hold_and_pass", "traverse_open", "peek")


def assets_root() -> str:
    env = os.environ.get("DOORBENCH_ASSETS")
    cands = [env] if env else []
    cands += [os.path.abspath(os.path.join(_HERE, "..", "..", "assets")), os.path.expanduser("~/DoorBench/assets")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "manifest.json")):
            return c
    raise FileNotFoundError("DoorBench assets not found: set DOORBENCH_ASSETS=/path/to/DoorBench/assets (the directory holding manifest.json)")


@lru_cache(maxsize=4)
def manifest(root: str | None = None) -> dict:
    root = root or assets_root()
    with open(os.path.join(root, "manifest.json")) as f:
        return json.load(f)


def door_dir(door_id: str, root: str | None = None) -> str:
    return os.path.join(root or assets_root(), "doors", door_id)


def usd_path(door_id: str, canonical: bool = True, root: str | None = None) -> str:
    """Path of the door USD: ``door_rl.usda`` (canonical 7-DoF, for multi-door training) or ``door.usda`` (full)."""
    p = os.path.join(door_dir(door_id, root), "door_rl.usda" if canonical else "door.usda")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return p


def all_ids(root: str | None = None, signed_off_only: bool = True) -> list[str]:
    return [d["id"] for d in manifest(root)["doors"] if (d.get("signed_off", True) or not signed_off_only)]


def load_spec(door_id: str, root: str | None = None) -> dict:
    with open(os.path.join(door_dir(door_id, root), "spec.json")) as f:
        return json.load(f)


def load_model(door_id: str, root: str | None = None) -> dict:
    with open(os.path.join(door_dir(door_id, root), "model.json")) as f:
        return json.load(f)


def is_easy(row: dict) -> bool:
    """Manifest row -> member of the curated easy set (unlocked, no keypad/card, simple operator, RL-friendly task)."""
    if not row.get("signed_off", True):
        return False
    if row.get("lock_engaged") or row.get("lock") not in ("none", "privacy_button", "thumbturn_only"):
        return False
    if row.get("family") not in EASY_FAMILIES:
        return False
    if not any(row.get("operator", "").startswith(p) for p in EASY_OPERATOR_PREFIXES):
        return False
    if row.get("task") not in EASY_TASKS:
        return False
    if row.get("condition") in ("jammed", "swollen", "damaged"):
        return False
    if (row.get("mass_kg") or 0) > 120:
        return False
    return True


def easy_ids(n: int = 100, seed: int = 0, root: str | None = None) -> list[str]:
    """Curated 'easy-N' door list: balanced over families, deterministic for a seed."""
    rows = [d for d in manifest(root)["doors"] if is_easy(d)]
    rng = random.Random(seed)
    rng.shuffle(rows)
    # round-robin over families so the subset is not 90 % swing_single
    by_fam: dict[str, list] = {}
    for r in rows:
        by_fam.setdefault(r["family"], []).append(r)
    order = sorted(by_fam, key=lambda f: -len(by_fam[f]))
    out = []
    while len(out) < n and any(by_fam.values()):
        for f in order:
            if by_fam[f] and len(out) < n:
                out.append(by_fam[f].pop()["id"])
    return sorted(out)


def select_ids(spec: str, root: str | None = None, seed: int = 0, *, benchmark_only: bool = True) -> list[str]:
    """Door selection strings used by the CLI / env cfg:

    ``all``            every signed-off benchmark-eligible door (985)
    ``benchmark_only=False`` is reserved for physical asset QA/parity, not policy evaluation.
    ``easy`` / ``easy-100`` / ``easy-300``   curated easy subset of that size
    ``family:<name>``  every door of one family (comma separate several)
    ``db0002_swing_single,db0016_swing_single``  explicit ids
    ``@file.txt``      one id per line
    ``random-50``      50 random doors (seeded)
    """
    spec = spec.strip()
    rows = manifest(root)["doors"]
    eligible = {r["id"] for r in rows if is_benchmark_eligible(r)}
    def filtered(ids):
        return [i for i in ids if not benchmark_only or i in eligible]
    if spec == "all":
        return filtered(all_ids(root))
    if spec.startswith("easy"):
        n = int(spec.split("-")[1]) if "-" in spec else 100
        return easy_ids(n, seed=seed, root=root)
    if spec.startswith("random-"):
        ids = filtered(all_ids(root))
        return sorted(random.Random(seed).sample(ids, min(int(spec.split("-")[1]), len(ids))))
    if spec.startswith("@"):
        with open(spec[1:]) as f:
            ids = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        return select_ids(",".join(ids), root=root, seed=seed, benchmark_only=benchmark_only)
    if spec.startswith("family:"):
        fams = set(spec[len("family:"):].split(","))
        if benchmark_only:
            for family in fams:
                require_benchmark_eligible(family, operation="Isaac Lab benchmark selection")
        return [d["id"] for d in manifest(root)["doors"] if d["family"] in fams and d.get("signed_off", True)]
    ids = [s for s in spec.split(",") if s]
    known = set(all_ids(root, signed_off_only=False))
    bad = [i for i in ids if i not in known]
    if bad:
        raise KeyError(f"unknown door ids: {bad[:5]}")
    if benchmark_only:
        require_eligible_ids(ids, root=root)
    return ids


def require_eligible_ids(ids: list[str], root: str | None = None) -> None:
    """Guard evaluation config lists and actual source specs before USD spawning.

    Raw USD lookup/all_ids/load_spec deliberately remain usable for asset QA.
    """
    by_id = {row["id"]: row for row in manifest(root)["doors"]}
    for door_id in ids:
        if door_id not in by_id:
            raise KeyError(f"unknown door id: {door_id}")
        require_benchmark_eligible(by_id[door_id], operation="Isaac Lab evaluation/training")
        require_benchmark_eligible(load_spec(door_id, root=root), operation="Isaac Lab evaluation/training")


def door_usd_paths(ids: list[str], canonical: bool = True, root: str | None = None) -> list[str]:
    return [usd_path(i, canonical=canonical, root=root) for i in ids]
