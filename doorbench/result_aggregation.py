"""Simulator-independent result aggregation shared by runs and historical subsets."""
import math

SCENARIO_TYPES = ("open_and_traverse", "open_then_close", "close_only", "unlock_and_traverse", "locked_recognize", "hold_open_for_human", "wait_for_human", "knock_and_wait")
SUITES = ("core", "human")
OUTCOMES = ("success", "fail", "damaged", "fell", "timeout", "error")

def _mean(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return round(sum(xs) / len(xs), 3) if xs else None


def _median(xs):
    xs = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not xs:
        return None
    n = len(xs)
    return round(xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]), 3)


def _group_stats(eps: list[dict]) -> dict:
    by_door = {}
    for e in eps:
        by_door.setdefault(e["door_id"], []).append(bool(e["success"]))
    n = len(eps)
    ns = sum(1 for e in eps if e["success"])
    pass_t = [e.get("time_to_pass") for e in eps if e["success"] and e.get("time_to_pass") is not None]
    hc = [e.get("human_collision") for e in eps if e.get("human_collision") is not None]
    out = {
        "n_doors": len(by_door), "n_episodes": n, "n_success": ns, "success_rate": round(ns / n, 4) if n else 0.0,
        "doors_solved": sum(1 for v in by_door.values() if all(v)), "doors_solved_any": sum(1 for v in by_door.values() if any(v)),
        "damage_rate": round(sum(1 for e in eps if e.get("damage")) / n, 4) if n else 0.0,
        "mean_return": _mean([e.get("episode_return") for e in eps]),
        "mean_time_to_pass_s": _mean(pass_t), "median_time_to_pass_s": _median(pass_t),
        "mean_time_to_open_s": _mean([e.get("time_to_open") for e in eps if e.get("time_to_open") is not None]),
        "mean_max_leaf_force_N": _mean([e.get("max_leaf_force_N") for e in eps]),
        "mean_energy_J": _mean([e.get("energy_J") for e in eps]),
        "outcomes": {k: sum(1 for e in eps if e.get("outcome") == k) for k in OUTCOMES if any(e.get("outcome") == k for e in eps)},
    }
    if hc:
        out["human_collision_rate"] = round(sum(1 for x in hc if x) / len(hc), 4)
    return out


def lock_state_of(door: dict) -> str:
    if not door.get("lock_engaged"):
        return "unlocked"
    return "locked_releasable" if door.get("robot_side_release", True) else "locked_no_release"


def aggregate_suite(suite: str, episodes: list[dict], doors_by_id: dict | None = None) -> dict:
    """One table of the aggregate: every episode of `suite` (errors excluded from the rates, counted separately)."""
    eps_all = [e for e in episodes if e.get("suite") == suite]
    eps = [e for e in eps_all if e.get("outcome") != "error"] or eps_all
    agg = {"suite": suite, "scenarios": sorted({e["scenario"] for e in eps_all}, key=SCENARIO_TYPES.index), **_group_stats(eps)}
    agg["n_errors"] = sum(1 for e in eps_all if e.get("outcome") == "error")
    agg["timeouts"] = sum(1 for e in eps_all if e.get("outcome") == "timeout")
    agg["mean_wall_s"] = _mean([e.get("wall_s") for e in eps_all])
    agg["mean_sim_time_s"] = _mean([e.get("sim_time") for e in eps])

    def group(key, order=None):
        g = {}
        for e in eps:
            g.setdefault(str(key(e)), []).append(e)
        keys = sorted(g, key=(lambda k: order.index(k) if order and k in order else 10 ** 6) if order else None)
        return {k: _group_stats(g[k]) for k in keys}
    agg["by_family"] = group(lambda e: e["family"])
    agg["by_difficulty"] = group(lambda e: e.get("difficulty"))
    agg["by_scenario"] = group(lambda e: e.get("scenario"), list(SCENARIO_TYPES))
    if doors_by_id:
        agg["by_lock_state"] = group(lambda e: lock_state_of(doors_by_id.get(e["door_id"], {})), ["unlocked", "locked_releasable", "locked_no_release"])
    if suite == "human":
        agg["human_collision_rate"] = agg.get("human_collision_rate", 0.0)
    return agg


def aggregate(episodes: list[dict], doors_by_id: dict | None = None) -> dict:
    """{suite: table}: core and human episodes are aggregated separately and never mixed."""
    present = [s for s in SUITES if any(e.get("suite") == s for e in episodes)]
    return {s: aggregate_suite(s, episodes, doors_by_id) for s in present}
