"""Benchmark scenarios: which DoorEnv task preset each door is reset to, the time budget, and the success predicate.

Success is defined from the episode labels (`doorbench.benchmark.labels.EpisodeLabels`) the way the benchmark
describes it: the robot traversed the pass plane without damaging the door within the time budget; for close
scenarios the door must also be closed again (and not slammed).  "Unlock" is evidenced by the door opening: a door
whose lock still holds cannot be opened without a damage event (forced maglock / sheared latch), so the
`unlock_open_traverse` predicate does not additionally require the `lock_released` label (several lock kinds with a
robot-side release have no separate release part to move).

If a door's `spec.json` carries a `benchmark` block with `time_budget_s` (per-door scenario spec landed by the
benchmark-spec workstream), that budget overrides the scenario default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# tasks whose success needs the base on the far side
TRAVERSE_TASKS = ("open_and_traverse", "unlock_open_traverse", "push_through", "hold_and_pass", "traverse_open")
CLOSED_THR_HINGE = math.radians(3.0)
CLOSED_THR_SLIDE = 0.03


@dataclass
class Scenario:
    name: str
    description: str
    task: str | None            # DoorEnv task preset; None = the door's own spec.task
    time_budget_s: float = 20.0
    require_closed: bool = False       # door must be closed again at the end (open-then-close)
    tags: list = field(default_factory=list)

    def task_for(self, spec: dict) -> str:
        return self.task or spec.get("task", "open_and_traverse")

    def budget_for(self, spec: dict) -> float:
        b = (spec.get("benchmark") or {}).get("time_budget_s")
        if b:
            return float(b)
        if spec.get("lock", {}).get("model") == "delayed_egress" and spec["lock"].get("engaged"):
            return max(self.time_budget_s, 40.0)      # IBC delayed egress: 15 s after the bar has been pushed for 3 s
        return self.time_budget_s

    def to_dict(self):
        return {"name": self.name, "description": self.description, "task": self.task or "spec.task", "time_budget_s": self.time_budget_s, "require_closed": self.require_closed}


SCENARIOS: dict[str, Scenario] = {
    "default": Scenario("default", "Each door's own task (spec.task: open_and_traverse, unlock_open_traverse, hold_and_pass, push_through, traverse_open, open_only, close, peek, locked_recognize); the leaderboard scenario.", None),
    "traverse": Scenario("traverse", "Every door reset closed / latched / locked as specified and the robot must open it and pass through (locked doors without a robot-side release are expected failures).", "open_and_traverse"),
    "traverse_close": Scenario("traverse_close", "Open, pass through, then close the door behind you without slamming it.", "open_and_traverse", require_closed=True),
    "hold_and_pass": Scenario("hold_and_pass", "Open, hold against the closer, pass, no slam (self-closing doors).", "hold_and_pass"),
}

SCENARIO_SETS = {"all": list(SCENARIOS)}


def parse_scenarios(arg: str) -> list[Scenario]:
    out = []
    for s in (x.strip() for x in arg.split(",") if x.strip()):
        names = SCENARIO_SETS.get(s, [s])
        for n in names:
            if n not in SCENARIOS:
                raise KeyError(f"unknown scenario {n!r}; known: {', '.join(SCENARIOS)}")
            if SCENARIOS[n] not in out:
                out.append(SCENARIOS[n])
    return out


def door_is_closed(door_q: float, is_hinge: bool) -> bool:
    return abs(door_q) < (CLOSED_THR_HINGE if is_hinge else CLOSED_THR_SLIDE)


def success_of(task: str, L: dict, scenario: Scenario, door_q_end: float, is_hinge: bool, goal_reached: bool = True) -> bool:
    """Success predicate over the final labels dict `L` (EpisodeLabels.to_dict()) for `task` under `scenario`.

    Traversal means the door opened (>= 10 deg / 0.1 m), the base crossed the pass plane AND reached the goal point:
    a robot leaning a few centimetres past the plane of a closed leaf, or a lock-limited slide that counts as "clear"
    at 95 % of a 2 mm range, is not a traversal.  Closing requires having touched the door (self-closing doors close
    on their own)."""
    if L.get("door_damaged") or L.get("robot_fell"):
        return False
    closed_end = door_is_closed(door_q_end, is_hinge)
    traversed = bool(L["robot_passed_through"] and L["door_opened"] and goal_reached)
    if task == "traverse_open":
        ok = traversed and not L["touched_door"]
    elif task in TRAVERSE_TASKS:
        ok = traversed
    elif task == "open_only":
        ok = bool(L["door_open_clear"] and L["door_opened"])
    elif task == "peek":
        ok = L["door_opened"] and not L["door_open_clear"] and not L["robot_passed_through"]
    elif task == "close":
        ok = closed_end and not L["door_slammed"] and L["touched_door"]
    elif task == "locked_recognize":
        ok = (not L["door_opened"]) and (not L["hardware_misuse"]) and L["touched_door"]
    else:
        ok = bool(L.get("success"))
    if scenario.require_closed:
        ok = ok and closed_end and not L["door_slammed"]
    return bool(ok)
