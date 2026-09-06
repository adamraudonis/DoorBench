"""Explicit controller memory for a bounded material-chain pull."""
from dataclasses import dataclass
import math


@dataclass
class HoistSpeedState:
    integral_force_N: float = 0.
    last_elapsed_s: float | None = None
    opening: bool | None = None


def speed_force(desired, measured, elapsed, limit, opening, state=None):
    """PI velocity feedback with conditional integration and a fixed force cap.

    State belongs to one controller/run, never the MuJoCo model or native data.
    Omitting it reproduces the original proportional controller for comparison.
    The integral learns the load from speed error, including as desired speed
    tends to zero near the height goal. It cannot bypass a jam or raise the cap.
    """
    if any(not math.isfinite(float(x)) for x in (desired,measured,elapsed,limit)) or elapsed<0 or limit<=0:
        raise ValueError('Finite velocity, nonnegative time and positive force limit required')
    error=desired-measured
    proportional=250.*error
    if state is None:
        return max(-limit,min(limit,proportional)),0.
    if not isinstance(state,HoistSpeedState):
        raise TypeError('HoistSpeedState required for stateful control')
    if state.opening!=opening or state.last_elapsed_s is None or elapsed<state.last_elapsed_s:
        state.integral_force_N=0.
        state.last_elapsed_s=float(elapsed)
        state.opening=opening
    dt=elapsed-state.last_elapsed_s
    raw=proportional+state.integral_force_N
    # Never integrate further into saturation. Reverse error can unload it.
    if abs(raw)<limit or (raw>=limit and error<0) or (raw<=-limit and error>0):
        state.integral_force_N=max(-limit,min(limit,state.integral_force_N+80.*error*dt))
    state.last_elapsed_s=float(elapsed)
    return max(-limit,min(limit,proportional+state.integral_force_N)),state.integral_force_N
