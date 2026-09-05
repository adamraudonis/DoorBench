"""Isaac parity gate: one behavioural protocol run in MuJoCo (reference) and Isaac Sim / PhysX, compared per door.

``doorbench.parity.protocol`` holds the protocol as data + pure functions (no simulator imports):
per-door inputs, the per-step drive schedule, the metrics computed from recorded curves, the pass/fail criteria,
the comparison tolerances and the verdict / discrepancy classification.  The runners are
``scripts/parity_reference_mujoco.py`` (CPU) and ``scripts/isaaclab/isaac_parity.py`` (GPU, Isaac Lab).
"""
from .protocol import (  # noqa: F401
    PROTOCOL_VERSION, PHASES, SAMPLE_HZ, CODES, door_inputs, expected_outcomes, phase_efforts, phase_duration,
    phase_initial_state, phase_metrics, phase_status, tendon_min_positions, servo_effort, compare_door, summarize,
)
