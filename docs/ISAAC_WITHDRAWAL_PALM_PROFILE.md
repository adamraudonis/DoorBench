# Prospective withdrawal palm-load profile

The default actual-Isaac withdrawal retains its source-bound palm target for
the entire stage. An optional outer runtime document field selects a distinct
prospective experiment:

```json
"withdrawal_palm_load_profile": "release-unload-restore-v1"
```

This fixed profile requires the qualified predecessor's target and hybrid
maximum to be exactly 6 N. Unknown names, null, numeric settings and other
types are rejected. Its duration comes from the independently admitted source
route and must exceed 1.5 seconds. No new duration or arbitrary force parameter
can be supplied through the option.

The active hybrid target stays at 6 N through entry and the existing actual
intentional-release qualification. When `StandingWithdrawalTeacher` sets its
actual `release_started`, the target follows a quintic smoothstep from 6 to
2.5 N over 0.5 seconds. It then stays at 2.5 N until the final one second of the
admitted withdrawal duration, when a second quintic smoothstep restores 6 N.
The first release command still requests 6 N. Values and first derivatives
are continuous at the boundaries. A late release whose unloading and restore
ramps would overlap is rejected before feedback update; the event cannot be
backdated, changed or erased. If release has not qualified, the request stays
at 6 N.

The same inherited `StandingSupportFeedback` object, contact surface, filter,
physical epoch and hybrid blend are retained. `left.support_load_target`,
`inherited_support.target_N` and `feedback.maximum_target_N` stay at their
source value of 6 N. Only the numeric target passed to its existing `update`
changes. No filter reset, reference rewrite, state installation, kinematic
solve, motor-cap change or physical-gate change is introduced.

The `inherited_support` telemetry keeps the original `target_N` as source
identity. Additional profile fields record `active_target_N`,
`inherited_source_target_N`, `inherited_maximum_target_N`, `palm_load_phase`,
the latched actual release epoch, restore/completion epochs and ramp durations.
The active value is checked against the value stored by the real feedback
update. This target is not an upper bound on measured contact force: the
existing feedback also reacts to actual load and motion. It is not a leaf-speed
limit and has no physical success claim before a new experiment.

Omitting the option retains the old update call and default numeric path.
CPU tests verify unchanged predecessor commands and first withdrawal handoff,
the real inherited feedback history, request propagation, all schedule phases,
continuous boundaries, missing/changed/late release rejection, invalid options
and derived-clock overflow. These tests establish software behavior only.
