# Keypad codes must physically work (task G9b)

Read `handoffs/README.md` first. Partial work may exist on `worktree-agent-a68f30762fa4dd816` (see
`operator-spring-return.md`); otherwise start from `master` on a new branch.

## Why (owner's words)

"Ensure that some of the door codes with keypads actually work."

## Current state

* `doorbench/spec.py` (~line 144-160) gives keypad locks a `spec["lock"]["code"]`: 4 or 6 digits for
  `keypad_code_4/6`, a sorted set of 3-4 buttons for `keypad_mechanical`.
* `doorbench/geometry/common.py` builds a keypad face (`keypad_face` parameter; buttons are geoms in the full
  tier) and the lock body.
* `doorbench/benchmark/env.py` (~line 565-600) releases keypad / card / electric-strike locks when
  `L.lock_released` or `self.unlocked_by_env` is set (an API call), not from physical button presses.
* `doorbench/benchmark/scenarios.py` has the `unlock_and_traverse` scenario for locks with a robot-side release.

## Goal

1. Buttons are real pressable bodies (slide joints with return springs, FULL tier) laid out as a keypad on the
   correct face.
2. `DoorEnv` detects presses from the button joints (depth threshold, debounce) and releases the lock ONLY when
   the spec's code is entered in order within a timeout; wrong code -> no release, with a documented lockout /
   backoff for electronic keypads; mechanical push-button locks: the right set pressed, then the lever. Keep
   `env.enter_code()` as a convenience but make the physical path the real one. Labels `code_entered`,
   `wrong_code_attempts`.
3. The `unlock_and_traverse` scenario for keypad doors carries the code (e.g. `scenario["lock"]["code"]`),
   documented in `docs/BENCHMARK.md`, so a policy can enter it.
4. Tests: a programmatic finger presses the buttons: right code opens, wrong code does not, timeout resets.
5. Viewer: clicking the keypad buttons in the 3D view (or a small keypad panel) presses them; the lock panel
   shows the code (the dataset is open) and unlocks when the sequence is right.

## Files

`doorbench/geometry/common.py` (keypad), `doorbench/benchmark/env.py`, `doorbench/benchmark/scenarios.py`,
`doorbench/qa.py` (a `keypad_code_works` check on keypad doors), `tests/`, `viewer/src/DoorView.tsx`,
`docs/BENCHMARK.md`.

## Done when

Every keypad door (list them with `python -m doorbench list` / manifest `lock` field) passes the new QA check;
1000 signed off; clearance 1000/1000; tests green; viewer build clean; one screenshot of the keypad UI under
`docs/media/`.
