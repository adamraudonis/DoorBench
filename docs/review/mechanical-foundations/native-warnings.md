# Native solver warnings are failures

MuJoCo can emit `Linesearch objective is not convex` through its global warning
callback without incrementing `MjData.warning`. Checking the data counters alone
therefore allowed a numerically suspect run to look clean. This was a validation
blind spot, not evidence that the physical assembly was sound.

`doorbench.native_warnings.capture_native_warnings` captures the callback for a
whole QA run or benchmark episode, preserves an existing handler and restores it
when the scope exits. Nested captures are supported. Native workers use separate
processes; concurrent independent simulations in one process are outside this
callback's isolation guarantee.

Whole-door QA records the messages and fails `native_warning_messages_absent`.
The benchmark terminates with `native_failure` when counters, global messages or
non-finite states occur. An explicit mechanism-controller failure is separately
reported as `mechanism_failure`. Both outcomes are accepted by the result schema
and neither can count as success. Recordings retain the failed attempt.

Regression checks cover a real native instability, an injected callback message
with zero data counters, an explicit failed mechanism and nested callback
restoration. The callback-boundary injection tests the blind spot; it is not
claimed to reproduce the solver's internal linesearch algorithm.

The new gate caught the chain-hoist open initializer at 6.6115 simulated seconds.
A private full-implicit comparison completes, but convergence and all six
installed hoists must be verified before changing the source integrator. The
failed original initializer remains part of the evidence.
