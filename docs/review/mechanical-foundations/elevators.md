# Elevator door assemblies

Eight generic horizontal landing-door assemblies now use actual hanger wheels,
rail/sill guides, storage space, terminal stops and a hook/bar interlock operated
by a spring-return retiring cam. The former2mm leaf-range restriction is gone.
The original design uses the retiring-cam/interlock topology described by
[C. J. Anderson](https://cjanderson.com/lr-use-with-retiring-cam/); it does not
reproduce that manufacturer's CAD or a certified elevator installation.

The native drive is bounded to135N. Calling the door seats the leaves, withdraws
the hook, opens, holds, closes and re-engages the hook. Unpowered motors apply
zero force. An actual physical call/REX press is required; restoration of power
is not a new call. Presence holds the opening, and measured obstruction load
causes reopening. Electrical position/presence logic and the paired drive are
explicit idealizations.

All8×3 native component cases complete two cycles with removed-cam and removed-
hook counterexamples. All8 full-geometry sweeps pass. Production XML matches the
proven models in every tier. Runtime tests cover all8 normal cycles, physical
obstruction and reopening, close-only initial state, presence holding, power
loss and actual call/REX inputs. Passing cases retain native solver warnings
and contact penetration below1mm as failure gates.

Evidence: `out/mechanical-foundations/elevator-interlocks/`, including
`promoted-native-parity.json`, and `tests/test_elevator_mechanics.py`.
These are stationary level-car component/control proofs. Moving-car operation,
structural strength, impact ratings, human reach, safety integrity and regulatory
compliance are not certified.
