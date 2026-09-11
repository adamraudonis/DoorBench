# Reproduce the standing-transfer experiment

This recipe reproduces the **H1/dual-Shadow privileged partial-opening and
receiving-palm experiment** used by run046. It is not a complete traversal,
a sensor-only policy, or a configuration for a different robot.

First provision the pinned environment using [the Isaac setup](ISAAC_ONE_CLICK.md)
and [GPU lifecycle instructions](RUNPOD.md). Keep the original robot, motor
contract, door and readiness receipt together. Arm an allocation teardown guard
or cluster job timeout. The recipe prints a command by default; it does not
provision or terminate GPU allocations.

On the compute node, supply these explicit inputs:

- `SOURCE`: immutable source snapshot with its verified `source-manifest.json`.
- `READY`: successful readiness receipt for the original robot/door assets.
- `REFERENCE`: the acquisition reference belonging to that original model.
- `ROUTE`: independently audited transfer route from a qualified measured state.
- `OUTPUT`: a new results directory.
- `WORK`: the environment workspace root.
- `DEADLINE`: Unix time earlier than the externally enforced teardown deadline.

```sh
python3 scripts/isaac/standing_transfer_recipe.py \
  --source "$SOURCE" --ready "$READY" --reference "$REFERENCE" \
  --route "$ROUTE" --output "$OUTPUT" --work "$WORK" \
  --deadline-unix "$DEADLINE"
```

Review the printed argument array, then add `--execute` to run it. The existing
coordinator retains its native prerequisite, independent contact audits,
original model and motor checks, and measured transfer acceptance checks.
The [profile](../configs/isaac/standing-transfer-h1-shadow-v1.json) supplies the
controller settings and fixed manipulation-eye configuration. The recipe's
output exactly matches the actual046 launch command
([comparison](evidence/isaac046-recipe-equivalence.json)); this is command
reproducibility, not a successful cross-cluster physical test.

For another cluster, regenerate readiness at its paths and independently
rebuild/audit the route against the relocated original assets. Old routes
contain hash-bound source paths: changing those strings by hand invalidates
the evidence chain. For another robot, regenerate its model, sensor and motor
contracts, qualify acquisition, and solve a new route for its actual embodiment.
Do not reuse H1 joint targets or assume a matching number of joints is sufficient.

Start an evidence collector before execution. Reserve storage for the complete
pipeline plus its atomic verification copy; retain the original failed results
as well as successful ones. See [storage policy](STORAGE_POLICY.md). Treat only
the final coordinator result and required independent audits as qualification.
