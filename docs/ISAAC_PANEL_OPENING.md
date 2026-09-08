# Contact-preserving completion of initialized opening

This is a development extension of the [initialized H1 handle demo](ISAAC_HANDLE_DEMO.md). It remains a privileged motor teacher on Door55, with the original free-base H1, Shadow Hands, joint stops, motor limits and passive door. It is not approach, traversal or a vision/tactile policy.

**Known v1 hardware-model failure:** the preserved hand lacks the manufacturer's passive finger loopback constraints, distal J1 ≤ middle J2. [Recorded Isaac grasps violate this relationship by up to 1.383 rad](../results/dexterous/2026-09-08/isaac-shadow-v1-loopback-audit.json). This bounded panel experiment remains evidence about controller behavior on that older simulator configuration. It cannot qualify Shadow hardware fidelity; no broad repetitions or training should use it before the versioned model correction and grasp refit.

The earlier teacher begins release at about 0.257 rad and withdraws near 0.49 rad, before its 0.7 rad opening threshold. Some repeats coast open; a retained camera repeat stopped at 0.449 rad. The new `--panel-push` option releases the lever, opens the fingers, moves across in free space, and reacquires the panel nearer the hinge. The palm follows measured door motion and applies pressure through bounded arm motors. It withdraws after measured panel contact reaches 1.2 rad. No controller forces or pose changes are applied to the door or free base.

Acceptance retains all 18 original opening checks and adds all-joint stop checking at 500 Hz, complete robot/scene collision checking at 50 Hz and unchanged plant-property readbacks. The independent panel audit requires at least 0.3 rad of forward travel under actual panel load and contact across the usable-aperture threshold. Coasting, brushing an already open leaf, or merely touching near the threshold cannot pass.

## Retained development attempts

| Attempt | Observed result | Qualification |
|---|---|---|
| Wider fixed-grasp kinematic fit | A continuous 0.85 rad arm workspace fit numerically | Rejected: substantial robot/wall collision; never run on GPU |
| Native original teacher | Opened 0.408 rad, then lost stance | Failed; direct 50 Hz analytic stance port is not native parity |
| Native panel teacher, direct stance port | Opened 0.409 rad, then lost stance | Failed |
| Native panel teacher, native 500 Hz stance | Recontacted panel and opened to 1.597 rad; maximum torso tilt 2.07° | Development only: replay found finger joint-stop overshoot up to 0.151 rad |

The last native attempt's sampled non-foot penetration was 0.73 mm and its body/hand replay was visually inspected. The joint-stop failure overrides its coarse opening pass. By comparison, saved Isaac H1 sensor repeats 001/002 show sampled all-joint overshoot around 1e-6 rad. These measurements do not establish cross-engine equivalence.

Native attempts, rejected geometry, source, states and inspected images are archived at `~/Desktop/Projects/DoorBench-runs/2026-09-08-robust-opening/native-development-001.tar.gz` (SHA256 `8dc0b4c81b4aac4b8ae617c97bf7d06ea3d58ae69a1eb214e7a81412c94f5c63`). Live Isaac verification remains pending.

For an isolated prepared-environment trial, use the usual `isaac_opening.py` arguments plus `--panel-push --seconds 16`. Run `audit_isaac_opening.py` first, then `audit_panel_opening.py --trial PATH`; retain both reports and `mechanical-audit.json`. The ready wrapper's default remains the historical teacher until the extension passes live verification.
