# Human reference motion before robot retargeting

DoorBench's target is a natural simulated human opening a door, followed by separate retargeting to humanoid robots. The slow procedural figure shown in the experimental Motion Lab does not meet that target. Its incremental gait/IK tuning was stopped after the owner's September 5 review. Existing trajectories and audits are retained as research evidence; the unpublished second corpus is not selected for release.

## First deliverable

Build one convincing, clothed human performance in Blender at natural speed, starting with the captured task's mechanism. Then adapt it to DoorBench fixtures, including the requested `db0002_swing_single`. That door needs a knob turn and pull; a recording of lever depression and pushing is not its ground truth. Start from genuine human motion or a comparably strong human animation prior, with an anatomical skinned character, articulated fingers, feet and toes. Preserve body rhythm, continuous stepping, weight transfer and reaching; do not manufacture style by slowing a sequence of independently solved robot poses.

The first complete animation must make the approach, grasp, mechanism operation, door swing, passage and release readable. Inspect normal-speed playback from several views before expanding to more doors. A nicer mesh alone does not satisfy this milestone.

## Architecture

1. **Human performance source.** Preserve the source capture, original timing, skeleton, anthropometry, coordinate conventions, rights and any measured contact or prop channels. Separate captured observations from inferred or authored corrections.
2. **Human and door interaction.** Register the performance to the native door's actual geometry and articulation. Use explicit palm/finger contact frames, support-foot phases and mechanism events. Adapt the human and task timing together; do not force the human to follow the previous scripted-hand recording's clock.
3. **Blender character and scene.** Use a reusable anatomical human rig with authored skin weights and ordinary clothing. Reuse the modular door, wall, floor and lighting system. Character deformation and rendered observations must refer to the same exported human pose and native door state.
4. **Independent evidence.** Check human proportions, floor/door/self clearance, planted-foot drift, hand orientation and grip, mechanism sequencing and temporal continuity. Check dynamic balance and contact/actuation feasibility separately when claiming physical execution. Preserve failures and uncertainty instead of hiding them with retiming or rendering.
5. **Robot retargeting.** Treat each robot as a downstream embodiment with its own joint limits, reach, hands, collision model and controller. Preserve the human's interaction intent and contact events, while validating the resulting robot motion independently.

## Quality gate before scaling

The first demonstration needs natural normal-speed playback, believable full-body movement, sustained grasp contact, correct latch/lock/leaf behavior and a clean passage through the aperture. It must be inspected as a complete animation, not approved from sparse phase sheets or a single aggregate metric. Then test a small varied set of mechanisms before attempting all 1,000 doors.

Simulation state can provide exact pose, articulation and rendering labels. That alone does not make an animation physically executable or a perfect model of human behavior. Until the relevant visual, geometric and physical checks pass, call an output a **human reference candidate**, rather than ground-truth human operation.

Generated human assets and motion must retain their actual redistribution terms. Tool-code licenses, mesh/rig licenses, capture-data licenses and learned-model licenses are checked separately. Blender is the character/rendering tool; Unity and Unreal are outside this work. The other agent's Isaac/RunPod node remains untouched.

## Current state

The old procedural tuning is stopped. A clothed 1.75 m MakeHuman character with 163 bones, authored skinning and packed textures has been built and visually inspected in Blender. The canonical mesh/rig/textures are CC0; the external MPFB build tool's GPL license is separate.

The first motion transfer uses an actual right-hand lever-door performance from [CeTI-Age-Kinematics v2](https://doi.org/10.6084/m9.figshare.26983645.v2), by Pogrzeba and colleagues, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its 762 motion samples retain the original 100 Hz clock and 7.61-second span; only two documented leading calibration rows are omitted. The source has full-body and finger rotations, but no measured door trajectory. Door placement, contact and any adaptation remain separate inferred channels. Target toes and extra finger segments do not become independently captured merely because the target rig has more joints.

A second source, [MeLLO](https://github.com/nluttmer1/MeLLO-Data-Library), provides human and door markers in the same measured scene under CC BY 4.0. The selected 6.87-second sample uses a fixed instrumented handle and closer. It cannot demonstrate knob or lever unlatching. Its measured leaf trajectory is preserved separately; force synchronization/calibration and a subject-scaled human reconstruction still need work. These two recordings must not be spliced together and described as one measured interaction.

No complete human–DoorBench interaction or robot retarget has passed the quality gate yet. Hugging Face remains on the existing published release, with updates batched at most once per day when a reviewed revision is ready.

## Reproduce the local Blender preview

The original capture, character packages, Blender files and rendered media stay under ignored `out/`. The fetcher verifies pinned hashes, downloads only the selected capture and its license from the larger archive, and isolates Blender's external MPFB extension from the normal user profile.

```sh
python scripts/setup_human_reference.py build

BLENDER_USER_RESOURCES="$PWD/out/toolcache/mpfb/blender-user" \
  /Applications/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_human_capture.py -- \
  --source out/human-reference/source/sub-d02_ses-02_task-o03_tracksys-rokokosmartsuit1_run-01_motion.bvh \
  --human out/human-reference/assets/human-preview.blend \
  --calibration out/human-reference/assets/tpose-calibration.json \
  --out out/human-reference/ceti-d02-o03 --glb --video

bun run --cwd viewer dev
```

Open `#/human-reference` on that development server. The **Blender render** is the primary appearance preview; **3D inspection** supports orbiting, following the human, source-time scrubbing and normal-speed playback. Set `DOORBENCH_HUMAN_REFERENCE_ROOT` to another export of this raw CeTI sample, or to the explicitly supported `out/human-reference/ceti-d02-o03-contact-fit-v2` candidate; relative paths resolve from the repository root. The latter is labeled “Captured motion · legs fitted to character”: its authored leg changes preserve the original clock, pelvis and upper-body transfer. The viewer checks the reviewed pose binding and adjustment report; this does not certify the complete interaction or dynamics. Other candidate stages are rejected until explicitly supported. These files are not bundled into the public site.

Use `--blender /path/to/blender` for the setup command on another platform, and the same executable for capture baking. `--no-render` skips still-image rendering during setup; the capture command also supports it for a quick `.blend`/NPZ/GLB export. Its full Blender video is sampled at 30 fps, with decoder time distinguished from the retained 100 Hz source clock. glTF reduces skinning to four influences and approximates the Blender shaders; its appearance is a convenience preview rather than the reference renderer.

A fresh build reproduced the rig metadata and all calibration matrices exactly. The resulting raw motion time, bone positions, rotations, local transforms and pelvis arrays also matched the first export exactly. Every baked pose was checked against Blender's evaluated skeleton. These are reproducibility and transform checks, not human interaction acceptance.

The separate **legs fitted to character** candidate restores the source ankle spacing and makes small, explicitly authored leg corrections for the target shoes and limb lengths. Its clock, pelvis, upper body and absolute foot rotations remain unchanged. Independent checks found no floor penetration in any of its 762 sampled rendered poses and no shoe overlap in the 61-sample crossing window. Those results cover the shoes at sampled times; door contact, full-body clearance, balance and dynamics remain unvalidated. The original transfer and an earlier rejected correction are retained separately. `doorbench.human_reference.contact_fit` implements the correction; `scripts/blender_human_contact_fit.py` bakes and measures a report-bound candidate.

## Research informing the next stage

Copying joint rotations alone does not preserve interactions across different bodies. The current transfer exposes that directly through narrower crossing steps and displaced hands. [OmniRetarget](https://omniretarget.github.io/) preserves spatial/contact relationships while enforcing robot kinematic constraints; that is relevant to the downstream robot stage. [InterMimic](https://sirui-xu.github.io/InterMimic/) refines imperfect human–object capture through subject-specific simulated teachers before scaling to a general controller. [MaskedMimic](https://research.nvidia.com/labs/par/maskedmimic/) provides a route to generating physical human motion from partial motion constraints.

The practical sequence here is therefore: establish a believable captured-human example, fit its actual contact geometry, verify simulated execution, then broaden mechanisms and robot embodiments. Those research systems are references for the architecture, not installed DoorBench capabilities or evidence that this candidate already passes their evaluations.
