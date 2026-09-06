# Scripted mechanism recordings

Current corrections are reviewed against `doorbench.native-motion.v1` recordings.
Each clip contains actual MuJoCo joint states, every commanded interaction site,
point-force vectors, and native cable tangencies. The companion NPZ includes
qpos, qvel, motor controls, applied generalized efforts and world body poses.
The simulation clock is preserved. No humanoid is fitted and no physics state is
interpolated between samples in the viewer.

This is an idealized mechanism oracle. Joint efforts are not contact forces from
a simulated hand; site forces have a 120 N cap but no grasp model. Simultaneous
markers can represent inputs that one person cannot reach. The abstract base is
not a collision-dynamic human. These clips must not be called human ground truth
or used as validated humanoid retargeting targets. Failed attempts are retained.

```sh
PYTHONPATH=. python -m doorbench.reference.record \
  --assets path/to/assets --out path/to/reference-motions \
  --native-only --doors all --workers 4 --fps 30 --wall-timeout 600
PYTHONPATH=. python scripts/validate_reference_motions.py \
  --assets path/to/assets --root path/to/reference-motions
```

The validator checks coverage of the eligible manifest, exact source hashes,
all recorded body poses and contact positions, and correspondence between the
browser clip and native arrays. Pet flaps remain supplementary and are excluded.
This validates recording integrity; separate mechanism and dynamic gates remain
necessary.

Free mechanism roots retain all seven position coordinates and six velocities.
The browser uses recorded world poses for every native body, including actual
circulating chain links. It does not treat a free root as one scalar angle.
Expensive native open-state initialization is timed separately from the episode;
the wall timeout is configurable independently of the unchanged scenario clock.

The older `doorbench.reference-motion.v1` format remains readable for archived
comparisons. Its illustrative actor is not regenerated in this review.

## Changes to task interpretation

Closet openings and hatches use opening/closing tasks when standing passage is
not supported by their geometry. `open_only` requires the scenario's full opening
target; a small initial movement does not count. A locking hatch stay must also
engage. Bypass operation selects one panel, and bifold banks operate sequentially.
A recessed pocket door must deploy its edge pull before extraction.

The authored `jam_stuck` examples represent elevated Coulomb breakaway friction,
not an immovable security lock. Their tasks now attempt opening and retain any
failure within the available effort. They do not earn success by declaring a
lock that does not exist.

Energy labels now integrate commanded generalized effort and actuator work
before applied forces are cleared. Constraint reaction is not added as another
actuator. These corrected labels and tasks are not comparable to the archived
benchmark scores without rerunning the benchmark.

The passage band includes the authored closed moving stock and a 0.6 m front/back
body extent, with a 0.5 m width and 1.8 m standing height. This fixes a false
positive for rolling curtains offset behind the wall. In the source-bound
DB0419 integration trial, clearance first becomes available at 16.14 native
seconds and the base crosses at 18.216 seconds. The 103 sampled frames match the
native arrays. This remains a synthetic-base task, not a human demonstration.

Manual joint efforts require an approach-side authored contact. A missing or
far-side mechanical release no longer triggers an invented badge action; that
API is limited to credential-controlled lock classes. A short, unsuccessful
chain-driven attempt is deliberately tested to remain blocked until the curtain
clears a standing traveller.
