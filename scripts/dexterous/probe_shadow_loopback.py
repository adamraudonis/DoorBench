#!/usr/bin/env python3
"""Measure unilateral compliance in a contact-free two-link Shadow fixture.

External equal/opposite torques excite this isolated mechanism. They are NOT
allowed assistance in robot-task qualification. No actuator exists here.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from doorbench.dexterous.shadow_loopback import add_loopbacks


FIXTURE_XML = '''<mujoco><compiler angle="radian"/>
<option timestep=".002" gravity="0 0 0" integrator="implicitfast"/>
<worldbody><body name="middle">
<joint name="rh_FFJ2" type="hinge" axis="1 0 0" range="0 1.5708"
       armature=".0002" damping=".05" frictionloss=".01"/>
<inertial pos="0 0 .0125" mass=".017" diaginertia=".0000027 .0000026 .00000087"/>
<geom size=".004" contype="0" conaffinity="0"/>
<body name="distal" pos="0 0 .025">
<joint name="rh_FFJ1" type="hinge" axis="1 0 0" range="0 1.5708"
       armature=".0002" damping=".05" frictionloss=".01"/>
<inertial pos="0 0 .0130769" mass=".013"
          diaginertia=".00000128092 .00000112092 .00000053"/>
<geom size=".004" contype="0" conaffinity="0"/>
</body></body></worldbody>
<tendon><fixed name="rh_FFJ0"><joint joint="rh_FFJ1" coef="1"/>
<joint joint="rh_FFJ2" coef="1"/></fixed></tendon></mujoco>'''


def fixture_spec(time_constant):
    spec = mujoco.MjSpec.from_string(FIXTURE_XML)
    if time_constant is not None:
        add_loopbacks(spec, spec.compile(), hands=['rh'], digits=['FF'])
        # A fixture-only override preserves the failed 20 ms prototype as a
        # comparator; the generated robot uses its versioned profile unchanged.
        next(t for t in spec.tendons if t.name == 'rh_FF_loopback').solref_limit = [time_constant, 1]
    return spec


def measure(model, initial_j1_j2, load):
    data = mujoco.MjData(model)
    ids = [model.joint(name).id for name in ('rh_FFJ1', 'rh_FFJ2')]
    qa, va = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    data.qpos[qa] = initial_j1_j2
    mujoco.mj_forward(model, data)
    samples = []
    for _ in range(750):
        torque = load * np.clip((data.time - .5) / .2, 0, 1)
        data.qfrc_applied[va] = [torque, -torque]
        mujoco.mj_step(model, data)
        samples.append([data.time, *data.qpos[qa], torque, *data.qfrc_constraint[va]])
    samples = np.asarray(samples)
    summary = {
        'initial_J1_J2': initial_j1_j2, 'load_Nm': load,
        'max_difference_rad': float(np.max(samples[:, 1] - samples[:, 2])),
        'final_difference_rad': float(samples[-1, 1] - samples[-1, 2]),
        'final_J1_J2': samples[-1, 1:3].tolist(),
        'max_constraint_Nm': float(np.max(np.abs(samples[:, 4:]))),
        'steady_constraint_Nm': samples[-1, 4:].tolist(),
    }
    return samples, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'base.xml').write_text(FIXTURE_XML)
    results = []
    for tc in (None, .02, .004):
        spec = fixture_spec(tc)
        model = spec.compile()
        name = 'none' if tc is None else str(tc)
        (args.output / f'fixture-{name}.xml').write_text(spec.to_xml())
        for label, start, load in (('slack', [.2, .6], 0.),
                                   ('ramp', [.2, .6], 1.),
                                   ('active', [.6, .6], 1.)):
            samples, summary = measure(model, start, load)
            np.savez_compressed(args.output / f'{name}-{label}.npz', samples=samples)
            row = dict(time_constant_s=tc, case=label, **summary)
            results.append(row)
            print(json.dumps(row), flush=True)
    report = {
        'scope': 'Passive-mechanism fixture; external excitation is not a robot task',
        'mujoco': mujoco.__version__, 'physics_dt_s': .002,
        'sample_columns': ['t', 'J1', 'J2', 'applied_distal_Nm',
                           'constraint_J1_Nm', 'constraint_J2_Nm'],
        'schedule': '0 Nm until 0.5 s; ramp to 1 Nm over 0.2 s; hold until 1.5 s; J1 positive, J2 negative',
        'robot_terms': 'Original Shadow middle/distal masses, COMs and inertias; armature 0.0002, damping 0.05, frictionloss 0.01',
        'results': results,
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
