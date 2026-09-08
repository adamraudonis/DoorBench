#!/usr/bin/env python3
"""Check complete state admission on recorded Isaac data, without simulation."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from doorbench.dexterous.destination_state_binding import freeze_destination_state,admit_destination_state,ROOT_CONVENTION
from doorbench.dexterous.sensor_contract import SENSOR_KEYS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    if args.output.exists():raise FileExistsError('Retain the prior check; choose a new output')
    files=[run/name for name in ('configuration.json','motor-contract.json','provenance.json',
        'balance-steps.json.gz','acquisition-physics.npz','sensors/layout.json','sensors/actor-sensors.npz')]
    hashes={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files}
    config=json.loads(files[0].read_text());motors=json.loads(files[1].read_text())
    provenance=json.loads(files[2].read_text());steps=json.loads(gzip.decompress(files[3].read_bytes()))
    layout=json.loads(files[5].read_text());names=layout['joint_order']
    if set(names)!=set(motors['joint_names']) or len(names)!=69:raise ValueError('Actual named sensor layout differs')
    if config.get('balance_root_state_convention')!='balance-steps uses root_link_state_w: actor-origin pose and world actor-origin linear/angular velocity; evaluator only':
        raise ValueError('Explicit actor-origin root evidence required; legacy mixed COM state is rejected')
    door_source=config['args']['door_usd'];door_sha=provenance['files'][door_source]
    results=[]
    with np.load(files[4],allow_pickle=False) as physics,np.load(files[6],allow_pickle=False) as sensors:
        clock=np.asarray([row['time_s'] for row in steps])
        if not np.array_equal(clock,physics['time_s']) or not np.array_equal(clock,sensors['time_s']):
            raise ValueError('Measured state archives have different epochs')
        for key in ('joint_position','joint_velocity'):
            i=SENSOR_KEYS.index(key)
            if not sensors['sensor_valid'][:,i].all() or not np.array_equal(sensors['sensor_time_s'][:,i],clock):
                raise ValueError('Current complete joint evidence is required')
        for index in sorted({0,len(clock)//2,len(clock)-1}):
            row=steps[index];q=dict(zip(names,map(float,sensors['joint_position'][index])))
            actual=row['actual_joint_position']
            if set(actual)!=set(names) or max(abs(actual[n]-q[n]) for n in names)>1e-9:
                raise ValueError('Robot encoder values differ from the actual same-epoch measurements')
            state=dict(motor_contract=motors,door_source_sha256=door_sha,time_s=float(clock[index]),
                measured_time_s=float(clock[index]),root_state_world=row['root13_actororigin'],
                joint_position=q,joint_velocity=dict(zip(names,map(float,sensors['joint_velocity'][index]))),
                door_joint_order=config['door_joint_names'],
                door_position=dict(zip(config['door_joint_names'],map(float,physics['door'][index]))),
                door_velocity=dict(zip(config['door_joint_names'],map(float,physics['door_velocity'][index]))),
                root_state_convention=ROOT_CONVENTION)
            binding=freeze_destination_state(**state);result=admit_destination_state(binding,**state)
            # Deliberate negative control: this left finger was absent from the
            # historical 47-joint withdrawal guard. Do not advance any physics.
            changed=copy.deepcopy(state);changed['joint_position']['lh_LFJ1']+=.002
            rejected=False
            try:admit_destination_state(binding,**changed)
            except ValueError:rejected=True
            if not rejected:raise AssertionError('Unbound left-finger negative control admitted')
            results.append(dict(time_s=float(clock[index]),binding=binding,admission=result,
                altered_left_finger_rejected=rejected))
    report=dict(schema='doorbench.destination-state-capture-check.v1',passed=True,
        scope='Three actual recorded state bindings and a deliberate negative control; no physics stepped or trajectory/task qualification',
        coherent_archive_intervals=len(steps),door_identity_source=door_source,input_sha256=hashes,
        source_sha256={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in
            (Path(__file__),ROOT/'doorbench/dexterous/destination_state_binding.py')},results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,coherent_archive_intervals=len(steps),checked_epochs=[r['time_s'] for r in results],
                         altered_left_finger_rejected=True,output=str(args.output))))


if __name__=='__main__':main()
