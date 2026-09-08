"""Test a trained actor against recorded observations; never step a simulator."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from doorbench.dexterous.sensor_demonstrations import SensorDemonstration
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('episode','checkpoint','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--qualification',choices=['operation-report.json','acquisition-report.json'],default='operation-report.json')
    ap.add_argument('--legacy-teacher-receipt',type=Path);ap.add_argument('--samples',type=int,default=64)
    args=ap.parse_args();torch.set_num_threads(1)
    e=SensorDemonstration(args.episode,qualification=args.qualification,legacy_teacher_receipt=args.legacy_teacher_receipt)
    if not 1<=args.samples<=len(e):ap.error('Choose available consecutive samples')
    motors=json.loads((args.episode/'motor-contract.json').read_text())
    kwargs=dict(motor_contract=motors,sensor_layout=e.layout,physics_dt_s=e.metadata['physics_dt_s'])
    actor=SensorPolicyController(args.checkpoint,**kwargs);actor.reset_episode();forces=[]
    for i in range(args.samples):
        packet=e.packet(i);packet['previous_action']=actor.previous_action
        forces.append(actor.force(packet,float(e.times[i])))
    forces=np.asarray(forces);caps=actor.force_ranges
    actor.reset_episode();packet=e.packet(0);packet['previous_action']=actor.previous_action
    reset=np.array_equal(actor.force(packet,float(e.times[0])),forces[0])
    changed=copy.deepcopy(motors);changed['actuators'][0]['force_range'][1]*=.99
    mismatch=False
    try:SensorPolicyController(args.checkpoint,**dict(kwargs,motor_contract=changed))
    except ValueError as error:mismatch='mechanics contract differs' in str(error)
    checks=dict(finite_consecutive_inferences=bool(np.isfinite(forces).all()),
        original_caps=bool(np.all(forces>=caps[:,0]) and np.all(forces<=caps[:,1])),
        explicit_reset_reproduces_first_action=bool(reset),changed_valid_motor_caps_rejected=mismatch)
    report=dict(scope='Actual trained checkpoint loaded by runtime actor; recorded physical observations and actor-owned command history. No simulator was stepped.',
        closed_loop_evaluated=False,task_success_rate=None,teacher_fallback=False,checks=checks,passed=all(checks.values()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),motor_contract_sha256=e.motor_contract_sha256,
        robot_xml_sha256=e.layout['robot_xml_sha256'],consecutive_inferences=args.samples,physics_dt_s=e.metadata['physics_dt_s'],
        maximum_absolute_motor_command=float(np.abs(forces).max()),
        limitation="Inference integration check only; recorded sensor states do not respond to this actor's forces.")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
