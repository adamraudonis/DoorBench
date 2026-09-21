#!/usr/bin/env python3
"""Prepare an admitted prospective motor experiment; never run physics."""
import argparse
import json
from pathlib import Path
import numpy as np

from doorbench.dexterous.coupled_release_geometry import sha
from doorbench.dexterous.coupled_release_reference import CoupledReleaseReference
from doorbench.dexterous.release_source_admission import admit_release_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envelope',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--launch-recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--trial-output',type=Path,required=True)
    a=p.parse_args()
    if a.trial_output.exists():raise ValueError('Prospective trial must have a fresh output directory')
    envelope=json.loads(a.envelope.read_text());c=json.loads(Path(envelope['source_candidate']).read_text())
    source=Path(c['configuration']['source_run']);screen=Path(c['configuration']['right_hand_route']);audit=screen.with_name('standing-audit.json')
    if json.loads(audit.read_text()).get('passed') is not True:raise ValueError('Original independent RH route admission required')
    config=dict(schema='doorbench.standing-withdrawal.v1',source_run=str(source.resolve()),screen_path=str(screen.resolve()),screen_sha256=sha(screen),audit_path=str(audit.resolve()),audit_sha256=sha(audit),
        measured_rest_transfer=True,palm_only_support=True,grasp_profile='volar-phalange-v1',contact_audit_name='independent-contact-audit.json',left_support_target_N=4.,left_arm_only=True,left_full_orientation=True,
        capture_returned_motor_command=True,coupled_envelope_path=str(a.envelope.resolve()),coupled_envelope_sha256=sha(a.envelope),coupled_audit_path=str(a.audit.resolve()),coupled_audit_sha256=sha(a.audit),
        scope='Prospective continuous native acquisition/opening/transfer then measured-rest release using admitted measured-angle body/hand nominal references. Original motor/contact limits; no state reset or door commands; no physical success claim.')
    admission=admit_release_source(source,profile=config['grasp_profile'],contact_audit_name=config['contact_audit_name'],measured_rest=True)
    with np.load(source/'trajectory.npz') as z:initial=z['terminal_qpos'].copy()
    reference=CoupledReleaseReference(config,admission,initial,envelope['duration_s']);reference.geometry.close()
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'withdrawal.json';path.write_text(json.dumps(config,indent=2)+'\n')
    command=list(json.loads(a.launch_recipe.read_text())['command'])
    for flag,value in [('--output',str(a.trial_output.resolve())),('--standing-withdrawal-path',str(path.resolve()))]:
        if command.count(flag)!=1:raise ValueError('Exact single original recipe flag required: '+flag)
        command[command.index(flag)+1]=value
    result=dict(command=command,cwd=str(Path.cwd()),executed=False,physics_steps=0,withdrawal_config_sha256=sha(path),source_launch_recipe=str(a.launch_recipe.resolve()),source_launch_recipe_sha256=sha(a.launch_recipe),scope=config['scope'])
    (a.output/'command.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(config=str(path.resolve()),command=str((a.output/'command.json').resolve()),executed=False)))


if __name__=='__main__':main()
