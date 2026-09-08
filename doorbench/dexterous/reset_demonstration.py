"""Admit a recorded cold-start observation with an independently qualified label.

Only the observation crosses from the failed actor run. The motor label comes
from the existing teacher's actual first physical action at an identical reset.
No failed actor action, inferred privileged feature or later actor state enters
this teacher data augmentation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .motor_contract_identity import motor_contract_fingerprint
from .sensor_actor import prepare_actor_packet


def matched_cold_start_packet(teacher_run,observation_run,dimensions):
    teacher_run=Path(teacher_run);observation_run=Path(observation_run)
    read=lambda run,name:json.loads((run/name).read_text())
    teacher_motors=read(teacher_run,'motor-contract.json');actor_motors=read(observation_run,'motor-contract.json')
    if motor_contract_fingerprint(teacher_motors)!=motor_contract_fingerprint(actor_motors):
        raise ValueError('Reset observation and teacher use different mechanics')
    if read(teacher_run,'sensors/layout.json')!=read(observation_run,'sensors/layout.json'):
        raise ValueError('Reset observation and teacher use different calibration')
    if read(observation_run,'sensors/report.json').get('control_source')!='sensor_actor':
        raise ValueError('Cold-start source must explicitly identify its actual actor recording')
    left=read(teacher_run,'acquisition-reset.json');right=read(observation_run,'acquisition-reset.json')
    lc=read(teacher_run,'configuration.json');rc=read(observation_run,'configuration.json')
    if lc['robot_joint_names']!=rc['robot_joint_names']:
        raise ValueError('Reset observation has a different physical joint order')
    if (not np.array_equal(left['root'],right['root']) or
            not np.array_equal(left['joints'],right['joints']) or left['door']!=right['door']):
        raise ValueError('Teacher label requires the identical actual robot and door reset')
    path=observation_run/'sensors/actor-initial-decision.npz'
    with np.load(path,allow_pickle=False) as z:
        fields=set(dimensions.shapes)|{'previous_action','sensor_time_s','sensor_valid'}
        if set(z.files)!=fields|{'time_s','motor_forces'} or np.asarray(z['time_s']).shape!=() or float(z['time_s'])!=0.:
            raise ValueError('Missing actual pre-step cold-start packet')
        packet={key:z[key].copy() for key in fields}
    prepare_actor_packet(packet,0.,dimensions)
    if (packet['sensor_valid'].any() or np.any(packet['sensor_time_s']!=-1.) or
            any(np.any(value!=0) for key,value in packet.items() if key not in ('sensor_valid','sensor_time_s'))):
        raise ValueError('This augmentation requires the recorded all-invalid zero-history cold start')
    names=('acquisition-reset.json','configuration.json','motor-contract.json','sensors/layout.json',
           'sensors/report.json','sensors/actor-initial-decision.npz')
    return packet,dict(scope='Only recorded cold-start observation; label is the qualified teacher first action, never the failed actor action',
        source=str(observation_run.resolve()),observation_files_sha256={name:hashlib.sha256((observation_run/name).read_bytes()).hexdigest() for name in names},
        teacher_reset_sha256=hashlib.sha256((teacher_run/'acquisition-reset.json').read_bytes()).hexdigest(),
        exact_robot_and_door_reset_match=True,failed_actor_motor_label_used=False)
