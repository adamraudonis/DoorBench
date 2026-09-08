"""Causal imitation examples from separately qualified sensor archives.

Post-step packet i contains the force just applied as previous_action. Its
training label is the force applied at step i+1. Labels and qualification metadata
never enter the actor packet. Sequences cannot cross an episode boundary.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .sensor_actor import ActorDimensions,prepare_actor_packet
from .sensor_contract import SENSOR_KEYS,INTERFACE_VERSION
from .motor_contract_identity import motor_contract_fingerprint


class SensorDemonstration:
    def __init__(self,run,*,qualification='operation-report.json'):
        self.path=Path(run)
        if qualification not in ('operation-report.json','acquisition-report.json'):
            raise ValueError('Select a declared robot-task qualification report')
        report=json.loads((self.path/qualification).read_text())
        mechanics=json.loads((self.path/'mechanical-audit.json').read_text())
        if report.get('passed') is not True or mechanics.get('passed') is not True or report.get('runtime_robot_pose_writes')!=0 or report.get('direct_door_commands') is not False:
            raise ValueError('Imitation archive is not qualified for the declared physical task')
        sensor_report=json.loads((self.path/'sensors/report.json').read_text())
        if (sensor_report.get('control_source')!='privileged_teacher' or
                report.get('control_source','privileged_teacher')!='privileged_teacher' or
                report.get('closed_loop_evaluated') is True):
            raise ValueError('Teacher imitation requires an explicitly privileged_teacher capture, not a sensor_actor rollout')
        tendons=json.loads((self.path/'passive-tendon-audit.json').read_text())
        motors=json.loads((self.path/'motor-contract.json').read_text())
        self.motor_contract_sha256=motor_contract_fingerprint(motors)
        from .isaac_readiness import validate_passive_audit
        validate_passive_audit(tendons,motors,'shadow-loopback-v2')
        sensor=self.path/'sensors'
        self.layout=json.loads((sensor/'layout.json').read_text())
        if self.layout['interface_version']!=INTERFACE_VERSION:
            raise ValueError('Unsupported sensor interface')
        if self.layout['robot_xml_sha256']!=motors.get('source_xml_sha256') or self.layout['action_order']!=[m['name'] for m in motors['actuators']]:
            raise ValueError('Sensor calibration or action order differs from the actual native motor contract')
        self.dimensions=ActorDimensions(joints=len(self.layout['joint_order']),actions=len(self.layout['action_order']),tactile=self.layout['tactile_dimension'])
        # Read numeric archives only; never unpickle simulator or controller objects.
        with np.load(sensor/'actor-sensors.npz',allow_pickle=False) as archive:
            expected=set(SENSOR_KEYS)-{'rgb_left','rgb_right'}|{'previous_action','sensor_time_s','sensor_valid','time_s'}
            if set(archive.files)!=expected:
                raise ValueError('Sensor archive contains missing or forbidden arrays')
            self.numeric={key:archive[key].copy() for key in archive.files}
        with np.load(sensor/'actor-rgb.npz',allow_pickle=False) as archive:
            if set(archive.files)!={'time_s','rgb_left','rgb_right'}:
                raise ValueError('RGB archive contains missing or forbidden arrays')
            self.rgb={key:archive[key].copy() for key in archive.files}
        self.times=self.numeric.pop('time_s')
        if self.times.ndim!=1 or len(self.times)<2 or not np.isfinite(self.times).all() or np.any(np.diff(self.times)<=0):
            raise ValueError('Episode sensor clock is invalid')
        if any(len(value)!=len(self.times) for value in self.numeric.values()):
            raise ValueError('Numeric sensor records have inconsistent lengths')
        delta=np.diff(self.times)
        if not np.allclose(delta,report['physics_dt_s'],rtol=0,atol=1e-8):
            raise ValueError('Labels require consecutive recorded physics steps')
        frames=self.rgb['time_s']
        if frames.ndim!=1 or len(frames)==0 or not np.isfinite(frames).all() or np.any(np.diff(frames)<=0):
            raise ValueError('RGB capture clock is invalid')
        if any(value.shape!=(len(frames),128,128,3) for key,value in self.rgb.items() if key!='time_s'):
            raise ValueError('RGB records disagree with the frozen camera resolution')
        self.metadata=dict(source=str(self.path.resolve()),qualification=qualification,
            control_source='privileged_teacher',
            grasp_profile=report.get('grasp_profile',report.get('final_pad_grasp',{}).get('grasp_profile','distal-pad-v1')),
            physics_dt_s=float(report['physics_dt_s']),examples=len(self),
            files={name:hashlib.sha256((sensor/name).read_bytes()).hexdigest() for name in ('layout.json','actor-sensors.npz','actor-rgb.npz')},
            robot_xml_sha256=self.layout['robot_xml_sha256'],
            motor_contract_sha256=self.motor_contract_sha256,
            qualification_files={name:hashlib.sha256((self.path/name).read_bytes()).hexdigest() for name in
                (qualification,'mechanical-audit.json','passive-tendon-audit.json','motor-contract.json','sensors/report.json')},
            limitation='Teacher imitation data; no student rollout or generalization result')

    def __len__(self):
        return len(self.times)-1

    def packet(self,index):
        if not 0<=index<len(self):raise IndexError('No future label exists at this episode index')
        packet={key:value[index].copy() for key,value in self.numeric.items()}
        for key in ('rgb_left','rgb_right'):
            stream=SENSOR_KEYS.index(key);capture=packet['sensor_time_s'][stream]
            frame=int(np.searchsorted(self.rgb['time_s'],capture+1e-9,side='right')-1)
            if not packet['sensor_valid'][stream]:
                packet[key]=np.zeros(self.dimensions.shapes[key],dtype=np.uint8)
            elif frame<0 or abs(self.rgb['time_s'][frame]-capture)>1e-8:
                raise ValueError('Missing exact causal RGB observation; future-frame substitution is forbidden')
            else:packet[key]=self.rgb[key][frame].copy()
        # Full packet validation and causal clock check also run at inference.
        prepare_actor_packet(packet,float(self.times[index]),self.dimensions)
        return packet

    def sequence(self,start,length):
        if start<0 or length<=0 or start+length>len(self):
            raise ValueError('A training sequence cannot cross an episode boundary')
        rows=[prepare_actor_packet(self.packet(i),float(self.times[i]),self.dimensions) for i in range(start,start+length)]
        inputs={key:np.stack([row[key] for row in rows]) for key in rows[0]}
        target=self.numeric['previous_action'][start+1:start+length+1].copy()
        if not np.isfinite(target).all() or np.max(np.abs(target))>1+1e-6:
            raise ValueError('Invalid next-step native motor-force labels')
        return inputs,target
