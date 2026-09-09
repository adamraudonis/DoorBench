"""Audited, bounded native recovery examples with separate teacher targets.

These are privileged correction diagnostics, not complete successful tasks.
Actual preceding motor commands always remain in the recorded actor inputs.
"""
import json
from pathlib import Path

import numpy as np

from .native_sensor_demonstrations import digest
from .sensor_actor import ActorDimensions, prepare_actor_packet
from .sensor_contract import ACTOR_KEYS
from .sensor_demonstrations import SensorDemonstration
from .motor_contract_identity import motor_contract_fingerprint

REQUIRED_CHECKS = set('sensor_joins bound_sensor_join completed_recovery explicit_privilege original_force_delivery no_state_assistance initial_fields initial_clock initial_validity initial_encoders exact_initial_images exact_recorded_images uninterrupted_states exact_counterfactual_labels exact_applied_teacher_forces exact_switch_clock original_motor_caps every_interval_upright finite_states every_interval_no_warnings complete_label_and_camera_coverage'.split())


class NativeCorrectionDemonstration:
    __len__ = SensorDemonstration.__len__
    packet = SensorDemonstration.packet

    def __init__(self, run):
        self.path = Path(run)
        audit = json.loads((self.path/'independent-correction-audit.json').read_text())
        if (audit.get('schema') != 'doorbench.native-approach-correction-audit.v1'
                or audit.get('passed') is not True or set(audit.get('checks', {})) != REQUIRED_CHECKS
                or not all(v is True for v in audit['checks'].values()) or audit.get('physics_steps') != 0):
            raise ValueError('Complete independent native correction audit required')
        for relative, expected in audit['files'].items():
            if Path(relative).is_absolute() or '..' in Path(relative).parts or digest(self.path/relative) != expected:
                raise ValueError('Changed correction evidence: '+relative)
        read=lambda p:json.loads(p.read_text())
        report=read(self.path/'report.json');sensor=self.path/'own-sensors';capture=read(sensor/'report.json')
        required={'manifest.json','report.json','counterfactual-labels.npz','motors-input.json',
            'own-sensors/layout.json','own-sensors/report.json','own-sensors/actor-initial-decision.npz',
            'own-sensors/actor-rgb.npz','own-sensors/independent-join-audit.json','raw-transitions/manifest.json'}
        required.update('own-sensors/'+c['file'] for c in capture['chunks'])
        required.update('raw-transitions/'+c['file'] for c in read(self.path/'raw-transitions/manifest.json')['chunks'])
        if not required.issubset(audit['files']):
            raise ValueError('Correction audit omitted evidence used by training')
        if (report['control_source']!='privileged_recovery' or capture['control_source']!='privileged_recovery'
                or report['unassisted_policy_success'] is not False or report['full_task_qualified'] is not False):
            raise ValueError('Recovery examples must retain explicit privilege and task limitations')
        self.layout=read(sensor/'layout.json');motors=read(self.path/'motors-input.json')
        self.motor_contract_sha256=motor_contract_fingerprint(motors)
        self.dimensions=ActorDimensions(joints=69,actions=61,tactile=self.layout['tactile_dimension'])
        if (self.layout['joint_order']!=motors['joint_names'] or
                self.layout['action_order']!=[v['name'] for v in motors['actuators']] or
                self.layout['robot_xml_sha256']!=motors['source_xml_sha256']):
            raise ValueError('Correction embodiment contract mismatch')
        expected=set(ACTOR_KEYS)-{'rgb_left','rgb_right'}|{'time_s'}
        arrays=[]
        for chunk in capture['chunks']:
            with np.load(sensor/chunk['file'],allow_pickle=False) as z:
                if set(z.files)!=expected:raise ValueError('Invalid numeric sensor fields')
                arrays.append({k:z[k].copy() for k in z.files})
        with np.load(sensor/'actor-initial-decision.npz',allow_pickle=False) as z:
            initial={k:z[k].copy() for k in z.files}
        self.numeric={k:np.concatenate([initial[k][None]]+[v[k] for v in arrays]) for k in expected}
        self.times=self.numeric.pop('time_s')
        with np.load(sensor/'actor-rgb.npz',allow_pickle=False) as z:
            self.rgb={k:np.concatenate([initial[k][None],z[k]]) for k in z.files}
        with np.load(self.path/'counterfactual-labels.npz',allow_pickle=False) as z:
            forces=z['teacher_motor_forces'].copy();times=z['time_s'].copy()
        caps=np.array([v['force_range'] for v in motors['actuators']])
        self.targets=(2*(forces-caps[:,0])/(caps[:,1]-caps[:,0])-1).astype(np.float32)
        if (forces.shape!=(len(self),61) or len(self)!=audit['samples'] or
                not np.allclose(times,self.times[:-1],atol=1e-8,rtol=0) or
                not np.allclose(np.diff(self.times),.002,atol=1e-8,rtol=0) or
                not np.isfinite(self.targets).all() or np.max(abs(self.targets))>1+1e-6):
            raise ValueError('Invalid causal counterfactual targets')
        self.metadata=dict(source=str(self.path.resolve()),qualification='bounded_privileged_recovery',
            control_source='privileged_recovery',physics_dt_s=.002,examples=len(self),
            motor_contract_sha256=self.motor_contract_sha256,
            audit_sha256=digest(self.path/'independent-correction-audit.json'),files=audit['files'],
            limitation='Correction-only approach curriculum; not a successful complete door task. Independent accelerometer reconstruction is not included.')

    def sequence(self,start,length):
        if start<0 or length<=0 or start+length>len(self):
            raise ValueError('Correction sequence crosses an episode boundary')
        rows=[prepare_actor_packet(self.packet(i),float(self.times[i]),self.dimensions) for i in range(start,start+length)]
        return {k:np.stack([r[k] for r in rows]) for k in rows[0]},self.targets[start:start+length].copy()
