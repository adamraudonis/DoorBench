"""Causal native teacher examples with separately audited fixed-eye images.

This format does not impersonate an Isaac demonstration. Qualification is for
one recorded native episode, not a sensor policy or a successful Isaac port.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .motor_contract_identity import motor_contract_fingerprint
from .sensor_actor import ActorDimensions
from .sensor_contract import ACTOR_KEYS, SENSOR_KEYS, INTERFACE_VERSION
from .sensor_demonstrations import SensorDemonstration


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class NativeSensorDemonstration:
    """Strictly admit an audited complete native sequence and camera variant."""
    __len__ = SensorDemonstration.__len__
    packet = SensorDemonstration.packet
    sequence = SensorDemonstration.sequence

    def __init__(self, run, camera_variant):
        self.path = Path(run)
        variant = Path(camera_variant)
        sensor = self.path / 'own-sensors'
        def read(path):
            return json.loads(path.read_text())
        def bound(path, expected):
            if digest(path) != expected:
                raise ValueError('Changed demonstration evidence: ' + str(path))
        task = read(self.path / 'report.json')
        physics = read(self.path / 'independent-continuous-audit.json')
        join = read(sensor / 'independent-join-audit.json')
        camera = read(variant / 'independent-audit.json')
        for name, report, count in [('task', task, 47), ('physics', physics, 16),
                                    ('sensor join', join, 15), ('camera', camera, 9)]:
            if (report.get('passed') is not True or len(report.get('checks', {})) != count
                    or not all(v is True for v in report['checks'].values())):
                raise ValueError('Incomplete native qualification: ' + name)
        if task.get('runtime_pose_writes') != 0 or task.get('native_mirror_steps') != 0:
            raise ValueError('Native task contains external state assistance')
        capture = read(sensor / 'report.json')
        receipt = read(variant / 'receipt.json')
        if (capture.get('capture_complete') is not True or
                capture.get('control_source') != 'privileged_teacher' or
                receipt.get('complete') is not True or receipt.get('physics_steps') != 0):
            raise ValueError('Require completed teacher capture and recorded-state camera variant')
        bound(sensor / 'report.json', join['sensor_report_sha256'])
        bound(self.path / 'raw-transitions/manifest.json', physics['raw_manifest_sha256'])
        bound(self.path / 'raw-transitions/manifest.json', join['raw_manifest_sha256'])
        bound(self.path / 'actual-warning-intervals.npz', physics['warning_sidecar_sha256'])
        bound(variant / 'receipt.json', camera['variant_receipt_sha256'])
        for relative in ('manifest.json', 'trace.json', 'trajectory.npz', 'own-sensors/layout.json',
                         'raw-transitions/transitions-00000.npz'):
            matches = [value for name,value in receipt['source_files'].items()
                       if name.endswith('/'+self.path.name+'/'+relative)]
            if len(matches) != 1:
                raise ValueError('Camera source identity is not this native episode')
            bound(self.path / relative, matches[0])
        for chunk in read(self.path / 'raw-transitions/manifest.json')['chunks']:
            bound(self.path / 'raw-transitions' / chunk['file'], chunk['sha256'])
        for name, expected in receipt['output_sha256'].items():
            bound(variant / name, expected)
        bound(sensor / 'layout.json', capture['layout_sha256'])
        self.layout = read(variant / 'layout.json')
        original = read(sensor / 'layout.json')
        # The separate variant may change only the fixed camera calibration.
        camera_fields = {'cameras', 'native_cameras', 'camera_profile'}
        if ({k:v for k,v in self.layout.items() if k not in camera_fields} !=
                {k:v for k,v in original.items() if k not in camera_fields} or
                self.layout.get('native_cameras') != original['cameras']):
            raise ValueError('Camera variant changed a non-camera sensor contract')
        motors = read(self.path / 'motors-input.json')
        self.motor_contract_sha256 = motor_contract_fingerprint(motors)
        if (self.layout['interface_version'] != INTERFACE_VERSION or
                self.layout['robot_xml_sha256'] != motors['source_xml_sha256'] or
                self.layout['action_order'] != [m['name'] for m in motors['actuators']] or
                len(self.layout['joint_order']) != 69 or len(motors['actuators']) != 61):
            raise ValueError('Incompatible native robot/motor contract')
        self.dimensions = ActorDimensions(joints=69, actions=61, tactile=self.layout['tactile_dimension'])
        expected = set(ACTOR_KEYS) - {'rgb_left', 'rgb_right'} | {'time_s'}
        arrays = []
        for chunk in capture['chunks']:
            path = sensor / chunk['file']; bound(path, chunk['sha256'])
            with np.load(path, allow_pickle=False) as z:
                if set(z.files) != expected or any(len(z[k]) != chunk['rows'] for k in z.files):
                    raise ValueError('Invalid numeric chunk fields or row count')
                arrays.append({k:z[k].copy() for k in z.files})
        numeric = {k:np.concatenate([a[k] for a in arrays]) for k in expected}
        with np.load(variant / 'actor-initial-decision.npz', allow_pickle=False) as z:
            if set(z.files) != set(ACTOR_KEYS) | {'time_s', 'motor_forces'}:
                raise ValueError('Invalid reset packet fields')
            initial = {k:z[k].copy() for k in z.files}
        caps = np.array([max(abs(x) for x in m['force_range']) for m in motors['actuators']])
        if (float(initial['time_s']) != 0 or initial['previous_action'].any() or
                not np.allclose(initial['motor_forces']/caps, numeric['previous_action'][0], atol=1e-7, rtol=0)):
            raise ValueError('Reset label is not the first actual force')
        self.numeric = {k:np.concatenate([initial[k][None], v]) for k,v in numeric.items()}
        self.times = self.numeric.pop('time_s')
        if (any(not np.isfinite(v).all() for v in self.numeric.values()) or
                np.max(np.abs(self.numeric['previous_action'])) > 1+1e-6):
            raise ValueError('Nonfinite observations or invalid normalized forces')
        if (len(self.times)-1 != capture['samples'] or capture['samples'] != physics['actual_intervals'] or
                not np.isfinite(self.times).all() or
                not np.allclose(np.diff(self.times), task['physics_dt_s'], atol=1e-8, rtol=0)):
            raise ValueError('Incomplete causal episode clock')
        with np.load(variant / 'actor-rgb.npz', allow_pickle=False) as z:
            if set(z.files) != {'time_s', 'rgb_left', 'rgb_right'}:
                raise ValueError('Invalid camera archive fields')
            self.rgb = {k:np.concatenate([initial[k][None], z[k]]) for k in z.files}
        if np.any(np.diff(self.rgb['time_s']) <= 0):
            raise ValueError('Invalid camera clock')
        for key in ('rgb_left', 'rgb_right'):
            if self.rgb[key].shape != (len(self.rgb['time_s']), 128, 128, 3) or self.rgb[key].dtype != np.uint8:
                raise ValueError('Invalid fixed camera dimensions or dtype')
            stream = SENSOR_KEYS.index(key)
            times = self.numeric['sensor_time_s'][:, stream]
            indices = np.searchsorted(self.rgb['time_s'], times+1e-9, side='right')-1
            valid = self.numeric['sensor_valid'][:, stream]
            if (np.any(indices[valid] < 0) or
                    not np.allclose(self.rgb['time_s'][indices[valid]], times[valid], atol=1e-8, rtol=0) or
                    np.any(times[valid] > self.times[valid]+1e-8)):
                raise ValueError('Missing exact causal camera frame')
        self.metadata = dict(source=str(self.path.resolve()), camera_variant=str(variant.resolve()),
            qualification='native-continuous-with-audited-camera-variant', control_source='privileged_teacher',
            physics_dt_s=task['physics_dt_s'], examples=len(self), robot_xml_sha256=self.layout['robot_xml_sha256'],
            motor_contract_sha256=self.motor_contract_sha256,
            qualification_files={str(p):digest(p) for p in [self.path/'report.json', self.path/'independent-continuous-audit.json',
                sensor/'independent-join-audit.json', variant/'independent-audit.json', variant/'receipt.json']},
            limitation='One native privileged teacher episode; accelerometer not independently reconstructed; no student rollout, Isaac success or generalization claim')
        self.packet(0)
        self.packet(len(self)-1)
