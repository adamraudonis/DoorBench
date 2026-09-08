"""Explicit DAgger candidate data: real student sensors, counterfactual expert labels.

This is intentionally a different admission path from qualified successful
teacher demonstrations. Source task failures stay failed and these corrections
are not advertised as physically validated recovery trajectories.
"""
import json
from pathlib import Path

import numpy as np

from .offline_teacher_queries import load_query_evidence,sha,correction_quality
from .sensor_actor import ActorDimensions,prepare_actor_packet
from .sensor_contract import SENSOR_KEYS,INTERFACE_VERSION
from .sensor_demonstrations import SensorDemonstration


class CorrectionDemonstration:
    def __init__(self,path,*,source_run=None):
        path=Path(path);report=json.loads((path/'report.json').read_text())
        if (report.get('schema')!='doorbench.counterfactual-acquisition-teacher.v1' or
                report.get('teacher_actions_delivered_to_physics') is not False or
                report.get('student_actions_used_as_expert_labels') is not False or
                report.get('maximum_teacher_internal_sim_time_s')!=0. or
                report.get('physical_recovery_evaluated') is not False):
            raise ValueError('Expected explicit counterfactual labels without a physical recovery claim')
        expected={'reference.json','counterfactual-teacher-actions.npz','queries.json','manifest.json','source.tar.gz'}
        if set(report['files_sha256'])!=expected or any(sha(path/name)!=report['files_sha256'][name] for name in expected):
            raise ValueError('Correction evidence hash mismatch')
        run=Path(source_run or report['source']['source_run'])
        queries,contract,physical,source=load_query_evidence(run)
        if source['files_sha256']!=report['source']['files_sha256']:
            raise ValueError('Correction labels refer to different student measurements')
        motors=json.loads((run/'motor-contract.json').read_text())
        original=json.loads((run/'provenance.json').read_text())['files']
        original_reference=[v for k,v in original.items() if k.endswith('/reference.json')]
        if (report.get('robot_sha256') != motors['source_xml_sha256'] or
                report.get('reference_sha256') != sha(path/'reference.json') or
                original_reference != [report['reference_sha256']]):
            raise ValueError('Counterfactual teacher model/reference differ from the actual actor reset')
        rows=json.loads((path/'queries.json').read_text())
        with np.load(path/'counterfactual-teacher-actions.npz',allow_pickle=False) as z:labels={k:z[k].copy() for k in z.files}
        expected={'time_s','motor_force','normalized_force','label_valid','contiguous_valid_prefix'}
        if set(labels)!=expected or not np.array_equal(labels['time_s'],queries['time_s']):
            raise ValueError('Correction label clock differs from the actual actor decisions')
        n=len(queries['time_s']);caps=np.asarray([a['force_range'] for a in motors['actuators']])
        if (len(rows)!=n or report['teacher_force_calls']!=n or report['queries']!=n or
                not np.array_equal([r['time_s'] for r in rows],queries['time_s']) or
                labels['motor_force'].shape!=(n,61) or labels['normalized_force'].shape!=(n,61) or
                labels['label_valid'].shape!=(n,) or labels['contiguous_valid_prefix'].shape!=(n,) or
                labels['label_valid'].dtype!=np.bool_ or labels['contiguous_valid_prefix'].dtype!=np.bool_):
            raise ValueError('Incomplete continuous expert-query evidence')
        source_checks=json.loads((run/'report.json').read_text())['checks']
        global_physical=all(source_checks.get(k) is True for k in ('joint_stops','documented_loopbacks','self_collision','environment_collision',
            'working_hand_collision','plant_parameters_unchanged','finite','motor_delivery_matches_command','native_motor_caps'))
        recomputed=np.array([correction_quality(queries['root_state'][i],labels['motor_force'][i],caps,solver_status=r['solver_status'],
            source_physical=global_physical,teacher_exception=r['teacher_exception'])['candidate_label_valid'] for i,r in enumerate(rows)],bool)
        prefix=np.logical_and.accumulate(recomputed);count=int(prefix.sum())
        normalized=2*(labels['motor_force']-caps[:,0])/(caps[:,1]-caps[:,0])-1
        if (not np.array_equal(recomputed,labels['label_valid']) or not np.array_equal(prefix,labels['contiguous_valid_prefix']) or
                report['candidate_valid_labels']!=int(recomputed.sum()) or report['contiguous_valid_prefix_labels']!=count or
                not np.allclose(normalized,labels['normalized_force'],rtol=0,atol=1e-12) or count<1):
            raise ValueError('Correction feasibility labels do not reproduce the strict actual-state gate')
        self.path=run;self.layout=json.loads((run/'sensors/layout.json').read_text())
        if self.layout['interface_version']!=INTERFACE_VERSION:raise ValueError('Unsupported actor observation contract')
        self.dimensions=ActorDimensions(joints=len(self.layout['joint_order']),actions=len(self.layout['action_order']),tactile=self.layout['tactile_dimension'])
        with np.load(run/'sensors/actor-sensors.npz',allow_pickle=False) as z:
            expected=set(SENSOR_KEYS)-{'rgb_left','rgb_right'}|{'previous_action','sensor_time_s','sensor_valid','time_s'}
            if set(z.files)!=expected:raise ValueError('Unknown or missing actual actor sensor arrays')
            numeric={key:z[key].copy() for key in z.files if key!='time_s'}
        with np.load(run/'sensors/actor-initial-decision.npz',allow_pickle=False) as z:
            expected=set(self.dimensions.shapes)|{'previous_action','sensor_time_s','sensor_valid'}
            if set(z.files)!=expected|{'motor_forces','time_s'} or float(z['time_s'])!=0.:raise ValueError('Missing actual initial actor decision')
            initial={key:z[key].copy() for key in expected}
        prepare_actor_packet(initial,0.,self.dimensions)
        # query i uses actual post-step sensor row i-1; the expert force at query
        # i is its label. The failed student's previous_action remains an input.
        self.numeric={key:np.concatenate((initial[key][None],value[:count-1]),axis=0) for key,value in numeric.items()}
        self.times=queries['time_s'][:count];self.targets=labels['normalized_force'][:count].astype(np.float32)
        with np.load(run/'sensors/actor-rgb.npz',allow_pickle=False) as z:
            if set(z.files)!={'time_s','rgb_left','rgb_right'}:raise ValueError('Unexpected actor image arrays')
            self.rgb={key:z[key].copy() for key in z.files}
        self.motor_contract_sha256=source['motor_contract_sha256']
        self.metadata=dict(source=str(run.resolve()),correction_source=str(path.resolve()),qualification='counterfactual_feasible_acquisition_prefix',
            control_source='offline_privileged_teacher_labels_on_sensor_actor_observations',grasp_profile='distal-pad-v1',
            physics_dt_s=.002,examples=len(self),motor_contract_sha256=self.motor_contract_sha256,robot_xml_sha256=self.layout['robot_xml_sha256'],
            files={name:sha(run/'sensors'/name) for name in ('layout.json','actor-sensors.npz','actor-rgb.npz')},
            correction_report_sha256=sha(path/'report.json'),source_actor_task_passed=source['source_actor_task_passed'],
            last_admitted_decision_time_s=float(self.times[-1]),physical_recovery_evaluated=False,
            limitation='DAgger candidate labels from a continuous analytic teacher; source task still failed, no physical recovery or student success claim')

    def __len__(self):return len(self.times)

    def packet(self,index):
        # Reuse the exact strict packet/causal-image validator. This class owns
        # its separate label admission and never bypasses teacher qualification.
        return SensorDemonstration.packet(self,index)

    def sequence(self,start,length):
        if start<0 or length<=0 or start+length>len(self):raise ValueError('Correction sequence cannot cross the qualified prefix boundary')
        rows=[prepare_actor_packet(self.packet(i),float(self.times[i]),self.dimensions) for i in range(start,start+length)]
        return {key:np.stack([r[key] for r in rows]) for key in rows[0]},self.targets[start:start+length].copy()
