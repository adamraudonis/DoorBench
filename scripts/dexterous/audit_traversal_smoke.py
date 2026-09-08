#!/usr/bin/env python3
"""Independently audit one executed second of the Isaac traversal adapter.

Reads evidence only. A smoke pass never changes the incomplete whole-task result.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def unpack_contacts(row, layout):
    """Reject aliasing/truncation and retain normal/friction identities separately."""
    values = {}
    for kind, dimensions in [('normal', {'force_N':1, 'point_world':3, 'normal_world':3, 'distance_m':1}),
                             ('friction', {'force_N':3, 'point_world':3})]:
        block = row[kind];slots = block['slots'];pairs = block['pairs']
        if any(type(k) is not int for k in slots) or len(slots) >= layout['capacity'] or len(slots) != len(set(slots)):
            raise ValueError('Contact buffer exhausted or reused a slot')
        expected=[];seen_pairs=set();labels=[]
        for i,j,start,count in pairs:
            if (any(type(x) is not int for x in (i,j,start,count)) or min(i,j,start)<0 or count<=0 or
                    i>=len(layout['sensor_paths']) or j>=len(layout['filter_paths'][i]) or
                    start+count>layout['capacity'] or (i,j) in seen_pairs):
                raise ValueError('Malformed contact pair/slice')
            seen_pairs.add((i,j));expected.extend(range(start,start+count));labels.extend([(i,j)]*count)
        if slots!=expected:raise ValueError('Sparse slots no longer match their pair slices')
        arrays={k:np.asarray(block[k],float).reshape(len(slots),width) for k,width in dimensions.items()}
        if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite occupied contact evidence')
        if kind=='normal' and len(slots):
            if np.any(arrays['force_N']<0) or not np.allclose(np.linalg.norm(arrays['normal_world'],axis=1),1.,atol=1e-5,rtol=0):
                raise ValueError('Invalid normal force or direction')
        values[kind]=(labels,arrays)
    return values


def audit(run, spec_path, motors_path, robot_path):
    run=Path(run);spec=read_json(spec_path);motors=read_json(motors_path)
    contract=read_json(run/'motor-readback-contract.json');layout=read_json(run/'traversal-contact-layout.json')
    reset=read_json(run/'acquisition-reset.json');mechanics=read_json(run/'mechanical-audit.json')
    report=read_json(run/'traversal-report.json');provenance=read_json(run/'provenance.json')
    with np.load(run/'acquisition-physics.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    with gzip.open(run/'traversal-contacts.jsonl.gz','rt') as f:contacts=[json.loads(line) for line in f]
    with gzip.open(run/'traversal-steps.json.gz','rt') as f:decisions=json.load(f)
    with gzip.open(run/'acquisition-pad-steps.json.gz','rt') as f:pads=json.load(f)
    dt=.002;n=500;checks={};metrics={}
    checks['frozen_one_second_spec']=spec['physics_dt_s']==dt and spec['physical_steps']==n and spec['simulated_seconds']==1.
    checks['source_and_explicit_input_hashes']=all(provenance['files'].get(path)==digest for path,digest in spec['input_sha256'].items())
    dependencies={path[len(spec['remote_source'])+1:]:digest for path,digest in provenance['files'].items() if path.startswith(spec['remote_source']+'/')}
    checks['frozen_captured_dependency_bytes']=bool(dependencies) and all(spec['source_sha256'].get(name)==digest and sha(run/('source-'+Path(name).name))==digest for name,digest in dependencies.items())
    saved=run/'source-isaac_opening.py'
    checks['frozen_runner_bytes']=sha(saved)==spec['source_sha256']['scripts/dexterous/isaac_opening.py']
    checks['matching_original_motor_robot_inputs']=sha(motors_path)==spec['input_sha256']['/workspace/h1-v2-import-001/h1-import.motors.json'] and sha(robot_path)==motors['source_xml_sha256']
    checks['complete_500_step_arrays']=all(len(a)==n for a in data.values()) and np.allclose(data['time_s'],np.arange(1,n+1)*dt,atol=1e-8,rtol=0)
    checks['finite_arrays']=all(np.isfinite(a).all() for a in data.values())
    root=data['root'];legacy=data['legacy_root_state_w']
    offset=np.asarray(contract['root_com_offset_in_actor_m'])
    correction=np.cross(legacy[:,10:13],-Rotation.from_quat(legacy[:,[4,5,6,3]]).apply(np.broadcast_to(offset,(n,3))))
    error=root[:,7:10]-legacy[:,7:10]-correction
    metrics['actor_velocity_relation_max_error_m_s']=float(np.max(np.abs(error)))
    metrics['maximum_origin_velocity_difference_m_s']=float(np.linalg.norm(correction,axis=1).max())
    metrics['imported_root_com_offset_in_actor_m']=offset.tolist()
    checks['actor_origin_world_velocity']=metrics['actor_velocity_relation_max_error_m_s']<1e-7
    checks['same_actor_pose_world_angular_velocity']=np.array_equal(root[:,:7],legacy[:,:7]) and np.array_equal(root[:,10:13],legacy[:,10:13])
    checks['explicit_root_and_input_semantics']=contract['root_controller_field']=='root_link_state_w' and contract['initial_interval_s']==[0.,0.] and 'not independently measured joint torque' in contract['semantics']
    names=layout['robot_joint_names'];index={name:i for i,name in enumerate(names)}
    ordered=[a['name'] for a in motors['actuators']]
    checks['unique_exact_joint_motor_orders']=len(names)==69 and len(index)==69 and set(names)==set(motors['joint_names']) and layout['motor_names']==ordered and len(set(ordered))==61
    matrix=np.array([[a['terms'].get(name,0.) for name in names] for a in motors['actuators']])
    caps=np.array([a['force_range'] for a in motors['actuators']]);damping=np.array([motors['passive'][name]['damping'] for name in names]);friction=np.array([motors['passive'][name]['friction'] for name in names])
    previous_velocity=np.vstack([np.zeros((1,69)),data['joint_velocity'][:-1]])
    generalized=data['motor_forces']@matrix-damping*previous_velocity-friction*np.tanh(previous_velocity/.001)
    delivered=data['actual_joint_effort'];inverse=np.linalg.pinv(matrix.T)
    reconstructed=(delivered+damping*previous_velocity+friction*np.tanh(previous_velocity/.001))@inverse.T
    metrics['submitted_generalized_input_max_error_Nm']=float(np.max(np.abs(delivered-generalized)))
    metrics['motor_input_inverse_max_error_Nm']=float(np.max(np.abs(reconstructed-data['actual_motor_forces'])))
    residual=delivered+damping*previous_velocity+friction*np.tanh(previous_velocity/.001)-reconstructed@matrix
    metrics['transmission_residual_max_Nm']=float(np.max(np.abs(residual)))
    checks['submitted_input_readback']=metrics['submitted_generalized_input_max_error_Nm']<1e-4 and metrics['motor_input_inverse_max_error_Nm']<1e-7 and metrics['transmission_residual_max_Nm']<=1e-5
    checks['original_motor_caps']=all(np.all((data[key]>=caps[:,0]-1e-5)&(data[key]<=caps[:,1]+1e-5)) for key in ('motor_forces','actual_motor_forces'))
    checks['zero_unstepped_motor_input']=np.max(np.abs(contract['initial_motor_input']))<1e-8
    checks['decision_clock_and_causal_input']=len(decisions)==n and all(abs(r['time_s']-i*dt)<1e-8 and abs(r['pose_time_s']-i*dt)<1e-8 and np.allclose(r['preceding_actual_motor_forces'],contract['initial_motor_input'] if i==0 else data['actual_motor_forces'][i-1],atol=1e-8,rtol=0) for i,r in enumerate(decisions))
    checks['decision_state_and_interval_match_preceding_endpoint']=len(decisions)==n and all(np.allclose(r['root'],reset['root'] if i==0 else root[i-1],atol=1e-8,rtol=0) and np.allclose(r['contact_interval_s'],[max(0.,(i-1)*dt),i*dt],atol=1e-8,rtol=0) for i,r in enumerate(decisions))
    checks['reset_and_completed_contact_epochs']=len(contacts)==n+1 and all(abs(r['time_s']-i*dt)<1e-8 and abs(r['pose_time_s']-i*dt)<1e-8 and np.allclose(r['contact_interval_s'],[max(0.,(i-1)*dt),i*dt],atol=1e-8,rtol=0) for i,r in enumerate(contacts))
    feet=[];depths=dict(self=0.,environment=0.,working_hand=0.);slot_max=dict(normal=0,friction=0)
    for episode_index,row in enumerate(contacts):
        buffers=unpack_contacts(row,layout);loads=np.zeros(2)
        for kind in slot_max:slot_max[kind]=max(slot_max[kind],len(buffers[kind][0]))
        labels,arrays=buffers['normal']
        for k,(i,j) in enumerate(labels):
            name=layout['sensor_paths'][i].rsplit('/',1)[-1];other=layout['filter_paths'][i][j]
            if name in ('left_ankle_link','right_ankle_link') and other.rsplit('/',1)[-1]=='floor':
                loads[('left_ankle_link','right_ankle_link').index(name)]+=arrays['force_N'][k,0]*arrays['normal_world'][k,2]
                continue
            category='self' if other.startswith('/World/H1/') else 'working_hand' if name.startswith('rh_') and other.startswith('/World/Door/Articulation/') else 'environment'
            if episode_index:depths[category]=max(depths[category],float(-arrays['distance_m'][k,0]))
        feet.append(loads)
    metrics['maximum_occupied_slots']=slot_max;metrics['independent_penetration_m']=depths
    checks['all_contact_buffers_structurally_valid']=True
    checks['floor_support_reproduces_archive']=np.allclose(feet[1:],data['actual_foot_loads'],atol=1e-5,rtol=0) and all(np.allclose(r['foot_loads_N'],feet[i],atol=1e-5,rtol=0) for i,r in enumerate(decisions))
    model=mujoco.MjModel.from_xml_path(str(robot_path));limits=np.array([model.jnt_range[model.joint(name).id] for name in names])
    violation=np.maximum(limits[:,0]-data['joints'],data['joints']-limits[:,1])
    metrics['maximum_authored_joint_stop_penetration_rad']=float(max(0.,violation.max()))
    loop=np.column_stack([data['joints'][:,index[f'{side}_{digit}J1']]-data['joints'][:,index[f'{side}_{digit}J2']] for side in ('rh','lh') for digit in ('FF','MF','RF','LF')])
    metrics['maximum_loopback_violation_rad']=float(max(0.,loop.max()))
    metrics['maximum_torso_tilt_deg']=float(data['torso_tilt_deg'].max());metrics['minimum_root_height_m']=float(root[:,2].min())
    checks['original_mechanical_limits']=metrics['maximum_authored_joint_stop_penetration_rad']<.02 and metrics['maximum_loopback_violation_rad']<.02 and max(depths.values())<.003
    checks['reported_plant_invariants_and_mechanics']=mechanics['passed'] is True and all(v is True for v in mechanics['checks'].values()) and mechanics['contact_samples']==n
    checks['upright']=metrics['maximum_torso_tilt_deg']<12 and metrics['minimum_root_height_m']>.7
    checks['complete_reset_and_pad_evidence']=len(pads)==n+1 and pads[0]['active_contact_count']==0 and all(abs(row['sim_time_s']-i*dt)<1e-8 for i,row in enumerate(pads))
    checks['incomplete_task_is_not_promoted']=report['passed'] is False and report['opening_prefix'] is None and report['controller_failure'] is None and report['termination']=='declared_timeout_or_failure'
    checks['no_runtime_pose_writes_or_door_commands']=report['runtime_robot_pose_writes']==0 and report['direct_door_commands'] is False and report['native_mirror_steps']==0
    checks['no_error_or_early_stop']=not any((run/name).exists() for name in ('early-stop.json','motor-delivery-failure.json')) and 'PHYSX_RUN_COMPLETE' in (run/'run.log').read_text() and 'Traceback (most recent call last)' not in (run/'run.log').read_text()
    checks={k:bool(v) for k,v in checks.items()}
    return dict(schema='doorbench.isaac-traversal-smoke-audit.v1',passed=all(checks.values()),checks=checks,metrics=metrics,source_spec_sha256=sha(spec_path),auditor_sha256=sha(__file__),evidence_sha256={str(p.relative_to(run)):sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name not in ('smoke-audit.json',)},scope='One actual second of initialization, motor input, contact epoch and velocity-frame checks; no opening/traversal or sensor-only qualification',whole_task_passed=False,limitation='Joint bounds independently use authored native XML; live plant invariant readback is retained from the executed audit. Backend input is not independent motor torque. Decision rows end at.998s, endpoint physics/contact rows end at1s.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','spec','motors','native-robot','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=audit(a.run,a.spec,a.motors,a.native_robot)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
