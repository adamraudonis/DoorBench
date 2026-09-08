#!/usr/bin/env python3
"""Independent completed-run audit of the explicit own-IMU delta-angle profile.

Only archived evaluator data are read. Robot FK never steps a plant; absolute
poses used here are not actor observations. Sensor consistency is separate from
the unchanged physical acquisition result.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct_rates(times, body_quaternions_xyzw, mount_wxyz):
    """Independent vectorized reconstruction, including reset orientation."""
    times = np.asarray(times, float)
    q = np.asarray(body_quaternions_xyzw, float)
    mount = np.asarray(mount_wxyz, float)
    if (times.ndim != 1 or len(times) < 2 or q.shape != (len(times), 4)
            or mount.shape != (4,) or not np.isfinite(times).all()
            or not np.isfinite(q).all() or not np.isfinite(mount).all()):
        raise ValueError('Finite ordered own-body quaternion evidence required')
    if (abs(times[0]) > 1e-10 or not np.allclose(np.diff(times), .002, atol=1e-8, rtol=0)
            or np.max(abs(np.linalg.norm(q, axis=1)-1)) > 1e-5
            or abs(np.linalg.norm(mount)-1) > 1e-7):
        raise ValueError('Reset, consecutive 2ms epochs and calibrated rotations required')
    frames = Rotation.from_quat(q).as_matrix() @ Rotation.from_quat(mount[[1,2,3,0]]).as_matrix()
    delta = Rotation.from_matrix(np.swapaxes(frames[:-1],1,2) @ frames[1:]).as_rotvec()
    if np.max(np.linalg.norm(delta,axis=1)) >= np.pi-1e-6:
        raise ValueError('Ambiguous interval rotation')
    return (delta/.002).astype(np.float32), frames


def stats(values):
    values = np.asarray(values, float)
    norms = np.linalg.norm(values,axis=-1)
    return dict(count=len(values),rms_norm=float(np.sqrt(np.mean(norms**2))),max_norm=float(norms.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','robot','original-layout','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists(): raise FileExistsError('Preserve earlier independent receipts')
    run=a.run;s=run/'sensors';errors=[];checks={}
    layout=json.loads((s/'layout.json').read_text());old=json.loads(a.original_layout.read_text())
    producer=json.loads((s/'gyro-producer.json').read_text());sensor_report=json.loads((s/'report.json').read_text())
    task=json.loads((run/'balance-report.json').read_text())
    with np.load(s/'gyro-producer-evidence.npz',allow_pickle=False) as archive:
        evidence={k:archive[k].copy() for k in archive.files}
    with np.load(s/'actor-sensors.npz',allow_pickle=False) as archive:
        packet={k:archive[k].copy() for k in archive.files}
    with np.load(s/'actor-initial-decision.npz',allow_pickle=False) as archive:
        initial={k:archive[k].copy() for k in archive.files}
    with gzip.open(run/'balance-steps.json.gz','rt') as f:rows=json.load(f)
    reset=json.loads((run/'balance-acquisition-reset.json').read_text())
    count=len(rows);expected=np.arange(1,count+1)*.002
    checks['complete_9500_intervals']=count==9500 and len(packet['time_s'])==9500 and len(evidence['time_s'])==9501
    checks['profile_and_count_receipt']=(producer['profile']=='pose-delta-angle-v1'
        and producer['samples']==count and producer['recorded_actor_packets']==count
        and producer['interval_count_matches_packets'] is True and producer['failed_reason'] is None
        and producer['actor_receives_orientation'] is False and producer['physics_writes']==0)
    checks['producer_evidence_hash']=producer['evaluator_evidence_sha256']==sha(s/'gyro-producer-evidence.npz')
    checks['capture_complete']=sensor_report['capture_complete'] is True
    checks['numeric_capture_hashes']=all(sha(s/n)==h for n,h in sensor_report['numeric_file_sha256'].items())
    expected_layout=json.loads(json.dumps(old));expected_layout['imu']['gyro_profile']='pose-delta-angle-v1'
    checks['only_gyro_profile_changes_layout']=layout==expected_layout
    checks['layout_hashes']=(sensor_report['input_layout_sha256']==sha(a.original_layout)
        and sensor_report['effective_layout_sha256']==sha(s/'layout.json'))
    allowed={'joint_position','joint_velocity','imu_gyro','imu_accelerometer','tactile','previous_action','sensor_time_s','sensor_valid','time_s'}
    checks['actor_numeric_fields_only']=set(packet)==allowed and packet['previous_action'].shape==(count,61)
    checks['cold_t0_invalid']=float(initial['time_s'])==0 and not initial['sensor_valid'].any()
    checks['exact_sensor_and_body_epochs']=(np.allclose(packet['time_s'],expected,atol=1e-8,rtol=0)
        and np.allclose(evidence['time_s'][1:],expected,atol=1e-8,rtol=0)
        and np.allclose(packet['sensor_time_s'][:,2],expected,atol=1e-8,rtol=0)
        and bool(packet['sensor_valid'][:,2].all())
        and np.allclose([r['time_s'] for r in rows],expected,atol=1e-8,rtol=0))
    reconstructed,frames=reconstruct_rates(evidence['time_s'],evidence['body_quaternion_xyzw_world'],layout['imu']['quaternion_wxyz_body'])
    rate_error=reconstructed.astype(float)-packet['imu_gyro'].astype(float)
    checks['every_emitted_rate_reconstructed']=bool(np.max(abs(rate_error))<1e-7)
    # Bind FK to the original robot bytes, named joint order and exact actual
    # pre-reset/endpoint records. No reference route enters this calculation.
    import mujoco
    for name in ('mj_step','mj_step1','mj_step2'):
        setattr(mujoco,name,lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Audit cannot step physics')))
    m=mujoco.MjModel.from_xml_path(str(a.robot));d=mujoco.MjData(m)
    names=layout['joint_order'];qa=np.array([m.joint(n).qposadr[0] for n in names])
    checks['robot_hash_and_joint_order']=(sha(a.robot)==layout['robot_xml_sha256']
        and names==[m.joint(i).name for i in range(1,m.njnt)])
    expected_path=json.loads((s/'scope.json').read_text())['body_paths_by_name'][layout['imu']['body_name']]
    checks['producer_is_declared_robot_imu_body']=producer['source_body_path']==expected_path
    roots=[reset['root13_actororigin']]+[r['root13_actororigin'] for r in rows]
    angles=[reset['joint_position']]+[r['actual_joint_position'] for r in rows]
    actual_frames=[];actual_palms=[];joint_error=0.
    gauge=Rotation.from_quat(np.asarray(roots[0][3:7])[[1,2,3,0]]).as_matrix()
    origin=np.array([roots[0][0],roots[0][1],0.])
    for i,(root,joints) in enumerate(zip(roots,angles)):
        d.qpos[:7]=root[:7];d.qpos[qa]=[joints[n] for n in names];mujoco.mj_kinematics(m,d)
        actual_frames.append(d.site('imu').xmat.reshape(3,3).copy())
        actual_palms.append(gauge.T@(d.site('rh_palm_touch').xpos-origin))
        if i:joint_error=max(joint_error,float(np.max(abs(d.qpos[qa]-packet['joint_position'][i-1]))))
    frame_error=Rotation.from_matrix(frames@np.swapaxes(actual_frames,1,2)).as_rotvec()
    checks['producer_body_matches_named_joint_FK']=bool(np.max(np.linalg.norm(frame_error,axis=1))<1e-5)
    checks['recorded_joint_packets_match_actual_endpoints']=joint_error<1e-7
    # The recorded controller estimate precedes the interval's endpoint. Compare
    # it to exactly the previous physical row and previous encoder packet.
    estimated_palm_error=[];estimated_root_rotation_error=[]
    for i,r in enumerate(rows):
        estimate=np.asarray(r['controller_info']['estimated_root_local'])
        d.qpos[:7]=estimate;d.qpos[qa]=[angles[i][n] for n in names];mujoco.mj_kinematics(m,d)
        estimated_palm_error.append(d.site('rh_palm_touch').xpos-actual_palms[i])
        actual_root=gauge.T@Rotation.from_quat(np.asarray(roots[i][3:7])[[1,2,3,0]]).as_matrix()
        estimated_root=Rotation.from_quat(estimate[3:][[1,2,3,0]]).as_matrix()
        estimated_root_rotation_error.append(Rotation.from_matrix(estimated_root@actual_root.T).as_rotvec())
    checks['calculator_never_stepped']=float(d.time)==0
    times=np.arange(count)*.002
    masks={'first_100ms':times<.1,'first_11s_descriptive_slice':times<11.,'last_0_5s':times>=18.5,'all':np.ones(count,bool)}
    details={key:dict(estimated_palm_position_error_m=stats(np.asarray(estimated_palm_error)[mask]),
                      estimated_root_rotation_error_rad=stats(np.asarray(estimated_root_rotation_error)[mask])) for key,mask in masks.items()}
    paths=[a.robot,a.original_layout,Path(__file__)]+[s/n for n in ('layout.json','report.json','scope.json','gyro-producer.json','gyro-producer-evidence.npz','actor-sensors.npz','actor-initial-decision.npz')]+[run/n for n in ('balance-report.json','balance-steps.json.gz','balance-acquisition-reset.json')]
    report=dict(schema='doorbench.actual-pose-gyro-audit.v1',scope=__doc__,verification_passed=all(checks.values()),checks=checks,
        actual_task_passed=task['passed'],actual_task_checks=task['checks'],errors=errors,intervals=count,
        emitted_gyro_error_radps=stats(rate_error),maximum_component_rate_error_radps=float(np.max(abs(rate_error))),
        actual_body_FK_orientation_error_rad=stats(frame_error),maximum_actual_joint_packet_error_rad=joint_error,
        estimator_same_decision_epoch=details,producer=producer,input_sha256={str(p):sha(p) for p in paths},physics_steps=0,
        limitations=['The first11s is only a descriptive slice of the19s acquisition schedule, not the separate11s reach protocol.',
            'Absolute pose evidence stays evaluator-only. The declared actor gets three interval-local rates.',
            'Backend accelerometer is unchanged and retains mixed velocity/pose numerical semantics.',
            'Audit consistency cannot turn an unsuccessful grasp into success.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('verification_passed','actual_task_passed','checks','maximum_component_rate_error_radps','actual_body_FK_orientation_error_rad','estimator_same_decision_epoch')}))


if __name__=='__main__':main()
