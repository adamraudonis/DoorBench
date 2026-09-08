#!/usr/bin/env python3
"""Offline assessment of an explicitly different ideal delta-angle gyro.

Actual own-IMU poses are evaluator/sensor-producer data only. No controller or
physics run is modified. A synthetic actor would receive only three angular-rate
numbers; this assessment does not itself qualify any runtime sensor profile.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from scripts.dexterous.diagnose_recorded_imu import integrate_sample
from scripts.dexterous.diagnose_support_orientation import rotation,summary
from doorbench.dexterous.sensor_balance import SensorBalanceController


def delta_angle_gyro(rotations,dt):
    """Causal interval-local SO(3) increment, available at interval end.

The net rotation's axis has identical coordinates in the interval's start and
end body frames. It is a finite delta angle / dt, not instantaneous omega.
"""
    r=np.asarray(rotations,dtype=float)
    if r.ndim!=3 or r.shape[1:]!=(3,3) or len(r)<2 or not np.isfinite(r).all():raise ValueError('Finite rotation sequence required')
    if not np.isfinite(dt) or dt<=0:raise ValueError('Positive interval required')
    if np.max(abs(np.swapaxes(r,1,2)@r-np.eye(3)))>1e-7 or np.max(abs(np.linalg.det(r)-1))>1e-7:raise ValueError('Proper rotations required')
    v=Rotation.from_matrix(np.swapaxes(r[:-1],1,2)@r[1:]).as_rotvec()
    if np.max(np.linalg.norm(v,axis=1))>=np.pi-1e-6:raise ValueError('Ambiguous interval rotation')
    return v/dt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('robot','motors','calibration','native','isaac','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    for n in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,n,lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('No physical stepping')))
    m=mujoco.MjModel.from_xml_path(str(a.robot));d=mujoco.MjData(m);names=[m.joint(i).name for i in range(1,m.njnt)];qa=np.array([m.joint(n).qposadr[0] for n in names]);cal=json.loads(a.calibration.read_text());assert cal['robot_xml_sha256']==sha(a.robot)
    d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qa]=[cal['desired_posture'][n] for n in names];mujoco.mj_kinematics(m,d);initial_R=d.site('imu').xmat.reshape(3,3).copy()
    result={'schema':'doorbench.pose-derived-imu-assessment.v1','scope':__doc__,'runtime_changed':False,'physical_steps':0,'input_sha256':{str(f):sha(f) for f in (a.robot,a.motors,a.calibration,Path(__file__),Path('doorbench/dexterous/sensor_balance.py'))},'results':{}}
    for label,run in [('native',a.native),('isaac',a.isaac)]:
        if label=='native':
            tr=np.load(run/'trajectory.npz',allow_pickle=False);pk=np.load(run/'actor-inputs.npz',allow_pickle=False);assert tr['joint_names'].tolist()==names
            roots=tr['qpos'][:,3:10];angles=tr['qpos'][:,10:];gyro=pk['imu_gyro'][1:].astype(float);accel=pk['imu_accelerometer'][1:].astype(float);times=pk['sensor_time_s'][1:,2];files=[run/'trajectory.npz',run/'actor-inputs.npz']
        else:
            with gzip.open(run/'balance-steps.json.gz','rt') as f:rows=json.load(f)
            reset=json.loads((run/'balance-acquisition-reset.json').read_text());roots=np.array([reset['root13_actororigin'][:7]]+[r['root13_actororigin'][:7] for r in rows])
            pk=np.load(run/'sensors/actor-sensors.npz',allow_pickle=False);layout=json.loads((run/'sensors/layout.json').read_text());assert layout['joint_order']==names
            angles=np.r_[np.array([reset['joint_position'][n] for n in names])[None],pk['joint_position']]
            gyro=pk['imu_gyro'][:-1].astype(float);accel=pk['imu_accelerometer'][:-1].astype(float);times=pk['sensor_time_s'][:-1,2]
            files=[run/f for f in ('balance-steps.json.gz','balance-acquisition-reset.json','sensors/actor-sensors.npz','sensors/layout.json')]
        result['input_sha256'].update({str(f):sha(f) for f in files});result['input_sha256'][str(a.isaac/'sensors/layout.json')]=sha(a.isaac/'sensors/layout.json');layout=json.loads((a.isaac/'sensors/layout.json').read_text());motors=json.loads(a.motors.read_text())
        packet_arrays={k:pk[k][1:] if label=='native' else pk[k][:-1] for k in ('joint_position','joint_velocity','imu_gyro','imu_accelerometer','sensor_time_s','tactile')}
        indices=np.rint(times/.002).astype(int);np.testing.assert_allclose(times,indices*.002,rtol=0,atol=1e-8)
        np.testing.assert_array_equal(indices,np.arange(9499)+(0 if label=='native' else 1))
        assert roots.shape==(9501,7) and angles.shape==(9501,69)
        r0=rotation(roots[0,3:]);origin=np.r_[roots[0,:2],0.];frames=[];positions=[];palms=[];palm_rotations=[]
        for root,joints in zip(roots,angles):
            d.qpos[:7]=root;d.qpos[qa]=joints;mujoco.mj_kinematics(m,d)
            frames.append(r0.T@d.site('imu').xmat.reshape(3,3));positions.append(r0.T@(d.site('imu').xpos-origin));palms.append(r0.T@(d.site('rh_palm_touch').xpos-origin));palm_rotations.append(r0.T@d.site('rh_palm_touch').xmat.reshape(3,3))
        frames=np.array(frames);positions=np.array(positions);palms=np.array(palms);palm_rotations=np.array(palm_rotations);ideal=np.r_[np.zeros((1,3)),delta_angle_gyro(frames,.002)][indices]
        # Retain the frozen estimator's first-valid-sample skip separately from a
        # candidate interval-aware reset (known dt, never actual reset attitude).
        variants={'legacy_gyro_gravity':(gyro,.2,False),'pose_gyro_existing_gravity':(ideal,.2,False),
            'pose_gyro_interval_gravity':(ideal,.2,True),'pose_gyro_interval_only':(ideal,0,True)}
        arrays={'pose_minus_backend_gyro_radps':ideal-gyro};details={}
        for name,(stream,gain,first) in variants.items():
            R=initial_R.copy();errs=[]
            for i in range(9499):
                R=integrate_sample(R,stream[i],accel[i],(.002 if i else float(times[0]) if first else 0.),gain)
                errs.append(Rotation.from_matrix(R@frames[indices[i]].T).as_rotvec())
            arrays[name+'_error_rad']=np.array(errs)
            estimator=SensorBalanceController(a.robot,motors,layout,cal['desired_posture'],gravity_correction=gain)
            if first:estimator.last_gyro_time=0.
            pe=[];pre=[];re=[]
            for i in range(9499):
                packet={k:v[i].copy() for k,v in packet_arrays.items()};packet['imu_gyro']=np.asarray(stream[i],dtype=np.float32)
                estimator._estimate(packet) # Estimator only: no force call, QP solve, or action-history substitution.
                pe.append(estimator.d.site('rh_palm_touch').xpos-palms[i+1])
                pre.append(Rotation.from_matrix(estimator.d.site('rh_palm_touch').xmat.reshape(3,3)@palm_rotations[i+1].T).as_rotvec())
                re.append(estimator.d.qpos[:3]-r0.T@(roots[i+1,:3]-origin))
            arrays[name+'_palm_position_error_m']=np.array(pe)
            arrays[name+'_palm_orientation_error_rad']=np.array(pre)
            arrays[name+'_root_position_error_m']=np.array(re)
        # Pose quantization only: repeat the same calculation after a float32
        # sensor orientation export; this is not a stochastic sensor-noise fit.
        q32=Rotation.from_matrix(r0@frames).as_quat().astype(np.float32)
        quant=np.r_[np.zeros((1,3)),delta_angle_gyro(np.swapaxes(np.broadcast_to(r0,frames.shape),1,2)@Rotation.from_quat(q32.astype(float)).as_matrix(),.002)][indices]
        arrays['float32_pose_quantization_rate_error_radps']=quant-ideal
        # Backward second pose difference has centre t-dt, availability t.
        # Compare to recorded acceleration at that centre, after moving both to
        # the same world frame. This detects mixed kinematic/dynamic semantics.
        pose_world_accel=np.diff(positions,n=2,axis=0)/.002**2
        recorded_world_accel=np.einsum('nij,nj->ni',frames[indices],accel)-[0,0,9.81]
        # First native capture has no earlier pose interval; exclude it from
        # acceleration derivative statistics rather than invent a previous pose.
        acceleration_valid=(indices>=1)&(indices<9500)
        matched_pose_accel=pose_world_accel[np.maximum(0,indices-1)]
        arrays['pose_second_difference_minus_backend_accel_mps2']=matched_pose_accel-recorded_world_accel
        arrays['pose_second_difference_accel_mps2']=matched_pose_accel
        arrays['backend_world_accel_mps2']=recorded_world_accel
        windows={'first_100ms':times<=.1+1e-8,'all':np.ones(9499,dtype=bool),'late_quiet':times>=18.5}
        details['windows']={w:{k:summary(v[mask & acceleration_valid] if 'accel' in k else v[mask]) for k,v in arrays.items()} for w,mask in windows.items()}
        details['final_error_rotvec_rad']={n:arr[-1].tolist() for n,arr in arrays.items() if n.endswith('_error_rad')}
        details['timing']={'interval_s':.002,'available_at':'interval end, same as current packet producer','delta_angle_effective_midpoint_delay_s':.001,'extra_transport_delay_s':0.,'pose_acceleration_effective_midpoint_delay_s':.002,'first_packet':'cold t0 invalid; first delta requires reset producer pose and t=.002 endpoint','actor_orientation_input':False,'original_capture_to_decision_delay_s':.002 if label=='native' else 0.,'native_t0_delta_unavailable':label=='native'}
        result['results'][label]=details
        np.savez_compressed(a.output/(label+'-metrics.npz'),time_s=times,**arrays)
        print(json.dumps({label:{w:{k:float(np.linalg.norm(v['rms'])) for k,v in values.items() if 'palm_position' in k or k.endswith('gravity_error_rad')} for w,values in details['windows'].items()}}),flush=True)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
