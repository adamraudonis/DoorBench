"""Detached IMU production, epoch and integration audit; no actor/plant changes."""
import argparse,gzip,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from scripts.dexterous.diagnose_support_orientation import rotation,sha,lines,summary


def integrate_sample(R,gyro,accel,elapsed,gain):
    R=R@Rotation.from_rotvec(np.asarray(gyro)*elapsed).as_matrix()
    norm=np.linalg.norm(accel)
    if gain and abs(norm-9.81)<1.:
        error=np.cross(np.asarray(accel)/norm,R.T@np.array([0,0,1]))
        R=R@Rotation.from_rotvec(gain*elapsed*error).as_matrix()
    return R


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('robot','calibration','native','isaac','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Retain earlier diagnostics')
    a.output.mkdir(parents=True)
    for name in ('mj_step','mj_step1','mj_step2'):
        setattr(mujoco,name,lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('No physics in IMU audit')))
    m=mujoco.MjModel.from_xml_path(str(a.robot));d=mujoco.MjData(m)
    names=[m.joint(i).name for i in range(1,m.njnt)];qa=np.array([m.joint(n).qposadr[0] for n in names]);va=np.array([m.joint(n).dofadr[0] for n in names])
    site=m.site('imu').id;jp=np.zeros((3,m.nv));jr=jp.copy();cal=json.loads(a.calibration.read_text())
    if cal['robot_xml_sha256']!=sha(a.robot):raise ValueError('Bound original robot required')
    report=dict(schema='doorbench.recorded-imu-diagnostic.v1',scope=__doc__,physics_steps=0,results={},
        input_sha256={str(f):sha(f) for f in (a.robot,a.calibration,Path(__file__))},
        acceleration_scope='Reconstructed from differentiated site velocity at native FK; imported rigid-body pose/COM velocity may differ numerically. No raw IMU world pose is supplied to any actor.')
    for label,run in [('native',a.native),('isaac',a.isaac)]:
        if label=='native':
            tr=np.load(run/'trajectory.npz',allow_pickle=False);pk=np.load(run/'actor-inputs.npz',allow_pickle=False)
            if tr['joint_names'].tolist()!=names:raise ValueError('Native named order differs')
            roots=tr['qpos'][:,3:10];angles=tr['qpos'][:,10:];qvel=tr['qvel'][:,3:]
            infos=lines(run/'controller.jsonl.gz');sensors={k:pk[k] for k in ('joint_position','imu_gyro','imu_accelerometer','sensor_valid','sensor_time_s')}
            files=[run/f for f in ('trajectory.npz','actor-inputs.npz','controller.jsonl.gz')]
        else:
            with gzip.open(run/'balance-steps.json.gz','rt') as f:rows=json.load(f)
            query=np.load(run/'teacher-query-evidence/pre-action-measurements.npz',allow_pickle=False)
            contract=json.loads((run/'teacher-query-evidence/contract.json').read_text());order=contract['joint_order'];indices=[order.index(n) for n in names]
            if len(set(order))!=69 or set(order)!=set(names):raise ValueError('Query joint order differs')
            np.testing.assert_allclose(query['time_s'],np.arange(9500)*.002,rtol=0,atol=1e-8)
            pk=np.load(run/'sensors/actor-sensors.npz',allow_pickle=False);layout=json.loads((run/'sensors/layout.json').read_text())
            if layout['joint_order']!=names:raise ValueError('Actor joint order differs')
            roots=np.r_[query['root_state'][:,:7],np.array(rows[-1]['root13_actororigin'][:7])[None]]
            angles=np.r_[query['joint_position'][:,indices],pk['joint_position'][-1:]]
            vroot=np.r_[query['root_state'][:,7:],np.array(rows[-1]['root13_actororigin'][7:])[None]]
            # Lab's legacy root_state_w uses COM linear velocity. The native
            # free-joint Jacobian requires velocity at the actor origin. These
            # actual link-state readbacks stay exclusively on the scoring side.
            reset=json.loads((run/'balance-acquisition-reset.json').read_text())
            actor_roots=np.array([reset['root13_actororigin']]+[r['root13_actororigin'] for r in rows])
            np.testing.assert_array_equal(roots,actor_roots[:,:7])
            np.testing.assert_array_equal(vroot[:,3:],actor_roots[:,10:])
            vroot[:,:3]=actor_roots[:,7:10]
            np.testing.assert_array_equal(query['joint_position'][1:,indices],pk['joint_position'][:-1])
            np.testing.assert_array_equal(query['joint_velocity'][1:,indices],pk['joint_velocity'][:-1])
            qvel=np.c_[vroot,np.r_[query['joint_velocity'][:,indices],pk['joint_velocity'][-1:]]]
            for i in range(9501):qvel[i,3:6]=rotation(roots[i,3:]).T@qvel[i,3:6]
            infos=[r['controller_info'] for r in rows]
            initial=np.load(run/'sensors/actor-initial-decision.npz',allow_pickle=False)
            sensors={k:np.r_[initial[k][None],pk[k][:-1]] for k in ('joint_position','imu_gyro','imu_accelerometer','sensor_valid','sensor_time_s')}
            files=[run/f for f in ('balance-steps.json.gz','teacher-query-evidence/pre-action-measurements.npz','teacher-query-evidence/contract.json',
                'sensors/actor-sensors.npz','sensors/actor-initial-decision.npz','sensors/layout.json','balance-acquisition-reset.json')]
        if roots.shape!=(9501,7) or angles.shape!=(9501,69) or qvel.shape!=(9501,75) or len(infos)!=9500:raise ValueError('Complete physical records required')
        report['input_sha256'].update({str(f):sha(f) for f in files})
        R0=rotation(roots[0,3:]);origin=np.r_[roots[0,:2],0.];frames=[];worldv=[];gyros=[];positions=[]
        for i in range(9501):
            # Actual root/velocities enter only this scoring-side FK calculator.
            d.qpos[:7]=roots[i];d.qpos[qa]=angles[i];d.qvel[:]=qvel[i]
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_jacSite(m,d,jp,jr,site)
            R=d.site_xmat[site].reshape(3,3);frames.append(R0.T@R)
            positions.append(R0.T@(d.site_xpos[site]-origin));worldv.append(R0.T@(jp@d.qvel));gyros.append(R.T@(jr@d.qvel))
        frames=np.array(frames);worldv=np.array(worldv);gyros=np.array(gyros);positions=np.array(positions)
        # Distinguish packet errors from a mismatch between saved generalized
        # velocities and pose increments. No actual pose is fed back to these
        # detached integrators after the known reset.
        root_rotation=np.array([R0.T@rotation(r[3:]) for r in roots])
        omega=np.einsum('nij,nj->ni',root_rotation,qvel[:,3:6])
        torso=names.index('torso');torso_va=va[torso]
        pose_consistency={};root_integrals={k:root_rotation[0].copy() for k in ('current','previous','trapezoid')}
        joint_integrals={k:float(angles[0,torso]) for k in root_integrals}
        for k in root_integrals:pose_consistency[k+'_root_rotvec_rad']=[];pose_consistency[k+'_torso_angle_error_rad']=[]
        pose_consistency['root_increment_minus_endpoint_velocity_rad']=[]
        pose_consistency['torso_increment_minus_endpoint_velocity_rad']=[]
        for i in range(1,9501):
            for k in root_integrals:
                blend={'current':1.,'previous':0.,'trapezoid':.5}[k]
                w=blend*omega[i]+(1-blend)*omega[i-1]
                root_integrals[k]=Rotation.from_rotvec(w*.002).as_matrix()@root_integrals[k]
                joint_integrals[k]+=.002*(blend*qvel[i,torso_va]+(1-blend)*qvel[i-1,torso_va])
                pose_consistency[k+'_root_rotvec_rad'].append(Rotation.from_matrix(root_integrals[k]@root_rotation[i].T).as_rotvec())
                pose_consistency[k+'_torso_angle_error_rad'].append(joint_integrals[k]-angles[i,torso])
            pose_consistency['root_increment_minus_endpoint_velocity_rad'].append(Rotation.from_matrix(root_rotation[i]@root_rotation[i-1].T).as_rotvec()-.002*omega[i])
            pose_consistency['torso_increment_minus_endpoint_velocity_rad'].append(angles[i,torso]-angles[i-1,torso]-.002*qvel[i,torso_va])
        pose_consistency={k:np.asarray(v) for k,v in pose_consistency.items()}
        np.savez_compressed(a.output/(label+'-pose-velocity.npz'),time_s=np.arange(1,9501)*.002,**pose_consistency)
        acceleration=np.r_[np.zeros((1,3)),np.diff(worldv,axis=0)/.002]+[0,0,9.81]
        expected_accel=np.einsum('nij,nj->ni',np.swapaxes(frames,1,2),acceleration)
        # Sensor-only replay starts with the declared upright calibrated posture,
        # not an actual-orientation correction after motion begins.
        d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qa]=[cal['desired_posture'][n] for n in names];mujoco.mj_kinematics(m,d)
        initial_R=d.site_xmat[site].reshape(3,3).copy()
        states={k:initial_R.copy() for k in ('existing','gyro_only','gyro_include_first_span','fk_gyro_only')};previous=None
        metrics={k:[] for k in ('gyro_error_radps','gyro_minus_previous_epoch_radps','gyro_minus_next_epoch_radps','accel_error_mps2',
            'accel_minus_gravity_mps2','capture_age_s','gravity_update_error_rad','existing_replay_error_rad',
            'imu_actual_increment_minus_gyro_rad','imu_increment_minus_transported_endpoint_gyro_rad','existing_capture_error_rad','existing_decision_error_rad',
            'gyro_only_capture_error_rad','gyro_only_decision_error_rad','gyro_first_capture_error_rad','fk_gyro_capture_error_rad')};samples=[]
        for i in range(9500):
            if not sensors['sensor_valid'][i,2]:
                if i!=0:raise ValueError('Unexpected gyro dropout')
                continue
            stamp=float(sensors['sensor_time_s'][i,2]);astamp=float(sensors['sensor_time_s'][i,3]);idx=round(stamp/.002);ai=round(astamp/.002)
            if abs(idx*.002-stamp)>1e-8 or abs(ai*.002-astamp)>1e-8 or not 0<=idx<=i or ai>i:raise ValueError('Unmapped/noncausal IMU epoch')
            gyro=sensors['imu_gyro'][i].astype(float);accel=sensors['imu_accelerometer'][i].astype(float)
            elapsed=0. if previous is None else stamp-previous
            before=states['existing'].copy()
            states['existing']=integrate_sample(before,gyro,accel,elapsed,.2)
            states['gyro_only']=integrate_sample(states['gyro_only'],gyro,accel,elapsed,0)
            states['gyro_include_first_span']=integrate_sample(states['gyro_include_first_span'],gyro,accel,stamp if previous is None else elapsed,0)
            states['fk_gyro_only']=integrate_sample(states['fk_gyro_only'],gyros[idx],accel,elapsed,0)
            d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qa]=sensors['joint_position'][i];mujoco.mj_kinematics(m,d)
            recorded=rotation(np.asarray(infos[i]['estimated_root_local'])[3:])@d.site_xmat[site].reshape(3,3)
            def err(R,j):return float(Rotation.from_matrix(R@frames[j].T).magnitude())
            vals=dict(gyro_error_radps=gyro-gyros[idx],gyro_minus_previous_epoch_radps=gyro-gyros[max(0,idx-1)],
                gyro_minus_next_epoch_radps=gyro-gyros[min(9500,idx+1)],accel_error_mps2=accel-expected_accel[ai],
                accel_minus_gravity_mps2=accel-frames[ai].T@np.array([0,0,9.81]),capture_age_s=i*.002-stamp,
                gravity_update_error_rad=float(Rotation.from_matrix(states['existing']@(before@Rotation.from_rotvec(gyro*elapsed).as_matrix()).T).magnitude()),
                existing_replay_error_rad=float(Rotation.from_matrix(states['existing']@recorded.T).magnitude()),
                imu_actual_increment_minus_gyro_rad=(Rotation.from_matrix(frames[max(0,idx-1)].T@frames[idx]).as_rotvec()-gyro*.002),
                imu_increment_minus_transported_endpoint_gyro_rad=(Rotation.from_matrix(frames[max(0,idx-1)].T@frames[idx]).as_rotvec()
                    -(frames[max(0,idx-1)].T@frames[idx])@gyro*.002),
                existing_capture_error_rad=err(states['existing'],idx),existing_decision_error_rad=err(states['existing'],i),
                gyro_only_capture_error_rad=err(states['gyro_only'],idx),gyro_only_decision_error_rad=err(states['gyro_only'],i),
                gyro_first_capture_error_rad=err(states['gyro_include_first_span'],idx),fk_gyro_capture_error_rad=err(states['fk_gyro_only'],idx))
            for k,v in vals.items():metrics[k].append(v)
            if i in (1,2,5,10,25,50,100,500,7500,8500,9000,9499):samples.append(dict(decision=i,time_s=i*.002,capture_s=stamp,
                gyro=gyro.tolist(),expected_gyro=gyros[idx].tolist(),accel=accel.tolist(),expected_accel=expected_accel[ai].tolist(),
                **{k:np.asarray(v).tolist() for k,v in vals.items()}))
            previous=stamp
        arrays={k:np.asarray(v) for k,v in metrics.items()};times=np.arange(1,9500)*.002
        np.savez_compressed(a.output/(label+'-metrics.npz'),time_s=times,**arrays)
        windows={'all':np.ones(9499,bool),'first_100ms':times<=.1+1e-9,'settled_first_second':(times>=.5)&(times<=1.),'last_half_second':times>=18.5}
        result=dict(replayed_decisions=9499,maximum_replay_error_rad=float(arrays['existing_replay_error_rad'].max()),
            windows={w:{k:summary(v[mask]) for k,v in arrays.items()} for w,mask in windows.items()},samples=samples,
            pose_velocity_consistency={k:summary(v) for k,v in pose_consistency.items()},
            pose_velocity_final={k:v[-1].tolist() for k,v in pose_consistency.items()})
        report['results'][label]=result
        print(json.dumps({label:{w:{k:r[k]['rms'] for k in ('gyro_error_radps','existing_capture_error_rad','gyro_only_capture_error_rad','fk_gyro_capture_error_rad','accel_error_mps2')} for w,r in result['windows'].items()}}),flush=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
