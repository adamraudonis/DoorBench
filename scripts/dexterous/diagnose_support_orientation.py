"""Read-only support-foot orientation hypothesis on archived actual states.

Foot-pose estimates use constant joint calibration, recorded current encoders,
and local tactile weights only. Actual world root/joints enter separate scoring
and foot-slip diagnostics, never the estimate. No dynamics is executed.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rotation(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def lines(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]


def weighted_support_orientation(calibrated_foot_rotation,relative_foot_rotation,loads):
    """Assume the two calibrated foot orientations remain fixed in local ground."""
    loads=np.clip(np.asarray(loads,float),0,500)
    if loads.shape!=(2,) or not np.isfinite(loads).all():raise ValueError('Two finite local support loads required')
    if loads.sum()<1:loads=np.ones(2)
    weights=loads/loads.sum()
    estimates=np.asarray(calibrated_foot_rotation)@np.swapaxes(relative_foot_rotation,1,2)
    return Rotation.from_matrix(estimates).mean(weights=weights).as_matrix(),weights


def summary(values):
    a=np.asarray(values,float);return dict(maximum=np.max(a,axis=0).tolist(),minimum=np.min(a,axis=0).tolist(),
        mean=np.mean(a,axis=0).tolist(),rms=np.sqrt(np.mean(a*a,axis=0)).tolist(),
        percentile95=np.percentile(abs(a),95,axis=0).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','calibration','native','isaac','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep every prior diagnostic unchanged')
    a.output.mkdir(parents=True)
    for name in ('mj_step','mj_step1','mj_step2'):
        setattr(mujoco,name,lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('Detached diagnostic cannot step physics')))
    m=mujoco.MjModel.from_xml_path(str(a.robot));d=mujoco.MjData(m);names=[m.joint(i).name for i in range(1,m.njnt)]
    if (m.nq,m.nv,m.nu)!=(76,75,61):raise ValueError('Original robot-only kinematics required')
    qa=np.array([m.joint(n).qposadr[0] for n in names]);feet=[m.body(s+'_ankle_link').id for s in ('left','right')]
    calibration=json.loads(a.calibration.read_text())
    if calibration['robot_xml_sha256']!=sha(a.robot) or set(calibration['desired_posture'])!=set(names):raise ValueError('Bound original posture required')
    desired=np.array([calibration['desired_posture'][n] for n in names])
    d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qa]=desired;mujoco.mj_kinematics(m,d)
    foot_meshes=[]
    for body in feet:
        parts=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=body or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Expected original collision foot mesh')
            mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy()
            parts.append((g,v))
        if not parts:raise ValueError('Missing collision foot geometry')
        foot_meshes.append(parts)
    def sole_points():
        return [np.concatenate([v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for g,v in parts]) for parts in foot_meshes]
    initial_points=sole_points();height=-min(x[:,2].min() for x in initial_points)
    d.qpos[2]=height;mujoco.mj_kinematics(m,d)
    calR=d.xmat[feet].reshape(2,3,3).copy();calp=d.xpos[feet].copy();points=sole_points()
    sole_anchors=[];sole_normals=[]
    for b,x in zip(feet,points):
        bottom=x[x[:,2]<=x[:,2].min()+1e-4]
        if len(bottom)<3:raise ValueError('Insufficient points on calibrated sole plane')
        center=bottom.mean(axis=0);_,_,vh=np.linalg.svd(bottom-center);normal=vh[-1]
        if normal[2]<0:normal=-normal
        sole_anchors.append(d.xmat[b].reshape(3,3).T@(center-d.xpos[b]))
        sole_normals.append(normal)
    report=dict(schema='doorbench.detached-support-orientation-diagnostic.v1',scope=__doc__,physics_steps=0,
        input_sha256={str(x):sha(x) for x in (a.robot,a.calibration,Path(__file__))},
        calibration=dict(root_height_m=float(height),foot_body_rotation_local=calR.tolist(),foot_origins_local_m=calp.tolist(),
            sole_anchor_body_m=np.asarray(sole_anchors).tolist(),sole_plane_normal_local=np.asarray(sole_normals).tolist()),results={})
    for label,run in [('native',a.native),('isaac',a.isaac)]:
        if label=='native':
            tr=np.load(run/'trajectory.npz',allow_pickle=False);pack=np.load(run/'actor-inputs.npz',allow_pickle=False)
            if tr['joint_names'].tolist()!=names or tr['qpos'].shape!=(9501,79):raise ValueError('Exact original native ordering required')
            np.testing.assert_allclose(tr['time'],np.arange(9501)*.002,rtol=0,atol=1e-8)
            infos=lines(run/'controller.jsonl.gz');roots=tr['qpos'][:-1,3:10];joints=tr['qpos'][:-1,10:];enc=pack['joint_position']
            layout=json.loads((a.isaac/'sensors/layout.json').read_text());touch=pack['tactile']
            files=[run/x for x in ('trajectory.npz','actor-inputs.npz','controller.jsonl.gz','reset.json')]
        else:
            with gzip.open(run/'balance-steps.json.gz','rt') as f:rows=json.load(f)
            reset=json.loads((run/'balance-acquisition-reset.json').read_text());pack=np.load(run/'sensors/actor-sensors.npz',allow_pickle=False)
            np.testing.assert_allclose([r['time_s'] for r in rows],np.arange(1,9501)*.002,rtol=0,atol=1e-8)
            np.testing.assert_allclose(pack['time_s'],np.arange(1,9501)*.002,rtol=0,atol=1e-8)
            roots=np.array([reset['root13_actororigin'][:7]]+[r['root13_actororigin'][:7] for r in rows[:-1]])
            joints=np.array([[reset['joint_position'][n] for n in names]]+[[r['actual_joint_position'][n] for n in names] for r in rows[:-1]])
            enc=np.r_[np.zeros((1,69)),pack['joint_position'][:-1]];touch=np.r_[np.zeros((1,1344)),pack['tactile'][:-1]]
            infos=[r['controller_info'] for r in rows];layout=json.loads((run/'sensors/layout.json').read_text())
            files=[run/x for x in ('balance-steps.json.gz','balance-acquisition-reset.json','sensors/actor-sensors.npz','sensors/layout.json')]
        if roots.shape!=(9500,7) or joints.shape!=(9500,69) or enc.shape!=(9500,69) or len(infos)!=9500:raise ValueError('Complete nineteen-second evidence required')
        if layout['joint_order']!=names or layout['robot_xml_sha256']!=sha(a.robot):raise ValueError('Original sensor order/model required')
        slices={};offset=0
        for s in layout['sensors']:
            if s['name'] in ('left_ankle_touch','right_ankle_touch'):slices[s['name']]=slice(offset,offset+s['dimension'])
            offset+=s['dimension']
        loads=np.stack([np.linalg.norm(touch[:,slices[s+'_ankle_touch']].reshape(9500,3,-1).sum(axis=2),axis=1) for s in ('left','right')],axis=1)
        reported_load=np.array([r['foot_tactile_force_norm_N'] for r in infos]);load_error=float(abs(loads-reported_load).max())
        if load_error>1e-3:raise ValueError('Local touch does not reproduce estimator support weights')
        report['input_sha256'].update({str(x):sha(x) for x in files})
        yaw=rotation(roots[0,3:]);origin=np.r_[roots[0,:2],0.];acc={k:[] for k in ('imu_error_rotvec_rad','foot_error_rotvec_rad',
            'foot_orientation_change_rotvec_rad','foot_origin_drift_m','sole_anchor_drift_m','sole_minimum_z_m','foot_local_tilt_rad',
            'imu_position_error_m','foot_position_error_m','encoder_leg_error_rad','two_foot_estimate_disagreement_rad')}
        firstfeet=None;firstsole=None;samples=[]
        for i in range(9500):
            Ractual=yaw.T@rotation(roots[i,3:]);pactual=yaw.T@(roots[i,:3]-origin)
            d.qpos[:7]=np.r_[pactual,Rotation.from_matrix(Ractual).as_quat()[[3,0,1,2]]];d.qpos[qa]=joints[i];mujoco.mj_kinematics(m,d)
            actualfeet=d.xpos[feet].copy();actualRs=d.xmat[feet].reshape(2,3,3).copy()
            anchors=np.array([actualfeet[j]+actualRs[j]@sole_anchors[j] for j in range(2)])
            bottom=np.array([v[:,2].min() for v in sole_points()])
            if i==0:firstfeet=actualfeet.copy();firstsole=anchors.copy()
            d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[qa]=enc[i] if i else desired;mujoco.mj_kinematics(m,d)
            relativeR=d.xmat[feet].reshape(2,3,3).copy();relativep=d.xpos[feet].copy()
            inferred,weights=weighted_support_orientation(calR,relativeR,loads[i])
            inferredp=np.sum(weights[:,None]*(calp-(inferred@relativep.T).T),axis=0)
            imu=np.array(infos[i]['estimated_root_local']);imuR=rotation(imu[3:])
            one=calR@np.swapaxes(relativeR,1,2)
            values=dict(imu_error_rotvec_rad=Rotation.from_matrix(imuR@Ractual.T).as_rotvec(),
                foot_error_rotvec_rad=Rotation.from_matrix(inferred@Ractual.T).as_rotvec(),
                foot_orientation_change_rotvec_rad=Rotation.from_matrix(actualRs@np.swapaxes(calR,1,2)).as_rotvec(),
                foot_origin_drift_m=actualfeet-firstfeet,sole_anchor_drift_m=anchors-firstsole,sole_minimum_z_m=bottom,
                foot_local_tilt_rad=np.arccos(np.clip((actualRs@np.swapaxes(calR,1,2))[:,2,2],-1,1)),
                imu_position_error_m=imu[:3]-pactual,foot_position_error_m=inferredp-pactual,
                encoder_leg_error_rad=(enc[i]-joints[i])[:10] if i else np.zeros(10),
                two_foot_estimate_disagreement_rad=Rotation.from_matrix(one[0]@one[1].T).magnitude())
            for key,value in values.items():acc[key].append(value)
            if i in (0,1,10,50,100,250,500,1000,5000,6870,7500,8500,9499):
                samples.append(dict(decision=i,time_s=i*.002,actual_root_local_m=pactual.tolist(),support_weights=weights.tolist(),
                    actual_foot_origins_local_m=actualfeet.tolist(),**{k:np.asarray(v).tolist() for k,v in values.items()}))
        arrays={k:np.asarray(v) for k,v in acc.items()}
        np.savez_compressed(a.output/(label+'-metrics.npz'),time_s=np.arange(9500)*.002,**arrays)
        scored=slice(1,None)
        report['results'][label]=dict(decisions=9500,estimator_geometry_epoch='Current pre-decision encoders and actual pre-decision state; existing estimate at same decision',
            reset_root_height_m=float(roots[0,2]),calibration_height_minus_actual_reset_m=float(height-roots[0,2]),
            local_touch_reduction_error_N=load_error,statistics={k:summary(v[scored]) for k,v in arrays.items()},
            orientation_error_norm_rad={k:summary(np.linalg.norm(arrays[k][scored],axis=1)) for k in ('imu_error_rotvec_rad','foot_error_rotvec_rad')},samples=samples)
        print(json.dumps({label:report['results'][label]['orientation_error_norm_rad']}),flush=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
