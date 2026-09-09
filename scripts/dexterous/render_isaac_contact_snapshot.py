#!/usr/bin/env python3
"""Render measured Isaac hand-body poses and contacts, without IK or physics.

Native source hand meshes are placed at archived PhysX body transforms. The
lever is its audited collision cylinder. This is a diagnostic geometry view,
not an Isaac camera image, a full-scene replay, or a new task result.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.json_record_stream import iter_json_object_array


def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def measured_body_pose(value):
    value=np.asarray(value,float)
    if value.shape!=(7,) or not np.isfinite(value).all() or abs(np.linalg.norm(value[3:])-1)>1e-5:
        raise ValueError('Require finite measured position and unit XYZW quaternion')
    return value[:3],Rotation.from_quat(value[3:]).as_matrix()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trial','robot','door','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--at',type=float,action='append',required=True)
    p.add_argument('--azimuth',type=float,action='append')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh diagnostic output directory')
    if not np.isfinite(a.at).all() or min(a.at)<0:raise ValueError('Finite nonnegative snapshot epochs required')
    cfg=json.loads((a.trial/'configuration.json').read_text());prov=json.loads((a.trial/'provenance.json').read_text())
    source=cfg['args']['native_robot']
    if not source or digest(a.robot)!=prov['files'][source]:raise ValueError('Exact original native hand source required')
    pad_file=a.trial/'acquisition-pad-steps.json.gz'
    snapshots={}
    with gzip.open(pad_file,'rt') as f:
        for row in iter_json_object_array(f):
            for requested in a.at:
                if requested not in snapshots and row['sim_time_s']>=requested-1e-8:
                    if row['sim_time_s']-requested>row['physics_dt_s']+1e-8:raise ValueError('Missing requested contact epoch')
                    snapshots[requested]=row
            if len(snapshots)==len(a.at):break
    if len(snapshots)!=len(set(a.at)):raise ValueError('Requested epoch outside completed contact recording')
    sim=DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d
    receipt=dict(scope=__doc__,renderer_source_sha256=digest(Path(__file__)),physics_steps=0,robot_source_sha256=digest(a.robot),pad_recording_sha256=digest(pad_file),frames=[])
    a.output.mkdir(parents=True)
    camera=mujoco.MjvCamera();camera.distance=.32;camera.elevation=-20
    options=mujoco.MjvOption();options.sitegroup[:]=0
    try:
        mujoco.mj_kinematics(m,d)
        with mujoco.Renderer(m,height=720,width=960) as renderer:
            for requested,row in snapshots.items():
                raw=row['raw_evidence'];t=row['sim_time_s']
                if abs(raw['geometry_time_s']-t)>1e-8 or abs(raw['interval_end_s']-t)>1e-8:raise ValueError('Pose/contact clocks differ')
                poses={k.rsplit('/',1)[-1]:measured_body_pose(v) for k,v in raw['body_transforms_xyzw'].items()}
                if len(poses)!=len(raw['body_transforms_xyzw']):raise ValueError('Aliased hand body identities')
                maximum_error=0.
                for patch in row['contacts']:
                    pos,rot=poses[patch['body'].rsplit('/',1)[-1]]
                    err=float(np.linalg.norm(pos+rot@np.asarray(patch['body_position_m'])-np.asarray(patch['position'])))
                    maximum_error=max(maximum_error,err)
                if maximum_error>2e-6:raise ValueError('Measured pose does not reproduce archived contact point')
                lever=raw['lever'];axis=np.asarray(lever['axis'],float);center=np.asarray(lever['center'],float)
                if axis.shape!=(3,) or center.shape!=(3,) or not np.isfinite(np.r_[axis,center]).all() or abs(np.linalg.norm(axis)-1)>1e-5:raise ValueError('Invalid measured lever cylinder')
                cylinder_rotation=Rotation.align_vectors(axis[None],np.array([[0.,0.,1.]]))[0].as_matrix()
                camera.lookat[:]=center
                for azimuth in a.azimuth or [90.,150.,230.]:
                    camera.azimuth=azimuth;renderer.update_scene(d,camera=camera,scene_option=options)
                    scene=renderer.scene;shown=set()
                    for i in range(scene.ngeom):
                        g=scene.geoms[i]
                        if g.objtype!=mujoco.mjtObj.mjOBJ_GEOM or g.objid<0:
                            g.rgba[3]=0;continue
                        geom=int(g.objid);body=m.body(int(m.geom_bodyid[geom])).name
                        if not body.startswith('robot/rh_'):
                            g.rgba[3]=0;continue
                        name=body.removeprefix('robot/')
                        if name not in poses:raise ValueError('Missing measured pose for visible hand link: '+name)
                        pos,rot=poses[name];local=np.empty(9);mujoco.mju_quat2Mat(local,m.geom_quat[geom])
                        g.pos[:]=pos+rot@m.geom_pos[geom];g.mat[:]=(rot@local.reshape(3,3)).reshape(g.mat.shape)
                        g.rgba[:]=[.15,.6,.95,1] if name.startswith('rh_th') else [.7,.72,.74,1];shown.add(name)
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_CYLINDER,np.array([lever['radius'],lever['radius'],lever['half_length']]),center,cylinder_rotation.ravel(),np.array([.8,.5,.12,1.]));scene.ngeom+=1
                    for patch in row['contacts']:
                        if patch['normal_force_N']<=.1:continue
                        color=[.1,.9,.15,1.] if patch['pad_qualified'] else [1.,.1,.1,1.]
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.0015),np.asarray(patch['position']),np.eye(3).ravel(),np.asarray(color));scene.ngeom+=1
                    frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,58),fill='black')
                    draw.text((12,10),'MEASURED ISAAC HAND POSES / SOURCE MESH DIAGNOSTIC / NO PHYSICS',fill='white')
                    draw.text((12,33),f't={t:.3f}s | thumb blue | green: qualified patch | red: excluded patch | grasp={row["valid_pad_grasp"]}',fill='white')
                    path=a.output/f'hand-t{t:.3f}-az{azimuth:g}.png';frame.save(path)
                    receipt['frames'].append(dict(path=path.name,sha256=digest(path),requested_time_s=requested,actual_time_s=t,maximum_contact_mapping_error_m=maximum_error,measured_hand_links=len(shown),qualified_grasp=row['valid_pad_grasp']))
    finally:sim.close()
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
