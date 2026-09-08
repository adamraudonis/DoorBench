#!/usr/bin/env python3
"""Fit a bounded kinematic grasp initializer, never a physical success claim.

Full Shadow joint limits remain intact. Coupled distal joints share the initial
curl, but later physical control must allow their native tendon dynamics.
"""
import argparse
import json
import hashlib
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from PIL import Image
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.provenance import capture


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--robot',default='out/dexterous/robot/h1-shadow.xml')
    p.add_argument('--door',default='out/dexterous/assets/doors/db0055_swing_single')
    p.add_argument('--output',type=Path,default=Path('out/dexterous/grasp-seed'))
    p.add_argument('--attempts',type=int,default=12)
    p.add_argument('--initial-seed',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    capture(Path(__file__).resolve().parents[2],a.output,vars(a))
    robot=Path(a.robot);env=DexterousDoorEnv(a.door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    env.reset(randomize=False,images=False);m,d=env.m,env.d
    # This deliberately starts near the operator, for offline IK only.
    d.qpos[env.root_qadr:env.root_qadr+2]=[-.04,-.60]
    mujoco.mj_kinematics(m,d)
    lever=m.geom('leaf_handle_lever_col_n').id
    center=d.geom_xpos[lever].copy();axis=d.geom_xmat[lever].reshape(3,3)[:,2].copy()
    if axis[0]<0:axis=-axis
    digits=('ff','mf','rf','lf','th')
    sites=[m.site('robot/rh_'+digit+'distal_touch').id for digit in digits]
    normal=np.array([0.,1.,0.])
    palm=m.site("robot/rh_palm_touch").id
    palm_target=center+np.array([0.,-.045,.065])
    targets=np.array([center+axis*x+normal*(.007 if digit!='th' else -.007)
                      for digit,x in zip(digits,[-.036,-.012,.012,.036,-.035])])
    normals=np.array([-normal]*4+[normal])
    names=['right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_yaw']
    names += [m.joint(j).name.removeprefix('robot/') for j in env.joints
              if m.joint(j).name.startswith('robot/rh_') and not any(m.joint(j).name.endswith(k+'J1') for k in ('FF','MF','RF','LF'))]
    joints=[m.joint('robot/'+name).id for name in names]
    qadr=m.jnt_qposadr[joints];low=m.jnt_range[joints,0].copy();high=m.jnt_range[joints,1].copy()
    margin=.04*(high-low);low+=margin;high-=margin
    neutral=d.qpos[qadr].copy();rng=np.random.default_rng(71)
    for i,name in enumerate(names):
        if name.endswith(('FFJ2','MFJ2','RFJ2','LFJ2')):neutral[i]=.7
        if name.endswith(('FFJ3','MFJ3','RFJ3','LFJ3')):neutral[i]=.6
    neutral=np.clip(neutral,low+1e-5,high-1e-5)
    neutral=np.r_[neutral,-.04,-.60,np.pi/2]
    low=np.r_[low,-.30,-.85,np.pi/2-.35];high=np.r_[high,.20,-.40,np.pi/2+.35]
    def apply(x):
        d.qpos[qadr]=x[:len(qadr)]
        d.qpos[env.root_qadr:env.root_qadr+2]=x[-3:-1]
        d.qpos[env.root_qadr+3:env.root_qadr+7]=[np.cos(x[-1]/2),0,0,np.sin(x[-1]/2)]
        for finger in ('FF','MF','RF','LF'):
            d.qpos[m.jnt_qposadr[m.joint('robot/rh_'+finger+'J1').id]]=d.qpos[m.jnt_qposadr[m.joint('robot/rh_'+finger+'J2').id]]
        mujoco.mj_kinematics(m,d)
    def points():
        rotations=d.site_xmat[sites].reshape(-1,3,3)
        pad_normals=-rotations[:,:,2]
        return d.site_xpos[sites]+.009*pad_normals,pad_normals
    def collision_penalties():
        mujoco.mj_collision(m,d)
        penalties=np.zeros(m.nbody)
        for c in d.contact[:d.ncon]:
            for geom in c.geom:
                body=int(m.geom_bodyid[geom])
                if m.body(body).name.startswith("robot/"):
                    penalties[body]=max(penalties[body],-float(c.dist)-.001)
        return penalties
    def residual(x):
        apply(x);positions,pad_normals=points()
        return np.r_[((positions-targets)*100).ravel(),((pad_normals-normals)*.6).ravel(),
                     (d.site_xpos[palm]-palm_target)*50,
                     (-d.site_xmat[palm].reshape(3,3)[:,2]-[0,0,-1])*.5,(x-neutral)*.05,
                     collision_penalties()*500]
    best=None
    fitted_initial=None
    if a.initial_seed:
        saved=json.loads(a.initial_seed.read_text())
        fitted_initial=np.r_[[saved['joints'][name] for name in names],saved.get('root_pose_xy_yaw',[-.04,-.60,np.pi/2])]
        fitted_initial=np.clip(fitted_initial,low+1e-5,high-1e-5)
    for attempt in range(a.attempts):
        initial=(fitted_initial if fitted_initial is not None else neutral) if attempt==0 else np.clip((fitted_initial if fitted_initial is not None else neutral)+rng.normal(0,.15,len(neutral)),low+1e-5,high-1e-5)
        fit=least_squares(residual,initial,bounds=(low,high),max_nfev=220,ftol=1e-7)
        if best is None or fit.cost<best.cost:best=fit
        print(json.dumps({'attempt':attempt,'cost':float(fit.cost),'best':float(best.cost)}),flush=True)
    apply(best.x);positions,pad_normals=points();mujoco.mj_forward(m,d)
    report={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'robot_sha256':hashlib.sha256(robot.read_bytes()).hexdigest(),
        'configuration':vars(a),'stage':'kinematic grasp initializer only','physical_success':False,'door_opening_claim':False,
        'root_initialized_near_handle':True,'root_pose_xy_yaw':best.x[-3:].tolist(),
        'seed_joint_limit_margin_fraction':.04,'max_pad_target_error_m':float(np.linalg.norm(positions-targets,axis=1).max()),
        'joints':dict(zip(names,best.x.tolist())),'targets':targets.tolist(),'pad_points':positions.tolist(),
        'pad_normals':pad_normals.tolist(),'maximum_contact_penetration_m':float(max([0.]+[-c.dist for c in d.contact[:d.ncon]])),
        'robot_contact_violations':[{'bodies':[m.body(m.geom_bodyid[g]).name for g in c.geom], 'penetration_m':-float(c.dist)} for c in d.contact[:d.ncon] if c.dist<-.002 and any(m.body(m.geom_bodyid[g]).name.startswith('robot/') for g in c.geom)],
        'limitations':['No physical trajectory executed','No loaded-contact success established','Distal equal-curl initializer must be validated under native tendons']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    np.savez_compressed(a.output/'pose.npz',qpos=d.qpos.copy())
    with mujoco.Renderer(m,height=720,width=960) as renderer:
        options=mujoco.MjvOption();options.sitegroup[:]=0
        camera=mujoco.MjvCamera();camera.lookat[:]=center;camera.distance=.48;camera.azimuth=235;camera.elevation=-12
        for geom in range(m.ngeom):
            if m.geom(geom).name.startswith("leaf_handle"):
                m.geom_matid[geom]=-1;m.geom_rgba[geom]=[.83,.52,.1,1.]
        for azimuth in (70,90,120):
            camera.azimuth=azimuth;renderer.update_scene(d,camera=camera,scene_option=options);env.hide_sensor_overlays(renderer.scene)
            Image.fromarray(renderer.render()).save(a.output/f'hand-{azimuth}.png')
    print(json.dumps(report,indent=2,default=str));env.close()

if __name__=='__main__':main()
