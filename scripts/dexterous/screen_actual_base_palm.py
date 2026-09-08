#!/usr/bin/env python3
"""Replay actual base/proprioception through bounded palm target correction.

This is an unstepped geometry/rate screen, never a physical contact or success
claim. Archived actual dynamics supply the current body and scalar joints.
Only the declared arm/optional waist target joints change on a separate FK model.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation


def screened_nominal_arm_targets(plan,row,names):
    """Use the original plan, never an already corrected recorded command."""
    coordinates=np.asarray(row['planned_coordinates'],float)
    if coordinates.shape!=(6+len(plan['joint_names']),) or not np.isfinite(coordinates).all():
        raise ValueError('Require the complete finite original whole-body target')
    return coordinates[6:][[plan['joint_names'].index(n) for n in names]]



def enclosing_collision_radius(model,geom):
    kind=int(model.geom_type[geom]);size=model.geom_size[geom]
    if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh=model.geom_dataid[geom];start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
        return float(np.linalg.norm(model.mesh_vert[start:start+count],axis=1).max())
    if kind==int(mujoco.mjtGeom.mjGEOM_BOX):return float(np.linalg.norm(size))
    if kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):return float(size[0])
    if kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):return float(size[0]+size[1])
    if kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):return float(np.hypot(size[0],size[1]))
    if kind==int(mujoco.mjtGeom.mjGEOM_PLANE):return np.inf
    raise ValueError('Unsupported original collision shape; cannot prune safely')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--normal-offset-m',type=float,default=0.);p.add_argument('--normal-admittance-envelope',action='store_true');p.add_argument('--include-waist',action='store_true');a=p.parse_args();
 if not np.isfinite(a.normal_offset_m) or not 0<=a.normal_offset_m<=.002:raise ValueError('Require a normal-goal offset in the frozen0–2mm envelope')
 if a.normal_admittance_envelope and a.normal_offset_m:raise ValueError('Select only one normal envelope')
 run=a.run.resolve();source=run.with_name(run.name+'-source');own=Path(__file__).resolve().parents[2];sys.path.insert(0,str(own))
 from doorbench.dexterous.environment import DexterousDoorEnv
 from doorbench.dexterous.actual_base_palm import ActualBasePalmCorrection
 from doorbench.dexterous.palm_normal_admittance import PalmNormalAdmittance
 admittance=PalmNormalAdmittance() if a.normal_admittance_envelope else None
 cfg=json.load(open(run/'manifest.json'))['configuration'];robot=Path(cfg['robot']);sim=DexterousDoorEnv(cfg['door'],robot,json.load(open(robot.with_suffix('.audit.json'))));m,d=sim.m,sim.d
 robot_model=mujoco.MjModel.from_xml_path(str(robot));names=['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1'];names=(['torso']+names) if a.include_waist else names;correction=ActualBasePalmCorrection(robot_model,names,'lh_palm_touch')
 scalar={m.joint(j).name.removeprefix('robot/'):int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.joint(j).name.startswith('robot/') and m.jnt_type[j] in (2,3)};cq=np.array([scalar[n] for n in names]);rq=m.jnt_qposadr[m.joint('robot/free_base').id];palm=m.site('robot/lh_palm_touch').id
 plan=json.load(open(source/'whole-body-panel-plan.json'));q0=np.asarray(plan['initial_qpos']);qa=np.asarray(plan['joint_qpos_addresses']);r0=Rotation.from_quat([*q0[rq+4:rq+7],q0[rq+3]]);rd=mujoco.MjData(m);cd=mujoco.MjData(m)
 with gzip.open(run/'panel-phase.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
 initial_target=np.asarray(rows[0]['left_arm_targets']);initial_velocity=np.asarray(rows[0]['left_arm_target_velocity'])
 if a.include_waist:
  torso_index=plan['joint_names'].index('torso');initial_target=np.r_[rows[0]['planned_coordinates'][6+torso_index],initial_target];initial_velocity=np.r_[rows[0]['planned_velocity'][6+torso_index],initial_velocity]
 correction.begin(rows[0]['episode_time_s'],initial_target,initial_velocity)
 manifest=json.load(open(run/'raw-transitions/manifest.json'));assert manifest['complete'];count=0;geometric=0;maxima={};failures=[];values=[];floor=m.geom('floor').id
 hand=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
 scene=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and not m.body(m.geom_bodyid[g]).name.startswith('robot/')]
 radii=np.zeros(m.ngeom)
 for g in hand+scene:radii[g]=enclosing_collision_radius(m,g)
 minimum_right_clearance=.040001;exact_distance_pairs=0;robot_body=m.jnt_bodyid[m.joint('robot/free_base').id];torso=m.body('robot/torso_link').id
 for chunk in manifest['chunks']:
  if chunk['interval_end_s']<rows[0]['episode_time_s']:continue
  file=run/'raw-transitions'/chunk['file']
  if hashlib.file_digest(file.open('rb'),'sha256').hexdigest()!=chunk['sha256']:raise ValueError('Changed archived source')
  with np.load(file) as raw:
   for i,t in enumerate(raw['interval_start_s']):
    if t<rows[0]['episode_time_s']-1e-9:continue
    row=rows[count]
    if abs(row['episode_time_s']-t)>1e-9:raise ValueError('Actual and target timestamps differ')
    q=raw['qpos_before'][i];d.qpos[:]=q;mujoco.mj_kinematics(m,d)
    begin,end=raw['body_offsets'][i:i+2];ids=raw['body_ids'][begin:end]
    if max(float(np.max(abs(d.xpos[ids]-raw['body_positions_world_m'][begin:end]))),float(np.max(abs(d.xmat[ids].reshape(-1,3,3)-raw['body_rotations_world'][begin:end]))))>1e-9:raise ValueError('Actual FK/body frames differ')
    x=np.asarray(row['planned_coordinates']);nominal=screened_nominal_arm_targets(plan,row,names);rd.qpos[:]=q0;rd.qpos[rq:rq+3]+=x[:3];quat=(Rotation.from_rotvec(x[3:6])*r0).as_quat();rd.qpos[rq+3:rq+7]=np.r_[quat[3],quat[:3]];rd.qpos[qa]=x[6:];mujoco.mj_kinematics(m,rd)
    normal_offset=a.normal_offset_m if admittance is None else admittance.update(float(t),0.)[0]
    base=Rotation.from_quat([*q[rq+4:rq+7],q[rq+3]]).as_matrix();goalp=base.T@(rd.site_xpos[palm]+d.xmat[m.body('leaf').id].reshape(3,3)[:,1]*normal_offset-q[rq:rq+3]);goalr=base.T@rd.site_xmat[palm].reshape(3,3)
    target,velocity,info=correction.update(float(t),{n:float(q[address]) for n,address in scalar.items()},goalp,goalr,nominal)
    cd.qpos[:]=q;cd.qpos[cq]=target;mujoco.mj_kinematics(m,cd)
    vals={k:v for k,v in info.items() if k not in ('time_s','minimum_joint_margin_rad')};vals['target_change_from_nominal_rad']=float(np.max(abs(target-nominal)))
    if a.include_waist:vals['waist_correction_from_nominal_rad']=float(abs(target[0]-nominal[0]))
    vals['maximum_world_fk_consistency_error']=max(float(np.max(abs(cd.site_xpos[palm]-(q[rq:rq+3]+base@correction.data.site_xpos[correction.palm])))),float(np.max(abs(cd.site_xmat[palm].reshape(3,3)-base@correction.data.site_xmat[correction.palm].reshape(3,3)))))
    if count%5==0 or count==len(rows)-1:
     geometric+=1;mujoco.mj_collision(m,cd);penetration=0.
     mujoco.mj_comPos(m,d);mujoco.mj_comPos(m,cd)
     vals['com_xy_correction_m']=float(np.linalg.norm(cd.subtree_com[robot_body,:2]-d.subtree_com[robot_body,:2]))
     vals['torso_tilt_deg']=float(np.degrees(np.arccos(np.clip(cd.xmat[torso].reshape(3,3)[2,2],-1,1))))
     center=np.linalg.norm(cd.geom_xpos[hand,None,:]-cd.geom_xpos[None,scene,:],axis=2)
     candidates=np.argwhere(center-radii[hand,None]-radii[None,scene]<.040001)
     nearest=.040001
     for hi,si in candidates:
      nearest=min(nearest,float(mujoco.mj_geomDistance(m,cd,hand[hi],scene[si],.040001,None)));exact_distance_pairs+=1
     minimum_right_clearance=min(minimum_right_clearance,nearest)
     if nearest<.04 or vals['com_xy_correction_m']>.015 or vals['torso_tilt_deg']>12:
      failures.append(dict(time_s=float(t),right_scene_clearance_m=nearest,com_xy_correction_m=vals['com_xy_correction_m'],torso_tilt_deg=vals['torso_tilt_deg']))
     for contact in cd.contact[:cd.ncon]:
      bodies=[m.body(m.geom_bodyid[g]).name for g in contact.geom];allowed=floor in contact.geom and any(b.endswith('_ankle_link') for b in bodies)
      if not allowed and any(b.startswith('robot/') for b in bodies):penetration=max(penetration,-float(contact.dist))
     vals['nonfoot_robot_penetration_m']=penetration
     if penetration>.003:failures.append(dict(time_s=float(t),nonfoot_robot_penetration_m=penetration))
    for k,v in vals.items():maxima[k]=max(maxima.get(k,0.),float(v))
    if not np.isfinite(list(vals.values())).all():raise ValueError('Nonfinite proposed geometry')
    if vals['target_change_from_nominal_rad']>.1 or vals.get('waist_correction_from_nominal_rad',0)>.03 or vals['maximum_world_fk_consistency_error']>1e-9:failures.append(dict(time_s=float(t),values=vals))
    values.append(np.r_[t,target,velocity,list(info.values())[1:]]);count+=1
  if count%1000<250:print(json.dumps(dict(samples=count,geometry_samples=geometric,failures=len(failures),maxima=maxima)),flush=True)
 if count!=len(rows):raise ValueError('Incomplete target replay')
 a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'targets.npz',values=np.array(values));result=dict(passed=not failures,scope='Unstepped actual-base target geometry/rate screen only. This replay starts at the archived first panel target; physical handoff continuity is checked separately.',samples=count,geometry_samples=geometric,normal_goal_offset_m=a.normal_offset_m,normal_admittance_envelope=a.normal_admittance_envelope,include_waist=a.include_waist,waist_correction_limit_rad=.03 if a.include_waist else None,nominal_target_source='original_screened_whole_body_plan',maximum_target_correction_rad=.1,minimum_all_right_scene_clearance_capped_m=minimum_right_clearance,right_collision_shapes=len(hand),scene_collision_shapes=len(scene),exact_distance_pairs=exact_distance_pairs,controlled_joint_names=names,maxima=maxima,failures=failures,source_run=str(run),source_phase_sha256=hashlib.file_digest((run/'panel-phase.jsonl.gz').open('rb'),'sha256').hexdigest());(a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(a.output/'screen-source.py').write_bytes(Path(__file__).read_bytes());(a.output/'correction-source.py').write_bytes((own/'doorbench/dexterous/actual_base_palm.py').read_bytes());sim.close();print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2))


if __name__=='__main__':main()
