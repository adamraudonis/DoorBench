"""Generate auditable whole-body IK candidates without changing the v1 release."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import mujoco
from .guidance import make_guide,yaw_quaternion
from .ik import DoorHumanoidIK
from .rig import rig_xml,DIMENSIONS
from .humanoid import JOINTS,BONES,two_bone
from .retime import retime_trajectory


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')


class ContactResolver:
    """Put the hand sphere on a grasp surface, rather than inside a handle."""
    def __init__(self,solver,model_ir,source,*,roles=None):
        self.solver=solver;self.source=source;self.ids=[];self.roles=None
        for b in model_ir['bodies']:
            for g in b['geoms']:
                if not g.get('collision'):continue
                if g.get('semantic') in ('operator','lock','handle') or any(x in g['name'] for x in ('handle','pull_col','wheel','thumbturn','keypad')):
                    try:i=int(solver.model.geom(g['name']).id)
                    except KeyError:continue
                    if i in solver.scene_geom_ids and i not in solver.floor_geom_ids:self.ids.append(i)

        if roles is not None:
            self.roles={}
            for role in roles:
                name=role['id']
                if name in self.roles:raise ValueError('Duplicate manual contact role')
                model=solver.model
                site=model.site(role['site_name']);geom=model.geom(role['geom_name'])
                body=int(model.body(role['body_name']).id)
                if (int(site.bodyid[0])!=body or int(geom.bodyid[0])!=body or
                    int(model.joint(role['joint_name']).bodyid[0])!=body or int(geom.id) not in self.ids):
                    raise ValueError('Manual role must bind one native joint body, grip site and eligible collision geometry')
                if int(geom.type[0])==int(mujoco.mjtGeom.mjGEOM_MESH):
                    raise ValueError('Manual role requires a supported collision primitive')
                self.roles[name]=(int(site.id),int(geom.id))
            if not self.roles:raise ValueError('Manual contact schedule has no roles')

    def resolve(self,native_time,pelvis,*,role_id=None):
        if self.roles is not None:
            if role_id is None:return np.zeros(3),np.zeros(3),None
            if role_id not in self.roles:raise ValueError('Unknown manual contact role')
            site,geom=self.roles[role_id]
            point=self.solver.data.site_xpos[site].copy();ids=[geom]
        else:
            if role_id is not None:raise ValueError('Contact role supplied without a bound schedule')
            point=np.array([np.interp(native_time,self.source['time'],self.source['target'][:,i]) for i in range(3)])
            ids=self.ids
        direction=np.r_[np.asarray(pelvis)[:2]-point[:2],0.]
        direction/=max(np.linalg.norm(direction),1e-8)
        data=self.solver.data;model=self.solver.model
        candidates=[]
        for i in ids:
            center=data.geom_xpos[i]
            if np.linalg.norm(center-point)>float(model.geom_rbound[i])+.09:continue
            typ=int(model.geom_type[i])
            if typ==int(mujoco.mjtGeom.mjGEOM_MESH):continue
            origin=point+direction*.5
            normal=np.zeros(3)
            hit=mujoco.mju_rayGeom(center,data.geom_xmat[i],model.geom_size[i],origin,-direction,typ,normal)
            if hit<0 or hit>.65:continue
            surface=origin-direction*hit
            if np.linalg.norm(surface-point)>.14:continue
            # A 4.5 mm proxy surface gap lies inside the declared 5 mm contact
            # tolerance and leaves the solver's 4 mm clearance valid when the
            # hand starts to withdraw. This avoids a discontinuous constraint
            # change at release; it does not exempt the forearm or other parts.
            target=surface+normal*.0395
            # The first visible surface wins. A neck behind a knob may be closer
            # to the grip centre, but reaching it would pass through the knob.
            candidates.append((hit,model.geom(i).name,target))
        if not candidates:return point,point,None
        _,name,target=min(candidates,key=lambda x:x[0])
        return point,target,name


def solve_door(door_dir,recording_dir,out,*,fps=60,max_frames=None,gait_profile='smooth'):
    if type(fps) is not int or not 10<=fps<=60:
        raise ValueError('fps must be an integer from 10 to 60')
    if max_frames is not None and (type(max_frames) is not int or max_frames<2):
        raise ValueError('max_frames must be an integer of at least 2')
    directory=Path(door_dir);out=Path(out)/directory.name;out.mkdir(parents=True,exist_ok=True)
    guide=make_guide(directory,recording_dir,fps=fps,gait_profile=gait_profile)
    n=len(guide.time) if max_frames is None else min(len(guide.time),max_frames)
    # Optimize with an extra millimetre beyond the independent 3 mm gate. The
    # margin accommodates curvature between saved joint-space samples.
    solver=DoorHumanoidIK(directory,native_qpos=guide.native_qpos[0],root_pos=guide.pelvis[0],root_yaw=float(guide.yaw[0]),clearance=.004)
    with np.load(Path(recording_dir)/'trajectories'/f'{directory.name}.npz',allow_pickle=False) as data:
        source={k:data[k].copy() for k in ['time','target']}
    ir=json.loads((directory/'model.json').read_text())
    schedule=guide.metadata.get('manual_contact_schedule')
    role_ids=schedule['contact_role_ids'] if schedule else None
    if schedule and (len(role_ids)!=len(guide.time) or any(
        role is None and (guide.hand_weight[i].max()>1e-8 or guide.hand_contact[i].any())
        for i,role in enumerate(role_ids))):
        raise ValueError('Manual contact role timeline does not cover every active or blended hand target')
    resolver=ContactResolver(solver,ir,source,roles=schedule['roles'] if schedule else None)
    native_body_names=[solver.rig.native_model.body(i).name for i in range(solver.rig.native_model.nbody)]
    native_body_ids=[int(solver.model.body(name).id) for name in native_body_names]
    actor_body_ids=[i for i in range(solver.model.nbody) if solver.model.body(i).name.startswith('actor_')]
    actor_body_names=[solver.model.body(i).name for i in actor_body_ids]
    geometries=[]
    for i in solver.actor_geom_ids:
        geometries.append({'name':solver.model.geom(i).name,'body_name':solver.model.body(int(solver.model.geom_bodyid[i])).name,
            'type':mujoco.mjtGeom(solver.model.geom_type[i]).name.removeprefix('mjGEOM_').lower(),
            'size':solver.model.geom_size[i].tolist(),'pos':solver.model.geom_pos[i].tolist(),
            'quat_wxyz':solver.model.geom_quat[i].tolist()})
    actor_joints=[solver.model.joint(i) for i in range(solver.model.njnt) if solver.model.joint(i).name.startswith('actor_')]
    amap={global_i:local_i for local_i,global_i in enumerate(solver.actor_qpos_indices)}
    frames=[];exclusions=[];diagnostics=[];clock=time.monotonic()

    def targets_at(i):
        targets={'pelvis':{'pos':guide.pelvis[i],'quat_wxyz':yaw_quaternion(guide.yaw[i]),
                   'position_cost':.5,'orientation_cost':.8,'position_tolerance_m':.12,'orientation_tolerance_rad':.35}}
        # An upright torso is a soft style preference. Reaching should first
        # change the working stance, rather than lean the entire adult backward.
        targets['chest']={'pos':guide.pelvis[i]+[0,0,.35],
            'quat_wxyz':yaw_quaternion(guide.yaw[i]),'position_cost':.15,
            'orientation_cost':.15,'position_tolerance_m':.18,'orientation_tolerance_rad':.35}
        for k,name in enumerate(['left_foot','right_foot']):
            targets[name]={'pos':guide.foot_pos[i,k],'quat_wxyz':guide.foot_quat[i,k],
                'contact':bool(guide.foot_contact[i,k]),'position_cost':8.,'orientation_cost':2.,
                'position_tolerance_m':.005,'orientation_tolerance_rad':.015}
        desired_hands=guide.hands[i].copy()
        anchor,contact_target,geom=resolver.resolve(guide.native_time[i],guide.pelvis[i],
                                                  role_id=role_ids[i] if role_ids is not None else None)
        for k,name in enumerate(['left_hand','right_hand']):
            weight=guide.hand_weight[i,k]
            desired_hands[k]+=(contact_target-anchor)*weight
            contact=bool(guide.hand_contact[i,k])
            # Explicit role schedules keep the finite proxy inside the same
            # independently checked 5 mm surface-gap limit during regrasp.
            active_cost=16. if schedule else 4.
            targets[name]={'pos':desired_hands[k],'position_cost':.18+(active_cost-.18)*weight,
                'position_tolerance_m':.01 if contact else .09,
                'grip_geoms':[geom] if geom and contact else []}
            if weight>1e-8:
                side=np.array([np.cos(guide.yaw[i]),np.sin(guide.yaw[i]),0.])
                sign=-1 if k==0 else 1
                shoulder=guide.pelvis[i]+side*(sign*.18)+[0,0,.41]
                elbow,_,_=two_bone(shoulder,desired_hands[k],.30,.28,
                                   side*(sign*.25)+[0,0,-1.])
                targets['left_elbow' if k==0 else 'right_elbow']={
                    'pos':elbow,'position_cost':.12*weight,
                    'position_tolerance_m':.20}
        # Support-centred COM is a soft posture objective, never a dynamic certificate.
        # The gait already authors a continuous support transfer. Taking the
        # centroid of a binary contact set jumps when a foot lifts or lands,
        # producing a spurious whole-body acceleration at every step.
        targets['com']={'pos':guide.pelvis[i,:2]+np.array(
            [-.04*np.sin(guide.yaw[i]),.04*np.cos(guide.yaw[i])]),'cost':.15}
        return targets,desired_hands

    # Solve a starting stance off the motion clock. It becomes the first pose;
    # invisible setup iterations are not emitted as a teleporting approach.
    initial,_=targets_at(0)
    for _ in range(4):result=solver.solve(initial,.5)
    for i in range(n):
        solver.set_door_state(guide.native_qpos[i])
        targets,hands=targets_at(i)
        dt=1/fps if i==0 else float(guide.time[i]-guide.time[i-1])
        # Large retimed gaps are bounded by solving the held target over substeps.
        pieces=max(1,int(np.ceil(dt/.1)))
        for _ in range(pieces):result=solver.solve(targets,min(.5,dt/pieces))
        data=solver._fresh_data(result.qpos)
        fp=[result.foot_poses[k] for k in ['left_foot','right_foot']]
        frames.append({'actor_qpos':result.actor_qpos,'actor_joints':result.joint_positions,
             'qpos':result.native_qpos,'body_pos':data.xpos[native_body_ids].copy(),'body_quat':data.xquat[native_body_ids].copy(),
             'actor_body_pos':data.xpos[actor_body_ids].copy(),'actor_body_quat':data.xquat[actor_body_ids].copy(),
             'foot_pos':np.array([f['pos'] for f in fp]),'foot_quat':np.array([f['quat_wxyz'] for f in fp]),
             'hand_target':hands,'com':result.com})
        exclusions.append(result.diagnostics['allowed_grip_pairs'])
        diagnostics.append({'success':result.success,'feasible':result.diagnostics['kinematically_feasible'],
            'clearance':result.diagnostics['min_noncontact_distance_m'],'error':result.diagnostics['solver_error'],
            'residuals':result.diagnostics['target_residuals']})
        if i%50==0:print(f'{directory.name} {i+1}/{n} {guide.phases[i]} pass={result.success} elapsed={time.monotonic()-clock:.1f}s',flush=True)
    arrays={k:np.asarray([f[k] for f in frames]) for k in frames[0]}
    arrays.update(actor_time=guide.time[:n],native_time=guide.native_time[:n],foot_contact=guide.foot_contact[:n],
        hand_contact=guide.hand_contact[:n],foot_target_pos=guide.foot_pos[:n],foot_target_quat=guide.foot_quat[:n])
    combined=np.repeat(solver.home_qpos[None,:],n,axis=0)
    combined[:,solver.actor_qpos_indices]=arrays['actor_qpos']
    combined[:,solver.native_qpos_indices]=arrays['qpos']
    clock_result=retime_trajectory(solver.model,combined,arrays['actor_time'],
        actor_dof_indices=solver.actor_dof_indices,native_time=arrays['native_time'])
    arrays['proposal_time']=arrays['actor_time'].copy()
    arrays['actor_time']=clock_result.time
    trajectory=out/'trajectory.npz';np.savez_compressed(trajectory,**arrays)
    metadata={'schema':'doorbench.planned-reference.v1','door_id':directory.name,'fps':fps,
        'units':'metres/radians/seconds','up_axis':'Z','status':'unvalidated','qa':{},
        'duration':float(arrays['actor_time'][-1]),'frames':n,'complete_proposal':n==len(guide.time),
        'sampling':'actor_time is authoritative; fps is the nominal rendering frame rate',
        'retiming':{'success':clock_result.success,**clock_result.metrics},
        'trajectory_sha256':digest(trajectory),'source_sha256':guide.metadata['source_sha256'],
        'native':{'body_names':native_body_names,'joint_names':[solver.rig.native_model.joint(i).name for i in range(solver.rig.native_model.njnt)],
                  'qpos_addresses':solver.rig.native_model.jnt_qposadr.tolist()},
        'actor':{'body_names':actor_body_names,'joint_names':[j.name for j in actor_joints],
                 'qpos_addresses':[amap[int(j.qposadr[0])] for j in actor_joints],
                 'geometries':geometries,'mjcf_xml':rig_xml(guide.pelvis[0],float(guide.yaw[0])),
                 'dimensions':DIMENSIONS,'landmark_names':JOINTS,'bones':BONES},
        'phases':guide.phases[:n],'contact_exclusions':exclusions,'proposal':guide.metadata,
        'solver_summary':{'failed_frames':sum(not r['success'] for r in diagnostics),
            'geometrically_infeasible_frames':sum(not r['feasible'] for r in diagnostics),
            'error_counts':dict(Counter(r['error'] for r in diagnostics if r['error'])),
            'runtime_s':time.monotonic()-clock},
        'limitations':['Kinematic candidate: independent acceptance and visual review pending.',
             'Native door poses are time-warped; source forces and dynamics are not replayed on this clock.',
             'Original approximate adult rig; no articulated fingers, actuator limits or contact-force certification.']}
    write_json(out/'clip.json',metadata);write_json(out/'solver-diagnostics.json',diagnostics)
    print(json.dumps(metadata['solver_summary']),flush=True)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',default='assets');p.add_argument('--recordings',default='out/reference-motions')
    p.add_argument('--out',default='out/reference-planned');p.add_argument('--doors',required=True)
    p.add_argument('--fps',type=int,default=60);p.add_argument('--max-frames',type=int)
    p.add_argument('--gait-profile',choices=['smooth','controlled','wide_turns','compact'],default='smooth')
    a=p.parse_args()
    if not 10<=a.fps<=60:p.error('fps must be10..60')
    if a.max_frames is not None and a.max_frames<2:p.error('max-frames must be at least 2')
    manifest=json.loads((Path(a.assets)/'manifest.json').read_text())
    ids=[d['id'] for d in manifest['doors']] if a.doors=='all' else a.doors.split(',')
    failures=0
    for door_id in ids:
        try:solve_door(Path(a.assets)/'doors'/door_id,a.recordings,a.out,fps=a.fps,max_frames=a.max_frames,gait_profile=a.gait_profile)
        except Exception as exc:
            failures+=1
            write_json(Path(a.out)/door_id/'failure.json',{'door_id':door_id,'status':'planning_failed','error':f'{type(exc).__name__}: {exc}'})
            print(f'{door_id}: {type(exc).__name__}: {exc}',flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
