"""Produce every-door native physics recordings and portable humanoid reference clips."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, subprocess, gzip, platform
from collections import Counter
from pathlib import Path
import numpy as np
from ..benchmark_eligibility import require_benchmark_eligible, collection_counts
from .humanoid import JOINTS, BONES, fit_motion
from ..benchmark.runner import Job, run_episode, load_manifest, select_doors
from ..benchmark.interactions import ContactSites

SCHEMA='doorbench.reference-motion.v1'
NATIVE_SCHEMA='doorbench.native-motion.v1'

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')

class Recorder:
    def __init__(self,fps=20): self.fps=fps; self.frames=[]; self.next=0.; self.info=None
    def __call__(self,event,env,base,action):
        m,d,mj=env.m,env.d,env.mj
        if event=='reset':
            names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j) for j in range(m.njnt)]
            self.info={'joint_names':names,'qpos_addresses':m.jnt_qposadr.tolist(),
                       'qvel_addresses':m.jnt_dofadr.tolist(), 'dt':float(m.opt.timestep),
                       'joint_types':[mj.mjtJoint(int(kind)).name for kind in m.jnt_type],
                       'qpos_widths':np.diff(np.r_[m.jnt_qposadr,m.nq]).tolist(),
                       'qvel_widths':np.diff(np.r_[m.jnt_dofadr,m.nv]).tolist(),
                       'body_names':[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,j) for j in range(m.nbody)]}
            self.env=env
            self.pose=mj.MjData(m)
            self.limits=dict(action["torque_limits"])
            self.site_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_SITE,j) or '' for j in range(m.nsite)]
            self.contacts=ContactSites(env)
            from ..benchmark.site_forces import SiteForces
            self.site_forces=SiteForces(env, self.limits)
            self.default_site=self.contacts.select(env.meta.get('primary_joint'))
            self.cable_names=[c['name'] for c in env.model_json.get('spatial_cables', [])]
        if float(d.time)+1e-9<self.next and event!='final': return
        if self.frames and abs(self.frames[-1]['time']-float(d.time))<1e-8: return
        self.next=float(d.time)+1/self.fps-1e-8
        # mj_step can leave derived world poses at the preceding integration state.
        # Refresh geometry on private data: recording must never mutate the live env.
        pose=self.pose; pose.qpos[:]=d.qpos
        if m.nmocap:
            pose.mocap_pos[:]=d.mocap_pos; pose.mocap_quat[:]=d.mocap_quat
        mj.mj_kinematics(m,pose)
        mj.mj_comPos(m,pose)
        cables = None
        if self.cable_names:
            from ..spatial_cables import native_cable_paths
            mj.mj_tendon(m,pose)
            cables = native_cable_paths(m,pose,self.cable_names)
        torques=action.get('torques') or {}
        nonzero=[(n,abs(float(v))) for n,v in torques.items() if abs(float(v))>1e-5]
        nonzero += [(n,float(np.linalg.norm(v))) for n,v in (action.get('site_forces') or {}).items() if np.linalg.norm(v)>1e-5]
        nonzero += [(n,float(np.linalg.norm(v))) for n,v in (action.get('site_torques') or {}).items() if np.linalg.norm(v)>1e-5]
        roles=self.contacts.roles
        target_joint,sid=self.contacts.active(action)
        contact_sites=[]
        for name,value in torques.items():
            if abs(float(value))>1e-5 and self.limits.get(name,0)>0:
                contact=self.contacts.select(name)
                contact_sites.append({'joint':name,'site':self.site_names[contact] if contact is not None else None})
        force_tau,wrenches=self.site_forces.resolve(pose,action.get('site_forces'),action.get('site_torques'))
        contact_sites.extend({'joint':None,'site':name} for name,wrench in wrenches.items()
                             if any(np.linalg.norm(value)>1e-5 for value in wrench.values()))
        for contact in contact_sites:
            contact_id=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,contact['site']) if contact['site'] else -1
            contact['position']=pose.site_xpos[contact_id].tolist() if contact_id>=0 else None
            if contact['site'] in wrenches:contact.update(wrenches[contact['site']])
        contact_valid=sid is not None
        if not nonzero: sid=self.default_site
        target=pose.site_xpos[sid].copy() if sid is not None else np.array([0.,-.05,1.])
        # Save the exact input used by the native step. Re-projecting a hand
        # force at the post-step pose changes its generalized moment arm.
        tau=env.last_applied_qfrc.copy()
        role=roles.get(target_joint,'')
        phase=('operate' if role in ('operator','lock') else 'open' if nonzero else 'traverse' if np.linalg.norm(action.get('base_velocity') or [0,0])>.03 else 'wait')
        if action.get('declare_locked'): phase='recognize_locked'
        self.frames.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),tau=tau,
          body_pos=pose.xpos.copy(),body_quat=pose.xquat.copy(),base=np.asarray(base).copy(),target=target,
          active=bool(nonzero) and contact_valid,phase=phase,
          contact_valid=contact_valid,target_site=self.site_names[sid] if sid is not None else None,
          target_joint=target_joint, site_forces=action.get('site_forces') or {},
          site_torques=action.get('site_torques') or {},native_cables=cables, contact_sites=contact_sites))


def write_native_recording(door,directory,out,rec,ep):
    """Export the observed mechanism without inventing a human or retiming physics."""
    frames=rec.frames
    arrays={k:np.asarray([f[k] for f in frames],dtype=np.float64)
            for k in ('qpos','qvel','ctrl','tau','body_pos','body_quat','base','target')}
    times=np.array([f['time'] for f in frames])
    arrays['time']=times
    contacts=[f['contact_sites'] for f in frames]
    arrays['oracle_contacts_json_utf8']=np.frombuffer(json.dumps(contacts,separators=(',',':')).encode(),dtype=np.uint8)
    cables=[f['native_cables'] for f in frames] if rec.cable_names else None
    if cables is not None:
        arrays['native_cables_json_utf8']=np.frombuffer(json.dumps(cables,separators=(',',':')).encode(),dtype=np.uint8)
    if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError(door['id']+': non-finite recording')
    npz=out/'trajectories'/f"{door['id']}.npz"; npz.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(npz,**arrays)
    source={n:digest(directory/n) for n in ('spec.json','model.json','door.xml')}
    clip={'schema':NATIVE_SCHEMA,'door_id':door['id'],'scenario':ep['scenario'],'seed':0,'fps':rec.fps,
          'units':'metres/radians/seconds','up_axis':'Z','duration':float(times[-1]),
          'times':times.tolist(),'joint_names':rec.info['joint_names'],
          'door_q':arrays['qpos'][:,rec.info['qpos_addresses']].tolist(),
          'targets':arrays['target'].tolist(),'phases':[f['phase'] for f in frames],
          'oracle_contacts':contacts,'outcome':ep,'source_sha256':source,
          'native':{**rec.info,'applied_effort_timing':'tau is the exact qfrc_applied input of the native step ending at the sample; zero at reset',
                    'poses':np.concatenate((arrays['body_pos'],arrays['body_quat']),axis=2).reshape(len(frames),-1).tolist()},
          'limitations':['Scripted mechanism oracle, not a human demonstration or robot policy.',
             'Joint efforts and bounded site forces are idealized inputs. Reach, grip, balance and human motion are not validated.',
             'Simultaneous contact markers show all commanded inputs, not a claim that one actor can reach them.',
             'Failed attempts are retained. A successful task does not certify every mechanism.']}
    if cables is not None: clip['native_cables']=cables
    cp=out/'clips'/f"{door['id']}.json"; write_json(cp,clip)
    gz=cp.with_suffix('.json.gz');gz.write_bytes(gzip.compress(cp.read_bytes(),mtime=0))
    return {'door_id':door['id'],'family':door['family'],'scenario':clip['scenario'],
            'clip':f'clips/{cp.name}','trajectory':f'trajectories/{npz.name}',
            'duration':clip['duration'],'frames':len(frames),'physics_frames':len(frames),
            'success':ep['success'],'outcome':ep['outcome'],'source_sha256':source,
            'web_clip':f'clips/{gz.name}','web_clip_sha256':digest(gz),
            'clip_sha256':digest(cp),'trajectory_sha256':digest(npz)}


def record_one(args):
    door,assets,out,fps=args[:4]; native_only=bool(args[4]) if len(args)>4 else False
    wall_timeout=float(args[5]) if len(args)>5 else 600. if native_only else 120.
    out=Path(out); directory=Path(assets)/'doors'/door['id']
    require_benchmark_eligible(door, operation='native reference recording')
    require_benchmark_eligible(json.loads((directory/'spec.json').read_text()), operation='native reference recording')
    scenario=(door.get('benchmark') or {}).get('primary','open_and_traverse')
    rec=Recorder(fps)
    ep=run_episode(Job(door,str(directory),scenario,0,'full','scripted_hand',randomize=False,wall_timeout_s=wall_timeout),observer=rec)
    if ep.get('error') or not rec.frames: return {'door_id':door['id'],'error':ep.get('error') or 'no frames'}
    if native_only: return write_native_recording(door,directory,out,rec,ep)
    f=rec.frames
    first=np.asarray(f[0]['target']); work=np.array([first[0]-.12,min(first[1]-.40,-.40)])
    # Give broad gates' distant starts enough walking time before physical operation.
    lead=max(2.,np.ceil(np.linalg.norm(work-f[0]['base'][:2])/.60*fps)/fps)
    lead_n=round(lead*fps)
    physics_times=np.array([x['time'] for x in f]); times=np.r_[np.arange(lead_n)/fps,physics_times+lead]
    def arr(k): return np.asarray([x[k] for x in f])
    def padded(k): return np.concatenate([np.repeat(arr(k)[:1],lead_n,axis=0),arr(k)])
    # Reference hands release after crossing the door plane; the oracle's continuing
    # generalized forces remain in the native tau array and are not called hand contact.
    hand_active=padded('active') & (padded('base')[:,1] < 0.)
    hand_active[:lead_n]=False
    poses,reach,feet,actor_roots=fit_motion(times,padded('base'),padded('target'),hand_active,f[0]['base'],lead)
    arrays={k:arr(k).astype(np.float32) for k in ('qpos','qvel','ctrl','tau','body_pos','body_quat','base','target')}
    arrays.update(time=physics_times.astype(np.float32),actor_time=times.astype(np.float32),actor_joints=poses,
                  actor_root=actor_roots,hand_target_error=reach,foot_contact=feet)
    if rec.cable_names:
        arrays['native_cables_json_utf8']=np.frombuffer(json.dumps([x['native_cables'] for x in f],separators=(',',':')).encode(),dtype=np.uint8)
    if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError(door['id']+': non-finite recording')
    npz=out/'trajectories'/f"{door['id']}.npz"; npz.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(npz,**arrays)
    source={n:digest(directory/n) for n in ('spec.json','model.json','door.xml')}
    addr=rec.info['qpos_addresses']
    clip={'schema':SCHEMA,'door_id':door['id'],'scenario':scenario,'seed':0,'fps':fps,'units':'metres/radians/seconds','up_axis':'Z',
          'lead_in_s':lead,'duration':float(times[-1]),'joint_names':rec.info['joint_names'],
          'avatar_joint_names':JOINTS,'avatar_bones':BONES,'times':times.round(4).tolist(),
          'door_q':padded('qpos')[:,addr].round(5).tolist(),'avatar':poses.round(4).reshape(len(times),-1).tolist(),
          'targets':padded('target').round(4).tolist(),'hand_active':hand_active.astype(int).tolist(),
          'target_sites':[f[0]['target_site']]*lead_n+[x['target_site'] for x in f],
          'target_joints':[None]*lead_n+[x['target_joint'] for x in f],
          'contact_dispatch_valid':[False]*lead_n+[x['contact_valid'] for x in f],
          'oracle_contact_sites':[[]]*lead_n+[x['contact_sites'] for x in f],
          'hand_error_m':reach.round(4).tolist(),'phases':['approach']*lead_n+[x['phase'] for x in f],
          'outcome':ep,'source_sha256':source,'native':rec.info,
          'limitations':['Kinematic humanoid reference, not a trained or dynamically simulated humanoid policy.',
             'Door physics uses generalized joint forces from scripted_hand; hand contact and balance are not validated.',
             'Avatar path differs from the benchmark synthetic base. Unreachable targets remain marked; failed episodes are retained.'],
          'max_hand_error_m':round(float(max(reach)),4),'unreachable_frames':int(np.sum(reach>.08))}
    if rec.cable_names:
        clip['native_cables']=[f[0]['native_cables']]*lead_n+[x['native_cables'] for x in f]
    cp=out/'clips'/f"{door['id']}.json"; write_json(cp,clip)
    gz=cp.with_suffix('.json.gz'); gz.write_bytes(gzip.compress(cp.read_bytes(),mtime=0))
    return {'door_id':door['id'],'family':door['family'],'scenario':scenario,'clip':f'clips/{cp.name}','trajectory':f'trajectories/{npz.name}',
            'duration':clip['duration'],'frames':len(times),'physics_frames':len(f),'success':ep['success'],'outcome':ep['outcome'],
            'max_hand_error_m':clip['max_hand_error_m'],'unreachable_frames':clip['unreachable_frames'],
            'source_sha256':source,'web_clip':f'clips/{gz.name}','web_clip_sha256':digest(gz),'clip_sha256':digest(cp),'trajectory_sha256':digest(npz)}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--assets',default='assets'); p.add_argument('--out',default='out/reference-motions')
    p.add_argument('--doors',default='all'); p.add_argument('--workers',type=int,default=6); p.add_argument('--fps',type=int,default=20)
    p.add_argument('--native-only',action='store_true',help='Record actual mechanism states and oracle contacts; generate no humanoid animation')
    p.add_argument('--wall-timeout',type=float,default=600.,help='Per-episode wall-clock budget after initialization; native scenario duration is unchanged')
    a=p.parse_args()
    if not 1<=a.fps<=60: p.error('fps must be 1..60')
    if not np.isfinite(a.wall_timeout) or a.wall_timeout<=0:p.error('wall-timeout must be positive and finite')
    manifest=load_manifest(a.assets); doors=select_doors(manifest,a.doors); jobs=[(d,str(Path(a.assets).resolve()),a.out,a.fps,a.native_only,a.wall_timeout) for d in doors]
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(record_one,jobs):
            rows.append(row)
            print(f"{len(rows)}/{len(doors)} {row['door_id']} {row.get('outcome',row.get('error'))}",flush=True)
    import mujoco
    index={'runtime':{'python':platform.python_version(),'numpy':np.__version__,'mujoco':mujoco.__version__},'schema':NATIVE_SCHEMA if a.native_only else SCHEMA,'generator_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'generator_sha256':{str(f):digest(f) for f in sorted(Path('doorbench').rglob('*.py'))},
           'manifest_sha256':digest(Path(a.assets)/'manifest.json'),'fps':a.fps,'seed':0,'tier':'full',
           'policy':'scripted_hand','embodiment':'scripted mechanism oracle' if a.native_only else 'kinematic humanoid reference over recorded door physics',
           'wall_timeout_s':a.wall_timeout,
           'scope':'one primary core scenario per benchmark-eligible door; supplementary pet doors excluded; failed attempts retained',
           **collection_counts(manifest['doors']),
           'counts':dict(Counter(r.get('outcome','error') for r in rows)),'clips':rows}
    write_json(Path(a.out)/'index.json',index)
    print(json.dumps(index['counts']))
    if any(r.get('error') for r in rows): raise SystemExit(1)
if __name__=='__main__': main()
