"""Produce every-door native physics recordings and portable humanoid reference clips."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, subprocess, gzip, platform
from collections import Counter
from pathlib import Path
import numpy as np
from .humanoid import JOINTS, BONES, fit_motion
from ..benchmark.runner import Job, run_episode, load_manifest, select_doors

SCHEMA='doorbench.reference-motion.v1'

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
                       'body_names':[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,j) for j in range(m.nbody)]}
            self.env=env
            self.pose=mj.MjData(m)
            self.limits=dict(action["torque_limits"])
            self.site_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_SITE,j) or '' for j in range(m.nsite)]
            self.default_site=next((j for j,n in enumerate(self.site_names) if 'grip_n' in n),
                next((j for j,n in enumerate(self.site_names) if 'grip' in n or 'edge_mid' in n),-1))
        if float(d.time)+1e-9<self.next and event!='final': return
        if self.frames and abs(self.frames[-1]['time']-float(d.time))<1e-8: return
        self.next=float(d.time)+1/self.fps-1e-8
        # mj_step can leave derived world poses at the preceding integration state.
        # Refresh geometry on private data: recording must never mutate the live env.
        pose=self.pose; pose.qpos[:]=d.qpos
        if m.nmocap:
            pose.mocap_pos[:]=d.mocap_pos; pose.mocap_quat[:]=d.mocap_quat
        mj.mj_kinematics(m,pose)
        torques=action.get('torques') or {}
        nonzero=[(n,abs(float(v))) for n,v in torques.items() if abs(float(v))>1e-5]
        # The most specific actuated hardware wins over a simultaneous leaf torque.
        roles={b['joint']['name']:b['joint'].get('role') for b in env.model_json['bodies'] if b.get('joint')}
        nonzero.sort(key=lambda p:(roles.get(p[0]) in ('lock','operator'),p[1]),reverse=True)
        target_joint=nonzero[0][0] if nonzero else None
        sid=self.default_site
        if target_joint:
            jid=env._jid(target_joint); bid=int(m.jnt_bodyid[jid])
            candidates=[j for j in range(m.nsite) if int(m.site_bodyid[j])==bid and ('grip' in self.site_names[j] or 'touch' in self.site_names[j])]
            if candidates: sid=candidates[0]
            elif roles.get(target_joint) in ('lock','operator'):
                candidates=[j for j in range(m.ngeom) if int(m.geom_bodyid[j])==bid]
                if candidates: sid=-1; target=pose.geom_xpos[candidates[0]].copy()
        if sid>=0: target=pose.site_xpos[sid].copy()
        elif 'target' not in locals(): target=np.array([0.,-.05,1.])
        tau=np.zeros(m.nv)
        for name,value in torques.items():
            jid=env._jid(name)
            if jid>=0:
                limit=self.limits.get(name,0.); tau[int(m.jnt_dofadr[jid])]=np.clip(float(value),-limit,limit)
        role=roles.get(target_joint,'')
        phase=('operate' if role in ('operator','lock') else 'open' if nonzero else 'traverse' if np.linalg.norm(action.get('base_velocity') or [0,0])>.03 else 'wait')
        if action.get('declare_locked'): phase='recognize_locked'
        self.frames.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),tau=tau,
          body_pos=pose.xpos.copy(),body_quat=pose.xquat.copy(),base=np.asarray(base).copy(),target=target,
          active=bool(nonzero),phase=phase))


def record_one(args):
    door,assets,out,fps=args; out=Path(out); directory=Path(assets)/'doors'/door['id']
    scenario=(door.get('benchmark') or {}).get('primary','open_and_traverse')
    rec=Recorder(fps)
    ep=run_episode(Job(door,str(directory),scenario,0,'full','scripted_hand',randomize=False),observer=rec)
    if ep.get('error') or not rec.frames: return {'door_id':door['id'],'error':ep.get('error') or 'no frames'}
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
    if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError(door['id']+': non-finite recording')
    npz=out/'trajectories'/f"{door['id']}.npz"; npz.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(npz,**arrays)
    source={n:digest(directory/n) for n in ('spec.json','model.json','door.xml')}
    addr=rec.info['qpos_addresses']
    clip={'schema':SCHEMA,'door_id':door['id'],'scenario':scenario,'seed':0,'fps':fps,'units':'metres/radians/seconds','up_axis':'Z',
          'lead_in_s':lead,'duration':float(times[-1]),'joint_names':rec.info['joint_names'],
          'avatar_joint_names':JOINTS,'avatar_bones':BONES,'times':times.round(4).tolist(),
          'door_q':padded('qpos')[:,addr].round(5).tolist(),'avatar':poses.round(4).reshape(len(times),-1).tolist(),
          'targets':padded('target').round(4).tolist(),'hand_active':hand_active.astype(int).tolist(),
          'hand_error_m':reach.round(4).tolist(),'phases':['approach']*lead_n+[x['phase'] for x in f],
          'outcome':ep,'source_sha256':source,'native':rec.info,
          'limitations':['Kinematic humanoid reference, not a trained or dynamically simulated humanoid policy.',
             'Door physics uses generalized joint forces from scripted_hand; hand contact and balance are not validated.',
             'Avatar path differs from the benchmark synthetic base. Unreachable targets remain marked; failed episodes are retained.'],
          'max_hand_error_m':round(float(max(reach)),4),'unreachable_frames':int(np.sum(reach>.08))}
    cp=out/'clips'/f"{door['id']}.json"; write_json(cp,clip)
    gz=cp.with_suffix('.json.gz'); gz.write_bytes(gzip.compress(cp.read_bytes(),mtime=0))
    return {'door_id':door['id'],'family':door['family'],'scenario':scenario,'clip':f'clips/{cp.name}','trajectory':f'trajectories/{npz.name}',
            'duration':clip['duration'],'frames':len(times),'physics_frames':len(f),'success':ep['success'],'outcome':ep['outcome'],
            'max_hand_error_m':clip['max_hand_error_m'],'unreachable_frames':clip['unreachable_frames'],
            'source_sha256':source,'web_clip':f'clips/{gz.name}','web_clip_sha256':digest(gz),'clip_sha256':digest(cp),'trajectory_sha256':digest(npz)}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--assets',default='assets'); p.add_argument('--out',default='out/reference-motions')
    p.add_argument('--doors',default='all'); p.add_argument('--workers',type=int,default=6); p.add_argument('--fps',type=int,default=20)
    a=p.parse_args()
    if not 1<=a.fps<=60: p.error('fps must be 1..60')
    manifest=load_manifest(a.assets); doors=select_doors(manifest,a.doors); jobs=[(d,str(Path(a.assets).resolve()),a.out,a.fps) for d in doors]
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(record_one,jobs):
            rows.append(row)
            print(f"{len(rows)}/{len(doors)} {row['door_id']} {row.get('outcome',row.get('error'))}",flush=True)
    import mujoco
    index={'runtime':{'python':platform.python_version(),'numpy':np.__version__,'mujoco':mujoco.__version__},'schema':SCHEMA,'generator_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'generator_sha256':{str(f):digest(f) for f in [Path('doorbench/reference/record.py'),Path('doorbench/reference/humanoid.py'),Path('doorbench/benchmark/runner.py'),Path('doorbench/benchmark/baselines/scripted_hand.py')]},
           'manifest_sha256':digest(Path(a.assets)/'manifest.json'),'fps':a.fps,'seed':0,'tier':'full',
           'policy':'scripted_hand','embodiment':'kinematic humanoid reference over recorded door physics',
           'scope':'one primary core scenario per door; failed attempts retained',
           'counts':dict(Counter(r.get('outcome','error') for r in rows)),'clips':rows}
    write_json(Path(a.out)/'index.json',index)
    print(json.dumps(index['counts']))
    if any(r.get('error') for r in rows): raise SystemExit(1)
if __name__=='__main__': main()
