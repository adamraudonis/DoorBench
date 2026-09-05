"""Compose a metric contact/footstep proposal from an immutable native recording.

Native states are re-timed poses. Their original time and outcome remain separate
from the proposal's clock; this module never claims the re-timed motion satisfies
the original dynamics or the benchmark's original time budget.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
from .planning import SceneNavigator, NoRoute, heading, smoothstep


def yaw_quaternion(yaw):
    y=np.asarray(yaw,float)
    return np.stack([np.cos(y/2),np.zeros_like(y),np.zeros_like(y),np.sin(y/2)],axis=-1)


def rest_hands(pelvis,yaw,feet):
    """Restrained contralateral arm swing, expressed as soft wrist targets."""
    p=np.asarray(pelvis);yaw=np.asarray(yaw)
    right=np.stack([np.cos(yaw),np.sin(yaw),np.zeros_like(yaw)],axis=-1)
    forward=np.stack([-np.sin(yaw),np.cos(yaw),np.zeros_like(yaw)],axis=-1)
    wrists=np.repeat(p[:,None,:],2,axis=1)
    for k,sign in [(0,-1),(1,1)]:
        stride=np.sum((feet[:,k,:]-p)*forward,axis=1)
        swing=np.clip(-stride*.35,-.07,.07)
        wrists[:,k]+=right*(sign*.22)+forward*(.07+swing)[:,None]+[0,0,-.12]
    return wrists


@dataclass
class MotionGuide:
    time: np.ndarray
    native_time: np.ndarray
    native_qpos: np.ndarray
    pelvis: np.ndarray
    yaw: np.ndarray
    foot_pos: np.ndarray
    foot_quat: np.ndarray
    foot_contact: np.ndarray
    hands: np.ndarray
    hand_contact: np.ndarray
    hand_weight: np.ndarray
    phases: list[str]
    metadata: dict


class GuideBuilder:
    def __init__(self,source,fps):
        self.source=source;self.fps=fps;self.parts=[]

    def add(self,gait,native_time,phase,*,hand=None,reach_weight=None,pelvis_height=None):
        t=np.asarray(gait['time']);n=len(t)
        nt=np.broadcast_to(native_time,(n,)).astype(float).copy()
        pelvis=gait['pelvis_xyz'].copy()
        if pelvis_height is not None:pelvis[:,2]=np.broadcast_to(pelvis_height,(n,))
        hands=rest_hands(pelvis,gait['pelvis_yaw'],gait['foot_pos'])
        contact=np.zeros((n,2),bool)
        weights=np.zeros((n,2),float)
        if hand:
            k=0 if hand=='left_hand' else 1
            target=np.stack([np.interp(nt,self.source['time'],self.source['target'][:,j]) for j in range(3)],axis=1)
            w=np.ones(n) if reach_weight is None else np.asarray(reach_weight)
            hands[:,k]=hands[:,k]*(1-w[:,None])+target*w[:,None]
            contact[:,k]=w>=1-1e-8
            weights[:,k]=w
        q=np.stack([np.interp(nt,self.source['time'],self.source['qpos'][:,j]) for j in range(self.source['qpos'].shape[1])],axis=1)
        self.parts.append({'time':t,'native_time':nt,'native_qpos':q,'pelvis':pelvis,
            'yaw':gait['pelvis_yaw'],'foot_pos':gait['foot_pos'],'foot_quat':gait['foot_quat'],
            'foot_contact':gait['foot_contact'],'hands':hands,'hand_contact':contact,'hand_weight':weights,'phases':[phase]*n})

    def finish(self,metadata):
        result={};offset=0.;parts=[]
        for i,p in enumerate(self.parts):
            keep=slice(1,None) if i else slice(None)
            part={k:v[keep] for k,v in p.items()}
            part['time']=part['time']+offset
            offset=float(part['time'][-1]);parts.append(part)
        for key in parts[0]:
            result[key]=sum([p[key] for p in parts],[]) if key=='phases' else np.concatenate([p[key] for p in parts])
        return MotionGuide(**result,metadata=metadata)


def stationary(previous,duration,fps):
    """A held support pose; transitions are authored separately."""
    n=max(2,int(math.ceil(duration*fps))+1)
    return {'time':np.arange(n)/fps,**{k:np.repeat(np.asarray(previous[k])[-1:],n,axis=0)
        for k in ['pelvis_xyz','pelvis_yaw','foot_pos','foot_quat','foot_contact']}}


def native_progress(source, start, end):
    """A path clock bounded by both handle travel and native joint travel.

    Replaying the recording's time coordinate preserves long pauses followed by
    very fast motion. Parameterizing its geometric path instead preserves every
    pose while allowing a comfortable manipulation speed. The returned duration
    includes the peak derivative of the quintic endpoint easing curve.
    """
    q=np.asarray(source['qpos'][start:end+1])
    target=np.asarray(source['target'][start:end+1])
    joint=np.max(np.abs(np.diff(q,axis=0)),axis=1)/.45
    hand=np.linalg.norm(np.diff(target,axis=0),axis=1)/.18
    travel=np.maximum(joint,hand)
    # Strictly increasing path coordinates retain otherwise identical poses.
    arc=np.r_[0.,np.cumsum(np.maximum(travel,1e-9))]
    return arc, max(.35,float(arc[-1])*1.875)


def smooth_body_guidance(guide, fps, sigma_seconds=.20):
    """Author continuous soft body guidance while leaving all contacts intact.

    The footstep planner deliberately stops at support changes. Applying those
    stops directly to the pelvis causes excessive side-to-side lurching. Only
    the body targets are filtered here, before constrained IK; no solved pose,
    planted foot, native door coordinate, or active grasp target is smoothed.
    Scene clearance and actor derivatives still require independent validation.
    """
    from scipy.ndimage import gaussian_filter1d
    old_rest=rest_hands(guide.pelvis,guide.yaw,guide.foot_pos)
    guide.pelvis=gaussian_filter1d(guide.pelvis,fps*sigma_seconds,axis=0,mode='nearest')
    guide.yaw=gaussian_filter1d(np.unwrap(guide.yaw),fps*sigma_seconds,mode='nearest')
    new_rest=rest_hands(guide.pelvis,guide.yaw,guide.foot_pos)
    guide.hands+=(1-guide.hand_weight[:,:,None])*(new_rest-old_rest)
    guide.metadata['body_guidance_filter']={'type':'Gaussian','sigma_seconds':sigma_seconds,
        'scope':'Soft body and resting-arm targets before IK; foot contacts and native poses unchanged.'}
    return guide


def make_guide(door_dir,recording_dir,*,fps=30,gait_profile='smooth'):
    from .gait import plan_walk
    door_dir=Path(door_dir);recording_dir=Path(recording_dir);door_id=door_dir.name
    clip=json.loads((recording_dir/'clips'/f'{door_id}.json').read_text())
    with np.load(recording_dir/'trajectories'/f'{door_id}.npz',allow_pickle=False) as f:
        source={k:f[k].astype(float) for k in ['time','qpos','target','base','tau']}
    spec=json.loads((door_dir/'spec.json').read_text())
    from .powered import powered_eligibility, make_powered_guide
    if not powered_eligibility(clip,spec,source):
        # The selected builder authenticates its inputs before planning. Do not
        # catch integrity or route errors and invent a manual fallback for it.
        return make_powered_guide(door_dir,recording_dir,fps=fps,gait_profile=gait_profile)
    if spec.get('family')=='sliding_single' and spec.get('lock',{}).get('model')=='padlock':
        from .manual_contacts import plan_manual_contacts,build_manual_guide,UnsupportedManualContact
        model_ir=json.loads((door_dir/'model.json').read_text())
        try:
            plan_manual_contacts(spec,model_ir,clip,source)
        except UnsupportedManualContact:
            pass  # Unsupported schedules retain the baseline's explicit limits.
        else:
            # Source-integrity and route errors from a selected builder surface.
            return build_manual_guide(door_dir,recording_dir,fps=fps,gait_profile=gait_profile).guide
    scenario=next(s for s in spec['benchmark']['scenarios'] if s['name']==clip['scenario'])
    nav=SceneNavigator(door_dir)
    start=source['base'][0,:2]
    nav.update(source['qpos'][0])
    stance=nav.stance(source['target'][0],start)
    route=nav.route(start,stance.xy)
    if gait_profile not in ('controlled','wide_turns','compact','smooth'):
        raise ValueError('Unknown gait profile')
    walk_options={'fps':fps,'step_length':.42,'step_duration':.65,'stance_width':.21,'blend_turns':True,
                  'max_step_yaw_deg':45. if gait_profile=='wide_turns' else 20.,
                  'pelvis_acceleration_m_s2':None if gait_profile in ('compact','smooth') else 1.5}
    approach=plan_walk(start,0.,route[1:],**walk_options)
    # A final in-place turn aligns the body with the chosen interaction face.
    turn=plan_walk(route[-1],float(approach['pelvis_yaw'][-1]),[route[-1]],**walk_options,waypoint_yaws=[stance.yaw])
    builder=GuideBuilder(source,fps)
    builder.add(approach,source['time'][0],'approach')
    builder.add(turn,source['time'][0],'face_hardware')
    reach=stationary(turn,.9,fps);w=smoothstep(reach['time']/reach['time'][-1])
    builder.add(reach,source['time'][0],'reach',hand=stance.hand,reach_weight=w,
                pelvis_height=.94*(1-w)+stance.pelvis_height*w)
    # Select geometric key states; retain operator motion even at a fixed handle position.
    q=source['qpos'];scale=np.maximum(np.ptp(q,axis=0),.05)
    keys=[0]
    active=np.flatnonzero(np.max(np.abs(source['tau']),axis=1)>1e-6)
    stop=int(active[-1]) if len(active) else len(q)-1
    crossed=np.flatnonzero(source['base'][:,1]>.05)
    if len(crossed):stop=min(stop,int(crossed[0]))
    for i in range(1,stop+1):
        j=keys[-1]
        if np.max(np.abs(q[i]-q[j])/scale)>.14 or np.linalg.norm(source['target'][i]-source['target'][j])>.10:
            keys.append(i)
    if keys[-1]!=stop:keys.append(stop)
    current=stance.xy.copy();yaw=stance.yaw;last=reach;hand=stance.hand;last_height=stance.pelvis_height
    for a,b in zip(keys,keys[1:]):
        nav.update(q[b])
        nxt=nav.stance(source['target'][b],current,preferred_hand=hand,previous_yaw=yaw,previous_height=last_height)
        if np.linalg.norm(nxt.xy-current)<.09:
            nxt=type(nxt)(current.copy(),yaw,nxt.pelvis_height,hand,nxt.clearance)
        travel=plan_walk(current,yaw,[nxt.xy],**walk_options,waypoint_yaws=[nxt.yaw])
        arc,duration=native_progress(source,a,b)
        if travel['time'][-1]<duration:
            # Stationary guides can be resampled; walking guides retain their authored contact timing.
            if np.linalg.norm(nxt.xy-current)<1e-8 and abs(nxt.yaw-yaw)<1e-8:
                travel=stationary(last,duration,fps)
            else:
                factor=math.ceil(duration/max(travel['time'][-1],1e-8))
                travel['time']=travel['time']*factor
        blend=smoothstep(travel['time']/max(travel['time'][-1],1e-8))
        nt=np.interp(blend*arc[-1],arc,source['time'][a:b+1])
        height=last_height*(1-blend)+nxt.pelvis_height*blend
        builder.add(travel,nt,'operate',hand=hand,pelvis_height=height)
        current=nxt.xy.copy();yaw=nxt.yaw;last=travel;last_height=nxt.pelvis_height
    release=stationary(last,.8,fps);w=1-smoothstep(release['time']/release['time'][-1])
    builder.add(release,source['time'][stop],'release',hand=hand,reach_weight=w,
                pelvis_height=.94*(1-w)+last_height*w)
    traverse='not_requested' if clip['scenario']=='locked_recognize' else 'not_attempted'
    reason=None
    if clip['scenario']!='locked_recognize' and clip['outcome']['success']:
        nav.update(q[stop])
        try:
            route=nav.passage_route(current,np.asarray(scenario['goal']['center'])[:2],scenario['pass_plane'])
            walking=plan_walk(current,yaw,route[1:],**walk_options)
            builder.add(walking,source['time'][stop],'traverse');last=walking;traverse='proposed'
        except NoRoute as exc:traverse='unresolved';reason=str(exc)
    finish=stationary(last,.5,fps);builder.add(finish,source['time'][stop],'settle')
    guide=builder.finish({'door_id':door_id,'scenario':clip['scenario'],'source_outcome':clip['outcome'],
        'source_sha256':clip['source_sha256'],'traversal':traverse,'traversal_reason':reason,
        'hand':hand,'native_keyframes':len(keys),'gait_profile':gait_profile,
        'scope':'Re-timed kinematic proposal; constrained IK and independent acceptance still required.'})
    return smooth_body_guidance(guide,fps) if gait_profile=='smooth' else guide
