"""Dynamic meeting-edge access and real surface-force paired-leaf sequencing."""
import json
import numpy as np
import pytest
from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.interactions import ContactSites
from doorbench.benchmark.site_forces import SiteForces
from doorbench.benchmark.runner import Job,run_episode,torque_limits


def test_contact_control_cadence_matches_native_proof_including_chain_guards():
    from types import SimpleNamespace
    from doorbench.benchmark.baselines.scripted_hand import ScriptedHandPolicy
    policy=ScriptedHandPolicy()
    for key,dt in(('security_guards',.000025),('paired_leaf_holds',.0005),('rollup_hoist',.0005)):
        env=SimpleNamespace(meta={key:[{}]},m=SimpleNamespace(opt=SimpleNamespace(timestep=dt)))
        assert policy.control_period(env)==dt
    assert policy.control_period(SimpleNamespace(meta={},m=None))==.004


@pytest.fixture(scope='module')
def doors(tmp_path_factory):
    root=tmp_path_factory.mktemp('paired-runtime');out={}
    for s in generate_all():
        if s['index']not in(127,222,467):continue
        summary=export_door(s,str(root/'doors'),str(root/'hardware'),formats=('json','mjcf'))
        out[s['index']]=(root/'doors'/s['id'],{'id':s['id'],'family':s['family'],'benchmark':summary['benchmark']})
    return out


def test_flush_permission_is_potential_but_every_site_application_checks_exposure(doors):
    p,_=doors[222];env=DoorEnv(str(p));env.reset(randomize=False)
    try:
        m,d,mj=env.m,env.d,env.mj;row=env.meta['paired_leaf_holds'][0]
        contacts=ContactSites(env);limits=torque_limits(env,str(p));forces=SiteForces(env,limits)
        assert limits[row['joint']]==20 and contacts.potential(row['joint'])
        assert contacts.select(row['joint'])is None
        j=env._jid(row['joint']);sid=m.site(row['site']).id;command=d.xaxis[j]*100.
        original=command.copy();tau,used=forces.resolve(d,{row['site']:command})
        assert not np.any(tau)and np.array_equal(command,original)
        assert np.linalg.norm(used[row['site']]['force_N'])==0
        # Explicit geometry fixtures test dynamic acquisition and revocation;
        # no native unlocking claim is made by these configuration assignments.
        a=m.jnt_qposadr[env._jid(row['primary_joint'])]
        for angle,allowed in((.3,True),(0.,False),(.3,True)):
            d.qpos[a]=angle;mj.mj_kinematics(m,d)
            assert (contacts.select(row['joint'])is not None)==allowed
            tau,used=forces.resolve(d,{row['site']:d.xaxis[j]*100.})
            assert (np.linalg.norm(tau)>0)==allowed
            assert np.linalg.norm(used[row['site']]['force_N'])<=20.+1e-9
        # A finger press cannot pull outward or apply a sideways free wrench.
        outward=d.site_xmat[sid].reshape(3,3)[:,2]
        tau,_=forces.resolve(d,{row['site']:outward*20.});assert not np.any(tau)
    finally:env.close()


def test_cane_uses_its_own_inside_permission_independent_of_active_lock(doors):
    for n,allowed in((127,False),(467,True)):
        p,_=doors[n];env=DoorEnv(str(p));env.reset(randomize=False)
        try:
            r=env.meta['paired_leaf_holds'][0];contacts=ContactSites(env);limits=torque_limits(env,str(p))
            env.d.qpos[env.m.jnt_qposadr[env._jid(r['primary_joint'])]]=-1e-5
            assert (contacts.select(r['joint'])is not None)==allowed
            assert (limits[r['joint']]>0)==allowed
            if not allowed:
                with pytest.raises(ValueError,match='No approach-side'):
                    SiteForces(env,limits).resolve(env.d,{r['site']:[0,0,20]})
        finally:env.close()


def test_direct_generalized_command_cannot_bypass_closed_edge(doors,monkeypatch):
    import doorbench.benchmark.runner as runner
    p,row=doors[222];samples=[]
    class Attack:
        control_dt=.1
        def reset(self,info,env=None):self.joints=[r['joint']for r in info['meta']['paired_leaf_holds']]
        def act(self,obs):return {'torques':{j:20. for j in self.joints}}
    monkeypatch.setattr(runner,'_policy_for',lambda _:Attack())
    def observe(event,env,base,action):
        if event=='step':samples.append([float(env.d.qpos[env.m.jnt_qposadr[env._jid(r['joint'])]])for r in env.meta['paired_leaf_holds']])
    result=run_episode(Job(row,str(p),'open_and_traverse',0,'full','scripted_hand',time_budget_s=.3,randomize=False),observer=observe)
    assert result['error']is None and not result['success']
    assert max(abs(q)for sample in samples for q in sample)<1e-6
    assert not result['labels']['touched_door']


@pytest.mark.parametrize('index',(222,467))
@pytest.mark.parametrize('scenario',('open_then_close','close_only'))
def test_native_primary_release_and_reverse_closing_sequence(doors,index,scenario):
    p,row=doors[index];phases=[];last=None;violations=[];peak=0.;final={};samples=[]
    def observe(event,env,base,action):
        nonlocal last,peak,final
        if event!='step':return
        m,d=env.m,env.d;phase=action.get('pair_phase')
        q={n:float(d.qpos[m.jnt_qposadr[env._jid(n)]])for n in ['leaf_a_hinge','leaf_b_hinge']+[r['joint']for r in env.meta['paired_leaf_holds']]}
        final=q
        if phase!=last:phases.append({'time':float(d.time),'phase':phase,'q':q});last=phase
        for r in env.meta['paired_leaf_holds']:
            if any(np.linalg.norm((action.get('site_forces')or{}).get(site,[0,0,0]))>0 for site in(r['site'],r['engage_site'])):
                if q[r['primary_joint']]<r['requires_primary_open_rad']-1e-4:violations.append('hidden edge force')
            if q['leaf_b_hinge']>.01 and q[r['joint']]<r['withdrawn_threshold_m']:violations.append('B moved before rod clear')
        for c in d.contact:peak=max(peak,-float(c.dist))
        if not samples or d.time-samples[-1]['time']>=.1:samples.append({'time':float(d.time),'q':q,'phase':phase})
    result=run_episode(Job(row,str(p),scenario,0,'full','scripted_hand',time_budget_s=40,randomize=False),observer=observe)
    proof={'episode':result,'pair_phases':phases,'final_q':final,'violations':violations,'max_all_native_penetration_m':peak,'samples':samples}
    (p/('runtime-'+scenario+'.json')).write_text(json.dumps(proof,indent=2))
    assert result['error']is None and result['success'],proof
    assert not violations and peak<=.001
    assert not result['mujoco_warning_messages']and not result['damage']and not result['labels']['door_slammed']
    assert abs(final['leaf_a_hinge'])<.02 and abs(final['leaf_b_hinge'])<.001
    assert all(abs(v)<.002 for n,v in final.items()if n.endswith('_slide'))
    if scenario=='open_then_close':
        names=[r['phase']for r in phases]
        assert any(n and n.startswith('withdraw:')for n in names)
        assert any(n and n.startswith('reinsert:')for n in names)
        assert names.index('close_B')<next(i for i,n in enumerate(names)if n and n.startswith('reinsert:'))<names.index('close_A')
    else:assert not any(r['phase']and r['phase'].startswith('withdraw:')for r in phases)
