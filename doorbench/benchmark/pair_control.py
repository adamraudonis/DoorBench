"""Contact-operated inactive-leaf sequence, preserving primary access rules."""
from __future__ import annotations
import numpy as np


class PairControl:
    """A opens normally, then exposed bolts release B; B reseats before A.

    Uses the native state and actual finger/pull sites. No configuration,
    native constraints, credential state or joint ranges are changed.
    """
    def __init__(self,env,limits):
        if env is None:raise ValueError('Inactive-leaf contact control requires the native environment')
        self.env=env;self.rows=env.meta['paired_leaf_holds'];self.limits=limits
        self.A=self.rows[0]['primary_joint'];self.B=self.rows[0]['leaf_joint']
        self.eligible=all(r['accessible_from_robot']and limits.get(r['joint'],0)>0 for r in self.rows)
        self.phase='open_A' if self.eligible else 'A_only';self.events=[];self.failed=None
        self.started=0.;self.closing=False;self.completed=False
        self.approach=1 if env.spec['robot'].get('approach_side','-y')=='+y'else -1
        self.pull=next(p for p in env.meta['inactive_leaf_pulls']if p['face']==self.approach)

    def q(self,name):
        m,d=self.env.m,self.env.d;j=m.joint(name).id
        return float(d.qpos[m.jnt_qposadr[j]]),float(d.qvel[m.jnt_dofadr[j]])

    def set_phase(self,name):
        if name!=self.phase:
            self.phase=name;self.started=float(self.env.d.time);self.events.append({'time_s':self.started,'phase':name})

    def bolt_force(self,row,withdraw):
        m,d=self.env.m,self.env.d;j=m.joint(row['joint']).id;q,v=self.q(row['joint'])
        sid=m.site(row['site']if withdraw else row['engage_site']).id
        axis=d.xaxis[j];mass=float(m.body_mass[m.site_bodyid[sid]])
        target=row['travel_m']+.001 if withdraw else -.001
        force=float(np.clip(2500*(target-q)-20*v+mass*9.81*axis[2],-row['force_cap_N'],row['force_cap_N']))
        if row['kind']=='flush_bolt':force=max(0.,force)if withdraw else min(0.,force)
        return m.site(sid).name,(axis*force).tolist()

    def pull_force(self,target):
        m,d,mj=self.env.m,self.env.d,self.env.mj;j=m.joint(self.B).id;vadr=int(m.jnt_dofadr[j])
        q,speed=self.q(self.B);friction=float(m.dof_frictionloss[vadr])
        if target==0.:
            effort=100*(-min(.2,8*max(q-.00015,0.))-speed)-friction-2.
        else:
            error=target-q;effort=100*error-20*speed
            if abs(error)>.001:effort+=np.sign(error)*friction
        effort=float(np.clip(effort,-self.limits.get(self.B,0),self.limits.get(self.B,0)))
        sid=m.site(self.pull['site']).id;direction=d.site_xmat[sid].reshape(3,3)[:,2]
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jacSite(m,d,jp,jr,sid)
        leverage=float(jp[:,vadr]@direction)
        if abs(leverage)<.1:raise ValueError('Inactive-leaf pull has no useful native lever arm')
        return self.pull['site'],(direction*np.clip(effort/leverage,-50.,50.)).tolist()

    def apply(self,action,policy,obs):
        result=dict(action);out=dict(result.get('torques')or{});out.pop(self.B,None)
        for r in self.rows:out.pop(r['joint'],None)
        result['torques']=out
        if not self.eligible or policy.scenario=='locked_recognize':return result
        if self.failed:return {'mechanism_failure':self.failed,'base_velocity':[0.,0.],'pair_phase':'failed'}
        closing=policy.scenario=='close_only'or(policy.require_closed and policy.t_pass is not None)
        if closing!=self.closing:
            self.closing=closing;self.set_phase('close_B'if closing else'open_A')
        qA,_=self.q(self.A);qB,vB=self.q(self.B);now=float(self.env.d.time)
        commands=dict(result.get('site_forces')or{})
        contact=None
        if closing:
            engaged=all(self.q(r['joint'])[0]<=.001 for r in self.rows)
            if abs(qB)<.0006 and abs(vB)<.01 and engaged:
                self.completed=True;self.set_phase('close_A')
            else:
                # Keep the meeting edge exposed until B is seated and secured.
                limit=self.limits.get(self.A,0.)
                out.pop(self.A,None)
                policy.servo(obs,self.A,.8,3*max(limit,1.),.5*max(limit,1.),limit,out)
                site,force=self.pull_force(0.);commands[site]=force;contact=(self.B,site)
                if abs(qB)<.0006 and abs(vB)<.01:
                    if qA<max(r['requires_primary_open_rad']for r in self.rows):
                        self.set_phase('expose_for_reinsert')
                    else:
                        row=next(r for r in self.rows if self.q(r['joint'])[0]>.001)
                        self.set_phase('reinsert:'+row['joint']);site,force=self.bolt_force(row,False)
                        commands[site]=force;contact=(row['joint'],site)
                else:self.set_phase('close_B')
        else:
            # No secondary action before the primary's normal opening phase.
            if policy.t_push is None or now-policy.delay<policy.t_push:return result
            if policy.t_pass is not None and policy.released:return result
            pending=[r for r in self.rows if self.q(r['joint'])[0]<r['travel_m']-.0005]
            if pending:
                if qA<max(r['requires_primary_open_rad']for r in self.rows):self.set_phase('open_A')
                else:
                    row=pending[0];self.set_phase('withdraw:'+row['joint'])
                    site,force=self.bolt_force(row,True);commands[site]=force;contact=(row['joint'],site)
            else:
                target=policy._secondary_target(self.B);site,force=self.pull_force(target);commands[site]=force;contact=(self.B,site)
                self.set_phase('open_B'if qB<.9*target else'both_open')
            if self.phase!='both_open':result['base_velocity']=[0.,0.]
        if (self.phase.startswith(('withdraw:','reinsert:'))and now-self.started>3.)or(self.phase=='close_B'and now-self.started>12.):
            self.failed='Inactive-leaf contact sequence stalled at '+self.phase
            return {'mechanism_failure':self.failed,'base_velocity':[0.,0.],'pair_phase':'failed'}
        if commands:result['site_forces']=commands
        if contact:result.update(contact_joint=contact[0],contact_site=contact[1])
        result['pair_phase']=self.phase
        return result
