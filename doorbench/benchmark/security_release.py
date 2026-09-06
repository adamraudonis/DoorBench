"""Oracle hand release through the installed chain/guard's real service input."""
from __future__ import annotations
import numpy as np


class SecurityRelease:
    def __init__(self,env):
        self.env=env
        self.pending=[r for r in env.meta.get('security_guards',[])
                      if r['engaged_initial'] and r['accessible_from_robot']]
        for row in self.pending:
            if row['kind']=='chain':
                expected={'keyhole_lateral':(row['keyhole_width_m']-row['head_diameter_m'])/2-.0001,
                          'slot_vertical':(row['slot_width_m']-row['neck_diameter_m'])/2-.0001,
                          'seated_position':.003}
                if row.get('handoff_tolerances_m')!=expected:
                    raise ValueError('Chain release tolerances must match its physical keyhole and slot')
        self.stage='close';self.start=None;self.initial=None;self.index=0
        self.jac=np.zeros((3,env.m.nv));self.angular=np.zeros((3,env.m.nv));self.velocity=np.zeros(6)

    def act(self):
        if self.index>=len(self.pending):return None
        e=self.env;m,d,mj=e.m,e.d,e.mj;r=self.pending[self.index]
        t=float(d.time);leaf=m.body(r['leaf_body']).id;primary=e.pj
        q=float(d.qpos[m.jnt_qposadr[primary]]);dq=float(d.qvel[m.jnt_dofadr[primary]])
        hold=float(np.clip(-100*q-20*dq,-40,40))
        action={'torques':{e.meta['primary_joint']:hold},'base_velocity':[0.,0.]}
        if self.stage=='close':
            if abs(q)>.003 or abs(dq)>.03:return action
            self.stage='align' if r['kind']=='chain' else 'park'
            self.start=t
        sid=m.site(r['release_site']).id;body=int(m.site_bodyid[sid]);R=d.xmat[leaf].reshape(3,3)
        if r['kind']=='chain':
            head=m.geom(r['head_geom']).id
            local=R.T@(d.geom_xpos[head]-d.xpos[leaf])
            if self.initial is None:self.initial=local.copy()
            key=np.asarray(r['keyhole_center_leaf']);direction=np.asarray(r['release_sequence'][2]['direction'])
            front=key+direction*r['release_sequence'][2]['distance_m']
            duration={'align':.5,'slide':2.,'keyhold':.5,'withdraw':1.5,'removed':.5}[self.stage]
            destination=self.initial if self.stage=='align' else key if self.stage in ('slide','keyhold') else front
            fraction=min(1.,(t-self.start)/duration)
            target=self.initial+(destination-self.initial)*fraction
            goal=d.xpos[leaf]+R@target
            # The finite hand acts at the authored surface, 14 mm from the
            # head centre. Track and damp that same surface; centre feedback
            # at an offset force point otherwise creates a competing moment.
            if int(m.geom_bodyid[head])!=body:
                raise ValueError('Chain head and grip must share a body')
            grip_goal=goal+R@(m.site_pos[sid]-m.geom_pos[head])
            mj.mj_jacSite(m,d,self.jac,self.angular,sid)
            force=np.clip(1000*(grip_goal-d.site_xpos[sid])-6*(self.jac@d.qvel),-20,20)
            normal=d.xmat[body].reshape(3,3)@np.array([0.,1,0]);wanted=R@np.array([0.,1,0])
            mj.mj_objectVelocity(m,d,mj.mjtObj.mjOBJ_BODY,body,self.velocity,0)
            # The sine/cross-product error vanishes at 180 degrees, exactly
            # where a hanging chain head can settle. Use the shortest full
            # rotation to align its actual head and shank with the keyhole.
            rotation=np.empty(4)
            mj.mju_mat2Quat(rotation,(R@d.xmat[body].reshape(3,3).T).ravel())
            if rotation[0]<0:rotation*=-1
            magnitude=float(np.linalg.norm(rotation[1:]))
            error=rotation[1:]*(2*np.arctan2(magnitude,rotation[0])/max(magnitude,1e-12))
            torque=.2*error-.0002*self.velocity[:3]
            cap=e.meta['site_wrench_limits_Nm'][r['release_site']]
            torque*=min(1.,cap/max(float(np.linalg.norm(torque)),1e-12))
            action.update(site_forces={r['release_site']:force.tolist()},site_torques={r['release_site']:torque.tolist()},
                          contact_site=r['release_site'],contact_joint=r['guard_joints'][-1])
            # A clock alone cannot declare the head removed. Each handoff
            # needs the observed physical end point and a stable orientation.
            position_error=destination-local
            tolerances=r['handoff_tolerances_m']
            aligned=np.linalg.norm(position_error)<tolerances['seated_position']
            if self.stage in ('slide','keyhold'):
                aligned=aligned and np.linalg.norm(position_error[[0,2]])<tolerances['keyhole_lateral']
            if fraction>=1 and aligned and np.dot(normal,wanted)>.97:
                if self.stage=='removed':self.index+=1;self.stage='close';self.initial=None
                else:
                    self.stage={'align':'slide','slide':'keyhold','keyhold':'withdraw','withdraw':'removed'}[self.stage]
                    self.start=t;self.initial=local.copy()
        else:
            joint=m.joint(r['guard_joint']).id;a=m.jnt_qposadr[joint];v=m.jnt_dofadr[joint]
            if self.initial is None:self.initial=float(d.qpos[a])
            target=self.initial+(np.pi-self.initial)*min(1.,(t-self.start)/3.)
            effort=float(np.clip(.5*(target-d.qpos[a])-.02*d.qvel[v],-.25,.25))
            mj.mj_jacSite(m,d,self.jac,self.angular,sid)
            lever=self.jac[:,v];den=float(lever@lever)
            if den<1e-10:raise ValueError('Security guard has no grip lever arm')
            force=lever*effort/den
            action.update(site_forces={r['release_site']:force.tolist()},contact_site=r['release_site'],contact_joint=r['guard_joint'])
            if t-self.start>3.5 and abs(d.qpos[a]-np.pi)<.01:
                self.index+=1;self.stage='close';self.initial=None
        return action
