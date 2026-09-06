"""Sequential service inputs for native vault boltwork, with finite surface forces."""
from __future__ import annotations
import math
import numpy as np
from .interactions import ContactSites


def vault_seated(model,data,meta):
    """A slow leaf actually contacting its closing rebate, not merely near it."""
    import mujoco
    j=model.joint(meta['primary_joint']).id
    if not (-.001<float(data.qpos[model.jnt_qposadr[j]])<.0001
            and abs(float(data.qvel[model.jnt_dofadr[j]]))<.015):return False
    stops={model.geom(name).id for name in meta['vault_closing_stops']}
    for i,c in enumerate(data.contact):
        if c.geom1 not in stops and c.geom2 not in stops:continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
        if np.linalg.norm(force[:3])>.1:return True
    return False


class VaultControl:
    def __init__(self,env):
        self.env=env;self.groups=env.meta['vault_boltwork']['groups']
        contacts=ContactSites(env);m=env.m
        self.inputs=[]
        for row in self.groups:
            j=m.joint(row['operator_joint']).id;sid=contacts.select(row['operator_joint'])
            if sid is None:raise ValueError('Vault input has no authored approach-side surface')
            self.inputs.append((row,j,sid,m.joint(row['carrier_joint']).id))
        self.leaf=m.joint(env.meta['primary_joint']).id
        self.leaf_site=contacts.select(env.meta['primary_joint'])
        if self.leaf_site is None:raise ValueError('Vault leaf has no approach-side grip')
        self.group=0;self.released_at=None
        self.closing_started=False;self.rethrow_group=None

    def released(self):
        m,d=self.env.m,self.env.d
        return all(float(d.qpos[m.jnt_qposadr[c]])>=r['stroke_m']-.0005 for r,_,_,c in self.inputs)

    def _contact(self,joint,site,effort):
        m,d=self.env.m,self.env.d
        tangent=np.cross(d.xaxis[joint],d.site_xpos[site]-d.xanchor[joint])
        length=float(np.linalg.norm(tangent))
        if length<.05:raise ValueError('Vault force has no usable moment arm')
        return {'site_forces':{m.site(site).name:(effort*tangent/length).tolist()},
                'contact_joint':m.joint(joint).name,'contact_site':m.site(site).name}

    def act(self,*,closing=False,goal=None,hands_off=False):
        m,d=self.env.m,self.env.d
        if closing and self.closing_started:return self._close()
        if not self.released():
            self.released_at=None
            # Work one real release at a time. Bolt coordinates, rather than
            # loose safety limits on the input joints, determine completion.
            while self.group<len(self.inputs):
                r,j,s,c=self.inputs[self.group];q=int(m.jnt_qposadr[c])
                if float(d.qpos[q])<r['stroke_m']-.0005:break
                self.group+=1
            if self.group==len(self.inputs):self.group=0
            r,j,s,_=self.inputs[self.group]
            effort=float(np.clip(140.*(r['operator_nominal_range'][1]+.6-d.qpos[m.jnt_qposadr[j]])
                                  -16.*d.qvel[m.jnt_dofadr[j]],-66.7,66.7))
            return self._contact(j,s,effort)
        if self.released_at is None:self.released_at=float(d.time)
        if float(d.time)-self.released_at<.4 or hands_off:return {}
        if closing:
            self.closing_started=True
            return self._close()
        return self._leaf_force(min(float(goal),float(self.env.meta['vault_primary_nominal_range'][1])-.02))

    def _close(self):
        m,d=self.env.m,self.env.d
        if self.rethrow_group is None:
            if not vault_seated(m,d,self.env.meta):return self._leaf_force(-.005)
            self.rethrow_group=0
        # Only rethrow after measured seating. Each independent carrier must
        # return; closing the leaf alone cannot latch a vault.
        while self.rethrow_group<len(self.inputs):
            r,j,s,c=self.inputs[self.rethrow_group]
            if abs(float(d.qpos[m.jnt_qposadr[c]]))<.0005 and abs(float(d.qpos[m.jnt_qposadr[j]]))<.01:
                self.rethrow_group+=1
                continue
            effort=float(np.clip(140.*(-.6-d.qpos[m.jnt_qposadr[j]])-16.*d.qvel[m.jnt_dofadr[j]],-66.7,66.7))
            return self._contact(j,s,effort)
        return {}

    def _leaf_force(self,target):
        m,d=self.env.m,self.env.d
        q=float(d.qpos[m.jnt_qposadr[self.leaf]]);v=float(d.qvel[m.jnt_dofadr[self.leaf]])
        radius=float(np.linalg.norm(np.cross(d.xaxis[self.leaf],d.site_xpos[self.leaf_site]-d.xanchor[self.leaf])))
        friction=float(m.dof_frictionloss[m.jnt_dofadr[self.leaf]])/radius
        compensation=math.copysign(friction,target-q) if abs(target-q)>.00005 else 0.
        damping=2.2*math.sqrt(250.*float(m.dof_M0[m.jnt_dofadr[self.leaf]])/radius)
        effort=float(np.clip(250.*(target-q)-damping*v+compensation,-120.,120.))
        return self._contact(self.leaf,self.leaf_site,effort)
