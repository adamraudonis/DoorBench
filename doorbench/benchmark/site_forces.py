"""Bounded world-space wrenches on authored approach-side hand sites.

This models force transfer through real articulated hardware. It does not
provide a robot hand, grasp constraint, reachability or balance simulation.
"""
from __future__ import annotations

import numpy as np

from .interactions import ContactSites


class SiteForces:
    max_force_N = 120.

    def __init__(self, env, limits):
        self.env = env
        contacts = ContactSites(env)
        self.contacts=contacts
        self.allowed = {sid for options in contacts.by_joint.values() for sid, _ in options}
        self.site_force_caps={name:float(row['force_cap_N']) for row in env.meta.get('paired_leaf_holds',[])
                              for name in (row['site'],row['engage_site'])}
        self.site_force_caps.update({row['site']:50. for row in env.meta.get('inactive_leaf_pulls',[])})
        self.press_sites={name for row in env.meta.get('paired_leaf_holds',[])if row['kind']=='flush_bolt'
                          for name in(row['site'],row['engage_site'])}
        if any(not np.isfinite(cap)or cap<=0 for cap in self.site_force_caps.values()):
            raise ValueError('Invalid paired-leaf site force cap')
        self.wrench_limits=env.meta.get('site_wrench_limits_Nm',{})
        if any(env.mj.mj_name2id(env.m,env.mj.mjtObj.mjOBJ_SITE,name)<0 or
               not np.isfinite(value) or value<=0 for name,value in self.wrench_limits.items()):
            raise ValueError('Invalid authored site wrench limits')
        # A hand force necessarily loads passive carrier and linkage joints.
        # Zero *direct actuation* permission must not delete that transmitted
        # force or invent a constraint holding those coordinates stationary.
        passive = {b['joint']['name'] for b in env.model_json['bodies']
                   if b.get('joint') and b['joint'].get('role') == 'mechanism'}
        sectional = env.meta.get('sectional_track')
        if sectional:
            passive.add(sectional['root_z_joint'])
        rollup = env.meta.get('rollup_curtain')
        if rollup:
            passive.add(rollup['primary_joint'])
        self.limits = {int(env.m.jnt_dofadr[env._jid(name)]): float(value)
                       for name, value in limits.items() if env._jid(name) >= 0 and name not in passive}

    def generalized(self, data, commands, torques=None):
        return self.resolve(data,commands,torques)[0]

    def resolve(self, data, commands, torques=None):
        """Project each bounded force through native Jacobians, preserving its direction.

        A wrench that would directly actuate forbidden hardware is rejected by
        a zero scale. Passive suspension coordinates retain their natural load.
        """
        env = self.env
        m, mj = env.m, env.mj
        result = np.zeros(m.nv)
        resolved={}
        for name in dict.fromkeys([*(commands or {}),*(torques or {})]):
            sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, name)
            if sid not in self.allowed:
                raise ValueError(f"No approach-side hand force site: {name}")
            force = np.asarray((commands or {}).get(name,[0.,0.,0.]), dtype=float)
            if force.shape != (3,) or not np.isfinite(force).all():
                raise ValueError(f"Invalid world force for {name}")
            magnitude = float(np.linalg.norm(force))
            force_cap=min(self.max_force_N,self.site_force_caps.get(name,self.max_force_N))
            if magnitude > force_cap:
                force = force * (force_cap / magnitude)
            if name in self.press_sites:
                inward=-data.site_xmat[sid].reshape(3,3)[:,2]
                force=inward*max(0.,float(force@inward))
            torque=np.asarray((torques or {}).get(name,[0.,0.,0.]),dtype=float)
            if torque.shape!=(3,) or not np.isfinite(torque).all():
                raise ValueError(f'Invalid world torque for {name}')
            cap=self.wrench_limits.get(name,0.)
            magnitude=float(np.linalg.norm(torque))
            if magnitude and cap<=0:raise ValueError(f'No authored grasp-wrench permission: {name}')
            if magnitude>cap:torque=torque*(cap/magnitude)
            if not self.contacts.available(sid,data):
                force=np.zeros(3);torque=np.zeros(3)
            tau = np.zeros(m.nv)
            mj.mj_applyFT(m, data, force, torque, data.site_xpos[sid],
                          int(m.site_bodyid[sid]), tau)
            scale = min([1.] + [limit/abs(tau[dof]) for dof, limit in self.limits.items()
                                if abs(tau[dof]) > 1e-9])
            result += tau * scale
            resolved[name]={'force_N':force*scale,'torque_Nm':torque*scale}
        scale = min([1.] + [limit/abs(result[dof]) for dof, limit in self.limits.items()
                            if abs(result[dof]) > 1e-9])
        return result * scale,{name:{k:(value*scale).tolist() for k,value in wrench.items()}
                              for name,wrench in resolved.items()}
