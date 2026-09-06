"""Select real, approach-side contact sites from the native articulated model.

This is semantic contact dispatch, not a reachability or grasp certificate.
Missing contacts remain missing; a lock cylinder or arbitrary geom centre is
never substituted for a door pull.
"""
from __future__ import annotations
import numpy as np


def direct_surface_access(model,data,site,geom,approach):
    """Five parallel rays must meet the intended grip before any other stock.

    This limited 8 mm contact patch check permits an exposed raised knob on
    either side of a low gate. It does not establish arm reach or a full grasp.
    """
    import mujoco
    direction=np.array([0.,-approach,0.]);hit=np.empty(1,dtype=np.int32)
    for dx,dz in ((0.,0.),(.004,0.),(-.004,0.),(0.,.004),(0.,-.004)):
        origin=data.site_xpos[site]+[dx,approach*.5,dz]
        distance=mujoco.mj_ray(model,data,origin,direction,None,True,-1,hit)
        if hit[0]!=geom or not .49<distance<.51:return False
    return True


def raised_gate_access(model,data,site,geom,approach,above):
    """Screen an over-top acquisition path with 4 mm offset rays.

    The path approaches above the gate, descends beside the knob, then touches
    its real radial surface. This is geometric input access, not a hand mesh,
    continuous swept-volume proof, arm reach or whole-body motion certificate.
    """
    import mujoco
    point=data.site_xpos[site];normal=data.site_xmat[site].reshape(3,3)[:,2]
    beside=point+normal*.015;high=beside.copy();high[2]=above
    front=high.copy();front[1]=approach*.5
    hit=np.empty(1,dtype=np.int32)
    offsets=np.vstack((np.zeros(3),np.eye(3)*.004,-np.eye(3)*.004))
    for start,end in ((front,high),(high,beside)):
        delta=end-start;length=float(np.linalg.norm(delta))
        if length<1e-8:continue
        for offset in offsets:
            distance=mujoco.mj_ray(model,data,start+offset,delta/length,None,True,-1,hit)
            if 0<=distance<length-.0001:return None
    # The last short approach must terminate on this knob, never on its
    # mounting post or on the gate panel in front of it.
    for dx,dz in ((0.,0.),(.004,0.),(-.004,0.),(0.,.004),(0.,-.004)):
        distance=mujoco.mj_ray(model,data,beside+[dx,0.,dz],-normal,None,True,-1,hit)
        if hit[0]!=geom or not .014<distance<.017:return None
    return [p.tolist() for p in (front,high,beside,point)]


class ContactSites:
    def __init__(self, env):
        self.env = env
        m, d = env.m, env.d
        approach=1 if env.spec.get('robot',{}).get('approach_side')=='+y' else -1
        self.roles = {b['joint']['name']: b['joint'].get('role')
                      for b in env.model_json['bodies'] if b.get('joint')}
        bodies = {b['name']: b for b in env.model_json['bodies']}
        names = {env.mj.mj_id2name(m, env.mj.mjtObj.mjOBJ_SITE, i): i for i in range(m.nsite)}
        self.by_joint = {};self.access_paths={}
        # Evaluate at reset. Face membership stays with the moving hardware.
        for b in bodies.values():
            chain, ancestor = [], b
            while ancestor:
                chain.append(ancestor)
                ancestor = bodies.get(ancestor.get('parent'))
            leaf = next((a for a in chain if (a.get('joint') or {}).get('role') in ('primary','secondary')), None)
            hardware = next((a for a in chain if (a.get('joint') or {}).get('role') in ('operator', 'lock', 'latch')), None)
            for site in b.get('sites', []):
                if site.get('role') not in ('grip', 'push', 'touch') or site['name'] not in names:
                    continue
                sid = names[site['name']]
                named_near = site['name'].endswith('_n') or '_n_' in site['name']
                named_far = site['name'].endswith('_p') or '_p_' in site['name']
                named_opposite=named_far if approach<0 else named_near
                named_approach=named_near if approach<0 else named_far
                # A far-face hand target is not made accessible by joint force.
                if leaf:
                    bid = env.mj.mj_name2id(m, env.mj.mjtObj.mjOBJ_BODY, leaf['name'])
                    relative = d.xmat[bid].reshape(3, 3).T @ (d.site_xpos[sid] - d.xpos[bid])
                    if named_opposite or (approach*relative[1] < -.002 and not named_approach):
                        continue
                elif hardware and (named_opposite or (approach*(d.site_xpos[sid,1]-env.meta.get('wall_y',0))<-.002 and not named_approach)):
                    continue
                if hardware:
                    self.by_joint.setdefault(hardware['joint']['name'], []).append((sid, site['role']))
                if leaf and (not hardware or hardware['joint']['role'] == 'operator'):
                    self.by_joint.setdefault(leaf['joint']['name'], []).append((sid, site['role']))
        self.prefer = 'push' if env.spec.get('robot', {}).get('is_push') else 'grip'
        for gate in env.meta.get('gate_hardware',[]):
            if not gate.get('release_face_sites'):continue
            joint=gate['operator_joint'];gid=env.mj.mj_name2id(m,env.mj.mjtObj.mjOBJ_GEOM,gate['knob_geom'])
            # Replace the initial face heuristic with a measured path to
            # the actual radial surface. A blocked knob stays unavailable.
            self.by_joint[joint]=[]
            if gid<0:continue
            tags=('n','p') if approach<0 else ('p','n')
            above=max(float(env.spec['leaf']['height'])+env.spec['opening'].get('ground_clearance',.05)+.15,
                      gate['grip_height_m']+.15)
            for tag in tags:
                sid=names.get(gate['release_face_sites'][tag])
                if sid is None:continue
                direct=tag==tags[0] and direct_surface_access(m,d,sid,gid,approach)
                route=None if direct else raised_gate_access(m,d,sid,gid,approach,above)
                if direct or route:
                    self.by_joint[joint]=[(sid,'grip')]
                    self.access_paths[gate['release_face_sites'][tag]]={'mode':'direct' if direct else 'over_top','waypoints_world_m':route}
                    break
        for bank in env.meta.get('folding_banks',[]):
            # The lead-panel grip operates its supported folding bank, whose
            # primary pivot is one ancestor above the passive fold hinge.
            self.by_joint[bank['pivot_joint']]=[(names[bank['grip_site']],'grip')]
        for control in env.meta.get('sliding_leaf_controls', []):
            # Normal opening uses the face grip. A retractable pocket edge
            # pull is only the extraction contact after the leaf is recessed.
            sites = [s for s in control.get('grip_sites', []) if not s.endswith('_p')]
            self.by_joint[control['joint']] = [(names[s], 'grip') for s in sites if s in names]
        pocket = env.meta.get('pocket_edge_pull')
        if pocket and pocket.get('final_push_site') in names:
            self.by_joint.setdefault(pocket['leaf_joint'], []).append((names[pocket['final_push_site']], 'push'))
        sectional = env.meta.get('sectional_track')
        if sectional and sectional['manual_grip_site'] in names:
            # The bottom grip is carried through several passive panel
            # hinges. It is the contact for the complete lifting assembly.
            self.by_joint[sectional['root_z_joint']] = [(names[sectional['manual_grip_site']], 'grip')]
        rollup = env.meta.get('rollup_curtain')
        if rollup and rollup['manual_grip_site'] in names:
            self.by_joint[rollup['primary_joint']] = [(names[rollup['manual_grip_site']], 'grip')]
        if env.meta.get('rollup_hoist'):
            self.by_joint[rollup['primary_joint']] = [(names[name],'grip') for name in env.meta['rollup_hoist']['material_grip_sites']
                if approach*(d.site_xpos[names[name],1]-env.meta.get('wall_y',0))>0]
            keeper=env.meta['rollup_hoist'].get('keeper')
            if keeper and keeper['grip_site'] in names:
                sid=names[keeper['grip_site']]
                if approach*(d.site_xpos[sid,1]-env.meta.get('wall_y',0))>0:
                    self.by_joint[keeper['joint']]=[(sid,'grip')]
        for guard in env.meta.get('security_guards',[]):
            site=guard['release_site']
            if guard['accessible_from_robot'] and site in names:
                joint=guard.get('guard_joint') or guard['guard_joints'][-1]
                self.by_joint.setdefault(joint,[]).append((names[site],'grip'))

    def select(self, joint):
        options = self.by_joint.get(joint, [])
        if not options:
            return None
        return min(options, key=lambda p: (p[1] != self.prefer, p[0]))[0]

    def active(self, action):
        explicit = action.get('contact_site')
        if explicit:
            sid = self.env.mj.mj_name2id(self.env.m, self.env.mj.mjtObj.mjOBJ_SITE, explicit)
            return action.get('contact_joint'), sid if sid >= 0 else None
        torques = action.get('torques') or {}
        active = [(j, abs(float(t))) for j, t in torques.items() if abs(float(t)) > 1e-5]
        hardware = [(j, t) for j, t in active if self.roles.get(j) in ('operator', 'lock', 'latch')]
        moving_hardware=[]
        for name,effort in hardware:
            jid=self.env._jid(name)
            if jid<0: continue
            lo,hi=self.env.m.jnt_range[jid]
            q=float(self.env.d.qpos[self.env.m.jnt_qposadr[jid]])
            if hi-lo>1e-8 and q<lo+.9*(hi-lo):moving_hardware.append((name,effort))
        leaves=[(name,effort) for name,effort in active if self.roles.get(name) in ('primary','secondary')]
        if leaves and not moving_hardware:
            # A withdrawn bolt being held is a concurrent support contact,
            # not the handle used to move the leaf. Prefer the active leaf.
            joint=action.get('active_leaf')
            if not any(name==joint for name,_ in leaves):joint=max(leaves,key=lambda pair:pair[1])[0]
            return joint,self.select(joint)
        joint = max(hardware, key=lambda p: p[1])[0] if hardware else action.get('active_leaf')
        if not joint or joint not in torques or abs(float(torques[joint])) <= 1e-5:
            joint = max(active, key=lambda p: p[1])[0] if active else None
        return joint, self.select(joint)
