"""Private native boundary inspection for the turnstile's contact-driven pawl.

The bolt is a prescribed electrical-input state during a sweep. Its default
position is obtained independently with the actual coil/spring at the closed
rotor. No source model, benchmark state or native input recording is changed.
"""
from __future__ import annotations
import copy
import math
import numpy as np


class TurnstileContactPreview:
    def __init__(self,model,metadata):
        from .turnstile_locks import compile_turnstile_locks
        self.model=copy.copy(model);self.row=metadata['turnstile_locks'];self.rules=compile_turnstile_locks(self.model,metadata)
        self.cache={}

    def default_qpos(self,qpos):
        return self._settle(qpos,[self.row['bolt_joint']],coil=True)

    def resolve(self,qpos,driven_joint=None):
        pawn=self.row['pawl_joint']
        return self._settle(qpos,[pawn] if pawn and pawn!=driven_joint else [])

    def _settle(self,qpos,names,coil=False):
        import mujoco
        from .ir import quat_from_axis_angle,quat_rotate
        m=self.model;q=np.asarray(qpos,float)
        if q.shape!=(m.nq,) or not np.isfinite(q).all():raise ValueError('Invalid turnstile inspection coordinates')
        key=(tuple(names),coil,tuple(q))
        if key in self.cache:return {**self.cache[key],'qpos':list(self.cache[key]['qpos'])}
        if not names:return {'ok':True,'qpos':q.tolist(),'failures':[]}
        source=mujoco.MjData(m);source.qpos[:]=q;mujoco.mj_kinematics(m,source)
        spec=mujoco.MjSpec();spec.compiler.degree=False;spec.option.timestep=m.opt.timestep
        spec.option.gravity=m.opt.gravity;spec.option.integrator=m.opt.integrator
        spec.option.iterations=m.opt.iterations;spec.option.tolerance=m.opt.tolerance
        bodies={};joints={}
        for name in names:
            j=m.joint(name).id;bid=int(m.jnt_bodyid[j]);a=int(m.jnt_qposadr[j]);v=int(m.jnt_dofadr[j]);delta=q[a]-m.qpos0[a]
            rotation=source.xmat[bid].reshape(3,3)
            if int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_SLIDE):
                base_rotation=rotation;position=source.xpos[bid]-source.xaxis[j]*delta
            elif int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_HINGE):
                inverse=quat_from_axis_angle(m.jnt_axis[j],-delta)
                matrix=np.column_stack([quat_rotate(inverse,e) for e in np.eye(3)])
                base_rotation=rotation@matrix
                position=source.xpos[bid]-base_rotation@m.jnt_pos[j]+rotation@m.jnt_pos[j]
            else:raise ValueError('Turnstile contact boundary supports scalar joints only')
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,base_rotation.ravel())
            body=spec.worldbody.add_body(name=m.body(bid).name,pos=position,quat=quat,mass=m.body_mass[bid],
                ipos=m.body_ipos[bid],iquat=m.body_iquat[bid],inertia=m.body_inertia[bid],explicitinertial=True)
            body.add_joint(name=name,type=int(m.jnt_type[j]),axis=m.jnt_axis[j],pos=m.jnt_pos[j],ref=m.qpos0[a],
                springref=m.qpos_spring[a],stiffness=m.jnt_stiffness[j],damping=m.dof_damping[v],armature=m.dof_armature[v],
                frictionloss=m.dof_frictionloss[v],limited=int(m.jnt_limited[j]),range=m.jnt_range[j],
                solref_limit=m.jnt_solref[j],solimp_limit=m.jnt_solimp[j])
            bodies[bid]=body;joints[name]=(j,a,v)
        meshes=set()
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            bid=int(m.geom_bodyid[g]);body=bodies.get(bid,spec.worldbody)
            if bid in bodies:pos=m.geom_pos[g];quat=m.geom_quat[g]
            else:
                pos=source.geom_xpos[g];quat=np.empty(4);mujoco.mju_mat2Quat(quat,source.geom_xmat[g])
            extra={}
            if int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_MESH):
                mi=int(m.geom_dataid[g]);meshname='source_mesh_'+str(mi)
                if mi not in meshes:
                    va=int(m.mesh_vertadr[mi]);vn=int(m.mesh_vertnum[mi]);fa=int(m.mesh_faceadr[mi]);fn=int(m.mesh_facenum[mi])
                    spec.add_mesh(name=meshname,uservert=m.mesh_vert[va:va+vn].ravel(),userface=m.mesh_face[fa:fa+fn].ravel());meshes.add(mi)
                extra['meshname']=meshname
            body.add_geom(name=m.geom(g).name,type=int(m.geom_type[g]),pos=pos,quat=quat,size=m.geom_size[g],
                contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g]),condim=int(m.geom_condim[g]),friction=m.geom_friction[g],
                solref=m.geom_solref[g],solimp=m.geom_solimp[g],margin=m.geom_margin[g],gap=m.geom_gap[g],priority=int(m.geom_priority[g]),solmix=m.geom_solmix[g],**extra)
        previous=mujoco.get_mjcb_passive()
        try:
            mujoco.set_mjcb_passive(None);native=spec.compile();data=mujoco.MjData(native);local=[]
            for name in names:
                j,a,v=joints[name];nj=native.joint(name).id;na=int(native.jnt_qposadr[nj]);nv=int(native.jnt_dofadr[nj]);data.qpos[na]=q[a]
                local.append((name,j,a,na,nv))
            mujoco.mj_kinematics(native,data);geometry_error=0.
            for g in range(m.ngeom):
                if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
                other=native.geom(m.geom(g).name).id
                if int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_MESH):
                    mi=int(m.geom_dataid[g]);ni=int(native.geom_dataid[other]);count=int(m.mesh_vertnum[mi])
                    original=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+count]@source.geom_xmat[g].reshape(3,3).T+source.geom_xpos[g]
                    copied=native.mesh_vert[native.mesh_vertadr[ni]:native.mesh_vertadr[ni]+count]@data.geom_xmat[other].reshape(3,3).T+data.geom_xpos[other]
                    geometry_error=max(geometry_error,float(np.max(np.linalg.norm(original-copied,axis=1))))
                else:geometry_error=max(geometry_error,float(np.linalg.norm(source.geom_xpos[g]-data.geom_xpos[other])),float(np.max(np.abs(source.geom_xmat[g]-data.geom_xmat[other])))*float(m.geom_rbound[g]))
            if geometry_error>2e-6:raise ValueError(f'Native boundary copy changes geometry by {geometry_error} m')
            if not coil:
                # A frozen rotor can place a rest-position pawl inside a tooth.
                # Seat the passive pawl from its valid retracted endpoint,
                # rather than asking an initially intersecting contact solver
                # to choose an escape face. This is an inspection seed only;
                # no coordinate is overwritten during subsequent stepping.
                for name,j,a,na,nv in local:data.qpos[na]=m.jnt_range[j,1]
            def callback(model,d):
                if model is native and coil:
                    for rule in self.rules:
                        if rule.name in names and rule.powered:
                            j=model.joint(rule.name).id;a=int(model.jnt_qposadr[j]);v=int(model.jnt_dofadr[j])
                            gap=max(0.,rule.stroke-float(d.qpos[a]));d.qfrc_passive[v]+=rule.force/(1.+gap/rule.gap)**2
            mujoco.set_mjcb_passive(callback)
            for _ in range(round(1.5/native.opt.timestep)):mujoco.mj_step(native,data)
            mujoco.mj_forward(native,data)
        finally:mujoco.set_mjcb_passive(previous)
        depth=max((max(0.,-float(c.dist)) for c in data.contact),default=0.);speed=float(np.max(np.abs(data.qvel)));failures=[];result=q.copy();limit_error=0.
        for name,j,a,na,nv in local:
            result[a]=data.qpos[na]
            if m.jnt_limited[j]:limit_error=max(limit_error,float(m.jnt_range[j,0]-data.qpos[na]),float(data.qpos[na]-m.jnt_range[j,1]))
        if depth>.001:failures.append('Native contact boundary penetrates more than1mm')
        if speed>.002:failures.append('Passive mechanism not settled below0.002m/s or rad/s')
        if limit_error>.0001:failures.append('Passive joint exceeds its authored limit by more than0.1mm/mrad')
        if np.any(data.warning.number) or not np.isfinite(data.qpos).all():failures.append('Native warning or nonfinite inspection state')
        report={'ok':not failures,'qpos':result.tolist(),'failures':failures,'max_contact_penetration_m':depth,'max_speed':speed,
                'limit_error':limit_error,'snapshot_max_geometry_error_m':geometry_error,
                'contacts':[{'geoms':[native.geom(int(c.geom1)).name,native.geom(int(c.geom2)).name],
                             'distance_m':float(c.dist)} for c in data.contact],
                'scope':'Exact rigid native boundary and passive original mechanism; no force-driven traversal certificate'}
        self.cache[key]=report;return {**report,'qpos':list(report['qpos'])}


def first_turnstile_contact_angle(model,metadata,qpos,*,direction,bolt=True,pawl=False):
    """First actual index-bolt/pawl contact bounds a blocked rotor sweep."""
    import mujoco
    row=metadata['turnstile_locks'];d=mujoco.MjData(model);d.qpos[:]=qpos;j=model.joint(row['rotor_joint']).id;a=int(model.jnt_qposadr[j]);initial=float(d.qpos[a]);pairs=[]
    if bolt:pairs.extend((model.geom(row['bolt_geom']).id,model.geom(name).id) for name in row['index_geoms'])
    if pawl and row['pawl_joint']:pairs.extend((model.geom(row['pawl_tip_geom']).id,model.geom(name).id) for name in row['ratchet_teeth'])
    if not pairs:raise ValueError('A blocked scan needs an authored load-contact pair')
    def gap(angle):
        d.qpos[a]=angle;mujoco.mj_kinematics(model,d)
        return min(float(mujoco.mj_geomDistance(model,d,x,y,.1,None)) for x,y in pairs)
    start_gap=gap(initial)
    if start_gap<-.00001:return {'ok':False,'failure':'Intended turnstile load surfaces overlap initially','gap_m':start_gap}
    previous=initial
    for value in np.linspace(initial,initial+math.copysign(row['sector_angle_rad'],direction),257)[1:]:
        if gap(float(value))<=0.:
            lo,hi=previous,float(value)
            for _ in range(30):
                middle=(lo+hi)/2
                if gap(middle)>0:lo=middle
                else:hi=middle
            return {'ok':True,'angle_rad':lo,'initial_gap_m':start_gap,'scope':'Exact first named blocking contact; no pair ignored'}
        previous=float(value)
    return {'ok':False,'failure':'No physical arrest within one indexed sector'}
