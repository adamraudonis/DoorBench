"""Native track-holder load/release checks on a private compiled-model copy."""
from __future__ import annotations
import copy
import numpy as np


class TrackContactPreview:
    """Native plunger response against a rigidly prescribed geometry boundary.

    A reduced private MuJoCo model retains the original plunger mass, inertia,
    joint, spring, damping and colliders. Every surrounding native collider is
    fixed at the requested inspection pose. Only the plunger may move; unrelated
    locks/couplings cannot corrupt this local boundary-value problem. The full
    geometric gate still checks all parts afterward. This is not proof of free
    traversal through an energized detent or a benchmark controller.
    """
    def __init__(self, model, metadata):
        from .closer_track_hold import compile_track_holds
        self.model=copy.copy(model)
        self.rules=compile_track_holds(self.model,metadata)
        if not self.rules:raise ValueError('Authored track-holder metadata is required')
        self.rows=metadata['closer_track_holds']
        self.cache={}

    def resolve(self, qpos, driven_joint=None):
        import mujoco
        from .closer_track_hold import track_coil_force
        m=self.model;q=np.asarray(qpos,dtype=float)
        if q.shape!=(m.nq,) or not np.isfinite(q).all():raise ValueError('Invalid inspection coordinates')
        key=(driven_joint,tuple(q))
        if key in self.cache:return {**self.cache[key],'qpos':list(self.cache[key]['qpos'])}
        dynamic=[r for r in self.rules if r.name!=driven_joint]
        if not dynamic:
            return {'ok':True,'failures':[],'qpos':q.tolist(),'cam_contact_depth_m':0.,'prescribed_boundary_residual':0.}
        source=mujoco.MjData(m);source.qpos[:]=q;mujoco.mj_kinematics(m,source)
        spec=mujoco.MjSpec();spec.compiler.degree=False;spec.option.timestep=m.opt.timestep
        spec.option.gravity=m.opt.gravity;spec.option.integrator=m.opt.integrator
        spec.option.iterations=m.opt.iterations;spec.option.tolerance=m.opt.tolerance
        bodies={};joint_ids={}
        for rule in dynamic:
            j=m.joint(rule.name).id;bid=int(m.jnt_bodyid[j]);joint_ids[rule.name]=j
            if int(m.jnt_type[j])!=int(mujoco.mjtJoint.mjJNT_SLIDE):raise ValueError('Holder plunger must be a native slide')
            R=source.xmat[bid].reshape(3,3)
            base=source.xpos[bid]-source.xaxis[j]*(q[rule.plunger_qpos]-m.qpos0[rule.plunger_qpos])
            body=spec.worldbody.add_body(name=m.body(bid).name,pos=base,quat=source.xquat[bid],
                mass=m.body_mass[bid],ipos=m.body_ipos[bid],iquat=m.body_iquat[bid],inertia=m.body_inertia[bid],explicitinertial=True)
            body.add_joint(name=rule.name,type=int(m.jnt_type[j]),axis=m.jnt_axis[j],pos=m.jnt_pos[j],
                ref=m.qpos0[rule.plunger_qpos],springref=m.qpos_spring[rule.plunger_qpos],stiffness=m.jnt_stiffness[j],
                damping=m.dof_damping[rule.plunger_dof],armature=m.dof_armature[rule.plunger_dof],
                frictionloss=m.dof_frictionloss[rule.plunger_dof],limited=int(m.jnt_limited[j]),range=m.jnt_range[j],
                solref_limit=m.jnt_solref[j],solimp_limit=m.jnt_solimp[j])
            bodies[bid]=body
        meshes=set()
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            bid=int(m.geom_bodyid[g]);body=bodies.get(bid,spec.worldbody)
            if bid in bodies:pos=m.geom_pos[g];quat=m.geom_quat[g]
            else:
                pos=source.geom_xpos[g];quat=np.empty(4);mujoco.mju_mat2Quat(quat,source.geom_xmat[g])
            kwargs={}
            if int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh=int(m.geom_dataid[g]);name='native_mesh_'+str(mesh)
                if mesh not in meshes:
                    va=int(m.mesh_vertadr[mesh]);vn=int(m.mesh_vertnum[mesh]);fa=int(m.mesh_faceadr[mesh]);fn=int(m.mesh_facenum[mesh])
                    spec.add_mesh(name=name,uservert=m.mesh_vert[va:va+vn].ravel(),userface=m.mesh_face[fa:fa+fn].ravel())
                    meshes.add(mesh)
                kwargs['meshname']=name
            body.add_geom(name=m.geom(g).name,type=int(m.geom_type[g]),pos=pos,quat=quat,size=m.geom_size[g],
                contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g]),condim=int(m.geom_condim[g]),
                friction=m.geom_friction[g],solref=m.geom_solref[g],solimp=m.geom_solimp[g],margin=m.geom_margin[g],
                gap=m.geom_gap[g],priority=int(m.geom_priority[g]),solmix=m.geom_solmix[g],**kwargs)
        # Compiling must never run a caller-owned passive callback against an
        # unrelated model with different dimensions. Restore it afterward.
        previous=mujoco.get_mjcb_passive()
        try:
            mujoco.set_mjcb_passive(None);native=spec.compile();data=mujoco.MjData(native)
            local=[]
            for rule in dynamic:
                j=native.joint(rule.name).id;a=int(native.jnt_qposadr[j]);dof=int(native.jnt_dofadr[j])
                data.qpos[a]=q[rule.plunger_qpos];local.append((rule,a,dof))
            mujoco.mj_kinematics(native,data)
            geometry_error=0.
            for g in range(m.ngeom):
                if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
                other=native.geom(m.geom(g).name).id
                if int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_MESH):
                    mi=int(m.geom_dataid[g]);ni=int(native.geom_dataid[other]);count=int(m.mesh_vertnum[mi])
                    original=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+count]@source.geom_xmat[g].reshape(3,3).T+source.geom_xpos[g]
                    copied=native.mesh_vert[native.mesh_vertadr[ni]:native.mesh_vertadr[ni]+count]@data.geom_xmat[other].reshape(3,3).T+data.geom_xpos[other]
                    geometry_error=max(geometry_error,float(np.max(np.linalg.norm(original-copied,axis=1))))
                else:
                    geometry_error=max(geometry_error,float(np.linalg.norm(source.geom_xpos[g]-data.geom_xpos[other])),
                        float(np.max(np.abs(source.geom_xmat[g]-data.geom_xmat[other])))*float(m.geom_rbound[g]))
            if geometry_error>2e-6:raise ValueError(f'Native inspection geometry copy differs by {geometry_error} m')
            def callback(model,d):
                if model is native:
                    for rule,a,dof in local:d.qfrc_passive[dof]+=track_coil_force(rule,d.qpos[a],q[rule.button_qpos])
            mujoco.set_mjcb_passive(callback)
            for _ in range(round(.6/native.opt.timestep)):mujoco.mj_step(native,data)
            mujoco.mj_forward(native,data)
        finally:mujoco.set_mjcb_passive(previous)
        depth=max((max(0.,-float(c.dist)) for c in data.contact),default=0.)
        speed=float(np.max(np.abs(data.qvel))) if native.nv else 0.
        failures=[]
        if depth>.001:failures.append('Native cam contacts penetrate by more than 1 mm')
        if speed>.002:failures.append('Passive cam has not settled below 2 mm/s')
        limit_error=0.
        for rule,a,_ in local:
            j=joint_ids[rule.name]
            if m.jnt_limited[j]:limit_error=max(limit_error,float(m.jnt_range[j,0]-data.qpos[a]),float(data.qpos[a]-m.jnt_range[j,1]))
        if limit_error>.0001:failures.append('Native cam exceeds its authored release limits by more than 0.1 mm')
        if np.any(data.warning.number) or not np.isfinite(data.qpos).all():failures.append('Native warning or nonfinite state')
        result=q.copy()
        for rule,a,_ in local:result[rule.plunger_qpos]=data.qpos[a]
        report={'ok':not failures,'qpos':result.tolist(),'failures':failures,
                'prescribed_boundary_residual':0.,'cam_contact_depth_m':depth,'cam_speed_m_s':speed,
                'snapshot_max_geometry_error_m':geometry_error,
                'cam_limit_error_m':limit_error,
                'scope':'Rigid inspection boundary with original native plunger mass/spring/contact/coil; not free traversal or dynamic success'}
        self.cache[key]=report
        return {**report,'qpos':list(report['qpos'])}


def run_closer_track_qa(model,metadata):
    import mujoco
    from .closer_pinion import compile_pinion_closers,apply_pinion_closers
    from .closer_track_hold import compile_track_holds,apply_track_holds
    from .geometry.closer_mounts import resolve_closer_configuration
    rows=metadata.get('closer_track_holds',[])
    if not rows:return {'ok':True,'applicable':False,'failures':[]}
    native=copy.copy(model);rules=compile_track_holds(native,metadata);pinions=compile_pinion_closers(native,metadata)
    result=[];failures=[];previous=mujoco.get_mjcb_passive();power=True
    def callback(m,d):
        if m is native:
            apply_pinion_closers(m,d,pinions);apply_track_holds(m,d,rules,powered=power)
    try:
        mujoco.set_mjcb_passive(callback)
        # Every actuator is tested even if a separate door lock prevents the
        #90-degree holding-load fixture. No authored lock is removed here.
        for mode in ('power_loss','test_button'):
            d=mujoco.MjData(native)
            for k in range(round(1.2/native.opt.timestep)):
                power=mode=='test_button' or d.time<.4
                d.qfrc_applied[:]=0.
                if mode=='test_button' and d.time>=.4:
                    for row in rows:d.qfrc_applied[native.jnt_dofadr[native.joint(row['button_joint']).id]]=8.
                mujoco.mj_step(native,d)
            for row,rule in zip(rows,rules):
                q=float(d.qpos[rule.plunger_qpos]);button=float(d.qpos[rule.button_qpos])
                item={'plunger_joint':rule.name,'probe':mode,'release_travel_m':q,'button_travel_m':button};result.append(item)
                if q<.008:failures.append({'release_failed':item})
                if mode=='test_button' and button<rule.button_threshold:failures.append({'button_obstructed':item})
            if np.any(d.warning.number):failures.append({'native_warning':d.warning.number.tolist(),'probe':mode})
        active_weld=any(int(native.eq_type[e])==int(mujoco.mjtEq.mjEQ_WELD) and native.eq_active0[e] for e in range(native.neq))
        limited=any(native.jnt_limited[native.joint(row['leaf_joint']).id] and native.jnt_range[native.joint(row['leaf_joint']).id,1]<row['nominal_hold_angle_rad']-1e-6 for row in rows)
        held=[]
        if not active_weld and not limited:
            for index,row in enumerate(rows):
                d=mujoco.MjData(native)
                for r in rows:d.qpos[native.jnt_qposadr[native.joint(r['leaf_joint']).id]]=r['nominal_hold_angle_rad']
                resolve_closer_configuration(native,d.qpos,metadata);adr=native.jnt_qposadr[native.joint(row['leaf_joint']).id]
                contacted=False;hold_angle=None
                for _ in range(round(6/native.opt.timestep)):
                    power={row['plunger_joint']:d.time<2.}
                    mujoco.mj_step(native,d)
                    if 1.<d.time<2.:
                        hold_angle=float(d.qpos[adr]);contacted|=any({native.geom(c.geom1).name,native.geom(c.geom2).name}=={row['roller_geom'],row['cam_geom']} for c in d.contact)
                item={'leaf_joint':row['leaf_joint'],'held_angle_rad':hold_angle,'target_rad':row['nominal_hold_angle_rad'],
                      'native_cam_contact':contacted,'angle_after_power_loss_rad':float(d.qpos[adr])};held.append(item)
                if not contacted or abs(hold_angle-row['nominal_hold_angle_rad'])>np.deg2rad(.5):failures.append({'holding_load_failed':item})
                if d.qpos[adr]>row['nominal_hold_angle_rad']-np.deg2rad(5):failures.append({'door_did_not_release':item})
                if np.any(d.warning.number):failures.append({'native_warning':d.warning.number.tolist(),'probe':'holding_load'})
        return {'ok':not failures,'applicable':True,'failures':failures,'actuator_probes':result,'holding_load_probes':held,
                'holding_load_skip_reason':'Separate active weld/range prevents an admissible open holding fixture; actuator checks still run' if active_weld or limited else None,
                'scope':'Native capture at selected point and power-loss/test-switch release; credential locks retained; overtravel recapture and OEM/fire certification are not established'}
    finally:mujoco.set_mjcb_passive(previous)
