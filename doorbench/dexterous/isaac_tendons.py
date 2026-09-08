"""Explicit passive Shadow loopback contract for a versioned PhysX plant.

USD angular gearing is expressed per degree; this contract uses radians. The
passive limit belongs to the robot mechanism, not a hand-object grasp helper.
Unknown passive tendons fail closed instead of disappearing during import.
"""
import re
import numpy as np

CONTRACT='shadow-loopback-v2'
SOURCE='https://shadow-robot-company-dexterous-hand.readthedocs-hosted.com/en/latest/user_guide/md_finger.html'


def native_passive_tendons(model, *, physx_limit_stiffness=10000.):
    import mujoco
    if not np.isfinite(physx_limit_stiffness) or physx_limit_stiffness<=0:
        raise ValueError('Positive finite numerical limit stiffness required')
    result=[]
    for tendon in range(model.ntendon):
        if not (model.tendon_limited[tendon] or model.tendon_stiffness[tendon] or model.tendon_damping[tendon]):continue
        name=model.tendon(tendon).name
        match=re.fullmatch(r'([lr]h)_(FF|MF|RF|LF)_loopback',name)
        if not match or not model.tendon_limited[tendon] or model.tendon_stiffness[tendon] or model.tendon_damping[tendon]:
            raise ValueError('Unsupported passive tendon mechanics: '+name)
        side,digit=match.groups();terms={}
        for k in range(model.tendon_adr[tendon],model.tendon_adr[tendon]+model.tendon_num[tendon]):
            if int(model.wrap_type[k])!=int(mujoco.mjtWrap.mjWRAP_JOINT):raise ValueError('Spatial passive tendon needs another adapter')
            joint=int(model.wrap_objid[k])
            if int(model.jnt_type[joint])!=int(mujoco.mjtJoint.mjJNT_HINGE):raise ValueError('Loopback axes must be hinges')
            joint_name=model.joint(joint).name;terms[joint_name]=terms.get(joint_name,0.)+float(model.wrap_prm[k])
        proximal=side+'_'+digit+'J2';distal=side+'_'+digit+'J1'
        if terms!={distal:1.,proximal:-1.}:raise ValueError('Loopback must represent J1 minus J2')
        parent_body=int(model.jnt_bodyid[model.joint(proximal).id]);child_body=int(model.jnt_bodyid[model.joint(distal).id])
        if model.body_parentid[child_body]!=parent_body:raise ValueError('Fixed tendon joints must be adjacent in the articulation')
        limits=model.tendon_range[tendon].copy()
        lower_possible=model.jnt_range[model.joint(distal).id,0]-model.jnt_range[model.joint(proximal).id,1]
        if not np.isfinite(limits).all() or limits[1]!=0. or limits[0]>=lower_possible:
            raise ValueError('Expected upper zero and inactive lower loopback bound')
        result.append(dict(name=name,profile=CONTRACT,root_joint=proximal,terms=terms,
            range_rad=limits.tolist(),spring_stiffness=0.,damping=0.,
            physx_limit_stiffness=float(physx_limit_stiffness),rest_length_rad=0.,offset_rad=0.,
            native_solref=model.tendon_solref_lim[tendon].tolist(),native_solimp=model.tendon_solimp_lim[tendon].tolist(),
            scope='Numerical approximation to the documented passive inequality; not measured tendon material stiffness',source=SOURCE))
    return result


def validate_contract(tendons):
    if not isinstance(tendons,list):raise ValueError('Passive tendon contract must be a list')
    seen=set()
    for t in tendons:
        name=t['name'];match=re.fullmatch(r'([lr]h)_(FF|MF|RF|LF)_loopback',name)
        if name in seen or not match or t['profile']!=CONTRACT:raise ValueError('Unknown or duplicate loopback contract')
        seen.add(name);side,digit=match.groups();middle=f'{side}_{digit}J2';distal=f'{side}_{digit}J1'
        if t['root_joint']!=middle or t['terms']!={distal:1.,middle:-1.}:raise ValueError('Incorrect loopback transmission')
        values=np.asarray(t['range_rad'],float)
        if values.shape!=(2,) or not np.isfinite(values).all() or values[0]>=-np.pi/2 or values[1]!=0:raise ValueError('Incorrect unilateral range')
        if not np.isfinite(t['physx_limit_stiffness']) or t['physx_limit_stiffness']<=0:raise ValueError('Invalid numerical stiffness')
        if any(t[k]!=0 for k in ('spring_stiffness','damping','rest_length_rad','offset_rad')):raise ValueError('Slack loopback must not apply a bilateral spring or damping')
    return tendons


def author_passive_tendons(stage,root_path,tendons):
    from pxr import Usd,UsdPhysics,PhysxSchema
    validate_contract(tendons)
    if not tendons:return dict(profile='upstream-v1',count=0,tendons=[])
    joints={}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            name=prim.GetName()
            if name in joints:raise ValueError('Ambiguous robot joint name '+name)
            joints[name]=prim
    records=[]
    for t in tendons:
        instance=t['name']
        if not set(t['terms']).issubset(joints):raise ValueError('Missing loopback joints')
        for name,coefficient in t['terms'].items():
            prim=joints[name]
            if name==t['root_joint']:
                PhysxSchema.PhysxTendonAxisRootAPI.Apply(prim,instance)
                axis=PhysxSchema.PhysxTendonAxisAPI(prim,instance)
            else:axis=PhysxSchema.PhysxTendonAxisAPI.Apply(prim,instance)
            axis.CreateGearingAttr([coefficient*np.pi/180.])
            axis.CreateForceCoefficientAttr([coefficient])
        root=PhysxSchema.PhysxTendonAxisRootAPI(joints[t['root_joint']],instance)
        root.CreateStiffnessAttr(0.);root.CreateDampingAttr(0.)
        root.CreateLimitStiffnessAttr(t['physx_limit_stiffness'])
        root.CreateLowerLimitAttr(t['range_rad'][0]);root.CreateUpperLimitAttr(0.)
        root.CreateRestLengthAttr(0.);root.CreateOffsetAttr(0.)
        records.append(dict(name=instance,root_path=str(joints[t['root_joint']].GetPath()),
            joint_paths={name:str(joints[name].GetPath()) for name in t['terms']},
            range_rad=t['range_rad'],limit_stiffness=t['physx_limit_stiffness'],
            gearing_units='radians per USD degree',force_coefficients=t['terms']))
    return dict(profile=CONTRACT,count=len(records),tendons=records,qualification='Authored contract only; live unilateral dynamics fixture required')
