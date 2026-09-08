"""Versioned passive Shadow loopback tendons, preserving original motor strength.

The manufacturer specifies J1 <= J2, not J1 == J2. Existing J0 sum tendons stay
as motor transmissions; these separate unactuated difference tendons supply only
the missing unilateral limit. Numerical compliance is declared, not measured.
"""
from pathlib import Path
import json

import mujoco
import numpy as np

PROFILE_PATH=Path(__file__).resolve().parents[2]/'configs/dexterous/shadow-loopback-v2.json'


def load_profile():
    return json.loads(PROFILE_PATH.read_text())


def add_loopbacks(spec,model,*,hands=None,digits=None):
    """Append passive fixed tendons to a spec; never edit existing joints/motors."""
    profile=load_profile();hands=profile['hands'] if hands is None else hands
    digits=profile['digits'] if digits is None else digits
    if profile['solref_limit'][0]<profile['minimum_time_constant_ratio_to_timestep']*model.opt.timestep-1e-12:
        raise ValueError('Loopback time constant must be at least two physics timesteps')
    if profile['coefficients'] != {'J1': 1., 'J2': -1.} or profile['upper_limit_rad'] != 0:
        raise ValueError('The versioned loopback contract requires J1 - J2 <= 0')
    existing={model.tendon(i).name for i in range(model.ntendon)};receipt=[]
    for hand in hands:
        if hand not in profile['hands']:raise ValueError('Unsupported Shadow hand')
        for digit in digits:
            if digit not in profile['digits']:raise ValueError('Unsupported Shadow finger')
            name=f'{hand}_{digit}_loopback'
            if name in existing:raise ValueError(f'Loopback already exists: {name}')
            first=f'{hand}_{digit}J1';second=f'{hand}_{digit}J2'
            ids=[model.joint(n).id for n in (first,second)]
            if any(int(model.jnt_type[j])!=int(mujoco.mjtJoint.mjJNT_HINGE) or not model.jnt_limited[j] for j in ids):
                raise ValueError('Loopback requires the original bounded scalar finger hinges')
            minimum=float(model.jnt_range[ids[0],0]-model.jnt_range[ids[1],1])
            limits=[minimum-profile['lower_limit_slack_beyond_joint_ranges_rad'],0.]
            tendon=spec.add_tendon(name=name,limited=True,range=limits,margin=profile['margin'],
                stiffness=profile['stiffness'],damping=profile['damping'],
                solref_limit=profile['solref_limit'],solimp_limit=profile['solimp_limit'])
            tendon.wrap_joint(first,1.);tendon.wrap_joint(second,-1.)
            existing.add(name)
            receipt.append(dict(name=name,joints={first:1.,second:-1.},range_rad=limits,
                solref_limit=profile['solref_limit'],solimp_limit=profile['solimp_limit'],
                stiffness=0.,damping=0.,margin=0.,actuated=False))
    return dict(**profile,passive_tendons=receipt)


def assert_only_passive_loopbacks_changed(before,after,receipt):
    """Reject accidental strength, geometry, inertia, support or transmission edits."""
    for attr in ('nq','nv','nu','nbody','ngeom','njnt','nmocap','neq','nsensor'):
        if getattr(before,attr)!=getattr(after,attr):raise ValueError(f'Unexpected structural change: {attr}')
    arrays=('body_mass','body_inertia','body_pos','body_quat','body_gravcomp',
        'geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp',
        'geom_contype','geom_conaffinity','jnt_type','jnt_range','jnt_axis','jnt_pos','jnt_solref','jnt_solimp',
        'dof_damping','dof_frictionloss','dof_armature','jnt_stiffness',
        'actuator_trntype','actuator_trnid','actuator_gear','actuator_gainprm','actuator_biasprm',
        'actuator_ctrlrange','actuator_forcerange','actuator_forcelimited','actuator_ctrllimited',
        'eq_type','eq_data','eq_obj1id','eq_obj2id','eq_active0')
    for attr in arrays:
        if not np.array_equal(getattr(before,attr),getattr(after,attr)):
            raise ValueError(f'Unexpected model change: {attr}')
    for attr in ('timestep','gravity','integrator','solver','iterations','tolerance'):
        if not np.array_equal(getattr(before.opt,attr),getattr(after.opt,attr)):
            raise ValueError(f'Unexpected solver change: {attr}')
    if after.ntendon!=before.ntendon+len(receipt['passive_tendons']):
        raise ValueError('Unexpected number of added passive tendons')
    # Fixed motor transmissions must retain their actual joint terms as well as
    # their limits. Tendon IDs in actuator_trnid alone do not prove this.
    for attr in ('wrap_type', 'wrap_objid', 'wrap_prm'):
        if not np.array_equal(getattr(before,attr),getattr(after,attr)[:before.nwrap]):
            raise ValueError(f'Existing motor tendon changed: {attr}')
    for tid in range(before.ntendon):
        other=after.tendon(before.tendon(tid).name).id
        for attr in ('tendon_range','tendon_limited','tendon_stiffness','tendon_damping',
                     'tendon_margin','tendon_solref_lim','tendon_solimp_lim'):
            if not np.array_equal(getattr(before,attr)[tid],getattr(after,attr)[other]):
                raise ValueError(f'Existing motor tendon changed: {attr}')
    for item in receipt['passive_tendons']:
        tid=after.tendon(item['name']).id
        if any(after.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_TENDON and after.actuator_trnid[a,0]==tid for a in range(after.nu)):
            raise ValueError('Passive loopback unexpectedly actuated')
        if not after.tendon_limited[tid] or not np.allclose(after.tendon_range[tid],item['range_rad'],rtol=0,atol=1e-12):
            raise ValueError('Incorrect passive tendon limits')
        for attr, field in (('tendon_stiffness','stiffness'),('tendon_damping','damping'),
                            ('tendon_margin','margin'),('tendon_solref_lim','solref_limit'),
                            ('tendon_solimp_lim','solimp_limit')):
            if not np.array_equal(getattr(after,attr)[tid], item[field]):
                raise ValueError(f'Incorrect passive tendon property: {attr}')
        address=after.tendon_adr[tid]; count=after.tendon_num[tid]
        terms={after.joint(int(after.wrap_objid[k])).name:float(after.wrap_prm[k])
               for k in range(address,address+count)
               if after.wrap_type[k]==mujoco.mjtWrap.mjWRAP_JOINT}
        if count!=2 or terms!=item['joints']:
            raise ValueError('Incorrect passive tendon joint coefficients')
