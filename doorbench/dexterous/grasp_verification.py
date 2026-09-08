"""Strict, opt-in evidence for the H1/Shadow straight-lever grasp.

The historical centroid-only ``opposition`` score remains useful for debugging.
This chosen canonical diagnostic additionally requires distal volar pad loads,
consistent contact patches, and a complete physics-step record. Legitimate
power grasps may use volar middle/proximal surfaces under a different explicit
task contract. These gates do not certify universal anatomy, visual naturalness,
acquisition, or a whole door task on their own.
"""
import re

import mujoco
import numpy as np

from .contact_audit import DIGITS

GRASP_PROFILES = ('distal-pad-v1', 'volar-phalange-v1')


def grasp_profile(value):
    """An explicit surface contract; unknown profiles cannot silently fall back."""
    if value not in GRASP_PROFILES:
        raise ValueError('Unknown Shadow grasp profile: '+str(value))
    return value


def shadow_surface_qualified(digit, segment, local_point, outward, *, profile='distal-pad-v1'):
    """Frozen Shadow local -Y surfaces, independent of the contact object.

    The opt-in profile admits four-finger proximal/middle inner surfaces within
    their 45/25 mm segment lengths. A 2 mm exclusion around the proximal joint
    origin rejects knuckle/joint-origin loads. Thumb anatomy remains distal-only.
    Caller additionally checks lever-side geometry and radial normal alignment.
    """
    grasp_profile(profile)
    point=np.asarray(local_point,dtype=float);normal=np.asarray(outward,dtype=float)
    if point.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(np.r_[point,normal]).all():
        raise ValueError('Expected finite local contact point and outward normal')
    if digit not in DIGITS:
        return False
    upper={'distal':.040}
    if profile=='volar-phalange-v1' and digit!='th':
        upper.update(proximal=.045,middle=.025)
    return bool(segment in upper and point[1]<-.001 and
                .002<=point[2]<=upper[segment] and -normal[1]>.5)


def profile_pad_opposition(contacts, center, axis, *, profile='distal-pad-v1'):
    """Score selected surfaces and retain the unchanged distal counter-score."""
    grasp_profile(profile)
    result=pad_opposition(contacts,center,axis)
    if profile=='volar-phalange-v1':
        distal=[dict(c,pad_qualified=c['distal_pad_qualified']) for c in contacts]
        result['distal_pad_grasp']=pad_opposition(distal,center,axis)
        result['reason']=result['reason'].replace('distal volar pad','volar phalange')
    return result


def pad_opposition(contacts, center, axis, *, minimum_force=.2,
                   maximum_misplaced_fraction=.05):
    """Audit every loaded patch, not just the five averaged digit centroids.

Each input patch must contain ``digit``, ``position``, ``normal_force_N`` and
``pad_qualified``. The latter is evaluator-owned anatomical evidence, not a
policy observation or an annotation supplied by the actor.
    """
    center=np.asarray(center,dtype=float);axis=np.asarray(axis,dtype=float)
    if center.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(np.r_[center,axis]).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError('Expected a finite lever center and nonzero axis')
    if not np.isfinite(minimum_force) or minimum_force<=0 or not np.isfinite(maximum_misplaced_fraction) or not 0<=maximum_misplaced_fraction<.5:
        raise ValueError('Invalid pad load thresholds')
    axis=axis/np.linalg.norm(axis);patches=[];totals={digit:0. for digit in DIGITS}
    vectors={digit:np.zeros(3) for digit in DIGITS};qualified={digit:0. for digit in DIGITS}
    for contact in contacts:
        digit=contact['digit'];point=np.asarray(contact['position'],dtype=float)
        force=float(contact['normal_force_N']);pad=contact['pad_qualified']
        if digit not in DIGITS or point.shape!=(3,) or not np.isfinite(point).all() or not np.isfinite(force) or force<0 or not isinstance(pad,(bool,np.bool_)):
            raise ValueError('Invalid anatomical contact evidence')
        if force==0:continue
        radial=point-center;radial-=axis*np.dot(radial,axis);length=np.linalg.norm(radial)
        direction=radial/length if length>1e-8 else np.zeros(3)
        good=bool(pad and length>1e-8);totals[digit]+=force
        if good:vectors[digit]+=force*direction;qualified[digit]+=force
        patches.append((digit,force,direction,good))
    result=dict(valid_pad_grasp=False,digit_forces_N=totals,qualified_pad_forces_N=qualified)
    if any(qualified[d]<minimum_force or np.linalg.norm(vectors[d])<1e-8 for d in DIGITS):
        return dict(**result,reason='missing loaded distal volar pad')
    directions={d:vectors[d]/np.linalg.norm(vectors[d]) for d in DIGITS}
    mean=sum(directions[d] for d in DIGITS[:-1]);length=np.linalg.norm(mean)
    if length<1e-8:return dict(**result,reason='finger pads span opposing sides')
    mean/=length
    misplaced={d:0. for d in DIGITS}
    for digit,force,direction,good in patches:
        correct_side=np.dot(direction,mean)<-.5 if digit=='th' else np.dot(direction,mean)>.5
        if not good or not correct_side:misplaced[digit]+=force
    pairs=[np.dot(directions[a],directions[b]) for i,a in enumerate(DIGITS[:-1]) for b in DIGITS[i+1:-1]]
    thumb_pairs=[np.dot(directions['th'],directions[d]) for d in DIGITS[:-1]]
    result.update(misplaced_force_N=misplaced,
        minimum_pairwise_finger_alignment=float(min(pairs)),
        maximum_thumb_finger_dot=float(max(thumb_pairs)))
    result['valid_pad_grasp']=bool(min(pairs)>.5 and max(thumb_pairs)<-.5 and
        all(misplaced[d]<=maximum_misplaced_fraction*totals[d]+1e-12 for d in DIGITS))
    result['reason']='distal volar pad loads and every contact patch checked'
    return result


def shadow_lever_pad_grasp(model,data,lever_geom,*,side='rh',axial_margin=.001,profile='distal-pad-v1'):
    """Qualify the chosen canonical straight-lever distal-palmar contact targets.

In the unmodified Shadow asset, the distal palmar face is local -Y. The tested
contact band is 2–40 mm along each distal link. By default, a contact on the
dorsal face, knuckle, middle link, or lateral edge cannot substitute for a pad.
The explicitly selected volar-phalange-v1 profile admits calibrated inner
four-finger middle/proximal surfaces and retains the original distal score.
Endcap hooks are rejected: require a 1 mm cylindrical-side margin and >0.8
radial contact-normal alignment. This asset-specific anatomy must be replaced
for a different robot.
    """
    if side not in ('rh','lh'):raise ValueError('Unknown Shadow hand side')
    grasp_profile(profile)
    if not np.isfinite(axial_margin) or axial_margin<0:raise ValueError('Invalid axial margin')
    target=model.geom(lever_geom).id
    if int(model.geom_type[target]) not in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        raise ValueError('Pad opposition requires a straight cylindrical lever')
    center=data.geom_xpos[target];axis=data.geom_xmat[target].reshape(3,3)[:,2]
    contacts=[]
    for index,contact in enumerate(data.contact[:data.ncon]):
        if target not in contact.geom:continue
        other=int(contact.geom[1] if contact.geom[0]==target else contact.geom[0])
        body=int(model.geom_bodyid[other]);name=model.body(body).name
        match=re.fullmatch(r'robot/'+side+r'_(ff|mf|rf|lf|th)(.+)',name)
        if not match:continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,index,force)
        rotation=data.xmat[body].reshape(3,3)
        local_point=rotation.T@(contact.pos-data.xpos[body])
        # Contact frame normal points from geom0 to geom1.
        outward=rotation.T@((1 if contact.geom[0]==other else -1)*contact.frame[:3])
        relative=contact.pos-center;axial=float(np.dot(relative,axis))
        radial=relative-axial*axis;radial_length=np.linalg.norm(radial)
        alignment=float(np.dot(rotation@outward,-radial/radial_length)) if radial_length>1e-8 else 0.
        clearance=float(model.geom_size[target,1]-abs(axial))
        geometry_ok=clearance>=axial_margin and alignment>.8
        pad=bool(shadow_surface_qualified(match.group(1),match.group(2),local_point,outward,
                                        profile=profile) and geometry_ok)
        contacts.append(dict(digit=match.group(1),body=name,
            position=contact.pos.copy().tolist(),normal_force_N=float(max(0,force[0])),
            distance_m=float(contact.dist),body_position_m=local_point.tolist(),
            hand_outward_normal_body=outward.tolist(),pad_qualified=pad,
            axial_clearance_m=clearance,inward_radial_normal_alignment=alignment))
        if profile=='volar-phalange-v1':
            contacts[-1]['distal_pad_qualified']=bool(shadow_surface_qualified(
                match.group(1),match.group(2),local_point,outward) and geometry_ok)
    result=profile_pad_opposition(contacts,data.geom_xpos[target],data.geom_xmat[target].reshape(3,3)[:,2],profile=profile)
    return dict(**result,contacts=contacts,hand=side,lever_geom=lever_geom,
                grasp_profile=profile,
                anatomy_contract='shadow-distal-volar-minus-y-v1' if profile=='distal-pad-v1' else 'shadow-volar-phalange-minus-y-v1',
                minimum_axial_clearance_m=axial_margin,
                contract_scope='canonical straight-lever diagnostic, not universal grasp anatomy')


def native_grasp_sample(sim,lever_geom,*,handle_joint,side='rh',pre_step_external_wrench_max=None,profile='distal-pad-v1'):
    """Capture an initial or post-step state without changing states or controls.

    DoorEnv clears applied-force buffers after each step. Post-step callers must
    supply the captured pre-step body-wrench maximum; generalized forces use the
    plant's saved ``last_applied_qfrc``. Prefer ``audited_native_step``.
    """
    m,d=sim.m,sim.d
    if d.time>0 and pre_step_external_wrench_max is None:
        raise ValueError('Post-step audit requires the pre-step body-wrench record')
    diagnostics=sim.diagnostics()
    if pre_step_external_wrench_max is not None:
        if not np.isfinite(pre_step_external_wrench_max) or pre_step_external_wrench_max<0:
            raise ValueError('Invalid body-wrench evidence')
        diagnostics['external_wrench_max']=max(diagnostics['external_wrench_max'],pre_step_external_wrench_max)
    if d.time>0:
        applied=np.asarray(sim.plant.last_applied_qfrc)
        diagnostics['applied_generalized_force_max']=float(np.max(np.abs(applied)))
    joints=np.asarray(sim.joints);q=d.qpos[m.jnt_qposadr[joints]]
    limited=m.jnt_limited[joints].astype(bool)
    violations=np.maximum(m.jnt_range[joints,0]-q,q-m.jnt_range[joints,1])
    joint_violation=max(0.,float(violations[limited].max(initial=0.)))
    penetration=0.;hand_contacts=0
    for contact in d.contact[:d.ncon]:
        bodies=[m.body(m.geom_bodyid[g]).name for g in contact.geom]
        geoms=[m.geom(int(g)).name for g in contact.geom]
        if contact.dist<=0 and any(n.startswith('robot/'+side+'_') for n in bodies):hand_contacts+=1
        if any(n.startswith('robot/') for n in bodies) and not ('floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)):
            penetration=max(penetration,-float(contact.dist))
    caps=m.actuator_forcerange[sim.actuators];forces=d.actuator_force[sim.actuators]
    force_ok=bool(np.isfinite(forces).all() and np.all(forces>=caps[:,0]-1e-5) and np.all(forces<=caps[:,1]+1e-5))
    loopback_differences={}
    for side_name in ('rh','lh'):
        for digit in ('FF','MF','RF','LF'):
            names=[f'robot/{side_name}_{digit}J{i}' for i in (1,2)]
            ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name) for name in names]
            if min(ids)>=0:loopback_differences[f'{side_name}_{digit}']=float(d.qpos[m.jnt_qposadr[ids[0]]]-d.qpos[m.jnt_qposadr[ids[1]]])
    if len(loopback_differences)!=8:raise ValueError('Shadow grasp audit requires all eight documented loopback pairs')
    return dict(**diagnostics,max_shadow_loopback_violation_rad=max([0.,*loopback_differences.values()]),shadow_loopback_differences_rad=loopback_differences,max_joint_limit_violation_rad=joint_violation,
        max_nonfoot_penetration_m=penetration,native_motor_limits=force_ok,
        hand_contact_count=hand_contacts,
        handle_angle_rad=float(d.qpos[m.jnt_qposadr[m.joint(handle_joint).id]]),
        pad_grasp=shadow_lever_pad_grasp(m,d,lever_geom,side=side,profile=profile))


def audited_native_step(sim,lever_geom,*,handle_joint,side='rh',profile='distal-pad-v1'):
    """Step the unchanged plant once and retain otherwise-erased force evidence."""
    grasp_profile(profile)
    body_wrench=float(np.max(np.abs(sim.d.xfrc_applied)))
    sim.plant.step()
    return native_grasp_sample(sim,lever_geom,handle_joint=handle_joint,side=side,
                               pre_step_external_wrench_max=body_wrench,profile=profile)


def audit_grasp_steps(rows,*,physics_dt,expected_duration,required_hold=.5,
                      require_contact_free_start=True):
    """Fail closed unless evidence covers t=0 and every physics tick to the end.

The 20 mrad joint / 3 mm nonfoot contact allowances retain the development
plant's existing soft-constraint bounds. They are not anatomical joint limits
or a claim that these tolerances suffice for hardware.
    """
    if not np.isfinite([physics_dt,expected_duration,required_hold]).all() or physics_dt<=0 or expected_duration<required_hold or required_hold<=0:
        raise ValueError('Invalid duration or physics timestep')
    fields=('sim_time_s','door_q','handle_angle_rad','root_height_m','torso_tilt_deg',
        'max_joint_limit_violation_rad','max_nonfoot_penetration_m','external_wrench_max',
        'applied_generalized_force_max','hand_contact_count','max_shadow_loopback_violation_rad')
    if not rows:return dict(passed=False,checks=dict(complete_evidence=False),reason='empty physics record')
    try:
        values=np.asarray([[r[k] for k in fields] for r in rows],dtype=float)
        valid=np.isfinite(values).all()
        flags=all(isinstance(r[k],(bool,np.bool_)) for r in rows for k in ('finite','native_motor_limits'))
        flags=flags and all(isinstance(r['pad_grasp']['valid_pad_grasp'],(bool,np.bool_)) for r in rows)
        warnings=np.asarray([r['numerical_warnings'] for r in rows],dtype=float)
    except (KeyError,ValueError,TypeError):
        return dict(passed=False,checks=dict(complete_evidence=False),reason='missing or malformed physics evidence')
    if not valid or not flags or not np.isfinite(warnings).all():
        return dict(passed=False,checks=dict(complete_evidence=False),reason='nonfinite or malformed physics evidence')
    times=values[:,0];epsilon=max(1e-8,physics_dt*1e-5)
    complete=bool(abs(times[0])<=epsilon and abs(times[-1]-expected_duration)<=epsilon and
                  len(rows)==round(expected_duration/physics_dt)+1 and
                  np.all(np.abs(np.diff(times)-physics_dt)<=epsilon))
    tail=[r for r in rows if r['sim_time_s']>=expected_duration-required_hold-epsilon]
    checks=dict(complete_physics_step_evidence=complete,
        closed_leaf_start=abs(rows[0]['door_q'])<=.001,
        resting_operator_start=abs(rows[0]['handle_angle_rad'])<=.001,
        contact_free_start=not require_contact_free_start or rows[0]['hand_contact_count']==0,
        finite=all(r['finite'] and r['numerical_warnings']==0 for r in rows),
        upright=all(0<=r['torso_tilt_deg']<12 and r['root_height_m']>.7 for r in rows),
        physical_joint_limits=all(0<=r['max_joint_limit_violation_rad']<=.02 for r in rows),
        nonfoot_penetration=all(0<=r['max_nonfoot_penetration_m']<=.003 for r in rows),
        native_motor_limits=all(r['native_motor_limits'] for r in rows),
        no_external_assistance=all(r['external_wrench_max']==0 and r['applied_generalized_force_max']==0 for r in rows),
        documented_loopback_limits=all(0<=r.get('max_shadow_loopback_violation_rad',float('inf'))<=.02 for r in rows),
        sustained_pad_grasp=bool(complete and len(tail)>=round(required_hold/physics_dt)+1 and all(r['pad_grasp']['valid_pad_grasp'] for r in tail)))
    return dict(passed=all(checks.values()),checks=checks,physics_dt_s=physics_dt,
                expected_duration_s=expected_duration,required_hold_s=required_hold)


def scalar_transmission_matrix(model,actuators,joints):
    """Constant scalar joint/fixed-tendon mapping, including tendon motor gear.

    A different robot with ball/free transmissions or spatial tendons requires
    its native state-dependent moment matrix; reject those here explicitly.
    """
    joints=np.asarray(joints,dtype=int);actuators=np.asarray(actuators,dtype=int)
    if len(set(joints))!=len(joints) or len(set(actuators))!=len(actuators):
        raise ValueError('Duplicate joint or motor index')
    if any(int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)) for j in joints):
        raise ValueError('Only scalar joints are supported')
    index={int(j):i for i,j in enumerate(joints)};matrix=np.zeros((len(actuators),len(joints)))
    for row,aid in enumerate(actuators):
        kind=model.actuator_trntype[aid];target=int(model.actuator_trnid[aid,0]);gear=model.actuator_gear[aid,0]
        if kind==mujoco.mjtTrn.mjTRN_JOINT:
            if target not in index:raise ValueError('Motor joint missing from ordering')
            matrix[row,index[target]]=gear
        elif kind==mujoco.mjtTrn.mjTRN_TENDON:
            for k in range(model.tendon_adr[target],model.tendon_adr[target]+model.tendon_num[target]):
                if model.wrap_type[k]!=mujoco.mjtWrap.mjWRAP_JOINT:
                    raise ValueError('Spatial tendon needs a state-dependent transmission')
                joint=int(model.wrap_objid[k])
                if joint not in index:raise ValueError('Tendon joint missing from ordering')
                matrix[row,index[joint]]+=gear*model.wrap_prm[k]
        else:raise ValueError('Unsupported motor transmission')
    return matrix
