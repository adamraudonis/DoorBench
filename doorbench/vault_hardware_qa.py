"""Independent bounded native service checks for the authored vault boltwork."""
from __future__ import annotations
import copy
import math
import numpy as np


def run_vault_native_qa(model,metadata,*,cycles=2,negative_controls=True):
    """Actual surface forces and native contacts; no qpos writes in a cycle."""
    import mujoco
    assembly=metadata.get('vault_boltwork')
    if not assembly:return {'ok':True,'applicable':False,'failures':[]}
    m=copy.copy(model);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    leaf=m.joint(metadata['primary_joint']).id;leafq=int(m.jnt_qposadr[leaf]);leafv=int(m.jnt_dofadr[leaf])
    leavesite=m.site('vault_leaf_grip_p' if metadata.get('approach_face',-1)>0 else 'vault_leaf_grip_n').id
    groups=[];failures=[];probes=[];phase_warnings=np.zeros(int(mujoco.mjtWarning.mjNWARNING),int)
    for row in assembly['groups']:
        joint=m.joint(row['operator_joint']).id;body=m.body(row['operator_body']).id
        sites=[i for i in range(m.nsite) if m.site_bodyid[i]==body and 'grip' in m.site(i).name]
        face=int(metadata.get('approach_face',-1))
        sites=[i for i in sites if float(m.site_pos[i,1])*face>0]
        if not sites:raise ValueError('Vault operator has no surface grip on the approach face')
        sid=sites[0];carrier=m.joint(row['carrier_joint']).id
        if not (m.jnt_range[carrier,0]<-.002 and m.jnt_range[carrier,1]>row['stroke_m']+.002):
            raise ValueError('Vault carrier safety range must exceed its physical stops')
        groups.append((row,joint,int(m.jnt_qposadr[joint]),int(m.jnt_dofadr[joint]),sid))
    def force_at(joint,site,force):
        point=d.site_xpos[site];axis=d.xaxis[joint];tangent=np.cross(axis,point-d.xanchor[joint]);length=float(np.linalg.norm(tangent))
        if length<.05:raise ValueError('Vault contact has no usable moment arm')
        mujoco.mj_applyFT(m,d,force*tangent/length,np.zeros(3),point,int(m.site_bodyid[site]),d.qfrc_applied)
    def phase(name,duration,*,operators=(),leaf_goal=None,leaf_force=0.,stop_when=None):
        nonlocal phase_warnings
        start=float(d.time);depth=eqmax=peakhand=peakleaf=peaklimit=0.;worst=None;contact_forces={}
        while float(d.time)-start<duration-m.opt.timestep/2:
            d.qfrc_applied[:]=0.;d.ctrl[:]=0.
            for index,target in operators:
                row,j,qa,va,site=groups[index]
                effort=float(np.clip(140.*(target-d.qpos[qa])-16.*d.qvel[va],-66.7,66.7));peakhand=max(peakhand,abs(effort));force_at(j,site,effort)
            effort=leaf_force
            if leaf_goal is not None:
                error=leaf_goal-float(d.qpos[leafq])
                radius=np.linalg.norm(np.cross(d.xaxis[leaf],d.site_xpos[leavesite]-d.xanchor[leaf]))
                friction=float(m.dof_frictionloss[leafv])/radius
                compensation=math.copysign(friction,error) if abs(error)>.00005 else 0.
                damping=2.2*math.sqrt(250.*float(m.dof_M0[leafv])/radius)
                effort=float(np.clip(250*error-damping*d.qvel[leafv]+compensation,-120.,120.))
            if effort:force_at(leaf,leavesite,effort);peakleaf=max(peakleaf,abs(effort))
            mujoco.mj_step(m,d);phase_warnings=np.maximum(phase_warnings,d.warning.number)
            limits=(d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)) & (d.efc_id==leaf)
            peaklimit=max(peaklimit,max(abs(d.efc_force[limits]),default=0.))
            equal=d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            eqmax=max(eqmax,max(abs(d.efc_pos[equal]),default=0.))
            for ci,c in enumerate(d.contact):
                pair=tuple(sorted((m.geom(c.geom1).name,m.geom(c.geom2).name)))
                if -c.dist>depth:depth=-float(c.dist);worst=pair
                if any(name.startswith(('vault_','strike_','stop_')) for name in pair):
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                    load=float(np.linalg.norm(wrench[:3]));contact_forces[pair]=max(load,contact_forces.get(pair,0.))
            if not np.isfinite(d.qpos).all() or np.any(phase_warnings):break
            if stop_when and stop_when():break
        result={'phase':name,'duration_s':float(d.time)-start,'leaf_rad':float(d.qpos[leafq]),
            'operators_rad':[float(d.qpos[q]) for _,_,q,_,_ in groups],
            'carriers_m':[float(d.qpos[m.jnt_qposadr[m.joint(r['carrier_joint']).id]]) for r,_,_,_,_ in groups],
            'max_hand_force_N':peakhand,'max_leaf_force_N':peakleaf,'max_primary_limit_reaction_Nm':float(peaklimit),'max_penetration_m':depth,'worst_pair':worst,
            'max_equality_residual':float(eqmax),'native_warnings':phase_warnings.tolist(),
            'contacts':[{'pair':list(pair),'force_N':load} for pair,load in sorted(contact_forces.items()) if load>.01]}
        probes.append(result)
        if depth>.001 or eqmax>.001 or peaklimit>.01 or np.any(phase_warnings) or not np.isfinite(d.qpos).all():
            failures.append({'check':'native_geometry_or_warning','phase':name,'details':result})
        return result
    def require(value,check,row):
        if not value:failures.append({'check':check,'phase':row['phase'],'details':row})
    def contact(row,first,second=None):
        return max((x['force_N'] for x in row['contacts'] if first in x['pair'] and (second is None or second in x['pair'])),default=0.)
    def bolt_contact(row):
        return all(any(contact(row,b)>.1 for b in r['bolt_geoms']) for r,_,_,_,_ in groups)
    def seated_on_rebate():
        if not (-.001<float(d.qpos[leafq])<0. and abs(float(d.qvel[leafv]))<.015):return False
        for ci,c in enumerate(d.contact):
            if not any(m.geom(g).name in metadata['vault_closing_stops'] for g in (c.geom1,c.geom2)):continue
            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
            if np.linalg.norm(wrench[:3])>.1:return True
        return False
    phase('settle',.5)
    held=phase('initial_locked_load',1.,leaf_force=120.)
    require(abs(held['leaf_rad'])<.01 and bolt_contact(held),'bolts_did_not_carry_initial_load',held)
    for cycle in range(cycles):
        # Remove the leaf opening load before manipulating the boltwork.
        phase(f'{cycle}_unload',.6)
        for i,(r,_,_,_,_) in enumerate(groups):
            released=phase(f'{cycle}_release_{i}',12. if r['ratio']<1 else 4.,operators=[(i,r['operator_nominal_range'][1]+.6)])
            require(abs(released['carriers_m'][i]-r['stroke_m'])<.0005,'carrier_not_fully_released',released)
            require(contact(released,r['carrier_stop_tab'],r['carrier_stop_geoms'][1])>.1,'released_stop_has_no_reaction',released)
        stable=phase(f'{cycle}_released_hands_off',1.)
        require(all(q>=r['stroke_m']-.0005 for q,(r,_,_,_,_) in zip(stable['carriers_m'],groups)),'released_bolts_drifted',stable)
        goal=float(metadata['vault_primary_nominal_range'][1])-.02
        opened=phase(f'{cycle}_leaf_open',60.,leaf_goal=goal,stop_when=lambda:abs(float(d.qpos[leafq])-goal)<.003 and abs(float(d.qvel[leafv]))<.015)
        require(opened['leaf_rad']>=goal-.003,'released_leaf_failed_to_open',opened)
        closed=phase(f'{cycle}_leaf_close',60.,leaf_goal=-.005,stop_when=seated_on_rebate)
        require(-.001<closed['leaf_rad']<.0001,'leaf_not_seated_for_rethrow',closed)
        require(any(contact(closed,g)>.1 for g in metadata['vault_closing_stops']),'closing_rebate_has_no_reaction',closed)
        for i,(r,_,_,_,_) in enumerate(groups):
            thrown=phase(f'{cycle}_rethrow_{i}',12. if r['ratio']<1 else 4.,operators=[(i,-.6)])
            require(abs(thrown['carriers_m'][i])<.0005 and abs(thrown['operators_rad'][i])<.01,'bolts_not_rethrown',thrown)
            require(contact(thrown,*r['crank_stop_pair'])>.1,'crank_return_stop_has_no_reaction',thrown)
        phase(f'{cycle}_thrown_hands_off',1.)
        held=phase(f'{cycle}_relocked_load',1.,leaf_force=120.)
        require(abs(held['leaf_rad'])<.01 and bolt_contact(held),'bolts_did_not_carry_relocked_load',held)
    negatives=[]
    if negative_controls:
        # A complete rod removal must sever the transmission. Keeping the
        # physical pin contacts while deleting only an equality is not removal.
        for row,_,_,_,_ in groups:
            m.eq_active0[m.equality(row['connect']).id]=0
            rod=m.body(row['rod_body']).id
            for g in range(m.ngeom):
                if m.geom_bodyid[g]==rod:m.geom_contype[g]=m.geom_conaffinity[g]=0
        d=mujoco.MjData(m);mujoco.mj_forward(m,d);phase_warnings[:]=0
        failed=phase('removed_rods',6.,operators=[(i,r['operator_nominal_range'][1]+.6) for i,(r,_,_,_,_) in enumerate(groups)])
        require(all(abs(q)<.001 for q in failed['carriers_m']),'removed_rod_still_drives_bolts',failed)
        negatives.append('complete_rod_contact_and_connection_removal')
        # Recompile the original state, then remove only the actual load bolts.
        m=copy.copy(model)
        for row,_,_,_,_ in groups:
            for name in row['bolt_geoms']:
                g=m.geom(name).id;m.geom_contype[g]=m.geom_conaffinity[g]=0
        d=mujoco.MjData(m);mujoco.mj_forward(m,d);phase_warnings[:]=0
        free=phase('removed_bolt_noses',12.,leaf_force=120.,stop_when=lambda:float(d.qpos[leafq])>=.15)
        require(free['leaf_rad']>=.15,'removed_bolts_leave_hidden_arrest',free)
        negatives.append('actual_bolt_load_geometry_removal')
    return {'ok':not failures,'applicable':True,'cycles':cycles,'failures':failures,'probes':probes,'negative_controls':negatives,
            'scope':'Finite surface forces, actual rod/pin and stop/strike reactions, hands-off retention and continuous native service cycles. Ideal bearings and explicitly declared keyed spur relation; no humanoid, strength, rated blast/security or full task certification.'}


def run_vault_mount_qa(model, metadata):
    """Native signed-distance evidence for connected support and open journals.

    Queries include same-body/filter-excluded primitives. They prove geometric
    mounting continuity and bore clearance, not bearing stress or compliance.
    """
    import mujoco
    if not metadata.get('vault_boltwork'):
        return {'ok':True,'applicable':False,'failures':[]}
    d=mujoco.MjData(model);mujoco.mj_forward(model,d);rows=[];failures=[]
    def distance(a,b):
        return float(mujoco.mj_geomDistance(model,d,model.geom(a).id,model.geom(b).id,2.,np.zeros(6)))
    def check(kind,names,gap,low=None,high=None):
        row={'check':kind,'parts':names,'signed_distance_m':gap};rows.append(row)
        if (low is not None and gap<low) or (high is not None and gap>high):failures.append(row)
    frame=[model.geom(i).name for i in range(model.ngeom) if model.geom(i).name.startswith(('jamb_','post_'))]
    stock=[model.geom(i).name for i in range(model.ngeom) if model.geom(i).name.startswith('leaf_slab')]
    if not frame or not stock:raise ValueError('Vault mounting proof requires authored frame and prepared leaf stock')
    strike_frame=[name for name in frame if name.startswith('jamb_strike')]
    if not strike_frame:raise ValueError('Vault closing rebate requires its actual strike jamb')
    for stop in metadata['vault_closing_stops']:
        check('closing_rebate_to_jamb',[stop],min(distance(stop,g) for g in strike_frame),high=1e-5)
    for row in metadata['vault_crane_journals']:
        journal=row['journal'];sleeves=row['sleeves'];arm=row['arm'];mount=row['leaf_mount']
        for block in row['frame_blocks']:
            check('journal_in_frame_block',[journal,block],distance(journal,block),high=1e-5)
            check('frame_block_to_jamb',[block],min(distance(block,g) for g in frame),high=1e-5)
        check('journal_radial_bore',[journal],min(distance(journal,g) for g in sleeves),low=.0001,high=.002)
        check('sleeve_to_crane_arm',[arm],min(distance(arm,g) for g in sleeves),high=1e-5)
        check('crane_arm_to_leaf_mount',[arm,mount],distance(arm,mount),high=1e-5)
        check('mount_to_prepared_stock',[mount],min(distance(mount,g) for g in stock),high=1e-5)
    for row in metadata['vault_boltwork']['groups']:
        for bolt in row['bolt_geoms']:
            for station in range(2):
                guides=[g for g in row['guide_geoms'] if g.startswith(bolt+f'_guide_{station}_')]
                if not guides:raise ValueError('Vault bolt needs both authored bored guides')
                check('bolt_guide_radial_bore',[bolt,guides[0]],min(distance(bolt,g) for g in guides),low=.0001,high=.002)
    return {'ok':not failures,'applicable':True,'failures':failures,'queries':rows,
            'scope':'Native signed 3D distances through prepared stock and bearing rings; continuity and clearance only. Ideal native bearing joints transmit loads; no material stress, fatigue or OEM tolerance certification.'}


def first_vault_contact_angle(model, metadata):
    """Inspect only closed boltwork up to its first actual frame contact."""
    import mujoco
    from .geometry.vault_hardware import resolve_vault_configuration
    m=copy.copy(model);d=mujoco.MjData(m);j=m.joint(metadata['primary_joint']).id;qa=int(m.jnt_qposadr[j])
    bolts=[m.geom(g).id for r in metadata['vault_boltwork']['groups'] for g in r['bolt_geoms']]
    strikes=[i for i in range(m.ngeom) if m.geom(i).name.startswith('strike_')]
    if not bolts or not strikes:raise ValueError('Vault arrest inspection requires actual bolts and strikes')
    if any(not (m.geom_contype[g] and m.geom_conaffinity[g]) for g in bolts):
        raise ValueError('Vault arrest requires every bolt to remain a native collider')
    def gap(angle):
        d.qpos[:]=m.qpos0;d.qpos[qa]=angle;resolve_vault_configuration(m,d.qpos,metadata);mujoco.mj_forward(m,d)
        return min((float(mujoco.mj_geomDistance(m,d,a,b,.2,None)),a,b) for a in bolts for b in strikes)
    initial=gap(0.)
    if initial[0]<=0:return {'ok':False,'reason':'Initial bolt/strike overlap','gap_m':initial[0]}
    lower=0.
    for upper in np.linspace(.001,.1,100):
        result=gap(float(upper))
        if result[0]<=0:break
        lower=float(upper)
    else:return {'ok':False,'reason':'No physical bolt arrest within0.1rad'}
    for _ in range(35):
        mid=(lower+upper)/2
        if gap(mid)[0]<=0:upper=mid
        else:lower=mid
    distance,a,b=gap(float(upper))
    return {'ok':True,'angle_rad':float(upper),'pair':[m.geom(a).name,m.geom(b).name],'gap_m':distance,
            'scope':'First signed-distance crossing of the actual closed bolt and frame; no prescribed sweep through the strike.'}
