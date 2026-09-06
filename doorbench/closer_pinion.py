"""Physical surface-closer pinion springs and passive directional hydraulic valves.

The torsional spring is native. This module only supplies dissipative valve
forces at that same pinion. The authored door-level curve is a sizing target,
not a second leaf force or a manufacturer performance certification.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def linkage_curve(pinion, shoe, hinge, axis_sign, lengths, elbow, maximum, samples=721):
    """Planar rigid two-arm geometry; q is relative pinion rotation from closed."""
    P0,B,H,E0=map(lambda p:np.asarray(p,float)[:2],(pinion,shoe,hinge,elbow))
    L1,L2=lengths
    v1,v2=E0-P0,B-E0
    sign=np.sign(v1[0]*v2[1]-v1[1]*v2[0])
    if sign==0:raise ValueError('Closer initial linkage is singular')
    theta=np.linspace(0.,maximum,samples);headings=[]
    for angle in theta:
        a=axis_sign*angle;c,s=math.cos(a),math.sin(a);R=np.array([[c,-s],[s,c]])
        P=H+R@(P0-H);v=B-P;dist=np.linalg.norm(v)
        if not abs(L1-L2)+1e-6<dist<L1+L2-1e-6:raise ValueError('Closer linkage reaches a geometric singularity')
        along=v/dist;across=np.array([-along[1],along[0]])
        x=(L1*L1-L2*L2+dist*dist)/(2*dist);h=math.sqrt(max(0.,L1*L1-x*x))
        # cross(E-P,B-E) has the opposite sign to the elbow's offset.
        E=P+x*along-sign*h*across
        headings.append(math.atan2(E[1]-P[1],E[0]-P[0])-a)
    q=np.unwrap(headings);q-=q[0];opening_sign=float(np.sign(q[-1]));phi=q*opening_sign
    ratio=np.gradient(phi,theta,edge_order=2)
    if np.min(ratio)<.05:raise ValueError('Closer pinion is nonmonotonic or has insufficient mechanical advantage')
    return theta,q,ratio,opening_sign


def configure_pinion(model, leaf, pinion_body, spec, phys, *,pinion,shoe,elbow,lengths,curve_data=None):
    """Size a positive native pinion spring and record its achieved door curve."""
    original=dict(phys['closer']);maximum=math.radians(spec['kinematics'].get('max_open_deg') or 90.)
    theta,q,ratio,sign=curve_data if curve_data is not None else linkage_curve(pinion,shoe,leaf.joint.pos,leaf.joint.axis[2],lengths,elbow,maximum)
    phi=q*sign;target=original['spring_preload_Nm']+original['spring_stiffness_Nm_per_rad']*theta
    preload=float(target[0]/ratio[0])
    col=ratio*phi;rate=max(.01,float(np.dot(col,target-ratio*preload)/np.dot(col,col)))
    hf=phys['hinge'];friction=float(hf.get('coulomb_torque_Nm',0)+.5*hf.get('stick_torque_Nm',0))
    rate=max(rate,float(np.max(np.maximum(0.,1.2*friction/ratio[1:]-preload)/phi[1:])))
    torque=ratio*(preload+rate*phi)
    if np.min(torque-friction)<=0:raise ValueError('Pinion spring cannot overcome authored hinge friction')
    settings=spec['closer'];latch_angle=math.radians(12.)
    from . import hardware as H
    base=H.CLOSERS[settings['model']].closing_damping
    condition_scale=float(original['damping_closing']/base) if base else 1.
    sweep_s=float(settings.get('sweep_time_s',5.*condition_scale));latch_s=float(settings.get('latch_time_s',1.*condition_scale))
    if not 0<sweep_s<=30 or not 0<latch_s<=10:raise ValueError('Closer valve time settings outside supported positive bounds')
    def valve(low,high,seconds):
        grid=np.linspace(low,high,501);integrand=np.interp(grid,theta,ratio**2/(torque-friction))
        return seconds/float(np.trapezoid(integrand,grid))
    sweep=valve(min(latch_angle,maximum/2),min(math.pi/2,maximum),sweep_s)
    latch=valve(0.,min(latch_angle,maximum/2),latch_s)
    reference_ratio=float(np.interp(min(math.pi/4,maximum),theta,ratio))
    opening=float(original['damping_opening']/reference_ratio**2)
    bc=original.get('backcheck_angle_rad');bc_ratio=float(np.interp(bc or maximum,theta,ratio))
    backcheck=float((original.get('backcheck_damping') or 0.)/bc_ratio**2)
    delayed=H.CLOSERS[settings['model']].delayed_action
    delay_angle=math.radians(float(settings.get('delay_end_deg',70.))) if delayed else None
    delay_s=float(settings.get('delay_time_s',12.))*condition_scale if delayed else 0.
    if delayed and not (0<delay_angle<min(maximum,math.pi/2) and 0<delay_s<=50):
        raise ValueError('Delayed action needs a reachable angle zone and a positive valve setting of at most50 s valve setting')
    delay_damping=valve(delay_angle,min(maximum,math.pi/2),delay_s) if delayed else 0.
    if delayed:
        # A high-resistance hydraulic zone loads a light pinion through a closed
        # loop. Native2 ms stepping is not converged for that device; the
        # unchanged law converges at0.25/0.125/0.0625 ms.
        model.meta['native_timestep_s']=min(float(model.meta.get('native_timestep_s',.002)),.00025)
    j=pinion_body.joint;j.stiffness=rate;j.springref=-sign*preload/rate
    # Keep the maximum in MuJoCo's implicit damping term. The callback removes
    # its unused part, leaving exactly -b_target*qvel at every evaluation;
    # adding a large explicit damper to a tiny shaft creates a limit cycle.
    j.damping=max(sweep,latch,opening+backcheck,delay_damping)
    j.damping_closing=j.damping_opening=j.backcheck_angle=j.backcheck_damping=None
    leaf.joint.stiffness=0.;leaf.joint.springref=0.;leaf.joint.damping=float(hf.get('air_damping_Nms_per_rad',0.))
    leaf.joint.damping_closing=leaf.joint.damping_opening=leaf.joint.backcheck_angle=leaf.joint.backcheck_damping=None
    law={'leaf_joint':leaf.joint.name,'pinion_joint':j.name,'opening_sign':sign,
         'sweep_damping_Nms_per_rad':sweep,'latch_damping_Nms_per_rad':latch,
         'opening_damping_Nms_per_rad':opening,'latch_angle_rad':latch_angle,
         'backcheck_angle_rad':bc,'backcheck_damping_Nms_per_rad':backcheck,
         'delay_angle_rad':delay_angle,'delay_damping_Nms_per_rad':delay_damping,'delay_time_target_s':delay_s}
    model.meta.setdefault('closer_pinion_laws',[]).append(law)
    report={'mechanism':'native_pinion_spring_two_arm','scope':'Positive torsional spring on physical pinion; ideal rack/pinion and hydraulic valve law; internal rack and fluid are idealized; directional valves and angle-zone delay use passive pinion torque; required electromagnetic hold is separately reported',
            'leaf_joint':leaf.joint.name,'pinion_joint':j.name,'original_door_curve_target':original,
            'pinion_spring_stiffness_Nm_per_rad':rate,'pinion_spring_preload_Nm':preload,
            'pinion_springref_rad':j.springref,'opening_sign':sign,'valves':law,
            'quasistatic_sweep_target_s':sweep_s,'quasistatic_latch_target_s':latch_s,
            'target_max_relative_torque_error':float(np.max(np.abs(torque/target-1))),
            'achieved_closing_torque_min_max_Nm':[float(torque.min()),float(torque.max())],
            'table':{'door_angle_rad':theta.tolist(),'pinion_angle_rad':q.tolist(),
                     'pinion_ratio_abs':ratio.tolist(),'achieved_door_torque_Nm':torque.tolist()},
            'retained_native_tiers':['full','simple','minimal'],
            'unmodeled_features':[name for name,active in [('hold_open',original.get('hold_open_rad') is not None),('delayed_action',delayed and delay_damping<=0)] if active]}
    model.meta.setdefault('closer_pinion_calibration',[]).append(report)
    phys['closer'].update({'mechanism':'native_pinion_spring_two_arm','original_door_curve_target':original,
                          'pinion_calibration':report,'formula':'tau_leaf = d(q_pinion)/d(theta_leaf) * [-k_pinion * (q_pinion - springref)]',
                          'closing_time_est_s':None})
    return report


@dataclass(frozen=True)
class PinionValve:
    pinion_dof:int
    leaf_qpos:int
    opening_sign:float
    base:float
    opening:float
    sweep:float
    latch:float
    latch_angle:float
    backcheck_angle:float|None
    backcheck:float
    delay_angle:float|None=None
    delay:float=0.


def compile_pinion_closers(model,metadata):
    rules=[]
    for row in metadata.get('closer_pinion_laws',[]):
        pinion=model.joint(row['pinion_joint']).id;leaf=model.joint(row['leaf_joint']).id
        dof=int(model.jnt_dofadr[pinion]);values=[row[k] for k in ('sweep_damping_Nms_per_rad','latch_damping_Nms_per_rad','opening_damping_Nms_per_rad','backcheck_damping_Nms_per_rad')]
        if not np.isfinite(values).all() or min(values)<0 or row['opening_sign'] not in (-1,1):raise ValueError('Invalid physical pinion valve parameters')
        rules.append(PinionValve(dof,int(model.jnt_qposadr[leaf]),float(row['opening_sign']),float(model.dof_damping[dof]),
            float(values[2]),float(values[0]),float(values[1]),float(row['latch_angle_rad']),row['backcheck_angle_rad'],float(values[3]),row.get('delay_angle_rad'),float(row.get('delay_damping_Nms_per_rad',0.))))
    return tuple(rules)


def apply_pinion_closers(model,data,rules):
    for r in rules:
        velocity=float(data.qvel[r.pinion_dof]);angle=float(data.qpos[r.leaf_qpos])
        if r.opening_sign*velocity>=0:
            damping=r.opening
            if r.backcheck_angle is not None and angle>r.backcheck_angle:damping+=r.backcheck
        else:
            damping=r.latch if angle<r.latch_angle else r.sweep
            if r.delay_angle is not None and angle>r.delay_angle:damping=r.delay
        data.qfrc_passive[r.pinion_dof]-=(damping-r.base)*velocity


def projected_static_resistance(model,data,metadata):
    """Native virtual-work projection for an inspection/effort-sizing query.

    Two private configurations differentiate the authored holonomic loop. The
    model, current state and benchmark trajectory are not modified.
    """
    import mujoco
    from .geometry.closer_mounts import resolve_closer_configuration
    primary=model.joint(metadata['primary_joint']).id;adr=int(model.jnt_qposadr[primary]);h=1e-6
    samples=[]
    for delta in (-h,h):
        q=data.qpos.copy();q[adr]+=delta
        for _ in range(2):
            for e in range(model.neq):
                if model.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e]>=0:
                    j1,j2=int(model.eq_obj1id[e]),int(model.eq_obj2id[e]);a1,a2=int(model.jnt_qposadr[j1]),int(model.jnt_qposadr[j2]);x=q[a2]-model.qpos0[a2]
                    q[a1]=model.qpos0[a1]+sum(model.eq_data[e,k]*x**k for k in range(5))
        resolve_closer_configuration(model,q,metadata);samples.append(q)
    tangent=np.zeros(model.nv);mujoco.mj_differentiatePos(model,tangent,2*h,*samples)
    resistance=float(np.dot(tangent,data.qfrc_bias-data.qfrc_passive))
    friction=float(np.dot(np.abs(tangent),model.dof_frictionloss))
    return {'static_resistance':resistance,'frictionloss':friction,
            'method':'Native virtual work through solved pinion/arm/rise loop, evaluated only for sizing',
            'tangent':{model.joint(int(model.dof_jntid[i])).name:float(x) for i,x in enumerate(tangent) if abs(x)>1e-8}}
