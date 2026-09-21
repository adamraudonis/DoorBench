"""Actual-source context and unstepped numerical release planning for Isaac.

No native manifest/trajectory is synthesized. Old plans may supply only the
bounded preference fields listed below, never coordinates, material points,
qualification, or a sampled door trajectory. The generated route is an
unadmitted geometry candidate; independent dense and physical gates still apply.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .landed_left_planner import LandedLeftScene
from .operation_teacher import smooth_phase
from .qualified_isaac_grasp import digest
from .release_material_targets import loaded_segment_targets


DEFAULT_PREFERENCES = dict(radial_clearance_m=.025, separation_reserve_m=.004,
    finger_profile='radial', retreat_profile='slide-lift', whole_body=True,
    coordinated_release=False, thumb_j3_margin_rad=.001,
    early_palm_clearance_m=0., early_palm_direction='away')


def numeric_release_preferences(document):
    """Copy numeric design choices only; all old physical coordinates discarded."""
    if type(document) is not dict:
        raise ValueError('A preference object is required')
    values = document.get('configuration', document)
    if type(values) is not dict:
        raise ValueError('Explicit numeric preference configuration required')
    result = {key: values.get(key, default) for key, default in DEFAULT_PREFERENCES.items()}
    for key, low, high in [('radial_clearance_m', .01, .03),
                          ('separation_reserve_m', .0005, .008),
                          ('thumb_j3_margin_rad', .001, .08),
                          ('early_palm_clearance_m', 0., .05)]:
        value = result[key]
        if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError('Original bounded release preference required: ' + key)
    # Omission preserves the historical preference payload as well as the
    # historical .01 arithmetic. This is a prospective numerical choice, not a
    # material-error tolerance or permission to change the independent gates.
    if 'thumb_posture_continuity_weight' in values:
        value = values['thumb_posture_continuity_weight']
        if type(value) not in (int, float) or not np.isfinite(value) or not .01 <= value <= .1:
            raise ValueError('Bounded thumb posture/continuity weight required (.01 to .1)')
        result['thumb_posture_continuity_weight'] = value
    for key in ('whole_body', 'coordinated_release'):
        if type(result[key]) is not bool:
            raise ValueError('Explicit boolean release preference required: ' + key)
    for key, choices in [('finger_profile', ('radial', 'extend')),
                         ('retreat_profile', ('lift', 'slide-lift')),
                         ('early_palm_direction', ('away', 'up'))]:
        if result[key] not in choices:
            raise ValueError('Known release preference required: ' + key)
    if result['coordinated_release'] and result['retreat_profile'] != 'slide-lift':
        raise ValueError('Coordinated release requires the free-end slide')
    return result


def _digit_regularization(v, preference, previous, *, thumb, extending,
                          thumb_posture_continuity_weight=.01):
    """Keep legacy digit arithmetic; opt in only the thumb's two soft terms."""
    posture_weight = thumb_posture_continuity_weight if thumb else (.25 if extending else .01)
    continuity_weight = thumb_posture_continuity_weight if thumb else .01
    return posture_weight*(v-preference), continuity_weight*(v-previous)


@dataclass(frozen=True)
class IsaacReleasePlanningContext:
    """Detached immutable admitted-source payload for a future constructor hook.

    Obtain this through ``admit_isaac_release_context``. ``qpos`` and receipts
    return fresh copies; mutating a consumer's arrays cannot alter the binding.
    """
    _admission_json: str

    @property
    def admission(self):
        return json.loads(self._admission_json)

    @property
    def qpos(self):
        return np.array(self.admission['initial_qpos'], dtype=float)

    @property
    def terminal_time_s(self):
        return float(self.admission['measured_rest']['terminal_time_s'])

    @property
    def state_archive_path(self):
        return Path(self.admission['source_run'])/'trial/acquisition-physics.npz'

    @property
    def sha256(self):
        return hashlib.sha256(self._admission_json.encode()).hexdigest()

    def verify_inputs(self):
        for name, expected in self.admission['input_sha256'].items():
            if digest(name) != expected:
                raise ValueError('Actual Isaac release source changed: ' + name)

    def scene(self):
        """Fresh isolated geometry calculator; never an active plant/reset."""
        self.verify_inputs()
        source = self.admission
        scene = LandedLeftScene(source['robot_path'], source['door_xml_path'])
        if self.qpos.shape != (scene.m.nq,):
            raise ValueError('Complete admitted scene coordinates required')
        scene.d.qpos[:] = self.qpos
        mujoco.mj_kinematics(scene.m, scene.d)
        return scene


def admit_isaac_release_context(source, *, robot, door_xml, door_usd,
                                profile='volar-phalange-v1'):
    """Run the actual PhysX physical/contact/rest and coordinate admission."""
    from .isaac_release_source import admit_isaac_release_source
    admission = admit_isaac_release_source(source, robot=robot, door_xml=door_xml,
                                            door_usd=door_usd, profile=profile)
    if (admission.get('schema') != 'doorbench.isaac-release-source.v1'
            or admission.get('source_engine') != 'isaac-physx'
            or admission['source_qualification'].get('passed') is not True
            or admission['coordinate_admission'].get('passed') is not True
            or admission.get('physics_steps') != 0
            or admission.get('source_sample_playback') != 0):
        raise ValueError('Explicit qualified actual Isaac release source required')
    # Keep the exact receipt, including coordinate normalization and historical
    # provenance hashes. Do not manufacture native fields for old constructors.
    encoded = json.dumps(admission, sort_keys=True, separators=(',', ':'), allow_nan=False)
    result = IsaacReleasePlanningContext(encoded)
    if not np.isfinite(result.qpos).all() or result.terminal_time_s <= 0:
        raise ValueError('Finite exact source coordinates and terminal epoch required')
    result.verify_inputs()
    return result


def generate_release_candidate(context, preferences):
    """Regenerate all coordinates/material goals from one admitted Isaac source.

    Numerical radial/retreat arithmetic is ported from the existing explicit
    profiled release planner. No old trajectory or native restart is consumed.
    """
    if not isinstance(context, IsaacReleasePlanningContext):
        raise ValueError('Admitted actual-source planning context required')
    values = numeric_release_preferences(preferences)
    rows, material_segments, material_points, residual = _generate_profiled_rows(context, values)
    initial = context.qpos
    if not rows or not np.array_equal(np.asarray(rows[0]['qpos']), initial):
        raise ValueError('Generated route changed the exact normalized source endpoint')
    context.verify_inputs()
    return dict(schema='doorbench.isaac-profiled-release-candidate.v1',
        source_engine='isaac-physx', source_context_sha256=context.sha256,
        source_admission=context.admission, initial_qpos=initial.tolist(),
        initial_time_s=context.terminal_time_s, configuration=values,
        grasp_profile=context.admission['grasp_profile'], trials=[dict(rows=rows)],
        tracked_material_segments=material_segments, release_material_points=material_points,
        maximum_pad_goal_residual_m=residual, input_sha256=context.admission['input_sha256'],
        physics_steps=0, API_calls=0, source_sample_playback=0,
        geometric_admission=False, physical_contact_qualification=False,
        runtime_route_exported=False, original_gates_unchanged=True,
        scope='Fresh unstepped candidate from qualified actual PhysX coordinates/material patches. Requires independent original geometry/anatomy/whole-handle/velocity gates and a new physical episode; no native state or success inherited.')


def _generate_profiled_rows(context, preferences):
    # This function owns only a new MjData geometry calculator. No mj_step,
    # environment reset, runtime articulation, archive playback or actuator call.
    scene = context.scene()
    m, d = scene.m, scene.d
    base = context.qpos
    admission = context.admission
    a = SimpleNamespace(**preferences, grasp_profile=admission['grasp_profile'])
    palm=m.site('robot/rh_palm_touch').id;lever=m.geom('leaf_handle_lever_col_n').id
    initial_p=d.site_xpos[palm].copy();initial_R=d.site_xmat[palm].reshape(3,3).copy()
    axis=d.geom_xmat[lever].reshape(3,3)[:,2].copy();center=d.geom_xpos[lever].copy()
    hub=m.geom('leaf_handle_hub_col_n').id
    free_end_direction=-np.sign((d.geom_xpos[hub]-center)@axis)*axis
    outward=-d.xmat[m.body('leaf').id].reshape(3,3)[:,1].copy()
    arm_names=['right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['rh_WRJ2','rh_WRJ1']
    body_names=[side+'_'+n for side in ('left','right') for n in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]+['torso']+arm_names+['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1']
    arms=np.array([m.joint('robot/'+n).id for n in arm_names]);armqa=m.jnt_qposadr[arms]
    body_ids=np.array([m.joint('robot/'+n).id for n in body_names]);bodyqa=m.jnt_qposadr[body_ids]
    rootqa=m.joint('robot/free_base').qposadr[0];rootP=base[rootqa:rootqa+3].copy()
    rootR=Rotation.from_quat(base[rootqa+3:rootqa+7][[1,2,3,0]])
    left=m.site('robot/lh_palm_touch').id;leftP=d.site_xpos[left].copy();leftR=d.site_xmat[left].reshape(3,3).copy()
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    footP=d.xpos[feet].copy();footR=d.xmat[feet].reshape(2,3,3).copy()
    tilt=np.arccos(np.clip(d.xmat[m.body('robot/torso_link').id].reshape(3,3)[2,2],-1,1))
    rotation_margin=.99*(np.radians(4)-tilt)/np.sqrt(2)
    if rotation_margin<=0:raise ValueError('Source needs upright rotation margin')
    obstacles=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name=='leaf_handle']
    digits=[];finger_names=[]
    for digit in ('FF','MF','RF','LF','TH'):
        names=[m.joint(j).name.removeprefix('robot/') for j in range(m.njnt) if m.joint(j).name.startswith('robot/rh_'+digit+'J')]
        ids=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[ids]
        body=m.body('robot/rh_'+digit.lower()+'distal').id
        shapes=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_'+digit.lower())]
        near=[]
        for g in shapes:
            if m.geom_bodyid[g]!=body:continue
            points=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,lever,.2,points)
            near.append((gap,points.copy()))
        gap,points=min(near,key=lambda x:x[0])
        if a.grasp_profile=='distal-pad-v1':
            if abs(gap)>.01:raise ValueError('Every attained fingertip must be near the lever')
            local=d.xmat[body].reshape(3,3).T@(points[:3]-d.xpos[body])
        else:
            # Track the dominant actually loaded material patch for this digit.
            # Every remaining finger shape still enters the obstacle residual
            # and the independent selected-profile dense geometry audit.
            contacts=[c for c in admission['measured_rest']['endpoint_contacts']
                if c['digit']==digit.lower() and c['pad_qualified'] and c['normal_force_N']>0]
            if not contacts:raise ValueError('Actual qualified material patch required for every release digit')
            contact=max(contacts,key=lambda c:c['normal_force_N'])
            body=m.body(contact['body']).id;local=np.array(contact['body_position_m'])
            points[:3]=d.xpos[body]+d.xmat[body].reshape(3,3)@local
        radial=points[:3]-center;radial-=axis*(radial@axis);radial/=np.linalg.norm(radial)
        low=np.minimum(base[qa],m.jnt_range[ids,0]+.001);high=np.maximum(base[qa],m.jnt_range[ids,1]-.001)
        relaxed=base[qa].copy()
        if digit!='TH':
            for j,name in enumerate(names):
                if name.endswith(('J1','J2','J3')):relaxed[j]=.1
        d.qpos[qa]=relaxed;mujoco.mj_kinematics(m,d)
        relaxed_point=d.xpos[body]+d.xmat[body].reshape(3,3)@local
        d.qpos[:]=base;mujoco.mj_kinematics(m,d)
        # Track every actually loaded anatomical segment, including the FF/MF
        # middle patches whose omission motivated this prospective candidate.
        tracked=[]
        for material in loaded_segment_targets(admission['measured_rest']['endpoint_contacts'], digit.lower()):
            bid=m.body(material['body']).id;loc=np.asarray(material['body_position_m'])
            pos=d.xpos[bid]+d.xmat[bid].reshape(3,3)@loc
            direction=pos-center;direction-=axis*(direction@axis);direction/=np.linalg.norm(direction)
            d.qpos[qa]=relaxed;mujoco.mj_kinematics(m,d)
            relaxed_material=d.xpos[bid]+d.xmat[bid].reshape(3,3)@loc
            d.qpos[:]=base;mujoco.mj_kinematics(m,d)
            tracked.append(dict(body=bid,local=loc,point=pos.copy(),radial=direction,relaxed_point=relaxed_material.copy(),source_force_N=material['source_force_N']))
        digits.append(dict(names=names,qa=qa,body=body,shapes=shapes,local=local,point=points[:3],radial=radial,low=low,high=high,previous=base[qa].copy(),relaxed=relaxed,relaxed_point=relaxed_point,tracked=tracked))
        finger_names+=names
    rows=[];previous_arm=base[armqa].copy();previous_body=np.r_[np.zeros(6),base[bodyqa]];worst=0.
    for time in np.linspace(0,8,161):
        separation=float(smooth_phase(time/3));lift=float(smooth_phase((time-3)/5))
        slide=0.
        if a.retreat_profile=='slide-lift':
            slide=float(smooth_phase((time-3)/2));lift=float(smooth_phase((time-5)/3))
            if a.coordinated_release:
                slide=float(smooth_phase((time-1)/3));lift=float(smooth_phase((time-4)/4))
        d.qpos[:]=base
        for digit in digits:
            qa=digit['qa'];goal=digit['point']+digit['radial']*a.radial_clearance_m*separation
            extending=a.finger_profile=='extend' and not digit['names'][0].startswith('rh_TH')
            preference=base[qa]
            if extending:
                goal=digit['point']+separation*(digit['relaxed_point']-digit['point'])
                preference=base[qa]+separation*(digit['relaxed']-base[qa])
            def residual(v):
                d.qpos[qa]=v;mujoco.mj_kinematics(m,d)
                point=d.xpos[digit['body']]+d.xmat[digit['body']].reshape(3,3)@digit['local']
                gaps=[min(0.,mujoco.mj_geomDistance(m,d,g,h,.01,None)-a.separation_reserve_m*separation) for g in digit['shapes'] for h in obstacles]
                loop=[]
                if digit['names'][0].startswith('rh_TH') is False:
                    i,j=[digit['names'].index('rh_'+digit['names'][0][3:5]+'J'+str(k)) for k in (1,2)]
                    loop=[10*(v[i]-v[j]-(1-separation)*(base[qa[i]]-base[qa[j]]))]
                material_errors=[]
                for material in digit['tracked']:
                    current=d.xpos[material['body']]+d.xmat[material['body']].reshape(3,3)@material['local']
                    requested=material['point']+material['radial']*a.radial_clearance_m*separation
                    if extending:requested=material['point']+separation*(material['relaxed_point']-material['point'])
                    material_errors.extend(100*(current-requested))
                posture, continuity = _digit_regularization(v, preference, digit['previous'],
                    thumb=digit['names'][0].startswith('rh_TH'), extending=extending,
                    thumb_posture_continuity_weight=preferences.get('thumb_posture_continuity_weight', .01))
                return np.r_[material_errors,1000*np.asarray(gaps),loop,posture,continuity]
            if time>0:
                low=digit['low'].copy();high=digit['high'].copy()
                if 'rh_THJ3' in digit['names']:
                    j=digit['names'].index('rh_THJ3');limits=m.jnt_range[m.joint('robot/rh_THJ3').id]
                    low[j]+=separation*(limits[0]+a.thumb_j3_margin_rad-low[j])
                    high[j]+=separation*(limits[1]-a.thumb_j3_margin_rad-high[j])
                fit=least_squares(residual,np.clip(digit['previous'],low+1e-12,high-1e-12),bounds=(low,high),max_nfev=100)
                digit['previous']=fit.x.copy()
                errors=fit.fun[:3*len(digit['tracked'])].reshape(-1,3)
                worst=max(worst,float(np.max(np.linalg.norm(errors,axis=1)))/100)
            d.qpos[qa]=digit['previous']
        fingers=d.qpos.copy()
        early=a.early_palm_clearance_m*separation
        early_direction=outward if a.early_palm_direction=='away' else np.array([0.,0.,1.])
        target=initial_p+.12*slide*free_end_direction+lift*(np.array([0.,0.,.1])+.04*outward)+early*early_direction
        def arm_residual(v):
            d.qpos[:]=fingers;d.qpos[armqa]=v;mujoco.mj_kinematics(m,d)
            return np.r_[100*(d.site_xpos[palm]-target),10*Rotation.from_matrix(initial_R@d.site_xmat[palm].reshape(3,3).T).as_rotvec(),.01*(v-base[armqa])]
        def body_residual(v):
            d.qpos[:]=fingers;d.qpos[rootqa:rootqa+3]=rootP+v[:3]
            quat=(Rotation.from_rotvec(v[3:6])*rootR).as_quat();d.qpos[rootqa+3:rootqa+7]=quat[[3,0,1,2]]
            d.qpos[bodyqa]=v[6:];mujoco.mj_kinematics(m,d)
            hand=np.r_[100*(d.site_xpos[palm]-target),10*Rotation.from_matrix(initial_R@d.site_xmat[palm].reshape(3,3).T).as_rotvec(),100*(d.site_xpos[left]-leftP),10*Rotation.from_matrix(leftR@d.site_xmat[left].reshape(3,3).T).as_rotvec()]
            foot=np.concatenate([np.r_[100*(d.xpos[b]-footP[i]),10*Rotation.from_matrix(footR[i]@d.xmat[b].reshape(3,3).T).as_rotvec()] for i,b in enumerate(feet)])
            return np.r_[hand,foot,.01*(v[6:]-base[bodyqa]),.02*v[:6]]
        if a.whole_body and (lift>0 or slide>0 or early>0):
            lo=np.r_[[-.06,-.06,-.03],[-rotation_margin,-rotation_margin,-.12],np.minimum(base[bodyqa],m.jnt_range[body_ids,0]+.001)]
            hi=np.r_[[.06,.06,.03],[rotation_margin,rotation_margin,.12],np.maximum(base[bodyqa],m.jnt_range[body_ids,1]-.001)]
            fit=least_squares(body_residual,np.clip(previous_body,lo+1e-12,hi-1e-12),bounds=(lo,hi),max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            previous_body=fit.x.copy();body_residual(previous_body)
        elif lift>0 or slide>0 or early>0:
            lo=np.minimum(base[armqa],m.jnt_range[arms,0]+.001);hi=np.maximum(base[armqa],m.jnt_range[arms,1]-.001)
            fit=least_squares(arm_residual,np.clip(previous_arm,lo+1e-12,hi-1e-12),bounds=(lo,hi),max_nfev=200)
            previous_arm=fit.x.copy()
            arm_residual(previous_arm)
        else:arm_residual(previous_arm)
        q=d.qpos.copy();actual_p=d.site_xpos[palm].copy();actual_R=d.site_xmat[palm].reshape(3,3).copy()
        rows.append(dict(time_s=float(time),phase='grasp_adjustment' if time==0 else 'measured_release' if time<=3 else 'clearance_lift',qpos=q.tolist(),palm_position=actual_p.tolist(),palm_rotation=actual_R.tolist(),requested_palm_position=target.tolist(),joints={n:float(q[m.joint('robot/'+n).qposadr[0]]) for n in body_names},finger_joints={n:float(q[m.joint('robot/'+n).qposadr[0]]) for n in finger_names}))
    segments = [dict(body=m.body(p['body']).name, body_position_m=p['local'].tolist(), source_force_N=p['source_force_N']) for digit in digits for p in digit['tracked']]
    points = [dict(body=m.body(digit['body']).name, body_position_m=digit['local'].tolist()) for digit in digits]
    return rows, segments, points, worst
