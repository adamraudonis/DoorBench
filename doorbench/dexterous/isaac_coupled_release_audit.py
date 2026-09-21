"""Independent static map audit bound to an actual qualified Isaac endpoint.

Original feet/palm/anatomy/whole-handle gates are retained. A map passing this
audit is still detached: measured contact, motion rates, post-limiter geometry,
motor loads and dynamic balance require a separate live physical episode.
"""
from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .isaac_coupled_release_geometry import IsaacCoupledReleaseGeometry
from .landed_left_audit import static_pose_check
from .qualified_isaac_grasp import digest
from scripts.dexterous.audit_standing_ungrip import release_surface_scores


SCHEMA = 'doorbench.isaac-coupled-release-envelope-audit.v1'
ORIGINAL_LIMITS = dict(left_position_error_m=.001,left_rotation_error_rad=.01,
    right_position_error_m=.001,right_rotation_error_rad=.01,
    foot_position_error_m=.001,foot_rotation_error_rad=.01,torso_tilt_deg=4.,
    root_translation_m=.03,com_xy_displacement_m=.015,palm_panel_gap_change_m=.001)
ORIGINAL_GEOMETRY_LIMITS = dict(nonfoot_penetration_m=.003,joint_violation_rad=.02,
    loopback_violation_rad=.02,all_handle_blend_clearance_m=.004,
    final_right_hand_environment_clearance_m=.04,lever_axial_margin_m=.001,
    lever_normal_alignment_strict_minimum=.8)


def iter_domain_samples(model,*,coarse=False):
    """Independent dense off-grid coverage, plus both bolt/operator extrema."""
    lo,hi = model.plan['operator_envelope_rad']
    schedules = ([(model.plan['operator_reference_rad'],81,18),(lo,81,18),(hi,81,18)]
        if coarse else [(model.plan['operator_reference_rad'],501,65),(lo,201,33),(hi,201,33)])
    if not coarse and hi-lo > .04:
        schedules.extend((value,101,33) for value in np.linspace(lo,hi,5)[1:-1])
    # Include actual knots and independent cell centers even if a future map
    # uses more knots than the original 81x18 planner resolution.
    midtimes = (model.elapsed[:-1]+model.elapsed[1:])/2
    midangles = (model.angles[:-1]+model.angles[1:])/2
    upper_nodes = model.plan.get('admitted_leaf_upper_nodes')
    knots = (np.array([0.,model.plan['duration_s']]) if upper_nodes is None
        else np.asarray(upper_nodes,float)[:,0])
    for operator,nt,na in schedules:
        times = np.linspace(0,model.plan['duration_s'],nt)
        if not coarse: times = np.unique(np.r_[times,model.elapsed,midtimes,knots])
        for t in times:
            upper = model.upper_angle(float(t))
            angles = np.linspace(model.angles[0],upper,na)
            if not coarse:
                angles = np.unique(np.r_[angles,model.angles,midangles])
                angles = angles[(angles>=model.angles[0]) & (angles<=upper)]
            for angle in angles:
                yield float(t),float(angle),float(operator),float(model.initial[model.bq])
    for latch in (-.001,.001):
        for operator in (lo,hi):
            for t in np.linspace(0,model.plan['duration_s'],21 if coarse else 81):
                for angle in np.linspace(model.angles[0],model.upper_angle(float(t)),9 if coarse else 17):
                    yield float(t),float(angle),float(operator),latch


def _rotation_error(a,b):
    return float(np.linalg.norm(Rotation.from_matrix(a@b.T).as_rotvec()))


class GeometryInspector:
    """Reusable single-configuration checks; receives only a private model."""
    def __init__(self,model):
        self.model = model
        self.m,self.d = m,d = model.m,model.d
        self.feet = [m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
        self.torso = m.body('robot/torso_link').id
        self.robotbody = int(m.jnt_bodyid[m.joint('robot/free_base').id])
        self.lever = m.geom('leaf_handle_lever_col_n').id
        active = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.right = [g for g in active if m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
        self.handle = [g for g in active if m.geom_bodyid[g]==model.handle]
        self.palm = [g for g in active if m.geom_bodyid[g]==m.site_bodyid[model.lh]]
        self.panel = [g for g in active if m.geom_bodyid[g]==model.leaf]
        self.environment = [g for g in active if not m.body(m.geom_bodyid[g]).name.startswith('robot/')]
        if not all((self.right,self.handle,self.palm,self.panel,self.environment)):
            raise ValueError('Complete active/receiving-only hand, handle, palm and scene geometry required')
        d.qpos[:] = model.initial;mujoco.mj_kinematics(m,d)
        self.initial_gap = self.gap(self.palm,self.panel,.1)

    def gap(self,first,second,limit):
        return min(float(mujoco.mj_geomDistance(self.m,self.d,g,h,limit,None))
            for g in first for h in second)

    def inspect(self,t,angle,operator,latch):
        model,m,d = self.model,self.m,self.d
        result = model.evaluate(t,angle,operator,latch)
        check = static_pose_check(m,d,coordinate=1.)
        mujoco.mj_comPos(m,d)
        c = model.c
        values = dict(
            left_position_error_m=float(np.linalg.norm(d.site_xpos[model.lh]-result['left_palm_position'])),
            left_rotation_error_rad=_rotation_error(result['left_palm_rotation'],d.site_xmat[model.lh].reshape(3,3)),
            right_position_error_m=float(np.linalg.norm(d.site_xpos[model.rh]-result['right_palm_position'])),
            right_rotation_error_rad=_rotation_error(result['right_palm_rotation'],d.site_xmat[model.rh].reshape(3,3)),
            foot_position_error_m=max(float(np.linalg.norm(d.xpos[b]-c['initial_feet_positions'][i])) for i,b in enumerate(self.feet)),
            foot_rotation_error_rad=max(_rotation_error(np.asarray(c['initial_feet_rotations'][i]),d.xmat[b].reshape(3,3)) for i,b in enumerate(self.feet)),
            torso_tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[self.torso].reshape(3,3)[2,2],-1,1)))),
            root_translation_m=float(np.linalg.norm(d.qpos[model.rq:model.rq+3]-model.initial[model.rq:model.rq+3])),
            com_xy_displacement_m=float(np.linalg.norm(d.subtree_com[self.robotbody,:2]-np.asarray(c['initial_com'])[:2])),
            palm_panel_gap_change_m=abs(self.gap(self.palm,self.panel,.1)-self.initial_gap))
        bad = [key for key,value in values.items() if not np.isfinite(value) or value>ORIGINAL_LIMITS[key]]
        if not check['passed']:
            bad.append('original_collision_joint_loopback_limits')
        wrists = [m.joint('robot/'+name).id for name in ('left_wrist_yaw','lh_WRJ2','lh_WRJ1')]
        wrist_margin = min(float(min(d.qpos[m.jnt_qposadr[j]]-m.jnt_range[j,0],
            m.jnt_range[j,1]-d.qpos[m.jnt_qposadr[j]])) for j in wrists)
        blend_gap = None
        if result['release_clock_s']-.5 >= model.through:
            blend_gap = self.gap(self.right,self.handle,.2)
            if not np.isfinite(blend_gap) or blend_gap < .004:
                bad.append('literal4mm_all_handle_blend_clearance')
        final_gap = None
        if t == model.plan['duration_s']:
            final_gap = self.gap(self.right,self.environment,.5)
            if not np.isfinite(final_gap) or final_gap < .04:
                bad.append('original_final_right_hand_environment_clearance')
        distal_invalid = 0
        for contact in d.contact[:d.ncon]:
            if contact.dist >= 0: continue
            bodies = [m.body(m.geom_bodyid[g]).name for g in contact.geom]
            if 'leaf_handle' not in bodies or not any(b.startswith('robot/rh_') for b in bodies): continue
            if self.lever not in contact.geom:
                bad.append('RH contact outside lever');continue
            side = 0 if contact.geom[1]==self.lever else 1
            body = int(m.geom_bodyid[contact.geom[side]]);name = m.body(body).name
            match = re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(distal|middle|proximal)',name)
            rotation = d.xmat[body].reshape(3,3)
            point = rotation.T@(contact.pos-d.xpos[body])
            normal = rotation.T@(contact.frame[:3]*(1 if side==0 else -1))
            axis = d.geom_xmat[self.lever].reshape(3,3)[:,2]
            rel = contact.pos-d.geom_xpos[self.lever];axial = float(rel@axis);radial = rel-axial*axis
            alignment = float((rotation@normal)@(-radial/max(np.linalg.norm(radial),1e-12)))
            selected,distal = (release_surface_scores(*match.groups(),point,normal,
                m.geom_size[self.lever,1]-abs(axial),alignment,
                profile=model.source_data['source_admission']['grasp_profile']) if match else (False,False))
            if not selected: bad.append('Original RH selected anatomy/side gates: '+name)
            distal_invalid += not distal
        return dict(failed=bad,values=values,collisions=check['contacts'],
            blend_gap_m=blend_gap,final_gap_m=final_gap,wrist_margin_rad=wrist_margin,
            original_distal_invalid_geometry_patches=distal_invalid)


def audit_geometry(model,*,coarse=False,progress=None):
    """Full static tensor-domain audit. Diagnostic mode can never pass."""
    if type(coarse) is not bool: raise ValueError('Explicit boolean diagnostic mode required')
    inspector = GeometryInspector(model)
    maximum,failures = {},[]
    samples = failed_count = distal_invalid = 0
    min_blend = min_final = min_wrist = float('inf')
    source = (0.,float(model.initial[model.lq]),float(model.initial[model.oq]),float(model.initial[model.bq]))
    exact = np.array_equal(model.evaluate(*source)['qpos'],model.initial)
    from itertools import chain
    for point in chain([source],iter_domain_samples(model,coarse=coarse)):
        row = inspector.inspect(*point);samples += 1
        for key,value in row['values'].items(): maximum[key] = max(maximum.get(key,0.),value)
        if row['blend_gap_m'] is not None: min_blend = min(min_blend,row['blend_gap_m'])
        if row['final_gap_m'] is not None: min_final = min(min_final,row['final_gap_m'])
        min_wrist = min(min_wrist,row['wrist_margin_rad'])
        distal_invalid += row['original_distal_invalid_geometry_patches']
        if row['failed']:
            failed_count += 1
            if len(failures)<100:
                failures.append(dict(elapsed_s=point[0],leaf_rad=point[1],operator_rad=point[2],latch_m=point[3],**row))
        if progress is not None and samples%2000==0:
            progress(dict(samples=samples,failed_samples=failed_count))
    complete = bool(np.isfinite([min_blend,min_final,min_wrist]).all())
    return dict(passed=bool(not failed_count and exact and complete and not coarse),
        coarse_diagnostic=coarse,samples=samples,failed_samples=failed_count,
        exact_initial_state=bool(exact),complete_original_geometry_coverage=complete,
        maximum=maximum,limits=ORIGINAL_LIMITS.copy(),geometry_limits=ORIGINAL_GEOMETRY_LIMITS.copy(),
        minimum_blend_all_handle_clearance_m=min_blend if np.isfinite(min_blend) else None,
        minimum_final_right_hand_environment_clearance_m=min_final if np.isfinite(min_final) else None,
        minimum_left_wrist_physical_margin_rad=min_wrist if np.isfinite(min_wrist) else None,
        original_distal_invalid_geometry_patches=distal_invalid,failures=failures)


def audit_isaac_coupled_envelope(path,*,source_config,coarse=False,progress=None):
    """Fresh actual-source admission followed by the independent map screen."""
    from . import (coupled_release_geometry,isaac_coupled_release_geometry,
        isaac_release_geometry_audit,landed_left_audit,grasp_verification,withdrawal_source_context)
    from scripts.dexterous import audit_standing_ungrip
    modules = (coupled_release_geometry,isaac_coupled_release_geometry,isaac_release_geometry_audit,
        landed_left_audit,grasp_verification,withdrawal_source_context,audit_standing_ungrip)
    code = {str(Path(p).resolve()):digest(p) for p in [__file__]+[module.__file__ for module in modules]}
    model = IsaacCoupledReleaseGeometry(path,source_config=source_config)
    try:
        result = audit_geometry(model,coarse=coarse,progress=progress)
        model.verify_inputs()
        if any(digest(name)!=expected for name,expected in code.items()):
            raise ValueError('Independent Isaac coupled audit primitives changed during computation')
        data,p = model.source_data,model.plan
        result.update(schema=SCHEMA,source_engine='isaac-physx',source_admission=data['source_admission'],
            source_context_sha256=data['source_context_sha256'],source_state_sha256=data['source_state_sha256'],
            initial_episode_time_s=data['start_time_s'],initial_qpos=model.initial.tolist(),
            source_physics_archive=data['source_archive_path'],motor_contract_sha256=data['motor_contract_sha256'],
            grasp_profile=data['source_admission']['grasp_profile'],physics_steps=0,active_state_writes=0,
            source_sample_playback=0,authorized_stages=0,runtime_route_exported=False,
            physical_admission=False,physical_contact_qualification=False,delivered_motor_force_checked=False,
            dynamic_balance_qualification=False,motion_rate_qualification=False,
            post_motion_limiter_geometry_checked=False,
            geometry_domain=dict(elapsed_s=[0,p['duration_s']],leaf_rad=[.08,.4],
                admitted_leaf_upper_nodes=p.get('admitted_leaf_upper_nodes'),
                operator_rad=p['operator_envelope_rad'],latch_m=[-.001,.001]),
            input_sha256={**model._file_hashes,**code},
            scope='Detached static measured-angle geometry only. Original actual PhysX source evidence is immutable; no source playback, physical release success, motor/load feasibility, arbitrary mechanism-rate admission or runtime promotion.',
            runtime_requirements='Live exact-prefix witness and original per-step geometry after constraint-preserving rate limiting remain required, with independent actual contact/load/balance gates.')
        return result
    finally:
        model.close()
