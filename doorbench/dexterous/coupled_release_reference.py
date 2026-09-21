"""Opt-in, independently admitted measured-frame withdrawal motor references."""
import json
from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .coupled_release_geometry import CoupledReleaseGeometry,sha
from .coupled_release_motion import CoupledReferenceMotion
from .grasp_verification import shadow_surface_qualified
from .landed_left_audit import static_pose_check


def validate_admission(config):
    envelope=Path(config['coupled_envelope_path']);audit_path=Path(config['coupled_audit_path'])
    if sha(envelope)!=config['coupled_envelope_sha256'] or sha(audit_path)!=config['coupled_audit_sha256']:raise ValueError('Coupled withdrawal evidence bytes changed')
    audit=json.loads(audit_path.read_text())
    if audit.get('schema')!='doorbench.coupled-release-envelope-audit.v1' or audit.get('passed') is not True or audit.get('coarse_diagnostic') is not False or audit.get('samples',0)<10000 or not audit.get('exact_initial_state') or audit.get('physics_steps')!=0:
        raise ValueError('Independent dense coupled envelope admission required')
    if audit['input_sha256'].get(str(envelope.resolve()))!=sha(envelope):raise ValueError('Coupled audit belongs to another envelope')
    import doorbench.dexterous.coupled_release_geometry as geometry
    if audit['input_sha256'].get(str(Path(geometry.__file__).resolve()))!=sha(geometry.__file__):raise ValueError('Coupled audit must bind the consumed geometry evaluator')
    limits=dict(left_position_error_m=.001,left_rotation_error_rad=.01,right_position_error_m=.001,right_rotation_error_rad=.01,foot_position_error_m=.001,foot_rotation_error_rad=.01,torso_tilt_deg=4.,root_translation_m=.03,com_xy_displacement_m=.015,palm_panel_gap_change_m=.001)
    if audit.get('limits')!=limits:raise ValueError('Original coupled geometric limits required')
    for name,digest in audit['input_sha256'].items():
        if sha(name)!=digest:raise ValueError('Coupled admitted input changed: '+name)
    return envelope,audit


class CoupledReleaseReference:
    def __init__(self,config,source_admission,initial_qpos,duration):
        if config.get('capture_returned_motor_command') is not True:raise ValueError('Coupled withdrawal requires actual predecessor motor capture')
        envelope,self.audit=validate_admission(config);self.geometry=CoupledReleaseGeometry(envelope);g=self.geometry;m,d=g.m,g.d
        if g.plan['source_admission']!=source_admission or not np.array_equal(g.initial,initial_qpos) or g.plan['duration_s']!=duration:raise ValueError('Coupled envelope must bind exact withdrawal source and duration')
        domain=self.audit['geometry_domain']
        if (domain.get('admitted_leaf_upper_nodes')!=g.plan.get('admitted_leaf_upper_nodes')
                or domain['operator_rad']!=g.plan['operator_envelope_rad']
                or domain.get('elapsed_s')!=[0,g.plan['duration_s']]
                or domain.get('leaf_rad')!=[float(g.angles[0]),float(g.angles[-1])]
                or domain.get('latch_m')!=[-.001,.001]):raise ValueError('Coupled audit and live domain differ')
        self.names=[m.joint(j).name[6:] for j in range(m.njnt) if m.joint(j).name.startswith('robot/') and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
        self.qa=np.array([m.joint('robot/'+n).qposadr[0] for n in self.names]);self.motion=CoupledReferenceMotion(np.r_[np.zeros(6),g.initial[self.qa]])
        self.feet=[m.body('robot/'+s+'_ankle_link').id for s in ('left','right')];self.torso=m.body('robot/torso_link').id;self.robotbody=m.jnt_bodyid[m.joint('robot/free_base').id]
        self.lever=m.geom('leaf_handle_lever_col_n').id;active=[v for v in range(m.ngeom) if m.geom_contype[v] or m.geom_conaffinity[v]]
        self.right=[v for v in active if m.body(m.geom_bodyid[v]).name.startswith('robot/rh_')];self.handle=[v for v in active if m.geom_bodyid[v]==g.handle]
        self.palm=[v for v in active if m.geom_bodyid[v]==m.site_bodyid[g.lh]];self.panel=[v for v in active if m.geom_bodyid[v]==g.leaf]
        d.qpos[:]=g.initial;mujoco.mj_kinematics(m,d)
        self.initial_gap=min(float(mujoco.mj_geomDistance(m,d,a,b,.1,None)) for a in self.palm for b in self.panel)
        self.info={};self.accepted=0;self.limited=0;self.failure_snapshot=None

    def _inspect(self,result):
        g=self.geometry;m,d=g.m,g.d;c=g.c;check=static_pose_check(m,d,coordinate=1.);mujoco.mj_comPos(m,d)
        def er(a,b):return float(np.linalg.norm(Rotation.from_matrix(a@b.T).as_rotvec()))
        bad=[]
        if not check['passed']:bad.append('original collision/joint/loopback gates')
        pairs=[(g.lh,result['left_palm_position'],result['left_palm_rotation']),(g.rh,result['right_palm_position'],result['right_palm_rotation'])]
        pe=max(float(np.linalg.norm(d.site_xpos[s]-p)) for s,p,r in pairs);rotation_error=max(er(r,d.site_xmat[s].reshape(3,3)) for s,p,r in pairs)
        if pe>.001 or rotation_error>.01:bad.append('measured-frame palm pose gate')
        if max(float(np.linalg.norm(d.xpos[b]-c['initial_feet_positions'][i])) for i,b in enumerate(self.feet))>.001 or max(er(np.array(c['initial_feet_rotations'][i]),d.xmat[b].reshape(3,3)) for i,b in enumerate(self.feet))>.01:bad.append('fixed-foot pose gate')
        if np.degrees(np.arccos(np.clip(d.xmat[self.torso].reshape(3,3)[2,2],-1,1)))>4.:bad.append('upright4degree gate')
        if np.linalg.norm(d.qpos[g.rq:g.rq+3]-g.initial[g.rq:g.rq+3])>.03 or np.linalg.norm(d.subtree_com[self.robotbody,:2]-np.array(c['initial_com'][:2]))>.015:bad.append('root/COM gate')
        gap=min(float(mujoco.mj_geomDistance(m,d,a,b,.1,None)) for a in self.palm for b in self.panel)
        if abs(gap-self.initial_gap)>.001:bad.append('palm/panel gap gate')
        clear=None
        if result['release_clock_s']-.5>=g.through:
            clear=min(float(mujoco.mj_geomDistance(m,d,a,b,.2,None)) for a in self.right for b in self.handle)
            if clear<.004:bad.append('literal4mm all-handle blend gate')
        for contact in d.contact[:d.ncon]:
            if contact.dist>=0:continue
            bodies=[m.body(m.geom_bodyid[a]).name for a in contact.geom]
            if 'leaf_handle' not in bodies or not any(b.startswith('robot/rh_') for b in bodies):continue
            if self.lever not in contact.geom:bad.append('RH contact outside lever');continue
            side=0 if contact.geom[1]==self.lever else 1;b=int(m.geom_bodyid[contact.geom[side]]);name=m.body(b).name
            match=re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(distal|middle|proximal)',name);R=d.xmat[b].reshape(3,3)
            point=R.T@(contact.pos-d.xpos[b]);normal=R.T@(contact.frame[:3]*(1 if side==0 else -1))
            axis=d.geom_xmat[self.lever].reshape(3,3)[:,2];rel=contact.pos-d.geom_xpos[self.lever];axial=float(rel@axis);radial=rel-axial*axis;alignment=float((R@normal)@(-radial/max(np.linalg.norm(radial),1e-12)))
            if not match or m.geom_size[self.lever,1]-abs(axial)<.001 or alignment<=.8 or not shadow_surface_qualified(*match.groups(),point,normal,profile='volar-phalange-v1'):bad.append('original selected RH anatomy/normal/end gate')
        if bad:raise ValueError('Coupled reference stopped before motor submission: '+', '.join(sorted(set(bad))))
        return dict(coupled_reference_palm_position_error_m=pe,coupled_reference_palm_rotation_error_rad=rotation_error,coupled_reference_all_handle_clearance_m=clear)

    def update(self,t,elapsed,angles,leaf_pose,handle_pose):
        if self.failure_snapshot is not None:raise ValueError('Coupled reference is terminal after its first failed diagnostic')
        try:return self._update(t,elapsed,angles,leaf_pose,handle_pose)
        except Exception as exc:
            self.failure_snapshot=dict(time_s=float(t),elapsed_s=float(elapsed),angles=dict(angles),error=type(exc).__name__+': '+str(exc),accepted_samples=self.accepted)
            raise

    def _update(self,t,elapsed,angles,leaf_pose,handle_pose):
        g=self.geometry;m,d=g.m,g.d
        elapsed=min(elapsed,g.plan['duration_s'])
        result=g.evaluate(elapsed,angles['leaf'],angles['operator'],angles['latch'])
        # Verify the actual poses used by the plant and the private mechanism
        # agree. Only measured angles enter the private model; none are outputs.
        for body,pose in [(g.leaf,leaf_pose),(g.handle,handle_pose)]:
            pose=np.asarray(pose,float)
            if pose.shape!=(7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]),1.,atol=1e-6):raise ValueError('Finite normalized measured mechanism pose required')
            rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
            if np.linalg.norm(pose[:3]-d.xpos[body])>1e-5 or Rotation.from_matrix(rotation@d.xmat[body].reshape(3,3).T).magnitude()>1e-5:raise ValueError('Measured mechanism frames differ from admitted model')
        desired=np.r_[result['coordinates'][:6],result['qpos'][self.qa]]
        value,motion=self.motion.update(t,desired)
        d.qpos[g.rq:g.rq+3]=g.initial[g.rq:g.rq+3]+value[:3]
        rotation=Rotation.from_rotvec(value[3:6])*g.initial_rotation;quat=rotation.as_quat();d.qpos[g.rq+3:g.rq+7]=quat[[3,0,1,2]];d.qpos[self.qa]=value[6:]
        checked=self._inspect(result);self.accepted+=1;self.limited+=int(motion['limited'])
        self.info={**checked,'coupled_reference':True,'coupled_reference_accepted_samples':self.accepted,'coupled_reference_limited_samples':self.limited,'coupled_reference_leaf_rad':float(angles['leaf']),'coupled_reference_leaf_upper_rad':g.upper_angle(elapsed),'coupled_reference_operator_rad':float(angles['operator']),'coupled_reference_handle_follow_weight':result['right_follow_weight'],'coupled_reference_release_clock_s':result['release_clock_s'],'coupled_reference_motion':motion,'coupled_reference_plant_pose_writes':0}
        return dict(zip(self.names,value[6:])),d.qpos[g.rq:g.rq+3].copy(),rotation.as_matrix(),result['right_palm_position'],result['right_palm_rotation'],self.info.copy()
