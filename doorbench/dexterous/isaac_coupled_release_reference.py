"""Source-bound Isaac initialization for the unchanged capped reference logic."""
import json
from pathlib import Path

import mujoco
import numpy as np

from .coupled_release_reference import CoupledReleaseReference
from .coupled_release_motion import CoupledReferenceMotion
from .isaac_coupled_release_geometry import IsaacCoupledReleaseGeometry,_verify_hashes
from .isaac_coupled_release_audit import SCHEMA,ORIGINAL_LIMITS,ORIGINAL_GEOMETRY_LIMITS
from .qualified_isaac_grasp import digest


def validate_isaac_coupled_audit(config,geometry,source_data):
    path=Path(config['coupled_audit_path']).resolve();envelope=Path(config['coupled_envelope_path']).resolve()
    if digest(path)!=config['coupled_audit_sha256'] or digest(envelope)!=config['coupled_envelope_sha256']:
        raise ValueError('Isaac coupled runtime evidence bytes changed')
    audit=json.loads(path.read_text());g=geometry;p=g.plan
    if (audit.get('schema')!=SCHEMA or audit.get('source_engine')!='isaac-physx'
            or audit.get('passed') is not True or audit.get('coarse_diagnostic') is not False
            or type(audit.get('samples')) is not int or audit['samples']<50000
            or audit.get('failed_samples')!=0 or audit.get('failures')!=[]
            or audit.get('exact_initial_state') is not True
            or audit.get('complete_original_geometry_coverage') is not True
            or audit.get('limits')!=ORIGINAL_LIMITS or audit.get('geometry_limits')!=ORIGINAL_GEOMETRY_LIMITS):
        raise ValueError('Complete distinct original Isaac coupled geometry audit required')
    for key in ('source_admission','source_context_sha256','source_state_sha256','motor_contract_sha256'):
        if audit.get(key)!=source_data[key] or g.source_data[key]!=source_data[key]:
            raise ValueError('Isaac coupled audit binds another '+key)
    if (audit.get('initial_episode_time_s')!=source_data['start_time_s']
            or not np.array_equal(np.asarray(audit.get('initial_qpos')),np.asarray(source_data['initial_qpos']))
            or not np.array_equal(g.initial,np.asarray(source_data['initial_qpos']))
            or p['duration_s']!=source_data['duration_s']
            or audit.get('grasp_profile')!=source_data['source_admission']['grasp_profile']
            or audit['grasp_profile']!='volar-phalange-v1'
            or Path(audit.get('source_physics_archive','')).resolve()!=Path(source_data['source_archive_path']).resolve()):
        raise ValueError('Exact prospective volar Isaac source, endpoint, profile and duration required')
    for key in ('physics_steps','active_state_writes','source_sample_playback','authorized_stages'):
        if type(audit.get(key)) is not int or audit[key]!=0:
            raise ValueError('Geometry receipt must retain zero '+key)
    for key in ('physical_admission','physical_contact_qualification','delivered_motor_force_checked',
                'dynamic_balance_qualification','motion_rate_qualification','post_motion_limiter_geometry_checked','runtime_route_exported'):
        if audit.get(key) is not False:raise ValueError('Static audit cannot manufacture '+key)
    domain=dict(elapsed_s=[0,p['duration_s']],leaf_rad=[.08,.4],admitted_leaf_upper_nodes=p.get('admitted_leaf_upper_nodes'),
        operator_rad=p['operator_envelope_rad'],latch_m=[-.001,.001])
    if audit.get('geometry_domain')!=domain:raise ValueError('Isaac live measured-state domain differs from audit')
    maximum=audit.get('maximum',{})
    if set(maximum)!=set(ORIGINAL_LIMITS) or any(type(maximum[k]) not in (int,float)
            or not np.isfinite(maximum[k]) or not 0<=maximum[k]<=limit for k,limit in ORIGINAL_LIMITS.items()):
        raise ValueError('Actual original geometry maxima required')
    for key,minimum in [('minimum_blend_all_handle_clearance_m',.004),('minimum_final_right_hand_environment_clearance_m',.04)]:
        if type(audit.get(key)) not in (int,float) or not np.isfinite(audit[key]) or audit[key]<minimum:
            raise ValueError('Original audited Isaac clearance required: '+key)
    hashes=_verify_hashes(audit.get('input_sha256'))
    for name,expected in g._file_hashes.items():
        if hashes.get(str(Path(name).resolve()))!=expected:raise ValueError('Isaac coupled audit omits source input: '+name)
    from . import isaac_coupled_release_geometry,coupled_release_geometry,isaac_coupled_release_audit
    for module in (isaac_coupled_release_geometry,coupled_release_geometry,isaac_coupled_release_audit):
        if hashes.get(str(Path(module.__file__).resolve()))!=digest(module.__file__):
            raise ValueError('Isaac coupled audit must bind current consumed geometry primitives')
    hashes[str(path)]=digest(path)
    return audit,hashes


class IsaacCoupledReleaseReference(CoupledReleaseReference):
    """No plant interface: same original rates/projection and post-FK guards."""
    def __init__(self,config,source_data):
        if config.get('coupled_motion_projection')!='fixed-poses-v1' or config.get('capture_returned_motor_command') is not True:
            raise ValueError('Exact predecessor capture and constraint-preserving motion required')
        self.geometry=IsaacCoupledReleaseGeometry(config['coupled_envelope_path'],source_config=config['source_config_path'])
        g=self.geometry;m,d=g.m,g.d
        self.audit,self.input_sha256=validate_isaac_coupled_audit(config,g,source_data)
        self.names=[m.joint(j).name[6:] for j in range(m.njnt) if m.joint(j).name.startswith('robot/') and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
        self.qa=np.array([m.joint('robot/'+n).qposadr[0] for n in self.names])
        self.motion=CoupledReferenceMotion(np.r_[np.zeros(6),g.initial[self.qa]])
        self.projection='fixed-poses-v1'
        self.body_columns=np.r_[np.arange(6),[6+self.names.index(n) for n in g.names]].astype(int)
        self.feet=[m.body('robot/'+s+'_ankle_link').id for s in ('left','right')]
        self.torso=m.body('robot/torso_link').id;self.robotbody=m.jnt_bodyid[m.joint('robot/free_base').id]
        self.lever=m.geom('leaf_handle_lever_col_n').id
        active=[v for v in range(m.ngeom) if m.geom_contype[v] or m.geom_conaffinity[v]]
        self.right=[v for v in active if m.body(m.geom_bodyid[v]).name.startswith('robot/rh_')]
        self.handle=[v for v in active if m.geom_bodyid[v]==g.handle]
        self.palm=[v for v in active if m.geom_bodyid[v]==m.site_bodyid[g.lh]]
        self.panel=[v for v in active if m.geom_bodyid[v]==g.leaf]
        self.environment=[v for v in active if not m.body(m.geom_bodyid[v]).name.startswith('robot/')]
        if not all((self.right,self.handle,self.palm,self.panel,self.environment)):
            raise ValueError('Complete active Isaac authored clearance geometry required')
        d.qpos[:]=g.initial;mujoco.mj_kinematics(m,d)
        self.initial_gap=min(float(mujoco.mj_geomDistance(m,d,a,b,.1,None)) for a in self.palm for b in self.panel)
        self.info={};self.accepted=0;self.limited=0;self.failure_snapshot=None

    def _inspect(self,result):
        # Parent checks the current post-limiter private qpos, not the original
        # desired map. Preserve those exact gates and add the audited final gap.
        info=super()._inspect(result)
        if result['release_clock_s']>=self.geometry.times[-1]-1e-12:
            gap=min(float(mujoco.mj_geomDistance(self.geometry.m,self.geometry.d,a,b,.5,None))
                for a in self.right for b in self.environment)
            if not np.isfinite(gap) or gap<.04:
                raise ValueError('Original final40mm hand/environment reference gate')
            info['coupled_reference_final_environment_clearance_m']=gap
        return info
