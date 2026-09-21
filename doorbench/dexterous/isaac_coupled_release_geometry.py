"""Detached measured-angle geometry initialized from qualified actual PhysX.

The geometric evaluator is inherited; an optional Isaac-only lower aperture
domain is enforced before its coordinate lookup. The native constructor is
never called. This private calculator grants no runtime stage permission.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial.transform import Rotation, Slerp

from .coupled_release_geometry import CoupledReleaseGeometry
from .isaac_release_geometry_audit import _path_arrays
from .landed_left_planner import LandedLeftScene
from .qualified_isaac_grasp import digest
from .withdrawal_source_context import load_withdrawal_source_context


SCHEMA = 'doorbench.isaac-coupled-release-envelope.v1'
JOINT_NAMES = (['torso'] + [side+'_'+name for side in ('left','right')
    for name in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]
    + [side+'_'+name for side in ('right','left')
       for name in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]
    + [side+'_'+name for side in ('rh','lh') for name in ('WRJ2','WRJ1')])


def _verify_hashes(bindings):
    if type(bindings) is not dict or not bindings:
        raise ValueError('Complete absolute Isaac coupled input bindings required')
    normalized = {}
    for name, expected in bindings.items():
        path = Path(name)
        if not path.is_absolute() or digest(path) != expected:
            raise ValueError('Isaac coupled input changed: '+name)
        resolved = str(path.resolve())
        if resolved in normalized and normalized[resolved] != expected:
            raise ValueError('Conflicting Isaac coupled input aliases')
        normalized[resolved] = expected
    return normalized


def validate_envelope_binding(plan, data, config_path):
    """The current independently admitted source remains the authority."""
    if (plan.get('schema') != SCHEMA or plan.get('source_engine') != 'isaac-physx'
            or plan.get('source_kind') != data.get('source_kind')
            or data.get('source_engine') != 'isaac-physx'
            or plan.get('source_admission') != data['source_admission']
            or plan.get('source_context_sha256') != data['source_context_sha256']
            or plan.get('source_state_sha256') != data['source_state_sha256']
            or plan.get('initial_episode_time_s') != data['start_time_s']
            or plan.get('duration_s') != data['duration_s']
            or plan.get('grasp_profile') != data['source_admission']['grasp_profile']
            or plan.get('motor_contract_sha256') != data['motor_contract_sha256']
            or not np.array_equal(np.asarray(plan.get('initial_qpos')), np.asarray(data['initial_qpos']))):
        raise ValueError('Coupled candidate must bind this exact admitted PhysX endpoint and contract')
    for name in ('physics_steps','source_sample_playback','active_state_writes','authorized_stages'):
        if type(plan.get(name)) is not int or plan[name] != 0:
            raise ValueError('Detached unstepped Isaac planning required: '+name)
    for name in ('physical_admission','runtime_route_exported'):
        if plan.get(name) is not False:
            raise ValueError('Preserve the unadmitted coupled candidate: '+name)
    expected_paths = dict(source_config=Path(config_path).resolve(),
        right_hand_candidate=Path(data['screen_path']).resolve(),
        right_hand_dense_audit=Path(data['audit_path']).resolve(),
        source_physics_archive=Path(data['source_archive_path']).resolve())
    for name, expected in expected_paths.items():
        if not isinstance(plan.get(name),str) or Path(plan[name]).resolve() != expected:
            raise ValueError('Coupled plan belongs to another '+name)
    hashes = _verify_hashes(plan.get('input_sha256'))
    for name, expected in {**data['input_sha256'],str(Path(config_path).resolve()):digest(config_path)}.items():
        if hashes.get(str(Path(name).resolve())) != expected:
            raise ValueError('Coupled plan omits actual source/candidate/audit evidence: '+name)
    return hashes


def validate_leaf_domain(plan, *, initial_leaf=None):
    """Validate both piecewise-linear bounds, including between their knots."""
    known = {'admitted_leaf_upper_nodes','admitted_leaf_lower_nodes'}
    if any(key.startswith('admitted_leaf_') and key not in known for key in plan):
        raise ValueError('Unknown admitted leaf domain field would be ignored')
    duration = plan['duration_s']
    if type(duration) not in (int,float) or not np.isfinite(duration) or duration <= 0:
        raise ValueError('Finite positive leaf domain duration required')
    bounds = []
    for name,default in (('admitted_leaf_lower_nodes',.08),('admitted_leaf_upper_nodes',.4)):
        raw = plan.get(name)
        nodes = np.array([[0.,default],[duration,default]]) if raw is None else np.asarray(raw,float)
        if (nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes)<2 or not np.isfinite(nodes).all()
                or not np.all(np.diff(nodes[:,0])>0) or nodes[0,0] != 0 or nodes[-1,0] != duration
                or np.any(nodes[:,1]<.08) or np.any(nodes[:,1]>.4)):
            raise ValueError('Explicit finite bounded leaf domain nodes required: '+name)
        bounds.append(nodes)
    lower,upper = bounds
    knots = np.unique(np.r_[lower[:,0],upper[:,0]])
    if np.any(np.interp(knots,lower[:,0],lower[:,1]) >= np.interp(knots,upper[:,0],upper[:,1])):
        raise ValueError('Leaf domain bounds cross or collapse')
    if initial_leaf is not None and not lower[0,1] <= initial_leaf <= upper[0,1]:
        raise ValueError('Explicit leaf domain must contain exact source aperture')
    return lower,upper


def geometry_domain(plan):
    """Canonical receipt contract; absent lower nodes preserve old receipts."""
    validate_leaf_domain(plan)
    domain = dict(elapsed_s=[0,plan['duration_s']],leaf_rad=[.08,.4],
        admitted_leaf_upper_nodes=plan.get('admitted_leaf_upper_nodes'),
        operator_rad=plan['operator_envelope_rad'],latch_m=[-.001,.001])
    if 'admitted_leaf_lower_nodes' in plan:
        domain['admitted_leaf_lower_nodes'] = plan['admitted_leaf_lower_nodes']
    return domain


def validate_map(plan, initial, qa, leaf_qpos, operator_qpos, latch_qpos):
    """Validate raw tensor data before an exact-source evaluation can mask it."""
    if plan.get('joint_names') != JOINT_NAMES:
        raise ValueError('Original explicit coupled joint coordinate order required')
    duration = plan.get('duration_s')
    if type(duration) not in (int,float) or not np.isfinite(duration) or duration <= 0:
        raise ValueError('Finite positive source-audited coupled duration required')
    times, angles, values = [np.asarray(plan.get(name),float)
        for name in ('elapsed_s','leaf_rad','coordinates')]
    if (times.ndim != 1 or angles.ndim != 1 or len(times) < 81 or len(angles) < 17
            or not np.isfinite(times).all() or not np.isfinite(angles).all()
            or not np.all(np.diff(times)>0) or not np.all(np.diff(angles)>0)
            or times[0] != 0. or times[-1] != duration or angles[0] != .08 or angles[-1] != .4
            or np.max(np.diff(times)) > duration/80+1e-12 or np.max(np.diff(angles)) > .02+1e-12
            or values.shape != (len(times),len(angles),6+len(JOINT_NAMES))
            or not np.isfinite(values).all()):
        raise ValueError('Complete finite ordered original-resolution progress/aperture map required')
    source_column = np.flatnonzero(angles == initial[leaf_qpos])
    wanted = np.r_[np.zeros(6),initial[qa]]
    if len(source_column) != 1 or not np.array_equal(values[0,source_column[0]],wanted):
        raise ValueError('Raw map first source column must equal exact attained coordinates')
    operator = np.asarray(plan.get('operator_envelope_rad'),float)
    if (operator.shape != (2,) or not np.isfinite(operator).all()
            or not -.05 <= operator[0] < operator[1] <= .05
            or not operator[0] <= initial[operator_qpos] <= operator[1]
            or plan.get('operator_reference_rad') != initial[operator_qpos]
            or abs(initial[latch_qpos]) > .001):
        raise ValueError('Explicit source-containing original resting operator/latch domain required')
    through = plan.get('follow_handle_through_route_seconds')
    if type(through) not in (int,float) or not np.isfinite(through) or not 0 <= through < 8:
        raise ValueError('Finite handle-follow clock before original eight-second route end required')
    validate_leaf_domain(plan,initial_leaf=float(initial[leaf_qpos]))
    return times,angles,values


def load_isaac_coupled_source(source_config):
    """Fresh detached admission shared by the independent adapter and planner."""
    path = Path(source_config).resolve()
    before = digest(path)
    config = json.loads(path.read_text())
    if config.get('source_engine') != 'isaac-physx':
        raise ValueError('Explicit actual-Isaac withdrawal source required')
    from .isaac_release_source_dispatch import validate_source_kind
    source_kind=validate_source_kind(config.get('source_kind'))
    motor_path = Path(config['motor_contract_path']) if source_kind is not None else Path(config['source_run'])/'trial/motor-contract.json'
    motors = json.loads(motor_path.read_text())
    context = load_withdrawal_source_context(config,motors,measured_rest=True)
    if digest(path) != before:
        raise ValueError('Isaac coupled source configuration changed during admission')
    return context,{str(path):before}


class IsaacCoupledReleaseGeometry(CoupledReleaseGeometry):
    """Private geometry only; inherited evaluation does not write a plant."""
    def __init__(self,path,*,source_config):
        self.path = Path(path).resolve()
        self.source_config_path = Path(source_config).resolve()
        self._file_hashes = {str(p):digest(p) for p in (self.path,self.source_config_path)}
        self.plan = p = json.loads(self.path.read_text())
        if p.get('schema') != SCHEMA:
            raise ValueError('Explicit actual-Isaac source and coupled schema required')
        self.source_context,source_hashes = load_isaac_coupled_source(self.source_config_path)
        self.source_data = data = self.source_context.data
        self._file_hashes.update(source_hashes)
        self._file_hashes.update(validate_envelope_binding(p,data,self.source_config_path))
        self._initialize_geometry(data,p)
        self.verify_inputs()

    def _initialize_geometry(self,data,p,*,scene=None):
        """Private shared initialization; callers must independently admit first."""
        self.scene = scene if scene is not None else LandedLeftScene(data['robot_path'],data['door_xml_path'])
        self.m,self.d = m,d = self.scene.m,self.scene.d
        self.initial = np.asarray(data['initial_qpos'],float)
        if self.initial.shape != (m.nq,) or not np.isfinite(self.initial).all():
            raise ValueError('Complete finite exact normalized actual source required')
        self.rq = self.scene.root
        if not np.isclose(np.linalg.norm(self.initial[self.rq+3:self.rq+7]),1.,atol=1e-8,rtol=0):
            raise ValueError('Admitted source quaternion must already be normalized')
        self.names = list(JOINT_NAMES)
        self.joints = np.array([m.joint('robot/'+name).id for name in self.names])
        if np.any(~np.isin(m.jnt_type[self.joints],[mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE])):
            raise ValueError('Original scalar coupled reference joints required')
        self.qa = m.jnt_qposadr[self.joints]
        self.lq,self.oq,self.bq = [int(m.joint(name).qposadr[0]) for name in
            ('leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide')]
        self.leaf,self.handle = m.body('leaf').id,m.body('leaf_handle').id
        self.rh,self.lh = [m.site('robot/'+hand+'_palm_touch').id for hand in ('rh','lh')]
        self.elapsed,self.angles,coordinates = validate_map(p,self.initial,self.qa,self.lq,self.oq,self.bq)
        self.initial_rotation = Rotation.from_quat(self.initial[self.rq+3:self.rq+7][[1,2,3,0]])
        self.splines = [RectBivariateSpline(self.elapsed,self.angles,coordinates[:,:,i],kx=3,ky=3,s=0)
            for i in range(coordinates.shape[-1])]
        self.times,self.qs,self.positions,rotations = _path_arrays(self.scene,data['screen'],self.initial)
        if self.times[-1] != 8.5:
            raise ValueError('Unchanged evaluator requires the original eight-second RH route')
        self.rotations = Slerp(self.times,Rotation.from_matrix(rotations))
        self.through = p['follow_handle_through_route_seconds']
        d.qpos[:] = self.initial
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        leaf_rotation = d.xmat[self.leaf].reshape(3,3)
        self.localp = leaf_rotation.T@(d.site_xpos[self.lh]-d.xpos[self.leaf])
        self.localr = leaf_rotation.T@d.site_xmat[self.lh].reshape(3,3)
        feet = [m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
        robotbody = int(m.jnt_bodyid[m.joint('robot/free_base').id])
        # Derived measured-state metadata only: nothing is borrowed from a
        # native coupled candidate or trusted as a caller-declared reference.
        self.c = dict(initial_feet_positions=d.xpos[feet].copy().tolist(),
            initial_feet_rotations=d.xmat[feet].reshape(2,3,3).copy().tolist(),
            initial_com=d.subtree_com[robotbody].copy().tolist())
        self.arm = {}
        for side,hand,site in [('right','rh',self.rh),('left','lh',self.lh)]:
            names = [side+'_'+name for name in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+[hand+'_'+name for name in ('WRJ2','WRJ1')]
            joints = np.array([m.joint('robot/'+name).id for name in names])
            self.arm[side] = (site,m.jnt_qposadr[joints],m.jnt_dofadr[joints],m.jnt_range[joints,0]+.01,m.jnt_range[joints,1]-.01)
        self.jp = np.zeros((3,m.nv));self.jr = np.zeros_like(self.jp)

    def verify_inputs(self):
        _verify_hashes(self._file_hashes)

    def lower_angle(self,elapsed):
        nodes = self.plan.get('admitted_leaf_lower_nodes')
        if nodes is None: return float(self.angles[0])
        nodes = np.asarray(nodes,float)
        return float(np.interp(elapsed,nodes[:,0],nodes[:,1]))

    def coordinates(self,elapsed,angle):
        # evaluate() calls this before any private pose writes or references.
        if not np.isfinite([elapsed,angle]).all() or angle < self.lower_angle(elapsed):
            raise ValueError('Measured aperture below screened Isaac coupled envelope')
        return super().coordinates(elapsed,angle)

    def close(self):
        # LandedLeftScene owns only MjModel/MjData, with no renderer or process.
        pass
