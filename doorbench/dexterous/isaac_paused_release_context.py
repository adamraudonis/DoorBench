"""Fresh original-gate planning admission for a genuine paused-live snapshot."""
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .destination_planner_admission import admit_destination_planner
from .isaac_paused_transfer_audit import audit_paused_isaac_transfer, SCHEMA as PHASE_AUDIT_SCHEMA, CHECKS
from .isaac_paused_transfer_source import inspect_paused_isaac_transfer_snapshot, _read, _verify, _encode
from .isaac_release_source import _bind_planning_door
from .landed_left_planner import LandedLeftScene
from .qualified_isaac_grasp import digest


SCHEMA='doorbench.paused-isaac-release-source.v1'


def paused_release_source_paths():
    package=Path(__file__).resolve().parent
    names=('isaac_paused_release_context.py','isaac_paused_transfer_audit.py','isaac_paused_transfer_source.py',
        'isaac_attained_state.py','isaac_prefix_witness.py','motor_contract_identity.py','npz_record_stream.py',
        'qualified_isaac_grasp.py','standing_body_record.py','destination_state_binding.py',
        'continuation_record_stream.py','isaac_pad_audit.py','isaac_release_source.py','isaac_transfer_rest_stop.py',
        'standing_transfer_evaluation.py','grasp_verification.py','destination_planner_admission.py',
        'destination_planning_coordinates.py','destination_return_kinematics.py','landed_left_planner.py')
    return tuple(package/name for name in names)+(package.parents[1]/'scripts/dexterous/plan_local_isaac_transfer.py',)


@dataclass(frozen=True)
class PausedIsaacReleasePlanningContext:
    """Detached geometry data, never a completed run or live resume capability."""
    _admission_json: str

    @property
    def admission(self):return json.loads(self._admission_json)
    @property
    def qpos(self):return np.asarray(self.admission['initial_qpos'],float)
    @property
    def terminal_time_s(self):return self.admission['measured_rest']['terminal_time_s']
    @property
    def sha256(self):return hashlib.sha256(self._admission_json.encode()).hexdigest()
    @property
    def physics_archive_path(self):return Path(self.admission['physics_archive_path'])
    @property
    def state_archive_path(self):return self.physics_archive_path
    @property
    def motor_contract_path(self):return Path(self.admission['motor_contract_path'])
    @property
    def configuration_path(self):return Path(self.admission['configuration_path'])
    def verify_inputs(self):_verify(self.admission['input_sha256'])
    def scene(self):
        self.verify_inputs();a=self.admission;scene=LandedLeftScene(a['robot_path'],a['door_xml_path'])
        if self.qpos.shape!=(scene.m.nq,):raise ValueError('Complete paused-source coordinates required')
        scene.d.qpos[:]=self.qpos;mujoco.mj_kinematics(scene.m,scene.d)
        return scene


def admit_paused_isaac_release_context(snapshot_path,*,robot,door_xml,door_usd,
                                      profile='volar-phalange-v1',phase_audit_path=None):
    if profile!='volar-phalange-v1':raise ValueError('Original prospectively declared volar source required')
    inspected=inspect_paused_isaac_transfer_snapshot(snapshot_path);s=inspected.inspection
    phase=audit_paused_isaac_transfer(snapshot_path)
    if (phase.get('passed') is not True or phase.get('schema')!=PHASE_AUDIT_SCHEMA
            or phase.get('source_kind')!=s['source_kind'] or phase.get('source_engine')!='isaac-physx'
            or phase.get('snapshot_sha256')!=s['snapshot_sha256'] or phase.get('snapshot_path')!=s['snapshot_path']
            or phase.get('live_pause')!=s['live_pause'] or phase.get('time_s')!=s['live_pause']['epoch_s']
            or phase.get('state_sha256')!=s['source_state_sha256']
            or phase.get('physical_intervals')!=s['core_intervals'] or phase.get('raw_intervals')!=s['core_intervals']
            or phase.get('invalid_loaded_patches')!=0 or phase.get('independent_raw_contact_audit_complete') is not True
            or phase.get('episode_complete') is not False or phase.get('authorized_stages')!=0
            or set(phase.get('checks',{}))!=CHECKS or any(v is not True for v in phase['checks'].values())):
        raise ValueError('Fresh full actual transfer-prefix qualification required')
    hashes=phase['input_sha256'].copy();paths={k:Path(v) for k,v in s['evidence_paths'].items()}
    if any(hashes.get(name)!=expected for name,expected in s['input_sha256'].items()):
        raise ValueError('Phase qualification omits original snapshot input bindings')
    if phase_audit_path is not None:
        audit=Path(phase_audit_path).resolve()
        if _read(audit)!=phase:raise ValueError('Saved paused phase audit does not reproduce freshly')
        hashes[str(audit)]=digest(audit)
    cfg,motors,provenance=[_read(paths[k]) for k in ('configuration','motor_contract','provenance')]
    report=_read(paths['phase_report'])
    robot,door_xml,door_usd=[Path(p).resolve() for p in (robot,door_xml,door_usd)]
    if (digest(robot)!=s['measured_state_binding']['robot_source_sha256']
            or digest(door_usd)!=s['measured_state_binding']['door_source_sha256']
            or Path(cfg['args']['native_robot']).resolve()!=robot
            or Path(cfg['args']['door_usd']).resolve()!=door_usd):
        raise ValueError('Actual paused original robot/USD source identity differs')
    if paths['transfer_route']!=Path(snapshot_path).resolve().parent/'standing-transfer-route.json':
        raise ValueError('Captured original transfer route must retain its explicit filename')
    asset_binding=_bind_planning_door(Path(snapshot_path).resolve().parent,cfg,report,provenance['files'],robot,door_xml,hashes)
    extracted=dict(binding=s['measured_state_binding'],measured_bodies=s['measured_bodies'])
    scene=LandedLeftScene(robot,door_xml)
    measured,coordinates=admit_destination_planner(scene.m,extracted,motor_contract=motors,door_source_sha256=digest(door_usd))
    if coordinates.get('passed') is not True:raise ValueError('Original destination kinematic admission must pass')
    scene.d.qpos[:]=measured.qpos;mujoco.mj_kinematics(scene.m,scene.d)
    rest=copy.deepcopy(phase['measured_rest']);raw=phase['endpoint_pad']['raw_evidence']
    point_error=body_error=rotation_error=0.
    for contact in rest['endpoint_contacts']:
        original=contact['body'];name=original.rsplit('/',1)[-1]
        match=re.fullmatch(r'rh_(ff|mf|rf|lf|th)(distal|middle|proximal)',name)
        if match is None or match[1]!=contact['digit']:raise ValueError('Original explicit material body required')
        body=scene.m.body('robot/'+name).id;local=np.asarray(contact['body_position_m'],float)
        position=np.asarray(contact['position'],float);pose=np.asarray(raw['body_transforms_xyzw'][original],float)
        if (local.shape!=(3,) or position.shape!=(3,) or pose.shape!=(7,)
                or not np.isfinite(np.r_[local,position,pose]).all() or abs(np.linalg.norm(pose[3:])-1)>2e-6):
            raise ValueError('Original finite material/body frames required')
        rotation=scene.d.xmat[body].reshape(3,3)
        p=float(np.linalg.norm(scene.d.xpos[body]+rotation@local-position))
        b=float(np.linalg.norm(scene.d.xpos[body]-pose[:3]))
        a=float(Rotation.from_matrix(rotation@Rotation.from_quat(pose[3:]).as_matrix().T).magnitude())
        if max(p,b)>2e-6 or a>2e-6:raise ValueError('Original2µm/2µrad material-frame admission failed')
        point_error=max(point_error,p);body_error=max(body_error,b);rotation_error=max(rotation_error,a)
        contact['isaac_body_path']=original;contact['body']='robot/'+name
    for path in (robot,door_xml,door_usd,*paused_release_source_paths()):
        actual=digest(path)
        if str(path) in hashes and hashes[str(path)]!=actual:raise ValueError('Paused source changed before final admission')
        hashes[str(path)]=actual
    _verify(hashes)
    result=dict(schema=SCHEMA,source_kind=s['source_kind'],source_engine='isaac-physx',
        source_run=phase['source_run'],snapshot_path=s['snapshot_path'],snapshot_sha256=s['snapshot_sha256'],
        source_state_sha256=s['source_state_sha256'],
        live_pause=s['live_pause'],source_qualification=phase,grasp_profile=profile,
        coordinate_admission=coordinates,planning_asset_binding=asset_binding,
        material_frame_admission=dict(passed=True,maximum_position_error_m=point_error,
            maximum_body_position_error_m=body_error,maximum_body_rotation_error_rad=rotation_error,
            position_limit_m=2e-6,rotation_limit_rad=2e-6),
        measured_rest=rest,initial_qpos=measured.qpos.tolist(),
        robot_path=str(robot),door_xml_path=str(door_xml),door_usd_path=str(door_usd),
        physics_archive_path=str(paths['physics']),motor_contract_path=str(paths['motor_contract']),
        configuration_path=str(paths['configuration']),observer_state_path=str(paths['observer_state']),
        observer_state=_read(paths['observer_state']),
        phase_audit_path=None if phase_audit_path is None else str(Path(phase_audit_path).resolve()),
        episode_complete=False,whole_task_qualified=False,live_process_state_verified=False,
        resume_authorized=False,authorized_stages=0,physics_steps=0,source_sample_playback=0,
        active_state_writes=0,input_sha256=hashes,
        scope='Original independently qualified paused transfer prefix plus original material/body kinematics. Detached release planning only; exact unchanged live-process/controller handshake remains mandatory.')
    _verify(hashes)
    return PausedIsaacReleasePlanningContext(_encode(result))
