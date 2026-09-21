"""Explicit actual-Isaac withdrawal admission and motor-controller factory.

No source is replayed into a plant. A separate live core+leaf prefix witness must
authorize the exact stage epoch before this opt-in controller can enter release.
"""
import copy
import json
from pathlib import Path

import numpy as np

from .isaac_coupled_release_geometry import load_isaac_coupled_source,_verify_hashes
from .isaac_coupled_release_reference import IsaacCoupledReleaseReference
from .isaac_prefix_witness import PREFIX_FIELDS
from .isaac_withdrawal_support import (InheritedIsaacPalmSupport,
    ReleaseQualifiedPalmLoadProfile,validate_palm_load_profile_name)
from .motor_contract_identity import motor_contract_fingerprint
from .qualified_isaac_grasp import digest
from .standing_body_record import POSE_CONVENTION
from .withdrawal_source_context import WithdrawalSourceContext


SCHEMA='doorbench.isaac-standing-withdrawal-runtime.v1'
REQUIRED_OPTIONS=dict(capture_returned_motor_command=True,inherit_transfer_support=True,
    left_arm_only=True,left_full_orientation=True,coupled_motion_projection='fixed-poses-v1')
PATH_KEYS=('source_config_path','coupled_envelope_path','coupled_audit_path')


def _document(path):
    path=Path(path).resolve();before=digest(path);config=json.loads(path.read_text())
    allowed={'schema','scope',*REQUIRED_OPTIONS,*PATH_KEYS,
             'source_config_sha256','coupled_envelope_sha256','coupled_audit_sha256',
             'withdrawal_palm_load_profile'}
    if config.get('schema')!=SCHEMA or set(config)-allowed:
        raise ValueError('Explicit minimal Isaac withdrawal runtime schema required')
    for key,expected in REQUIRED_OPTIONS.items():
        if type(config.get(key)) is not type(expected) or config[key]!=expected:
            raise ValueError('Required source-bound Isaac runtime option: '+key)
    if 'withdrawal_palm_load_profile' in config:
        validate_palm_load_profile_name(config['withdrawal_palm_load_profile'])
    hashes={str(path):before}
    for key in PATH_KEYS:
        value=config.get(key)
        if type(value) is not str or not Path(value).is_absolute():
            raise ValueError('Absolute bound runtime evidence path required: '+key)
        evidence=Path(value).resolve();hash_key=key.removesuffix('_path')+'_sha256'
        if digest(evidence)!=config.get(hash_key):raise ValueError('Isaac withdrawal evidence changed: '+key)
        hashes[str(evidence)]=config[hash_key]
    if digest(path)!=before:raise ValueError('Runtime configuration changed during read')
    return config,hashes


class IsaacWithdrawalRuntimeAdmission:
    """Freshly admitted context, private reference and explicit entry gate."""
    def __init__(self,path,motors=None):
        self.path=Path(path).resolve();config,hashes=_document(self.path)
        context,source_hashes=load_isaac_coupled_source(config['source_config_path'])
        data=context.data
        if data.get('source_kind') is not None:
            raise ValueError('Paused live source requires a distinct retained-controller runtime factory')
        if motors is None:
            motors=json.loads((Path(data['source_run'])/'trial/motor-contract.json').read_text())
        if data['motor_contract_sha256']!=motor_contract_fingerprint(motors):
            raise ValueError('Live original motor contract differs from admitted source')
        source_config=json.loads(Path(config['source_config_path']).read_text())
        configuration_path=Path(data['source_run'])/'trial/configuration.json'
        if data['input_sha256'].get(str(configuration_path.resolve()))!=digest(configuration_path):
            raise ValueError('Original transfer settings must be bound by actual source admission')
        recorded=json.loads(configuration_path.read_text())['args']
        target=recorded.get('standing_transfer_support_load_target',4.)
        if (recorded.get('standing_transfer_hybrid_support') is not True
                or type(target) not in (int,float) or not np.isfinite(target) or not 2<target<=8
                or not recorded.get('standing_transfer_route')
                or type(recorded.get('standing_transfer_start_seconds')) not in (int,float)
                or not np.isfinite(recorded['standing_transfer_start_seconds'])
                or not 0<recorded['standing_transfer_start_seconds']<data['start_time_s']):
            raise ValueError('Exact qualified predecessor transfer recipe and bounded palm target required')
        self.predecessor_route=Path(recorded['standing_transfer_route']).resolve()
        self.predecessor_start=float(recorded['standing_transfer_start_seconds'])
        self.inherited_support=None;self.target_N=float(target)
        self.support_profile=config.get('withdrawal_palm_load_profile')
        if self.support_profile is not None:
            ReleaseQualifiedPalmLoadProfile(self.support_profile,target,data['duration_s'])
        self.reference=IsaacCoupledReleaseReference(config,data)
        hashes.update(source_hashes);hashes.update(data['input_sha256']);hashes.update(self.reference.input_sha256)
        hashes[str(configuration_path.resolve())]=digest(configuration_path)
        self.input_sha256=_verify_hashes(hashes)
        data.update(runtime_path=str(self.path),runtime_sha256=digest(self.path),runtime_schema=SCHEMA,
            input_sha256=self.input_sha256.copy(),inherited_support_target_N=float(target),
            inherited_support_mode='qualified-predecessor-palm-only-v1',
            source_config_path=str(Path(config['source_config_path']).resolve()),
            coupled_envelope_path=str(Path(config['coupled_envelope_path']).resolve()),
            coupled_audit_path=str(Path(config['coupled_audit_path']).resolve()))
        if self.support_profile is not None:data['withdrawal_palm_load_profile']=self.support_profile
        self.source_context=WithdrawalSourceContext(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False))
        identity_keys=('schema','source_engine','source_run','screen_path','screen_sha256',
            'audit_path','audit_sha256','measured_rest_transfer','grasp_profile','contact_audit_name',
            'robot_path','door_xml_path','door_usd_path','source_state_sha256','start_time_s','duration_s')
        self.controller_config={**{key:source_config[key] for key in identity_keys},**REQUIRED_OPTIONS,
            'palm_only_support':True,'left_support_target_N':float(target)}
        # The outer document's coupled proof is consumed by the distinct Isaac
        # reference, never relabelled as native constructor evidence.
        self.controller_config.pop('coupled_motion_projection',None)
        self.config=copy.deepcopy(config);self.authorized=False;self.entered=False;self.failure=None
        self.prefix_receipt=None

    @property
    def source_admission(self):return self.source_context.data['source_admission']

    @property
    def start_time(self):return self.source_context.data['start_time_s']

    @property
    def duration(self):return self.source_context.data['duration_s']

    @property
    def runtime_path(self):return self.path

    def bind_predecessor(self,transfer):
        if (self.inherited_support is not None or self.entered
                or Path(transfer.path).resolve()!=self.predecessor_route
                or transfer.start_seconds!=self.predecessor_start):
            raise ValueError('One exact qualified live predecessor route and start required')
        if self.support_profile is None:
            self.inherited_support=InheritedIsaacPalmSupport(transfer,self.target_N)
        else:
            self.inherited_support=InheritedIsaacPalmSupport(transfer,self.target_N,
                profile=self.support_profile,duration_s=self.duration)

    def verify(self,path,motors):
        if Path(path).resolve()!=self.path or motor_contract_fingerprint(motors)!=self.source_context.data['motor_contract_sha256']:
            raise ValueError('Controller constructor differs from its fresh runtime admission')
        _verify_hashes(self.input_sha256)

    def authorize_source_prefix(self,receipt):
        """Accept only the matching completed core+measured-leaf witness."""
        if self.failure is not None:raise ValueError('Isaac release authorization is terminal: '+self.failure)
        try:
            data=self.source_context.data;qualification=data['source_admission']['source_qualification']
            t=data['start_time_s'];count=round(t/.002)
            if (self.authorized or self.entered or type(receipt) is not dict
                    or receipt.get('schema')!='doorbench.live-isaac-withdrawal-prefix-witness.v1'
                    or any(receipt.get(key) is not True for key in ('passed','stage_entry_authorized','prefix_complete'))
                    or receipt.get('failure') is not None
                    or Path(receipt.get('source_run','')).resolve()!=Path(data['source_run']).resolve()
                    or receipt.get('source_terminal_time_s')!=t or receipt.get('source_qualification')!=qualification):
                raise ValueError('One completed exact source core+leaf prefix required')
            core=receipt.get('core',{});leaf=receipt.get('leaf_pose',{})
            if (core.get('schema')!='doorbench.live-isaac-prefix-witness.v1'
                    or core.get('fields')!=list(PREFIX_FIELDS) or core.get('failure') is not None
                    or core.get('source_qualification')!=qualification
                    or Path(core.get('source_run','')).resolve()!=Path(data['source_run']).resolve()
                    or core.get('source_terminal_time_s')!=t
                    or core.get('runtime_motor_contract_sha256')!=data['motor_contract_sha256']):
                raise ValueError('Original exact physical/commanded-motor prefix differs')
            if (leaf.get('pose_convention')!=POSE_CONVENTION
                    or leaf.get('comparison')!='Exact bytes after explicit float64 canonicalization; historical tensor dtype unrecorded'
                    or leaf.get('historical_tensor_dtype_compared') is not False):
                raise ValueError('Explicit historical measured-leaf float64 comparison required')
            for part in (core,leaf):
                if (any(part.get(key) is not True for key in ('passed','stage_entry_authorized','prefix_complete'))
                        or type(part.get('intervals_verified')) is not int or part['intervals_verified']!=count
                        or part.get('intervals_required')!=count or part.get('last_verified_time_s')!=t):
                    raise ValueError('Complete synchronized core and leaf intervals through stage entry required')
            hashes=_verify_hashes(receipt.get('input_sha256'))
            for part in (core,leaf):
                for name,expected in _verify_hashes(part.get('input_sha256')).items():
                    if hashes.get(name)!=expected:raise ValueError('Combined prefix omits its component source hashes')
            for name,expected in qualification['input_sha256'].items():
                if hashes.get(str(Path(name).resolve()))!=expected:
                    raise ValueError('Prefix omits original source qualification evidence')
            _verify_hashes(self.input_sha256)
            self.prefix_receipt=copy.deepcopy(receipt);self.authorized=True
        except Exception as error:
            self.failure=str(error);self.authorized=False
            raise ValueError('Isaac release prefix authorization failed: '+self.failure) from error

    def require_entry(self,t):
        if (self.failure is not None or not self.authorized or self.entered or self.inherited_support is None
                or type(t) not in (int,float) or not np.isfinite(t) or t!=self.source_context.data['start_time_s']):
            self.failure=self.failure or 'Exact authorized one-time source epoch required'
            raise ValueError(self.failure)
        self.inherited_support.begin(t)
        self.entered=True


def admit_isaac_withdrawal_runtime(runtime_path,motors=None):
    """Read-only launcher preflight; no active plant/controller is required."""
    return IsaacWithdrawalRuntimeAdmission(runtime_path,motors)


def withdrawal_runtime_source_paths():
    """Explicit additional runtime/admission sources; callers deduplicate paths."""
    package=Path(__file__).resolve().parent;root=package.parents[1]
    names=('isaac_withdrawal_runtime.py','isaac_withdrawal_support.py','isaac_coupled_release_reference.py',
        'standing_withdrawal.py','withdrawal_motor_capture.py','withdrawal_source_context.py','resting_transfer.py',
        'coupled_release_reference.py','coupled_release_motion.py','coupled_release_geometry.py',
        'isaac_coupled_release_geometry.py','isaac_coupled_release_audit.py','isaac_release_geometry_audit.py',
        'isaac_release_planning.py','isaac_release_source.py','isaac_release_context_reconciliation.py','isaac_pad_audit.py','isaac_prefix_witness.py',
        'landed_left_planner.py','landed_left_audit.py','grasp_verification.py','qualified_isaac_grasp.py',
        'isaac_attained_state.py','standing_body_record.py','motor_contract_identity.py',
        'destination_state_binding.py','destination_planner_admission.py','destination_planning_coordinates.py',
        'destination_return_kinematics.py','release_material_targets.py','attained_arm_tracking.py',
        'attained_hand_tracking.py','return_palm_feedback.py','motor_handoff.py','standing_support_feedback.py',
        'robot_design_identity.py','robot_identity.py','json_record_stream.py',
        'isaac_transfer_rest_audit.py','isaac_transfer_rest_stop.py','locomotion_manipulation.py')
    return tuple(package/name for name in names)+(root/'scripts/dexterous/plan_local_isaac_transfer.py',
        root/'scripts/dexterous/plan_local_standing_transfer.py',root/'scripts/dexterous/audit_standing_ungrip.py')


def create_isaac_withdrawal_controller(standing_transfer,motors,runtime_path):
    """Preserve the predecessor; return a separately gated outer controller."""
    from .resting_transfer import RestingTransferBridge
    from .standing_withdrawal import StandingWithdrawalTeacher
    admission=admit_isaac_withdrawal_runtime(runtime_path,motors)
    admission.bind_predecessor(standing_transfer)
    bridge=RestingTransferBridge(standing_transfer)
    return StandingWithdrawalTeacher(bridge,motors,runtime_path,_isaac_runtime=admission)
