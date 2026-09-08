"""Opt-in left-palm replanning from detached actual runtime measurements.

Owns only an unstepped native geometry calculator. Never receives an active
plant, writes its pose, supplies a force, or qualifies physical execution.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import landed_left_audit, landed_left_planner, left_approach_clearance
from .landed_left_planner import LandedLeftScene, digest, make_landed_left_plan

PROFILES=('strict-v1','intermediate-clearance-3mm-v1')


class RuntimeLeftPlanFailure(ValueError):
    """Retain .receipt; no returned passed target is available after rejection."""
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt=copy.deepcopy(receipt)


def _numeric_copy(value):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError('Only detached numeric/JSON data is permitted')
    return json.loads(json.dumps(value,default=convert,allow_nan=False))


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_attained_measurements(scene, measured, *, at_time_s):
    """Verify complete coordinates and the actual door body-origin frame."""
    if set(measured)!={'pose_time_s','root','joints','door_positions','door_body_poses'}:
        raise ValueError('Supply exactly complete numeric attained state and leaf/handle body poses')
    if not np.isfinite(at_time_s) or at_time_s<0 or abs(measured['pose_time_s']-at_time_s)>1e-8:
        raise ValueError('Attained state must share the current requested runtime timestamp')
    if set(measured['door_body_poses'])!={'leaf','leaf_handle'}:
        raise ValueError('Actual leaf and leaf_handle body origins are required')
    state={key:copy.deepcopy(measured[key]) for key in ('pose_time_s','root','joints','door_positions')}
    q=scene.freeze(state);d=mujoco.MjData(scene.m);d.qpos[:]=q;mujoco.mj_kinematics(scene.m,d)
    checks={};errors={}
    for name,pose in measured['door_body_poses'].items():
        value=np.asarray(pose,float)
        if value.shape!=(7,) or not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value[3:]),1.,atol=1e-6,rtol=0):
            raise ValueError('Actual body pose requires xyz + unit wxyz')
        body=scene.m.body(name).id;rotation=Rotation.from_quat(value[[4,5,6,3]]).as_matrix()
        distance=float(np.linalg.norm(value[:3]-d.xpos[body]));rotation_error=float(np.max(np.abs(rotation-d.xmat[body].reshape(3,3))))
        errors[name]=dict(position_error_m=distance,rotation_matrix_max_error=rotation_error)
        # Existing complete-sequence geometry frame tolerances, not a relaxed
        # collision margin. Actual contact/penetration gates remain independent.
        checks[name]=distance<=.003 and rotation_error<=.02
    if not all(checks.values()):raise ValueError('Actual door body origins disagree with the named coordinate frame: '+str(errors))
    return state,dict(passed=True,checks=checks,body_errors=errors,pose_time_s=float(at_time_s),
                      position_tolerance_m=.003,rotation_matrix_tolerance=.02,physics_steps=0)


def plan_attained_left_contact(robot_xml, door_xml, original_targets, measured, *,
                              at_time_s, clearance_profile='strict-v1', subdivisions=20):
    """Return a passed named target config and a source-bound geometric receipt.

    The explicit 3mm option is attempted only after an endpoint-valid strict
    candidate fails its independent path screen. Both attempts remain recorded;
    the same collision/anatomy/contact-time gates apply to both. A rejected
    result raises RuntimeLeftPlanFailure with a detached failure receipt.
    """
    started=time.monotonic()
    receipt=dict(schema='doorbench.runtime-left-plan.v1',passed=False,clearance_profile=clearance_profile,
        requested_time_s=None,physics_steps=0,active_state_writes=0,
        contact_force_claim=False,physical_execution_qualified=False,attempts=[],
        scope='Static actual-state geometric candidate only; activeplant physics and continuous contact remain independently required')
    try:
        if not np.isfinite(at_time_s) or at_time_s<0:raise ValueError('Finite nonnegative runtime time required')
        receipt['requested_time_s']=float(at_time_s)
        if clearance_profile not in PROFILES:raise ValueError('Use an explicitly declared clearance profile')
        if type(subdivisions) is not int or not 2<=subdivisions<=100:raise ValueError('Use2..100 audit subdivisions per original segment')
        robot_xml,door_xml,original_targets=map(Path,(robot_xml,door_xml,original_targets))
        receipt['input_files_sha256']={str(path):_sha(path) for path in (robot_xml,door_xml,original_targets)}
        modules=(landed_left_planner,landed_left_audit,left_approach_clearance)
        sources=[Path(__file__),*(Path(inspect.getfile(module)) for module in modules)]
        receipt['source_sha256']={path.name:_sha(path) for path in sources}
        numeric=_numeric_copy(measured);receipt['attained_state']=numeric;receipt['attained_state_sha256']=digest(numeric)
        scene=LandedLeftScene(robot_xml,door_xml)
        state,frame=validate_attained_measurements(scene,numeric,at_time_s=at_time_s)
        receipt['actual_frame_audit']=frame
        original=json.loads(original_targets.read_text())
        config,strict=make_landed_left_plan(robot_xml,door_xml,original,state,subdivisions=subdivisions)
        receipt['attempts'].append(dict(profile='strict-v1',report=strict,candidate=config))
        selected='strict-v1'
        if not strict['passed'] and strict['fit']['passed'] and clearance_profile=='intermediate-clearance-3mm-v1':
            candidate,fit=left_approach_clearance.add_left_approach_clearance(scene,config,state,distance_m=.003)
            audit=landed_left_audit.audit_landed_left_path(scene,candidate,state,subdivisions=subdivisions)
            candidate['passed']=bool(fit['passed'] and audit['passed'])
            receipt['attempts'].append(dict(profile=clearance_profile,report=dict(passed=candidate['passed'],fit=fit,audit=audit),candidate=candidate))
            config=candidate;selected=clearance_profile
        if config['passed'] is not True:raise ValueError('Actual-state endpoint or independent dense path screen failed')
        if scene.d.time!=0.:raise ValueError('Geometry calculator time advanced unexpectedly')
        config['source']['runtime_attained_state_sha256']=receipt['attained_state_sha256']
        config['source']['runtime_pose_time_s']=float(at_time_s)
        receipt.update(passed=True,selected_profile=selected,subdivisions=subdivisions,
            actual_calculator_time_s=float(scene.d.time),target_content_sha256=digest(config),
            destination_compiled_robot_identity=config['compiled_robot_identity'],
            source_design_identity=config['source_design_identity'],
            door_source_design_identity=config['source']['door_source_design_identity'],
            wall_seconds=time.monotonic()-started)
        # All names/rows, including the exact initial arm and frozen torso, are
        # supplied to the caller; no stale source row is installed implicitly.
        return copy.deepcopy(config),copy.deepcopy(receipt)
    except Exception as error:
        receipt.update(passed=False,error=str(error),wall_seconds=time.monotonic()-started)
        raise RuntimeLeftPlanFailure(str(error),receipt) from error
