"""Sensor-only stationary-balance runtime and separate physical qualification.

The runtime accepts static robot calibration and numeric sensor packets. The
independent evaluator receives actual simulator measurements; those measurements
must never be passed to the runtime. Neither is an acquisition policy/checkpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .motor_contract_identity import motor_contract_fingerprint
from .sensor_actor import ActorDimensions
from .sensor_contract import validate_actor_packet

CALIBRATION_SCHEMA = 'doorbench.sensor-balance-calibration.v1'
BALANCE_PROTOCOL = 'doorbench.sensor-stationary-balance-5s.v1'
CALIBRATION_FIELDS = {'schema','scope','robot_xml_sha256','motor_contract_sha256',
    'physics_dt_s','gravity_correction','initial_orientation','desired_posture','provenance'}
ROW_FIELDS = {'time_s','root13_actororigin','torso_tilt_deg','foot_floor_loads',
    'hand_contact_count','controller_info'}


def _number(value):
    return (isinstance(value,(int,float,np.integer,np.floating)) and
            not isinstance(value,(bool,np.bool_)) and np.isfinite(value))


def _json_copy(value):
    return json.loads(json.dumps(value,allow_nan=False))


class SensorBalanceRuntime:
    """Delegate the declared sensor controller, returning original61 motor forces.

    Explicit reset is required. Any rejected inference ends the current runtime
    episode; the caller must reset before continuing. There is no teacher fallback.
    ``last_info`` is controller diagnostics, never an additional actor input.
    """
    def __init__(self,robot_xml,motors,layout,calibration_json):
        robot_xml=Path(robot_xml)
        if type(calibration_json) is dict:
            calibration=_json_copy(calibration_json)
            raw=json.dumps(calibration,sort_keys=True,separators=(',',':')).encode()
        else:
            raw=Path(calibration_json).read_bytes();calibration=json.loads(raw)
        if type(calibration) is not dict or set(calibration)!=CALIBRATION_FIELDS or calibration['schema']!=CALIBRATION_SCHEMA:
            raise ValueError('Require exact declared stationary-balance calibration schema')
        if type(motors) is not dict or type(layout) is not dict:
            raise ValueError('Require actual motor and sensor calibration dictionaries')
        robot_sha=hashlib.sha256(robot_xml.read_bytes()).hexdigest()
        fingerprint=motor_contract_fingerprint(motors)
        if (calibration['robot_xml_sha256']!=robot_sha or motors.get('source_xml_sha256')!=robot_sha or
                layout.get('robot_xml_sha256')!=robot_sha or calibration['motor_contract_sha256']!=fingerprint):
            raise ValueError('Balance calibration differs from the actual robot/motor identity')
        names=motors.get('joint_names')
        if (type(names) is not list or len(names)!=69 or any(type(n) is not str or not n for n in names) or len(set(names))!=69):
            raise ValueError('Require69 unique named robot joints')
        posture=calibration['desired_posture']
        if type(posture) is not dict or set(posture)!=set(names) or not all(_number(v) for v in posture.values()):
            raise ValueError('Desired posture must contain exactly69 finite joint angles')
        if (not _number(calibration['physics_dt_s']) or calibration['physics_dt_s']!=.002 or
                not _number(calibration['gravity_correction']) or not 0<=calibration['gravity_correction']<=2 or
                calibration['initial_orientation']!='upright; yaw/XY arbitrary robot-local gauge' or
                type(calibration['scope']) is not str or not calibration['scope']):
            raise ValueError('Invalid stationary-balance timing/orientation calibration')
        provenance=calibration['provenance']
        if (type(provenance) is not dict or set(provenance)!={'source_reference_sha256','selection'} or
                type(provenance['source_reference_sha256']) is not str or not re.fullmatch('[0-9a-f]{64}',provenance['source_reference_sha256']) or
                type(provenance['selection']) is not str or not provenance['selection']):
            raise ValueError('Require bounded joint-posture provenance without task state')
        # The controller's previous-command convention is symmetric original
        # motor force/cap. Do not silently reinterpret asymmetric replacements.
        try:self._caps=np.asarray([a['force_range'] for a in motors['actuators']],float)
        except (KeyError,TypeError,ValueError) as exc:raise ValueError('Invalid motor limits') from exc
        if (self._caps.shape!=(61,2) or not np.isfinite(self._caps).all() or
                np.any(self._caps[:,1]<=0) or not np.array_equal(self._caps[:,0],-self._caps[:,1])):
            raise ValueError('Require original61 symmetric native motor force limits')
        dimensions=ActorDimensions(tactile=layout['tactile_dimension'])
        self._shapes=dimensions.shapes
        # Import only after the static boundary passes; no simulator is supplied.
        from .sensor_balance import SensorBalanceController
        self._controller=SensorBalanceController(robot_xml,_json_copy(motors),_json_copy(layout),
            posture,physics_dt_s=.002,gravity_correction=calibration['gravity_correction'])
        self.calibration_sha256=hashlib.sha256(raw).hexdigest()
        self.motor_contract_sha256=fingerprint
        self.robot_xml_sha256=robot_sha
        self.physics_dt_s=.002
        self._active=False;self._previous=np.zeros(61,np.float32);self._info={}

    @property
    def previous_action(self):return self._previous.copy()

    @property
    def last_info(self):return copy.deepcopy(self._info)

    def reset_episode(self):
        self._active=False
        self._controller.reset_episode()
        self._previous=np.zeros(61,np.float32);self._info={};self._active=True

    def force(self,packet,now_s):
        if not self._active:raise ValueError('reset_episode() is required before balance inference')
        try:
            if not _number(now_s) or now_s<0:raise ValueError('Require a finite nonnegative local clock')
            validate_actor_packet(packet,self._shapes,61)
            if not np.allclose(packet['previous_action'],self._previous,atol=2e-7,rtol=0):
                raise ValueError('Previous action must be owned by this runtime episode')
            force,info=self._controller.force({k:v.copy() for k,v in packet.items()},now_s=float(now_s))
            force=np.asarray(force,float)
            if (force.shape!=(61,) or not np.isfinite(force).all() or
                    np.any(force<self._caps[:,0]) or np.any(force>self._caps[:,1])):
                raise ValueError('Balance controller exceeded the original motor contract')
            if type(info) is not dict:raise ValueError('Require detached controller diagnostics')
            self._info=_json_copy(info)
            self._previous=(force/self._caps[:,1]).astype(np.float32)
            return force.copy()
        except BaseException:
            self._active=False
            raise


def evaluate_sensor_balance(rows,physics_checks,*,physics_dt_s=.002,expected_duration_s=5.):
    """Score actual post-step records under the fixed5s stationary protocol.

    Root13 uses xyz+unit wxyz+world linear velocity+world angular velocity at
    the actor origin. Each controller_info belongs to the preceding command.
    Missing, malformed, short or gapped evidence produces a failed report.
    This is evaluation privilege only; no observed root enters the controller.
    """
    if not _number(physics_dt_s) or physics_dt_s!=.002 or not _number(expected_duration_s) or expected_duration_s!=5.:
        raise ValueError('The frozen stationary protocol requires5 seconds at2ms')
    errors=[];parsed=[]
    physics_valid=(type(physics_checks) is dict and bool(physics_checks) and
                   all(type(k) is str and type(v) is bool for k,v in physics_checks.items()))
    if not physics_valid:errors.append('Expected explicit nonempty original physical checks')
    if type(rows) not in (list,tuple):errors.append('Actual step evidence must be a sequence');rows=[]
    for i,row in enumerate(rows):
        try:
            if type(row) is not dict or set(row)!=ROW_FIELDS:raise ValueError('Incomplete actual step fields')
            t=row['time_s'];tilt=row['torso_tilt_deg'];count=row['hand_contact_count']
            root=np.asarray(row['root13_actororigin'],float);feet=np.asarray(row['foot_floor_loads'],float);info=row['controller_info']
            if (not _number(t) or not _number(tilt) or tilt<0 or root.shape!=(13,) or feet.shape!=(2,) or
                    not np.isfinite(np.r_[root,feet]).all() or np.any(feet<0) or
                    not np.isclose(np.linalg.norm(root[3:7]),1.,atol=1e-5,rtol=0) or
                    not isinstance(count,(int,np.integer)) or isinstance(count,(bool,np.bool_)) or count<0):
                raise ValueError('Malformed actual numeric pose/contact evidence')
            if (type(info) is not dict or info.get('controller')!='sensor_balance_v1' or
                    type(info.get('cold_start')) is not bool or type(info.get('qp_failures')) is not int or
                    info['qp_failures']<0 or not _number(info.get('calculator_time_s')) or type(info.get('qp_status')) is not str):
                raise ValueError('Missing preceding sensor-balance controller evidence')
            parsed.append((float(t),root,float(tilt),feet,int(count),info))
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            errors.append(f'Row{i}: {exc}')
    records_valid=bool(parsed) and len(parsed)==len(rows) and not errors
    times=np.array([r[0] for r in parsed]);n=round(expected_duration_s/physics_dt_s)
    complete=bool(records_valid and len(parsed)==n and np.allclose(times,np.arange(1,n+1)*physics_dt_s,atol=1e-8,rtol=0))
    tail=[r for r in parsed if r[0]>=4.-1e-8]
    checks=dict(actual_step_records_valid=records_valid,original_physics_checks_valid=physics_valid,
        original_physics_all_passed=physics_valid and all(physics_checks.values()),
        complete_5s_at_2ms=complete,
        lowered_height=bool(parsed) and all(.82<=r[1][2]<=.92 for r in parsed),
        torso_upright=bool(parsed) and all(r[2]<=12 for r in parsed),
        final_quiet=bool(tail) and all(np.linalg.norm(r[1][7:9])<.03 and np.linalg.norm(r[1][10:13])<.05 for r in tail),
        final_both_feet=bool(tail) and all(min(r[3])>30 for r in tail),
        hands_away=bool(parsed) and all(r[4]==0 for r in parsed),
        all_qp_solved=bool(parsed) and all(r[5]['qp_failures']==0 and r[5]['qp_status']=='solved' for r in parsed),
        estimator_never_stepped=bool(parsed) and all(r[5]['calculator_time_s']==0. for r in parsed),
        cold_start_at_t0=bool(parsed) and abs(parsed[0][0]-.002)<1e-8 and parsed[0][5]['cold_start'] is True and all(r[5]['cold_start'] is False for r in parsed[1:]))
    checks={k:bool(v) for k,v in checks.items()}
    return dict(schema=BALANCE_PROTOCOL,scope='Sensor-only stationary lowered balance component; no acquisition, opening or traversal result',
        passed=all(checks.values()),checks=checks,original_physics_checks=copy.deepcopy(physics_checks) if physics_valid else {},
        expected_duration_s=5.,physics_dt_s=.002,physics_steps=len(rows),valid_records=len(parsed),
        duration_s=None if not len(times) else float(times[-1]),errors=errors,
        max_torso_tilt_deg=max((r[2] for r in parsed),default=None),
        height_range_m=None if not parsed else [min(r[1][2] for r in parsed),max(r[1][2] for r in parsed)],
        evaluator_state_is_actor_input=False,checkpoint_evaluated=False,
        limitation='This fixed-posture balance evaluation does not test vision, handle grasping, manipulation or locomotion.')
