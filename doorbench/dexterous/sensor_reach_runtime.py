"""Portable eleven-second joint-only reach over sensor-feedback balance.

Runtime input is a numeric sensor packet and local clock. The static route is
separately projected from a reference before launch; no root or door fields are
accepted here. Actual-state qualification lives in sensor_reach_evaluation.py.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from .reach_balance_schedule import ReferenceReachSchedule
from .sensor_balance_runtime import SensorBalanceRuntime,_number
from .sensor_contract import validate_actor_packet
from .sensor_reach_balance import SensorReachBalanceController

PROTOCOL_SCHEMA='doorbench.sensor-reach-runtime.v1'
ROUTE_SCHEMA='doorbench.joint-only-reach-route.v1'
PARAMETER_VALUES=dict(schema='doorbench.scripted-reach-balance.v2',allow_torso_yaw=True,
    stop_fraction=.45,start_s=1.,reach_seconds=8.,settle_seconds=2.,duration_s=11.,
    maximum_goal_speed_radps=1.5,tracking_profile='motor-transmission-v2',
    finger_impedance_multiplier=4.,finger_velocity_damping=.05,finger_target_velocity_damping=True,
    feedback_profile='finger-moving-reference-damping-v3')


def _read(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode() if type(value) is dict else Path(value).read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def reach_names(joint_names):
    if type(joint_names) not in (tuple,list) or len(joint_names)!=69 or len(set(joint_names))!=69:
        raise ValueError('Require the unique69-joint robot contract')
    names=tuple(n for n in joint_names if n=='torso' or n.startswith('rh_') or
        (n.startswith('right_') and not any(k in n for k in ('hip_','knee','ankle'))))
    if len(names)!=30:raise ValueError('Require the declared30-joint torso/right-arm/right-hand scope')
    return names


def validate_parameters(parameters):
    if type(parameters) is not dict or set(parameters)!=set(PARAMETER_VALUES)|{'scope','source_reference_sha256'}:
        raise ValueError('Require the exact qualified reach parameter schema')
    for key,wanted in PARAMETER_VALUES.items():
        value=parameters[key]
        if isinstance(wanted,bool):valid=type(value) is bool and value==wanted
        elif isinstance(wanted,(int,float)):valid=_number(value) and value==wanted
        else:valid=type(value) is str and value==wanted
        if not valid:raise ValueError('Changed qualified reach parameter: '+key)
    if type(parameters['scope']) is not str or not parameters['scope']:
        raise ValueError('An explicit scripted reach scope is required')
    value=parameters['source_reference_sha256']
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Require original source reference identity')


def prepare_reach_runtime_inputs(reference_path,schedule_path,motors,route_output,protocol_output):
    """Offline projection only: strip root, door, left-body and leg trajectory data."""
    ref,ref_sha=_read(reference_path);parameters,schedule_sha=_read(schedule_path);validate_parameters(parameters)
    if parameters['source_reference_sha256']!=ref_sha:raise ValueError('Schedule references different source bytes')
    names=reach_names(motors['joint_names']);source=ref['acquisition']['joint_names'];path=np.asarray(ref['acquisition']['path_qpos'],float)
    if (source!=motors['joint_names'] or path.shape!=(401,69) or not np.isfinite(path).all()):
        raise ValueError('Require the frozen401-sample named joint source')
    route=dict(schema=ROUTE_SCHEMA,source_reference_sha256=ref_sha,joint_names=list(names),
        path_joint_position_rad=path[:,[source.index(n) for n in names]].tolist())
    route_raw=(json.dumps(route,sort_keys=True,separators=(',',':'))+'\n').encode()
    protocol=dict(schema=PROTOCOL_SCHEMA,source_schedule_sha256=schedule_sha,
        joint_route_sha256=hashlib.sha256(route_raw).hexdigest(),parameters=parameters)
    for path in (route_output,protocol_output):
        if Path(path).exists():raise FileExistsError('Preserve previous frozen runtime inputs')
    Path(route_output).parent.mkdir(parents=True,exist_ok=True)
    with Path(route_output).open('xb') as f:f.write(route_raw)
    Path(protocol_output).parent.mkdir(parents=True,exist_ok=True)
    with Path(protocol_output).open('x') as f:f.write(json.dumps(protocol,indent=2)+'\n')
    return dict(source_reference_sha256=ref_sha,source_schedule_sha256=schedule_sha,
        joint_route_sha256=protocol['joint_route_sha256'],protocol_sha256=hashlib.sha256(Path(protocol_output).read_bytes()).hexdigest())


def load_reach_inputs(protocol_json,joint_route_json,joint_names,desired_posture):
    protocol,protocol_sha=_read(protocol_json);joint_route,route_sha=_read(joint_route_json)
    if type(protocol) is not dict or set(protocol)!={'schema','source_schedule_sha256','joint_route_sha256','parameters'} or protocol['schema']!=PROTOCOL_SCHEMA:
        raise ValueError('Require the exact portable reach protocol schema')
    source_sha=protocol['source_schedule_sha256']
    if type(source_sha) is not str or len(source_sha)!=64 or any(c not in '0123456789abcdef' for c in source_sha):raise ValueError('Require source schedule identity')
    parameters=protocol['parameters'];validate_parameters(parameters)
    if protocol['joint_route_sha256']!=route_sha:raise ValueError('Joint-only route hash differs from protocol')
    if type(joint_route) is not dict or set(joint_route)!={'schema','source_reference_sha256','joint_names','path_joint_position_rad'} or joint_route['schema']!=ROUTE_SCHEMA:
        raise ValueError('Runtime route accepts joint data only; no root or door fields')
    names=reach_names(joint_names);path=np.asarray(joint_route['path_joint_position_rad'],float)
    if (joint_route['source_reference_sha256']!=parameters['source_reference_sha256'] or joint_route['joint_names']!=list(names) or
            path.shape!=(401,30) or not np.isfinite(path).all()):raise ValueError('Changed projected joint route or source ordering')
    if not np.array_equal(path[0],np.array([desired_posture[n] for n in names])):
        raise ValueError('Reach starts from different fixed joint calibration')
    route=ReferenceReachSchedule(names,names,path,stop_fraction=parameters['stop_fraction'],start_s=parameters['start_s'],
        reach_seconds=parameters['reach_seconds'],settle_seconds=parameters['settle_seconds'])
    if route.duration_s!=11. or route.sampled_maximum_goal_speed()>parameters['maximum_goal_speed_radps']:
        raise ValueError('Reach duration or planned slew differs from frozen protocol')
    return parameters,route,protocol_sha,route_sha


class SensorReachBalanceRuntime(SensorBalanceRuntime):
    """Fixed30-joint/26-motor high-level route, original61 capped force output."""
    def __init__(self,robot_xml,motors,layout,calibration_json,reach_protocol_json,joint_route_json):
        super().__init__(robot_xml,motors,layout,calibration_json)
        calibration,_=_read(calibration_json)
        parameters,self._route,self.protocol_sha256,self.joint_route_sha256=load_reach_inputs(
            reach_protocol_json,joint_route_json,motors['joint_names'],calibration['desired_posture'])
        self._controller=SensorReachBalanceController(robot_xml,copy.deepcopy(motors),copy.deepcopy(layout),calibration['desired_posture'],
            physics_dt_s=.002,gravity_correction=calibration['gravity_correction'],allow_torso_yaw=True,
            maximum_goal_speed_radps=parameters['maximum_goal_speed_radps'],finger_impedance_multiplier=parameters['finger_impedance_multiplier'],
            finger_velocity_damping=parameters['finger_velocity_damping'],finger_target_velocity_damping=True)
        self.goal_names=tuple(self._controller.goal_names)
        self.goal_motor_names=tuple(self._controller.balance.actions[i] for i in self._controller.arm_motors)
        if self.goal_names!=self._route.names or len(self.goal_motor_names)!=26:raise ValueError('Changed reach motor/joint scope')
        self.duration_s=11.;self._active=False

    def force(self,packet,now_s):
        if not self._active:raise ValueError('reset_episode() is required before reach inference')
        try:
            if not _number(now_s) or not 0<=now_s<self.duration_s:raise ValueError('Require the declared eleven-second local clock')
            validate_actor_packet(packet,self._shapes,61)
            if not np.allclose(packet['previous_action'],self._previous,atol=2e-7,rtol=0):raise ValueError('Previous action must be owned by this runtime episode')
            force,info=self._controller.force({k:v.copy() for k,v in packet.items()},now_s=float(now_s),joint_goals=self._route.goals(float(now_s)))
            force=np.asarray(force,float)
            if (force.shape!=(61,) or not np.isfinite(force).all() or np.any(force<self._caps[:,0]) or np.any(force>self._caps[:,1])):
                raise ValueError('Reach controller exceeded original motor force limits')
            self._info=json.loads(json.dumps(info,allow_nan=False));self._previous=(force/self._caps[:,1]).astype(np.float32)
            return force.copy()
        except BaseException:
            self._active=False
            raise
