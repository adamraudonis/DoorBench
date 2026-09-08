"""Scripted arm motion over sensor-feedback balance; separate six-second scope.

The runtime has only fixed robot calibration, a declared joint schedule, local
clock and sensor packets. Actual root/contact/joint records belong exclusively
to the independent evaluator. This is not a learned or vision-based door policy.
"""
import copy, hashlib, json
from pathlib import Path
import numpy as np
from .arm_balance_schedule import scripted_goals, validate_schedule
from .sensor_arm_balance import SensorArmBalanceController
from .sensor_balance_runtime import SensorBalanceRuntime, ROW_FIELDS, _number
from .sensor_contract import validate_actor_packet

ARM_PROTOCOL='doorbench.sensor-scripted-arm-balance-6s.v1'
ARM_NAMES=('left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow',
    'right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow',
    'lh_WRJ2','lh_WRJ1','rh_WRJ2','rh_WRJ1')


def read_schedule(value):
    if type(value) is dict:
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    else:raw=Path(value).read_bytes()
    schedule=json.loads(raw)
    validate_schedule(schedule,ARM_NAMES,6.)
    if (type(schedule['scope']) is not str or not schedule['scope'] or
            not all(_number(schedule[k]) for k in ('start_s','outward_seconds','hold_seconds','return_seconds','duration_s')) or
            not schedule['deltas_rad'] or not all(_number(v) and abs(v)>0 for v in schedule['deltas_rad'].values())):
        raise ValueError('Require an explicit finite nonzero scripted arm schedule')
    return schedule,hashlib.sha256(raw).hexdigest()


class SensorArmBalanceRuntime(SensorBalanceRuntime):
    """Return original61 motor forces; high-level goals follow a frozen schedule."""
    def __init__(self,robot_xml,motors,layout,calibration_json,schedule_json):
        # Reuse the exact static robot/motor/calibration boundary. Only the new
        # high-level arm controller is installed; no active simulator is supplied.
        super().__init__(robot_xml,motors,layout,calibration_json)
        calibration=copy.deepcopy(calibration_json) if type(calibration_json) is dict else json.loads(Path(calibration_json).read_bytes())
        self._schedule,self.schedule_sha256=read_schedule(schedule_json)
        self._posture=copy.deepcopy(calibration['desired_posture'])
        self._controller=SensorArmBalanceController(robot_xml,copy.deepcopy(motors),copy.deepcopy(layout),self._posture,
            physics_dt_s=.002,gravity_correction=calibration['gravity_correction'])
        self.goal_names=tuple(self._controller.goal_names)
        if set(self.goal_names)!=set(ARM_NAMES):raise ValueError('Unexpected arm/wrist command names')
        self._active=False

    def force(self,packet,now_s):
        if not self._active:raise ValueError('reset_episode() is required before arm balance inference')
        try:
            if not _number(now_s) or not 0<=now_s<6.:raise ValueError('Require the declared six-second local command clock')
            validate_actor_packet(packet,self._shapes,61)
            if not np.allclose(packet['previous_action'],self._previous,atol=2e-7,rtol=0):
                raise ValueError('Previous action must be owned by this runtime episode')
            goals=scripted_goals(float(now_s),self._posture,self.goal_names,self._schedule)
            force,info=self._controller.force({k:v.copy() for k,v in packet.items()},now_s=float(now_s),joint_goals=goals)
            force=np.asarray(force,float)
            if (force.shape!=(61,) or not np.isfinite(force).all() or np.any(force<self._caps[:,0]) or np.any(force>self._caps[:,1])):
                raise ValueError('Arm balance exceeded the original motor contract')
            self._info=json.loads(json.dumps(info,allow_nan=False))
            self._previous=(force/self._caps[:,1]).astype(np.float32)
            return force.copy()
        except BaseException:
            self._active=False
            raise


def evaluate_sensor_arm_balance(rows,physics_checks,*,initial_arm_joint_position,schedule,
                                physics_dt_s=.002,expected_duration_s=6.):
    """Score actual3000 post-step samples using their preceding applied goals.

Initial angles are actual reset measurements used only for motion-excursion
scoring. Recorded goals must reproduce the frozen schedule from the first
calibrated command. No actual state enters the runtime through this evaluator.
    """
    if not _number(physics_dt_s) or physics_dt_s!=.002 or not _number(expected_duration_s) or expected_duration_s!=6.:
        raise ValueError('The frozen arm protocol requires six seconds at2ms')
    errors=[];parsed=[];names=ARM_NAMES
    try:schedule,schedule_sha=read_schedule(schedule)
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        errors.append('Invalid frozen schedule: '+str(exc));schedule=None;schedule_sha=None
    initial_valid=(type(initial_arm_joint_position) is dict and set(initial_arm_joint_position)==set(names)
        and all(_number(v) for v in initial_arm_joint_position.values()))
    if not initial_valid:errors.append('Require exact12 finite actual initial arm/wrist angles')
    physics_valid=(type(physics_checks) is dict and bool(physics_checks)
        and all(type(k) is str and type(v) is bool for k,v in physics_checks.items()))
    if not physics_valid:errors.append('Expected explicit nonempty original physical checks')
    if type(rows) not in (list,tuple):errors.append('Actual step evidence must be a sequence');rows=[]
    for i,row in enumerate(rows):
        try:
            if type(row) is not dict or set(row)!=ROW_FIELDS|{'actual_arm_joint_position'}:raise ValueError('Incomplete actual arm step fields')
            t=row['time_s'];tilt=row['torso_tilt_deg'];count=row['hand_contact_count'];info=row['controller_info']
            root=np.asarray(row['root13_actororigin'],float);feet=np.asarray(row['foot_floor_loads'],float);actual=row['actual_arm_joint_position']
            if (not _number(t) or not _number(tilt) or tilt<0 or root.shape!=(13,) or feet.shape!=(2,) or
                    not np.isfinite(np.r_[root,feet]).all() or np.any(feet<0) or
                    not np.isclose(np.linalg.norm(root[3:7]),1.,atol=1e-5,rtol=0) or
                    not isinstance(count,(int,np.integer)) or isinstance(count,(bool,np.bool_)) or count<0):
                raise ValueError('Malformed actual numeric pose/contact evidence')
            if type(actual) is not dict or set(actual)!=set(names) or not all(_number(v) for v in actual.values()):
                raise ValueError('Require exact12 finite actual arm/wrist angles')
            if (type(info) is not dict or info.get('controller')!='sensor_arm_balance_v1' or
                    type(info.get('cold_start')) is not bool or type(info.get('qp_failures')) is not int or info['qp_failures']<0 or
                    not _number(info.get('calculator_time_s')) or type(info.get('qp_status')) is not str or
                    type(info.get('goal_joint_names')) is not list or len(info['goal_joint_names'])!=12 or set(info['goal_joint_names'])!=set(names)):
                raise ValueError('Missing preceding arm-balance controller evidence')
            values=info.get('goal_joint_position_rad')
            if type(values) is not list or len(values)!=12 or not all(_number(v) for v in values):raise ValueError('Malformed applied joint goals')
            goals=dict(zip(info['goal_joint_names'],values))
            parsed.append((float(t),root,float(tilt),feet,count,info,actual,goals))
        except (KeyError,TypeError,ValueError,OverflowError) as exc:errors.append(f'Row{i}: {exc}')
    valid=bool(parsed) and len(parsed)==len(rows) and not errors
    times=np.array([r[0] for r in parsed]);complete=valid and len(parsed)==3000 and np.allclose(times,np.arange(1,3001)*.002,atol=1e-8,rtol=0)
    tail=[r for r in parsed if r[0]>=5.-1e-9]
    tracking=max((abs(r[6][n]-r[7][n]) for r in parsed for n in names),default=None)
    goals_match=False;excursions={};motion=False
    if parsed and schedule is not None:
        base=parsed[0][7]
        goals_match=all(all(abs(r[7][n]-q[n])<1e-9 for n in names)
            for i,r in enumerate(parsed) for q in [scripted_goals(i*.002,base,names,schedule)])
        if initial_valid:
            excursions={n:max(abs(r[6][n]-initial_arm_joint_position[n]) for r in parsed) for n in names}
            motion=all(excursions[n]>.7*abs(delta) for n,delta in schedule['deltas_rad'].items())
    checks=dict(actual_step_records_valid=valid,original_physics_checks_valid=physics_valid,
        original_physics_all_passed=physics_valid and all(physics_checks.values()),complete_6s_at_2ms=complete,
        lowered_height=bool(parsed) and all(.82<=r[1][2]<=.92 for r in parsed),torso_upright=bool(parsed) and all(r[2]<=12 for r in parsed),
        final_quiet=bool(tail) and all(np.linalg.norm(r[1][7:9])<.03 and np.linalg.norm(r[1][10:13])<.05 for r in tail),
        final_both_feet=bool(tail) and all(min(r[3])>30 for r in tail),hands_away=bool(parsed) and all(r[4]==0 for r in parsed),
        all_qp_solved=bool(parsed) and all(r[5]['qp_failures']==0 and r[5]['qp_status']=='solved' for r in parsed),
        estimator_never_stepped=bool(parsed) and all(r[5]['calculator_time_s']==0. for r in parsed),
        cold_start_at_t0=bool(parsed) and abs(parsed[0][0]-.002)<1e-8 and parsed[0][5]['cold_start'] is True and all(r[5]['cold_start'] is False for r in parsed[1:]),
        actual_initial_angles_valid=initial_valid,applied_goals_match_schedule=goals_match,
        arm_tracking=tracking is not None and tracking<.04,physical_arm_motion_delivered=motion)
    checks={k:bool(v) for k,v in checks.items()}
    return dict(schema=ARM_PROTOCOL,scope='Sensor-feedback balance with explicit scripted arm/wrist motion; no learned policy, vision or door task result',
        passed=all(checks.values()),checks=checks,original_physics_checks=copy.deepcopy(physics_checks) if physics_valid else {},
        expected_duration_s=6.,physics_dt_s=.002,physics_steps=len(rows),valid_records=len(parsed),duration_s=None if not len(times) else float(times[-1]),
        errors=errors,max_torso_tilt_deg=max((r[2] for r in parsed),default=None),
        height_range_m=None if not parsed else [min(r[1][2] for r in parsed),max(r[1][2] for r in parsed)],
        maximum_arm_tracking_error_rad=tracking,measured_arm_excursions_rad=excursions,
        schedule_sha256=schedule_sha,initial_arm_joint_position=copy.deepcopy(initial_arm_joint_position) if initial_valid else None,
        evaluator_state_is_actor_input=False,checkpoint_evaluated=False,
        limitation='This moving-arm support evaluation does not test grasping, object interaction, locomotion or a learned high-level policy.')
