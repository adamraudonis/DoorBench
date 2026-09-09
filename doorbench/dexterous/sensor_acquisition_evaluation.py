"""Independent actual-state qualification for the nineteen-second acquisition component.

This module is evaluation privilege only. Robot-only FK uses recorded initial
and actual body/joint poses; no door model is needed. These values must never be
passed to SensorAcquisitionBalanceRuntime or supplied as sensor-policy observations.
"""
import copy,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from .grasp_verification import scalar_transmission_matrix
from .motor_contract_identity import motor_contract_fingerprint
from .sensor_balance_runtime import ROW_FIELDS,_number
from .sensor_acquisition_runtime import _read,load_acquisition_inputs
from .sensor_acquisition_evidence import evaluate_pad_evidence

ACQUISITION_EVALUATION_SCHEMA='doorbench.sensor-scripted-acquisition-19s.v1'
PHYSICS_FAMILIES=(('joint_stops','joint_limits'),('documented_loopbacks','passive_loopbacks'),
    ('self_collision','nonfoot_collision'),('environment_collision','nonfoot_collision'),
    ('working_hand_collision','nonfoot_collision'),('native_motor_caps','original_motor_caps'),
    ('motor_delivery_matches_command','delivered_motor_forces'),('plant_parameters_unchanged','original_model_unchanged'),
    ('finite',),('complete_physics_steps','complete_requested_duration'))


def _pose(value):
    a=np.asarray(value,float)
    if a.shape!=(13,) or not np.isfinite(a).all() or not np.isclose(np.linalg.norm(a[3:7]),1.,atol=1e-5,rtol=0):
        raise ValueError('Require actual actor-origin xyz,wxyz,world linear/angular velocity')
    return a


def _named(value,names):
    if type(value) is not dict or set(value)!=set(names) or not all(_number(v) for v in value.values()):
        raise ValueError('Require complete finite named actual joint positions')
    return np.array([value[n] for n in names],float)


class AcquisitionEvaluationRobot:
    """Detached robot-only FK and named transmission map; never steps physics."""
    def __init__(self,robot_xml,motors,calibration,protocol,joint_route):
        self.calibration,_=_read(calibration);self.motors=copy.deepcopy(motors)
        robot_sha=hashlib.sha256(Path(robot_xml).read_bytes()).hexdigest();c=self.calibration
        if robot_sha!=motors.get('source_xml_sha256') or c.get('robot_xml_sha256')!=robot_sha or c.get('motor_contract_sha256')!=motor_contract_fingerprint(motors):
            raise ValueError('Evaluator robot/motor/calibration identity mismatch')
        self.names=tuple(motors['joint_names']);self.desired=_named(c['desired_posture'],self.names)
        self.parameters,self.route,self.protocol_sha,self.route_sha=load_acquisition_inputs(protocol,joint_route,self.names,c['desired_posture'])
        self.m=mujoco.MjModel.from_xml_path(str(robot_xml));self.d=mujoco.MjData(self.m);m=self.m
        self.joints=np.array([m.joint(n).id for n in self.names]);self.qa=m.jnt_qposadr[self.joints]
        self.actions=tuple(a['name'] for a in motors['actuators']);self.act=np.array([m.actuator(n).id for n in self.actions])
        if m.nq!=76 or m.nv!=75 or m.nu!=61 or m.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE or len(self.names)!=69:
            raise ValueError('Changed authored free-base robot structure')
        self.matrix=scalar_transmission_matrix(m,self.act,self.joints)
        contract=np.zeros_like(self.matrix)
        for i,a in enumerate(motors['actuators']):
            for name,coef in a['terms'].items():contract[i,self.names.index(name)]=coef
        if not np.array_equal(contract,self.matrix):raise ValueError('Evaluator transmission differs from motor contract')
        self.goal_names=self.route.names;self.indices=np.array([self.names.index(n) for n in self.goal_names])
        self.goal_motors=np.flatnonzero(np.any(abs(self.matrix[:,self.indices])>0,axis=1))
        if len(self.goal_motors)!=26 or np.any(abs(np.delete(self.matrix[self.goal_motors],self.indices,axis=1))>0):
            raise ValueError('Reach motor transmission extends beyond declared scope')
        self.goal_motor_names=tuple(self.actions[i] for i in self.goal_motors)
        self.limited=m.jnt_limited[self.joints].astype(bool);self.joint_limits=m.jnt_range[self.joints]
        self.palm=int(m.site('rh_palm_touch').id)
        self.pairs={side+'_'+digit:(self.names.index(side+'_'+digit+'J1'),self.names.index(side+'_'+digit+'J2'))
            for side in ('rh','lh') for digit in ('FF','MF','RF','LF')}
        self.robot_sha=robot_sha

    def target(self,time_s):
        q=self.desired.copy();goals=self.route.goals(time_s)
        q[self.indices]=[goals[n] for n in self.goal_names]
        return q,(self.matrix@q)[self.goal_motors]

    def palm_position(self,root13,joint_position):
        root=_pose(root13);q=_named(joint_position,self.names)
        self.d.qpos[:7]=root[:7];self.d.qpos[self.qa]=q
        mujoco.mj_kinematics(self.m,self.d)
        return self.d.site_xpos[self.palm].copy()


def evaluate_sensor_acquisition_balance(rows,physics_checks,*,robot_xml,motors,initial_root13_actororigin,
        initial_joint_position,initial_hand_contact_count,initial_door_position,calibration,protocol,joint_route,physics_dt_s=.002,expected_duration_s=19.,reflex_inputs=None):
    """Preserve physical gates; score coupled fingers in commanded motor space.

Every row contains the previous applied30joint/26motor goals plus the actual
post-step69joint measurement. Nominal paired-finger split errors are reported,
while actual individual joint stops and unilateral loopbacks remain hard gates.
The actual palm endpoint is compared with the static initial-body target via
robot-only FK. Neither target nor actual body pose is a runtime observation.
    """
    if not _number(physics_dt_s) or physics_dt_s!=.002 or not _number(expected_duration_s) or expected_duration_s!=19.:
        raise ValueError('Frozen reach qualification requires nineteen seconds at2ms')
    errors=[];model=None;initial_root=None;initial_q=None;reflex=None
    initial_valid=(type(initial_hand_contact_count) is int and initial_hand_contact_count>=0 and type(initial_door_position) is dict and
        set(initial_door_position)=={'leaf_hinge','leaf_handle_hinge'} and all(_number(v) for v in initial_door_position.values()))
    if not initial_valid:errors.append('Missing actual initial door/contact evidence')
    try:
        model=AcquisitionEvaluationRobot(robot_xml,motors,calibration,protocol,joint_route)
        initial_root=_pose(initial_root13_actororigin);initial_q=_named(initial_joint_position,model.names)
        if np.max(abs(initial_q-model.desired))>1e-6:raise ValueError('Actual initial joints differ from frozen calibration')
        reflex=None
        if model.parameters.get('tactile_reflex_profile'):
            from .tactile_grasp_reflex import TactileGraspReflex
            if reflex_inputs is None:raise ValueError('Require actual causal tactile inputs for feedback-goal reconstruction')
            layout,packets=reflex_inputs
            if len(packets['tactile'])!=len(rows):raise ValueError('Incomplete tactile feedback inputs')
            reflex=TactileGraspReflex(layout,dict(zip(model.names,model.joint_limits)))
    except (ValueError,KeyError,TypeError,OverflowError) as exc:errors.append('Static evaluator/reset: '+str(exc))
    physics_valid=type(physics_checks) is dict and bool(physics_checks) and all(type(k) is str and type(v) is bool for k,v in physics_checks.items())
    families_valid=physics_valid and all(any(k in physics_checks for k in family) for family in PHYSICS_FAMILIES)
    if not families_valid:errors.append('Missing explicit authored mechanics/collision/force physical checks')
    if type(rows) not in (list,tuple):errors.append('Actual step evidence must be a sequence');rows=[]
    parsed=[];pad_results=[];unintended=[];tracking=0.;nominal_error=0.;max_stop=0.;max_loopback=0.;goals_match=True;motor_goals_match=True
    split_errors={k:0. for k in ('rh_FF','rh_MF','rh_RF','rh_LF')};actual_motor_values=[]
    if model is not None:
        for i,row in enumerate(rows):
            try:
                if type(row) is not dict or set(row)!=ROW_FIELDS|{'actual_joint_position','unintended_hand_contact_count','pad_evidence'}:raise ValueError('Incomplete actual reach row')
                t=row['time_s'];tilt=row['torso_tilt_deg'];count=row['hand_contact_count'];info=row['controller_info']
                root=_pose(row['root13_actororigin']);q=_named(row['actual_joint_position'],model.names);feet=np.asarray(row['foot_floor_loads'],float)
                if (not _number(t) or not _number(tilt) or tilt<0 or feet.shape!=(2,) or not np.isfinite(feet).all() or np.any(feet<0) or
                        not isinstance(count,(int,np.integer)) or isinstance(count,(bool,np.bool_)) or count<0):raise ValueError('Invalid actual pose/contact record')
                if (type(info) is not dict or info.get('controller')!='sensor_reach_balance_v1' or
                        type(info.get('cold_start')) is not bool or type(info.get('qp_failures')) is not int or info['qp_failures']<0 or
                        not _number(info.get('calculator_time_s')) or type(info.get('qp_status')) is not str or
                        info.get('goal_joint_names')!=list(model.goal_names) or info.get('goal_motor_names')!=list(model.goal_motor_names) or
                        info.get('coupled_joint_goals_are_nominal') is not True):raise ValueError('Missing preceding reach-controller evidence')
                values=info.get('goal_joint_position_rad');motor_values=info.get('goal_motor_coordinates')
                if (type(values) is not list or len(values)!=30 or not all(_number(v) for v in values) or
                        type(motor_values) is not list or len(motor_values)!=26 or not all(_number(v) for v in motor_values)):
                    raise ValueError('Malformed applied joint/motor goals')
                bad=row['unintended_hand_contact_count']
                if type(bad) is not int or not 0<=bad<=count:raise ValueError('Invalid unintended hand-contact count')
                pad=evaluate_pad_evidence(row['pad_evidence'],time_s=float(t))
                target,coordinates=model.target(i*.002)
                if reflex is not None:
                    packet={key:np.asarray(values[i]) for key,values in packets.items()}
                    goals=reflex.apply(packet,i*.002,dict(zip(model.goal_names,target[model.indices])))
                    target[model.indices]=[goals[n] for n in model.goal_names]
                    coordinates=(model.matrix@target)[model.goal_motors]
                goals_match=goals_match and bool(np.max(abs(np.array(values)-target[model.indices]))<1e-9)
                motor_goals_match=motor_goals_match and bool(np.max(abs(np.array(motor_values)-coordinates))<1e-9)
                actual_coordinates=(model.matrix@q)[model.goal_motors];actual_motor_values.append(actual_coordinates)
                tracking=max(tracking,float(np.max(abs(actual_coordinates-coordinates))))
                nominal_error=max(nominal_error,float(np.max(abs(q[model.indices]-target[model.indices]))))
                limited=model.limited;bounds=model.joint_limits
                max_stop=max(max_stop,float(np.max(np.maximum(bounds[limited,0]-q[limited],q[limited]-bounds[limited,1]))))
                for pair,(j1,j2) in model.pairs.items():
                    max_loopback=max(max_loopback,float(q[j1]-q[j2]))
                    if pair in split_errors:split_errors[pair]=max(split_errors[pair],float(abs((q[j1]-q[j2])-(target[j1]-target[j2]))))
                parsed.append((float(t),root,float(tilt),feet,int(count),info,q));pad_results.append(pad);unintended.append(bad)
            except (KeyError,ValueError,TypeError,OverflowError) as exc:errors.append(f'Row{i}: {exc}')
    valid=bool(parsed) and len(parsed)==len(rows) and not errors
    complete=valid and len(parsed)==9500 and np.allclose([r[0] for r in parsed],np.arange(1,9501)*.002,atol=1e-8,rtol=0)
    tail=[r for r in parsed if r[0]>=18.-1e-9];motion=False;excursions={};requested={};palm_error=None;actual_palm=None;target_palm=None
    if model is not None and initial_q is not None and initial_root is not None and parsed:
        zero=(model.matrix@initial_q)[model.goal_motors];_,first=model.target(0.);end_q,end=model.target(19.)
        delta=end-first;excursion=np.max(abs(np.array(actual_motor_values)-zero),axis=0)
        excursions=dict(zip(model.goal_motor_names,excursion.tolist()));requested=dict(zip(model.goal_motor_names,delta.tolist()))
        active=abs(delta)>.01;motion=bool(np.any(active) and np.all(excursion[active]>.7*abs(delta[active])))
        planned=initial_q.copy();planned[model.indices]=end_q[model.indices]
        target_palm=model.palm_position(initial_root,dict(zip(model.names,planned)))
        actual_palm=model.palm_position(parsed[-1][1],dict(zip(model.names,parsed[-1][6])))
        palm_error=float(np.linalg.norm(actual_palm-target_palm))
    hold_indices=[i for i,r in enumerate(parsed) if r[0]>=18.5-1e-8]
    minimum_hold={d:min((pad_results[i]['qualified_pad_forces_N'][d] for i in hold_indices),default=None) for d in ('ff','mf','rf','lf','th')}
    current=best=0
    for p in pad_results:
        current=current+1 if p['valid_pad_grasp'] else 0;best=max(best,current)
    checks=dict(actual_step_records_valid=valid,original_physics_checks_valid=physics_valid and families_valid,
        original_physics_all_passed=physics_valid and all(physics_checks.values()),complete_19s_at_2ms=complete,
        lowered_height=bool(parsed) and all(.82<=r[1][2]<=.92 for r in parsed),torso_upright=bool(parsed) and all(r[2]<=12 for r in parsed),
        final_quiet=bool(tail) and all(np.linalg.norm(r[1][7:9])<.03 and np.linalg.norm(r[1][10:13])<.05 for r in tail),
        final_both_feet=bool(tail) and all(min(r[3])>30 for r in tail),only_intended_hand_contacts=bool(parsed) and all(n==0 for n in unintended),
        original_distal_pad_patches=bool(parsed) and all(p['invalid_loaded_patches']==0 and p['non_digit_handle_force_N']<=1e-6 for p in pad_results),
        original_opposed_distal_grasp_hold=complete and len(hold_indices)>=251 and all(pad_results[i]['valid_pad_grasp'] for i in hold_indices),
        actual_closed_contact_free_start=initial_valid and initial_hand_contact_count==0 and all(abs(v)<=.001 for v in initial_door_position.values()),
        all_qp_solved=bool(parsed) and all(r[5]['qp_failures']==0 and r[5]['qp_status']=='solved' for r in parsed),
        estimator_never_stepped=bool(parsed) and all(r[5]['calculator_time_s']==0. for r in parsed),
        cold_start_at_t0=bool(parsed) and abs(parsed[0][0]-.002)<1e-8 and parsed[0][5]['cold_start'] is True and all(r[5]['cold_start'] is False for r in parsed[1:]),
        actual_initial_angles_valid=initial_q is not None and initial_root is not None and not any(s.startswith('Static evaluator/reset:') for s in errors),
        applied_joint_goals_match_schedule=bool(parsed) and goals_match,applied_motor_goals_match_transmission=bool(parsed) and motor_goals_match,
        motor_coordinate_tracking=bool(parsed) and tracking<.04,actual_authored_joint_stops=bool(parsed) and max_stop<=.02,
        actual_unilateral_loopbacks=bool(parsed) and max_loopback<=.02,physical_reach_motion_delivered=motion,
        palm_endpoint_tracking=palm_error is not None and palm_error<.02)
    checks={k:bool(v) for k,v in checks.items()}
    return dict(schema=ACQUISITION_EVALUATION_SCHEMA,scope='Scripted full-route right-hand acquisition with sensor-feedback balance; no learned, vision or opening result',
        passed=all(checks.values()),checks=checks,original_physics_checks=copy.deepcopy(physics_checks) if physics_valid else {},
        physics_dt_s=.002,expected_duration_s=19.,physics_steps=len(rows),valid_records=len(parsed),duration_s=parsed[-1][0] if parsed else None,
        errors=errors,grasp_profile='distal-pad-v1',required_grasp_hold_s=.5,
        first_hand_contact_s=next((r[0] for r in parsed if r[4]>0),None),strongest_qualified_hold_s=best*.002,
        final_half_second_minimum_pad_force_N=minimum_hold,invalid_loaded_patch_count=sum(p['invalid_loaded_patches'] for p in pad_results),
        final_pad_grasp=pad_results[-1] if pad_results else None,maximum_motor_coordinate_tracking_error_rad=tracking if parsed else None,maximum_nominal_joint_tracking_error_rad=nominal_error if parsed else None,
        maximum_nominal_passive_split_difference_rad=split_errors,maximum_joint_stop_violation_rad=max_stop if parsed else None,maximum_loopback_violation_rad=max_loopback if parsed else None,
        measured_motor_excursions_rad=excursions,requested_motor_displacements_rad=requested,palm_endpoint_error_m=palm_error,
        actual_palm_endpoint_world_m=None if actual_palm is None else actual_palm.tolist(),target_palm_endpoint_world_m=None if target_palm is None else target_palm.tolist(),
        max_torso_tilt_deg=max((r[2] for r in parsed),default=None),height_range_m=None if not parsed else [min(r[1][2] for r in parsed),max(r[1][2] for r in parsed)],
        protocol_sha256=None if model is None else model.protocol_sha,joint_route_sha256=None if model is None else model.route_sha,
        robot_xml_sha256=None if model is None else model.robot_sha,calculator_time_s=None if model is None else float(model.d.time),
        evaluator_state_is_actor_input=False,checkpoint_evaluated=False,
        limitation='The coupled finger split is passive and need not equal nominal per-joint targets; actual joint bounds and J1<=J2 remain separate hard checks. This is a declared canonical straight-lever grasp, not opening, vision or learned control.')
