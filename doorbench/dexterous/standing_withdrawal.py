"""Explicit, source-bound motor withdrawal after the qualified standing return."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .operation_teacher import smooth_phase,pose_components
from .attained_arm_tracking import AttainedArmTracking
from .attained_hand_tracking import AttainedHandTracking
from .return_palm_feedback import ReturnPalmFeedback,validate_correction_gain
from .motor_handoff import MotorHandoff


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def bound_input_digest(bindings,path):
    """Compare identical files across relative/absolute Windows path spellings."""
    resolved=Path(path).resolve()
    matches={digest for name,digest in bindings.items() if Path(name).resolve()==resolved}
    if len(matches)>1:raise ValueError('Conflicting digest aliases for the same withdrawal input')
    return next(iter(matches),None)


class StandingWithdrawalTeacher:
    def __init__(self,returned,motors,path,*,_isaac_runtime=None):
        self.returned=returned;self.acquisition=returned.acquisition;self.operation=returned.operation
        self.left=returned.transfer.left;self.started_withdrawal=None;self.release_started=None
        self.qualified_since=None;self.info={};self.handoff=None
        self.isaac_runtime=_isaac_runtime;self.inherited_support=None
        if _isaac_runtime is None:
            config=json.loads(Path(path).read_text())
            if config.get('source_engine','native-mujoco')!='native-mujoco':
                raise ValueError('Isaac withdrawal requires its explicit actual-source runtime factory')
        else:
            from .isaac_withdrawal_runtime import IsaacWithdrawalRuntimeAdmission
            if not isinstance(_isaac_runtime,IsaacWithdrawalRuntimeAdmission) or _isaac_runtime.inherited_support is None:
                raise ValueError('Fresh actual-Isaac runtime admission and predecessor required')
            _isaac_runtime.verify(path,motors)
            config=dict(_isaac_runtime.controller_config)
            self.source_context=_isaac_runtime.source_context
            self.inherited_support=_isaac_runtime.inherited_support
        if config.get('schema')!='doorbench.standing-withdrawal.v1':raise ValueError('Explicit standing withdrawal config required')
        self.palm_correction_gain_s_inv=validate_correction_gain(config.get('palm_correction_gain_s_inv',3.))
        capture_returned=config.get('capture_returned_motor_command',False)
        if type(capture_returned) is not bool:raise ValueError('Explicit delegated-command capture option required')
        self.motor_capture=None;self.motor_capture_info={}
        if capture_returned:
            from .withdrawal_motor_capture import WithdrawalMotorCapture
            from .isaac_withdrawal_runtime import PausedIsaacWithdrawalRuntimeAdmission
            self.motor_capture=(_isaac_runtime.motor_capture_for(returned)
                if isinstance(_isaac_runtime,PausedIsaacWithdrawalRuntimeAdmission)
                else WithdrawalMotorCapture(self.acquisition.caps))
        self.measured_rest=bool(getattr(returned,'requires_measured_rest',False))
        if config.get('measured_rest_transfer',False) != self.measured_rest:
            raise ValueError('Withdrawal must explicitly identify its measured-rest transfer bridge')
        self.palm_only_support=self.measured_rest or config.get('palm_only_support',False)
        if type(self.palm_only_support) is not bool:raise ValueError('Explicit palm-only qualification option required')
        self.support_target=float(config.get('left_support_target_N',4.))
        support_maximum=4. if self.inherited_support is None else self.inherited_support.target_N
        if not np.isfinite(self.support_target) or not 2<self.support_target<=support_maximum:raise ValueError('Withdrawal support target must remain above the original 2 N gate')
        self.initial_support_target=self.left.support_load_target
        self.hybrid_support=config.get('hybrid_support',False)
        if type(self.hybrid_support) is not bool:raise ValueError('Explicit hybrid support option required')
        self.support_feedback=None
        self.finger_velocity_feedforward=config.get('finger_velocity_feedforward',False)
        if type(self.finger_velocity_feedforward) is not bool:raise ValueError('Explicit finger velocity feedforward option required')
        self.previous_finger_target=None
        self.thumb_pad_feedback=config.get('thumb_pad_feedback',False)
        if type(self.thumb_pad_feedback) is not bool:raise ValueError('Explicit thumb feedback option required')
        self.thumb_feedback=None
        self.finger_pad_feedback=config.get('finger_pad_feedback',False)
        if type(self.finger_pad_feedback) is not bool:raise ValueError('Explicit finger material feedback option required')
        self.finger_feedback={};self.finger_points={}
        self.middle_feedback_enabled=config.get('release_middle_segment_feedback',False)
        if type(self.middle_feedback_enabled) is not bool:raise ValueError('Explicit middle segment tracking option required')
        self.middle_feedback={};self.middle_points={}
        segment_avoidance=config.get('release_ring_segment_avoidance',False)
        if type(segment_avoidance) is not bool:raise ValueError('Explicit release segment avoidance option required')
        self.release_segment_avoidance=None
        if segment_avoidance:
            from .lever_segment_avoidance import LeverSegmentAvoidance
            self.release_segment_avoidance=LeverSegmentAvoidance(self.acquisition)
        self.final_hub_avoidance=config.get('final_hub_avoidance',False)
        if type(self.final_hub_avoidance) is not bool:raise ValueError('Explicit final hub avoidance option required')
        if self.final_hub_avoidance and self.operation.hub_avoidance is None:raise ValueError('Final hub avoidance requires the original hub geometry/controller')
        whole_lf=config.get('withdrawal_whole_little_finger_hub_avoidance',False)
        hub_clearance=float(config.get('withdrawal_hub_clearance_m',.004))
        if not np.isfinite(hub_clearance) or not .004<=hub_clearance<=.008 or (hub_clearance!=.004 and not whole_lf):raise ValueError('Extended hub clearance requires whole little-finger feedback')
        if type(whole_lf) is not bool or (whole_lf and not self.final_hub_avoidance):raise ValueError('Whole little-finger avoidance requires final hub feedback')
        self.withdrawal_hub_avoidance=self.operation.hub_avoidance
        if whole_lf:
            from .handle_hub_avoidance import HandleHubAvoidance
            self.withdrawal_hub_avoidance=HandleHubAvoidance(self.acquisition,include_distal=True,clearance_m=hub_clearance)
        self.left_arm_only=config.get('left_arm_only',False)
        if type(self.left_arm_only) is not bool:raise ValueError('Explicit left-arm solve option required')
        self.left_full_orientation=config.get('left_full_orientation',False)
        if type(self.left_full_orientation) is not bool or (self.left_full_orientation and not self.left_arm_only):raise ValueError('Full palm orientation requires isolated left-arm IK')
        self.left_target_velocity=config.get('left_target_velocity',False)
        if type(self.left_target_velocity) is not bool or (self.left_target_velocity and not self.hybrid_support):raise ValueError('Left target velocity requires explicit hybrid support')
        if _isaac_runtime is None:
            audit_path=Path(config['audit_path']);screen_path=Path(config['screen_path']);source=Path(config['source_run'])
            if sha(audit_path)!=config['audit_sha256'] or sha(screen_path)!=config['screen_sha256']:raise ValueError('Withdrawal evidence changed')
            audit=json.loads(audit_path.read_text());screen=json.loads(screen_path.read_text())
            self.source_admission=None
            if self.measured_rest:
                from .release_source_admission import admit_release_source
                self.source_admission=admit_release_source(source,
                    profile=config['grasp_profile'],contact_audit_name=config['contact_audit_name'],measured_rest=True)
                if screen.get('grasp_profile')!=config['grasp_profile'] or audit.get('grasp_profile')!=config['grasp_profile']:
                    raise ValueError('Withdrawal route and dense audit must retain the selected source profile')
                if screen.get('source_admission')!=self.source_admission:
                    raise ValueError('Withdrawal screen must bind this exact measured-rest source')
            from .release_motion_admission import validate_motion_screen
            validate_motion_screen(config,screen_path,source)
            if audit.get('passed') is not True or audit.get('samples')!=2001 or audit.get('physics_steps')!=0:raise ValueError('Independent dense withdrawal admission required')
            if bound_input_digest(audit['input_sha256'],screen_path)!=sha(screen_path) or bound_input_digest(audit['input_sha256'],source/'trajectory.npz')!=sha(source/'trajectory.npz'):raise ValueError('Withdrawal audit belongs to another state or route')
            for name,digest in audit['input_sha256'].items():
                if sha(name)!=digest:raise ValueError('Withdrawal input bytes changed: '+name)
            manifest=json.loads((source/'manifest.json').read_text());cfg=manifest['configuration']
            if manifest['inputs']['robot']['sha256']!=motors['source_xml_sha256']:raise ValueError('Withdrawal requires its original robot motor contract')
            with np.load(source/'trajectory.npz') as z:actual=z['terminal_qpos'].copy();self.start_time=float(z['terminal_time_s'])
            self.duration=float(audit['duration_s'])
            robot_path=Path(cfg['robot']);door_xml_path=Path(cfg['door'])/'door.xml'
        else:
            source_data=self.source_context.data
            self.source_admission=source_data['source_admission'];screen=source_data['screen'];audit=source_data['audit']
            actual=self.source_context.initial_qpos;self.start_time=source_data['start_time_s'];self.duration=source_data['duration_s']
            robot_path=Path(source_data['robot_path']);door_xml_path=Path(source_data['door_xml_path'])
        if not np.isfinite([self.start_time,self.duration]).all() or self.duration<=0:raise ValueError('Finite positive withdrawal duration required')
        from .landed_left_planner import LandedLeftScene
        scene=LandedLeftScene(robot_path,door_xml_path);m=scene.m;d=mujoco.MjData(m);d.qpos[:]=actual;mujoco.mj_kinematics(m,d)
        rq=int(m.joint('robot/free_base').qposadr[0]);palm=m.site('robot/rh_palm_touch').id;leaf=m.body('leaf').id
        self.initial_leaf=(d.xpos[leaf].copy(),d.xmat[leaf].reshape(3,3).copy())
        initial=dict(time_s=0.,qpos=actual.tolist(),palm_position=d.site_xpos[palm].tolist(),palm_rotation=d.site_xmat[palm].reshape(3,3).tolist())
        rows=[initial]+[{**r,'time_s':r['time_s']+.5} for r in screen['trials'][0]['rows']]
        qs=np.asarray([r['qpos'] for r in rows]);self.times=np.array([r['time_s'] for r in rows])
        if qs.shape!=(len(rows),m.nq) or not np.isfinite(qs).all() or not np.all(np.diff(self.times)>0):raise ValueError('Finite monotonic withdrawal coordinates required')
        if not np.allclose(np.linalg.norm(qs[:,rq+3:rq+7],axis=1),1.,atol=1e-8,rtol=0):raise ValueError('Normalized withdrawal roots required')
        if self.thumb_pad_feedback:
            body=m.body('robot/rh_thdistal').id;lever=m.geom('leaf_handle_lever_col_n').id;nearest=[]
            for g in range(m.ngeom):
                if m.geom_bodyid[g]!=body or not m.geom_contype[g]:continue
                pair=np.zeros(6);distance=mujoco.mj_geomDistance(m,d,g,lever,.2,pair)
                nearest.append((distance,pair.copy()))
            distance,pair=min(nearest,key=lambda v:v[0])
            if abs(distance)>.01:raise ValueError('Source thumb must be near the actual lever')
            self.thumb_local=d.xmat[body].reshape(3,3).T@(pair[:3]-d.xpos[body]);points=[]
            for q in qs:
                d.qpos[:]=q;mujoco.mj_kinematics(m,d);points.append(d.xpos[body]+d.xmat[body].reshape(3,3)@self.thumb_local)
            self.thumb_positions=np.asarray(points)
        if self.finger_pad_feedback:
            from .finger_withdrawal_feedback import FingerWithdrawalFeedback
            lever=m.geom('leaf_handle_lever_col_n').id
            for digit in ('ff','mf','rf','lf'):
                d.qpos[:]=actual;mujoco.mj_kinematics(m,d)
                body=m.body('robot/rh_'+digit+'distal').id;nearest=[]
                for g in range(m.ngeom):
                    if m.geom_bodyid[g]!=body or not m.geom_contype[g]:continue
                    pair=np.zeros(6);distance=mujoco.mj_geomDistance(m,d,g,lever,.2,pair)
                    nearest.append((distance,pair.copy()))
                distance,pair=min(nearest,key=lambda v:v[0])
                if abs(distance)>.01:raise ValueError('Source finger must be near the actual lever')
                local=d.xmat[body].reshape(3,3).T@(pair[:3]-d.xpos[body]);points=[]
                for q in qs:
                    d.qpos[:]=q;mujoco.mj_kinematics(m,d)
                    points.append(d.xpos[body]+d.xmat[body].reshape(3,3)@local)
                self.finger_points[digit]=(local,np.asarray(points))
                self.finger_feedback[digit]=FingerWithdrawalFeedback(self.acquisition,local,digit=digit)
        if self.middle_feedback_enabled:
            from .finger_withdrawal_feedback import FingerWithdrawalFeedback
            for digit,obstacle in [('rf','leaf_handle_lever_col_n'),('lf','leaf_handle_hub_col_n')]:
                d.qpos[:]=actual;mujoco.mj_kinematics(m,d)
                body=m.body('robot/rh_'+digit+'middle').id;nearest=[]
                for g in range(m.ngeom):
                    if m.geom_bodyid[g]!=body or not m.geom_contype[g]:continue
                    pair=np.zeros(6);distance=mujoco.mj_geomDistance(m,d,g,m.geom(obstacle).id,.1,pair)
                    nearest.append((distance,pair.copy()))
                distance,pair=min(nearest,key=lambda v:v[0])
                if abs(distance)>.05:raise ValueError('Source middle segment must be near the original handle')
                local=d.xmat[body].reshape(3,3).T@(pair[:3]-d.xpos[body]);points=[]
                for q in qs:
                    d.qpos[:]=q;mujoco.mj_kinematics(m,d)
                    points.append(d.xpos[body]+d.xmat[body].reshape(3,3)@local)
                self.middle_points[digit]=(local,np.asarray(points))
                self.middle_feedback[digit]=FingerWithdrawalFeedback(self.acquisition,local,digit=digit,segment='middle')
        self.names=list(screen['trials'][0]['rows'][0]['joints'])
        self.finger_names=list(screen['trials'][0]['rows'][0]['finger_joints'])
        for row in rows[1:]:
            for group in ('joints','finger_joints'):
                for name,value in row[group].items():
                    if abs(row['qpos'][m.joint('robot/'+name).qposadr[0]]-value)>1e-12:raise ValueError('Consumed withdrawal targets differ from screened coordinates')
        self.all_names=self.names+self.finger_names
        self.joints=qs[:,[m.joint('robot/'+n).qposadr[0] for n in self.all_names]]
        self.roots=qs[:,rq:rq+7];self.root_rotations=Slerp(self.times,Rotation.from_quat(self.roots[:,[4,5,6,3]]))
        self.positions=np.asarray([r['palm_position'] for r in rows]);self.rotations=Slerp(self.times,Rotation.from_matrix(np.asarray([r['palm_rotation'] for r in rows])))
        self.release_phase=config.get('release_phase','measured_release')
        if self.release_phase not in ('measured_release','grasp_adjustment'):raise ValueError('Explicit screened release phase required')
        self.release_clock=min(r['time_s'] for r in rows[1:] if r['phase']==self.release_phase)
        self.panel_gain=config.get('panel_aperture_feedback_gain_N_per_rad',0.)
        if type(self.panel_gain) not in (int,float) or not np.isfinite(self.panel_gain) or not 0<=self.panel_gain<=20:raise ValueError('Require bounded explicit panel force feedback')
        self.panel_force=None
        profile=config.get('panel_aperture_force_profile')
        initial_stiction=config.get('panel_initial_stiction_assist',False)
        terminal_floor=config.get('panel_terminal_minimum_support_N')
        if terminal_floor is not None and (not config.get('panel_plan_path') or profile not in ('bounded-pi-stop-v1','bounded-pi-stop-v2')):
            raise ValueError('Terminal support floor requires a screened stopping plan')
        initial_load=config.get('panel_initial_load_profile','standard-6N-v1')
        if type(initial_stiction) is not bool or initial_load not in ('standard-6N-v1','bounded-7N-v1'):
            raise ValueError('Explicit initial panel assistance and load profile required')
        if (initial_stiction or initial_load!='standard-6N-v1') and (not config.get('panel_plan_path') or profile not in ('bounded-pi-stop-v1','bounded-pi-stop-v2')):
            raise ValueError('Initial stiction assistance requires a screened terminal panel target')
        if initial_load!='standard-6N-v1' and not initial_stiction:raise ValueError('Expanded panel load requires explicit stiction assistance')
        if profile is not None:
            if profile in ('bounded-pi-stop-v1','bounded-pi-stop-v2') and not config.get('panel_plan_path'):raise ValueError('Terminal braking requires a screened panel target')
            if profile not in ('bounded-pi-v1','bounded-pi-stop-v1','bounded-pi-stop-v2'):raise ValueError('Unknown panel aperture force profile')
            from .panel_aperture_force import PanelApertureForce
            self.panel_force=PanelApertureForce()
        self.panel=None;self.panel_handoff=None
        if config.get('panel_plan_path'):
            if sha(config['panel_plan_path'])!=config.get('panel_plan_sha256'):raise ValueError('Panel plan bytes changed')
            from .standing_panel_reference import StandingPanelReference
            self.panel=StandingPanelReference(scene,config['panel_plan_path'],self.left)
            if profile in ('bounded-pi-stop-v1','bounded-pi-stop-v2'):self.panel_force=PanelApertureForce(terminal_aperture=self.panel.plan['final_leaf_angle_rad'],terminal_support_margin_N=.5 if profile=='bounded-pi-stop-v2' else 0.,stiction_assist=initial_stiction,load_profile=initial_load,terminal_minimum_support_N=terminal_floor)
            if self.panel.plan['robot_xml_sha256']!=motors['source_xml_sha256']:raise ValueError('Panel plan uses another robot contract')
            if self.panel.start_time<self.start_time:raise ValueError('Panel continuation cannot precede withdrawal')
        self.panel_schedule=None
        if config.get('panel_continuations') and self.panel is None:raise ValueError('Panel continuations require an initial screened segment')
        if self.panel is not None:
            panels=[self.panel]
            self.panel_stiction_assistance=[initial_stiction]
            self.panel_load_profiles=[initial_load]
            for entry in config.get('panel_continuations',[]):
                if sha(entry['path'])!=entry['sha256']:raise ValueError('Continuation plan bytes changed')
                candidate=StandingPanelReference(scene,entry['path'],self.left)
                candidate.preserve_stance_reference=entry.get('preserve_stance_reference',False)
                if type(candidate.preserve_stance_reference) is not bool:raise ValueError('Explicit boolean stance reference continuity required')
                candidate.material_reference_handoff=entry.get('material_reference_handoff',False)
                if type(candidate.material_reference_handoff) is not bool:raise ValueError('Explicit boolean material reference handoff required')
                if candidate.plan['robot_xml_sha256']!=motors['source_xml_sha256']:raise ValueError('Continuation uses another robot contract')
                assist=entry.get('stiction_assist',False)
                if type(assist) is not bool or (assist and profile!='bounded-pi-stop-v2'):raise ValueError('Explicit stiction assistance requires terminal support profile v2')
                from .panel_aperture_force import PanelApertureForce
                load_profile=entry.get("load_profile","standard-6N-v1")
                PanelApertureForce(terminal_aperture=candidate.plan["final_leaf_angle_rad"],stiction_assist=assist,load_profile=load_profile)
                panels.append(candidate);self.panel_stiction_assistance.append(assist);self.panel_load_profiles.append(load_profile)
            from .panel_sequence import AttainedPanelSchedule
            self.panel_schedule=AttainedPanelSchedule(panels)
        self.panel_force_profile=profile
        self.initial_angles={name:float(actual[m.joint(joint).qposadr[0]]) for name,joint in [('operator','leaf_handle_hinge'),('leaf','leaf_hinge'),('latch','leaf_latch_bolt_slide')]}
        self.coupled=None
        coupled_keys=('coupled_envelope_path','coupled_envelope_sha256','coupled_audit_path','coupled_audit_sha256')
        if any(key in config for key in coupled_keys):
            if not all(key in config for key in coupled_keys):raise ValueError('Complete explicit coupled reference evidence required')
            if not self.measured_rest or not self.left_arm_only or not self.left_full_orientation:
                raise ValueError('Coupled withdrawal requires measured-rest source and isolated full-orientation left-arm tracking')
            if self.panel is not None or self.hybrid_support or self.thumb_pad_feedback or self.finger_pad_feedback or self.middle_feedback_enabled:
                raise ValueError('Coupled withdrawal cannot combine separately framed continuation/material feedback routes')
            from .coupled_release_reference import CoupledReleaseReference
            self.coupled=CoupledReleaseReference(config,self.source_admission,actual,self.duration)
        if _isaac_runtime is not None:
            self.coupled=_isaac_runtime.reference
            if not np.array_equal(self.coupled.geometry.initial,actual) or self.coupled.geometry.plan['duration_s']!=self.duration:
                raise ValueError('Isaac reference must retain constructor source endpoint and duration')

    @property
    def started(self):return self.returned.started

    @property
    def return_started(self):return self.returned.return_started

    def authorize_source_prefix(self,receipt):
        if self.isaac_runtime is None:raise ValueError('Combined Isaac source prefix applies only to explicit Isaac mode')
        self.isaac_runtime.authorize_source_prefix(receipt)

    def force(self,t,root,joints,velocities,handle_pose,leaf_pose,angles,hand_loads,*,grasp_qualified,left_panel_load,left_palm_load=None):
        teacher=self.acquisition
        if self.palm_only_support and (left_palm_load is None or not np.isfinite(left_palm_load) or left_palm_load<0):
            raise ValueError('Actual finite palm-only support required for withdrawal')
        support=left_palm_load if self.palm_only_support else left_panel_load
        if self.measured_rest and self.started_withdrawal is None:
            self.returned.observe_rest(t,grasp_qualified=grasp_qualified,left_palm_load=left_palm_load,angles=angles)
        if grasp_qualified and (not self.palm_only_support or support>=2):
            if self.qualified_since is None:self.qualified_since=t
        else:self.qualified_since=None
        if self.started_withdrawal is None and t>=self.start_time-1e-8:
            if getattr(self,'isaac_runtime',None) is not None:self.isaac_runtime.require_entry(t)
            if self.measured_rest and not self.returned.rest.ready:raise ValueError('Half-second actual resting grip and palm-only support required before withdrawal')
            if not grasp_qualified or support<2 or abs(angles['operator'])>.05 or abs(angles['latch'])>.001:raise ValueError('Qualified resting grip and left support required before withdrawal')
            if not np.allclose(root[:7],self.roots[0],atol=1e-5,rtol=0) or not np.allclose([joints[n] for n in self.all_names],self.joints[0],atol=1e-5,rtol=0):raise ValueError('Withdrawal requires its exact attained root and joints')
            if any(abs(angles[n]-v)>1e-5 for n,v in self.initial_angles.items()):raise ValueError('Withdrawal door state differs from its audited source')
            self.previous_force=teacher.last_force.copy()
            if self.motor_capture is not None:
                self.motor_capture_info=self.motor_capture.diagnostic(t,self.previous_force)
                self.previous_force=self.motor_capture.capture(t)
            self.started_withdrawal=t
            self.arm=AttainedArmTracking(teacher,joints,self.previous_force)
            self.hand=AttainedHandTracking(teacher,joints,self.previous_force)
            self.palm=ReturnPalmFeedback(teacher,root,joints,handle_pose,self.operation.geometry,correction_gain_s_inv=self.palm_correction_gain_s_inv)
            self.preload=self.hand.preload.copy()
            # Geometric routes describe attained poses. The balance controller
            # already has small reference offsets needed to hold that pose under
            # load; replacing those offsets with zero causes an avoidable kick.
            self.stance_root_bias=teacher.stance.target_root.copy()-self.roots[0,:3]
            self.stance_rotation_bias=teacher.stance.target_rotation@self.root_rotations(0.).as_matrix().T
            self.stance_names=[teacher.m.joint(int(j)).name for j in teacher.stance.joints]
            self.stance_joint_bias=teacher.stance.joint_target.copy()-np.array([joints[n] for n in self.stance_names])
            if self.left_arm_only:self.left.isolate_left_arm(root,joints,leaf_pose=leaf_pose if self.left_full_orientation else None)
        if self.started_withdrawal is None:
            force,self.info=self.returned.force(t,root,joints,velocities,handle_pose,leaf_pose,angles,hand_loads,grasp_qualified=grasp_qualified,left_panel_load=left_panel_load,**({'left_palm_load':left_palm_load} if self.measured_rest else {}))
            if self.motor_capture is not None:self.motor_capture.observe(t,force)
            return force,self.info
        elapsed=t-self.started_withdrawal;clock=float(smooth_phase(elapsed/self.duration))*self.times[-1]
        if clock>=self.release_clock and self.release_started is None:
            if self.qualified_since is None or t-self.qualified_since<.5 or support<2:raise ValueError('Half-second opposed grip and left support required at intentional release')
            self.release_started=t
        i=min(len(self.times)-2,max(0,int(np.searchsorted(self.times,clock,side='right')-1)));f=(clock-self.times[i])/(self.times[i+1]-self.times[i])
        q=(1-f)*self.joints[i]+f*self.joints[i+1];targets=dict(zip(self.all_names,q))
        preceding_stance=self.panel_schedule.pending_stance_reference(t,teacher.stance) if self.panel_schedule is not None else None
        teacher.stance.target_root[:]=(1-f)*self.roots[i,:3]+f*self.roots[i+1,:3]+self.stance_root_bias
        teacher.stance.target_rotation=self.stance_rotation_bias@self.root_rotations(clock).as_matrix()
        teacher.stance.joint_target[:]=np.array([targets[n] for n in self.stance_names])+self.stance_joint_bias
        coupled_goal=None;coupled_info={}
        if self.coupled is not None:
            try:targets,position,rotation,goal,goal_rotation,coupled_info=self.coupled.update(t,elapsed,angles,leaf_pose,handle_pose)
            except Exception:
                self.info={**self.info,'coupled_reference_failed':True,'coupled_reference_failure':self.coupled.failure_snapshot}
                raise
            if abs(coupled_info['coupled_reference_release_clock_s']-clock)>1e-10:
                raise ValueError('Coupled reference and withdrawal route clocks differ')
            teacher.stance.target_root[:]=position+self.stance_root_bias
            teacher.stance.target_rotation=self.stance_rotation_bias@rotation
            teacher.stance.joint_target[:]=np.array([targets[n] for n in self.stance_names])+self.stance_joint_bias
            self.left.path[-1]['nominal']=np.array([targets[n] for n in self.left.names])
            self.left.panel_palm_rotation=self.coupled.geometry.localr.copy()
            # Keep the original contact reference offset and force loop. The
            # coupled map supplies the reachable body/arm nominal posture.
            coupled_goal=(goal,goal_rotation)
            coupled_info['coupled_nominal_reference_only']=True
            coupled_info['coupled_existing_contact_and_motor_feedback_retained']=True
        material_handoff=False
        if self.panel_schedule is not None and self.panel_schedule.advance(t,angles['leaf'],left_panel_load):
            self.panel=self.panel_schedule.active;self.panel_handoff=None
            material_handoff=getattr(self.panel,'material_reference_handoff',False)
            if preceding_stance is not None:
                for name,value in preceding_stance.items():getattr(teacher.stance,name)[:]=value
            if self.panel_force is not None:
                from .panel_aperture_force import PanelApertureForce
                terminal=self.panel.plan['final_leaf_angle_rad'] if self.panel_force_profile in ('bounded-pi-stop-v1','bounded-pi-stop-v2') else None
                self.panel_force=PanelApertureForce(terminal_aperture=terminal,terminal_support_margin_N=.5 if self.panel_force_profile=='bounded-pi-stop-v2' else 0.,stiction_assist=self.panel_stiction_assistance[self.panel_schedule.index],load_profile=self.panel_load_profiles[self.panel_schedule.index],terminal_minimum_support_N=self.panel_force.terminal_minimum_support_N)
                if self.support_feedback is not None:self.support_feedback.maximum_target_N=self.panel_force.maximum_target_N
        panel_goal=None;panel_info={}
        if self.panel is not None and t>=self.panel.start_time-1e-8:
            if self.panel.started is None:
                self.panel_previous_force=teacher.last_force.copy()
                self.arm=AttainedArmTracking(teacher,joints,self.panel_previous_force)
                self.palm=ReturnPalmFeedback(teacher,root,joints,handle_pose,self.operation.geometry,correction_gain_s_inv=self.palm_correction_gain_s_inv)
            targets,position,rotation,panel_info=self.panel.update(t,root,joints,angles['leaf'],teacher.stance)
            panel_goal=(position,rotation)
        self.left.support_load_target=self.initial_support_target+float(smooth_phase(elapsed/2.))*(self.support_target-self.initial_support_target)
        if panel_goal is not None and self.panel_force is not None:
            self.left.support_load_target,force_info=self.panel_force.update(t,panel_info['panel_reference_aperture_rad'],angles['leaf'],self.left.support_load_target)
            panel_info.update(force_info)
        elif panel_goal is not None and self.panel_gain:
            self.left.support_load_target=float(np.clip(self.left.support_load_target+self.panel_gain*(panel_info['panel_reference_aperture_rad']-angles['leaf']),2.05,3.5))
            panel_info['panel_aperture_feedback_gain_N_per_rad']=self.panel_gain
        self.left.update_targets(t,root,joints,leaf_pose,left_panel_load,handle_pose)
        if getattr(self,'inherited_support',None) is not None:
            if getattr(self.inherited_support,'profile',None) is None:
                self.inherited_support.update(t,root,joints,leaf_pose,left_palm_load)
            else:
                self.inherited_support.update(t,root,joints,leaf_pose,left_palm_load,
                    release_started_s=self.release_started)
        if self.hybrid_support:
            self.left._read(root,joints)
            if self.support_feedback is None:
                from .standing_support_feedback import StandingSupportFeedback
                self.support_feedback=StandingSupportFeedback(self.left,maximum_target_N=self.panel_force.maximum_target_N if self.panel_force is not None else 4.)
            self.support_feedback.update(t,leaf_pose,left_panel_load,self.left.support_load_target)
            if self.left_target_velocity:self.support_feedback.update_target_velocity()
        force,info=self.operation.force(t,root,joints,velocities,handle_pose,leaf_pose,angles,hand_loads,grasp_qualified=grasp_qualified)
        force=self.left.apply_forces(force,joints,velocities)
        # The screened withdrawal is expressed in the attained resting world.
        # Following the moving leaf here lets both hands chase an opening door
        # during the regrasp, instead of retaining the screened working pose.
        goal=(1-f)*self.positions[i]+f*self.positions[i+1];rotation=self.rotations(clock).as_matrix()
        if coupled_goal is not None:goal,rotation=coupled_goal
        if panel_goal is not None:goal,rotation=panel_goal
        targets,palm_info=self.palm.world_targets(t,targets,root,joints,goal,rotation)
        force,arm_info=self.arm.force(force,t,targets,joints,velocities)
        desired=np.asarray([targets.get(n,joints[n]) for n in teacher.names])
        self.hand.targets=(teacher.matrix@desired)[self.hand.indices]
        scale=1. if self.release_started is None else 1.-float(smooth_phase((t-self.release_started)/1.))
        self.hand.preload=self.preload*scale
        reference_velocity=None
        if self.finger_velocity_feedforward:
            # The route specifies joint positions; damping must track their
            # velocity rather than resisting every intentional finger motion.
            # Differentiate only screened finger coordinates, never palm IK.
            current=np.asarray([targets[n] for n in self.finger_names])
            speed=np.zeros_like(current)
            if self.previous_finger_target is not None:
                previous_time,previous=self.previous_finger_target
                dt=t-previous_time
                if not 0<dt<=.05:raise ValueError('Monotonic bounded finger reference clock required')
                speed=(current-previous)/dt
            reference_velocity=dict.fromkeys(teacher.names,0.)
            reference_velocity.update(zip(self.finger_names,speed))
            self.previous_finger_target=(float(t),current.copy())
        force,hand_info=self.hand.force(force,joints,velocities,target_joint_velocities=reference_velocity)
        if self.thumb_pad_feedback and self.release_started is not None:
            if self.thumb_feedback is None:
                from .thumb_withdrawal_feedback import ThumbWithdrawalFeedback
                self.thumb_feedback=ThumbWithdrawalFeedback(teacher,self.thumb_local)
            goal=(1-f)*self.thumb_positions[i]+f*self.thumb_positions[i+1]
            if panel_goal is not None:
                body=self.panel.m.body('robot/rh_thdistal').id
                goal=self.panel.d.xpos[body]+self.panel.d.xmat[body].reshape(3,3)@self.thumb_local
            force,thumb_info=self.thumb_feedback.force(force,t,goal,t-self.release_started,start_handoff=material_handoff)
            hand_info={**hand_info,**thumb_info}
        if self.finger_pad_feedback and self.release_started is not None:
            digit_info={}
            for digit,feedback in self.finger_feedback.items():
                local,points=self.finger_points[digit];goal=(1-f)*points[i]+f*points[i+1]
                if panel_goal is not None:
                    body=self.panel.m.body('robot/rh_'+digit+'distal').id
                    goal=self.panel.d.xpos[body]+self.panel.d.xmat[body].reshape(3,3)@local
                force,digit_info[digit]=feedback.force(force,t,goal,t-self.release_started,start_handoff=material_handoff)
            hand_info={**hand_info,'finger_material_feedback':digit_info}
        if self.middle_feedback_enabled and self.release_started is not None:
            middle_info={}
            for digit,feedback in self.middle_feedback.items():
                local,points=self.middle_points[digit];goal=(1-f)*points[i]+f*points[i+1]
                if panel_goal is not None:
                    body=self.panel.m.body('robot/rh_'+digit+'middle').id
                    goal=self.panel.d.xpos[body]+self.panel.d.xmat[body].reshape(3,3)@local
                force,middle_info[digit]=feedback.force(force,t,goal,t-self.release_started,start_handoff=material_handoff)
            hand_info={**hand_info,'middle_segment_tracking':middle_info}
        if self.release_segment_avoidance is not None and self.release_started is not None:
            force,segment_info=self.release_segment_avoidance.force(force,float(smooth_phase(t-self.release_started)))
            hand_info={**hand_info,**segment_info}
        if self.final_hub_avoidance:
            force,hub_info=self.withdrawal_hub_avoidance.force(force,float(smooth_phase(elapsed)))
            hand_info={**hand_info,**hub_info,'withdrawal_hub_avoidance_applied_after_finger_tracking':True}
        if self.handoff is None:self.handoff=MotorHandoff(self.previous_force,force,teacher.caps,1.)
        force=self.handoff.force(force,elapsed);teacher.last_force=force.copy()
        if panel_goal is not None:
            if self.panel_handoff is None:self.panel_handoff=MotorHandoff(self.panel_previous_force,force,teacher.caps,1.)
            force=self.panel_handoff.force(force,t-self.panel.started);teacher.last_force=force.copy()
        if panel_goal is not None:
            self.panel_schedule.observe(t,panel_info['panel_progress'],angles['leaf'],left_panel_load)
            panel_info['panel_segment_index']=self.panel_schedule.index
        self.info={**panel_info,**info,**self.left.info,**arm_info,**hand_info,**palm_info,'phase':'standing_panel_continuation' if panel_goal is not None else 'standing_withdrawal','withdrawal_started_s':self.started_withdrawal,'release_started_s':self.release_started,'withdrawal_progress':float(smooth_phase(elapsed/self.duration)),'withdrawal_clock_s':clock,'grip_preload_scale':scale,'goal_frame':'attained-resting-world','stance_reference_preserved':True,'stance_reference_root_offset_m':self.stance_root_bias.tolist(),'stance_reference_joint_offset_rad':dict(zip(self.stance_names,self.stance_joint_bias.tolist())),'controller_scope':'Privileged screened upright withdrawal; only original capped motors'}
        self.info['withdrawal_qualification_support_surface']='left_palm_only' if self.palm_only_support else 'left_panel_total'
        self.info.update(coupled_info)
        if coupled_goal is not None:self.info['goal_frame']='measured-leaf-and-handle-with-admitted-world-blend'
        self.info.update(self.motor_capture_info)
        if getattr(self,'inherited_support',None) is not None:
            self.info['inherited_support']=self.inherited_support.info.copy()
        if self.measured_rest:
            self.info.update(self.returned.rest.diagnostic())
            self.info['measured_rest_evidence_epoch']='withdrawal_entry'
        # Explicit Isaac-only observation of existing locals; no reference or
        # feedback is evaluated again, and the original force is returned as-is.
        if getattr(self,'isaac_runtime',None) is not None and getattr(self,'continuation_reference_capture',None) is not None:
            self.continuation_reference_capture.capture_command(self,t,post_feedback_targets=targets,
                right_goal_position=goal,right_goal_rotation=rotation,returned_command=force,
                finger_reference_velocity=reference_velocity,grip_preload_scale=scale,
                measured_root=root,measured_joints=joints,measured_velocities=velocities,
                measured_handle_pose=handle_pose,measured_leaf_pose=leaf_pose,measured_angles=angles)
        return force,self.info
