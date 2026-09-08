"""Measured-state H1 approach/lowering teacher for another simulation backend.

The native model is an unstepped FK/dynamics calculator. Actual robot root,
joint states and foot loads are supplied by the active simulator; only motor
forces are returned. All robot/contact safety gates belong to that simulator.
This is privileged development control, not a vision/tactile policy.
"""
from types import SimpleNamespace
from pathlib import Path
import hashlib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .approach_lowering import ApproachLoweringController
from .locomotion import NativeH1MotorAdapter
from .grasp_verification import scalar_transmission_matrix


class ApproachBodyTeacher:
    def __init__(self,robot_xml,motors,reset,checkpoint,**controller_options):
        if motors.get('hand_mechanics_profile')!='shadow-loopback-v2':raise ValueError('Corrected hand model required')
        if hashlib.sha256(Path(robot_xml).read_bytes()).hexdigest()!=motors.get('source_xml_sha256'):
            raise ValueError('Native dynamics mirror differs from motor contract')
        m=mujoco.MjModel.from_xml_path(str(robot_xml));d=mujoco.MjData(m)
        self.m=m;self.d=d;self.names=motors['joint_names']
        self.joints=np.array([m.joint(n).id for n in self.names]);self.qa=m.jnt_qposadr[self.joints];self.va=m.jnt_dofadr[self.joints]
        self.act=np.array([m.actuator(v['name']).id for v in motors['actuators']])
        self.matrix=scalar_transmission_matrix(m,self.act,self.joints)
        self.kp=np.array([v['kp'] for v in motors['actuators']]);self.bias=np.array([v['bias'] for v in motors['actuators']])
        self.caps=np.array([v['force_range'] for v in motors['actuators']]);self.ranges=np.array([v['control_range'] for v in motors['actuators']])
        declared=np.array([[v['terms'].get(n,0.) for n in self.names] for v in motors['actuators']])
        comparisons=((self.matrix,declared),(self.kp,m.actuator_gainprm[self.act,0]),
            (self.bias,m.actuator_biasprm[self.act,:3]),(self.caps,m.actuator_forcerange[self.act]),
            (self.ranges,m.actuator_ctrlrange[self.act]))
        if any(not np.array_equal(actual,expected) for actual,expected in comparisons):
            raise ValueError('Imported actuator contract differs from original native model')
        if m.nu!=61 or len(self.names)!=69 or len(motors.get('passive_tendons',[]))!=8:
            raise ValueError('Expected full audited H1/Shadow v2 embodiment')
        root=np.r_[reset['initial_root'],np.zeros(6)]
        self._state(0.,root,reset['joints'],{n:0. for n in self.names})
        d.ctrl[self.act]=[reset['motor_targets'][v['name']] for v in motors['actuators']]
        sim=SimpleNamespace(m=m,d=d,joint_prefix='',actuators=self.act,root_qadr=0,root_vadr=0,
            pelvis=m.body('pelvis').id,feet=[m.body(side+'_ankle_link').id for side in ('left','right')],
            adapter=NativeH1MotorAdapter(m,prefix=''),external_generalized_force=np.zeros(m.nv))
        self.controller=ApproachLoweringController(sim,checkpoint,reset['goal_xy'],reset['goal_yaw_rad'],**controller_options)
        self.last_t=None;self.calls=0

    def _state(self,t,root,joints,velocities):
        root=np.asarray(root,float);q=np.array([joints[n] for n in self.names]);dq=np.array([velocities[n] for n in self.names])
        if root.shape!=(13,) or not np.isfinite(np.r_[t,root,q,dq]).all():raise ValueError('Invalid measured robot state')
        rotation=Rotation.from_quat([*root[4:7],root[3]]).as_matrix()
        self.d.time=float(t);self.d.qpos[:7]=root[:7];self.d.qpos[self.qa]=q
        self.d.qvel[:3]=root[7:10];self.d.qvel[3:6]=rotation.T@root[10:13];self.d.qvel[self.va]=dq
        mujoco.mj_forward(self.m,self.d)

    def force(self,t,root,joints,velocities,foot_loads,hand_forces=None):
        """Return 61 original capped forces; root angular velocity is world-frame.

        Invoke exactly once per active 2ms physics step. Foot loads are measured
        world-up support forces [left,right], not fabricated support constraints.
        Optional measured hand/body forces inform the analytic stance at body
        origins, without applied live forces. Contact moments remain an explicit
        approximation to validate in the full interaction rollout.
        """
        if self.last_t is not None and not np.isclose(t-self.last_t,.002,rtol=0,atol=1e-7):
            raise ValueError('Body teacher requires an uninterrupted 2 ms clock')
        loads=np.asarray(foot_loads,float)
        if loads.shape!=(2,) or not np.isfinite(loads).all():raise ValueError('Two finite measured foot loads required')
        self._state(t,root,joints,velocities)
        external=self.controller.sim.external_generalized_force;external[:]=0.
        if hand_forces:
            jp=np.zeros((3,self.m.nv));jr=jp.copy()
            for name,value in hand_forces.items():
                load=np.asarray(value,float)
                if load.shape!=(3,) or not np.isfinite(load).all():raise ValueError('Finite measured world hand force required')
                body=self.m.body(name.rsplit('/',1)[-1]).id
                mujoco.mj_jacBody(self.m,self.d,jp,jr,body);external+=jp.T@load
        controls=self.controller.command(loads)[self.act]
        length=self.matrix@self.d.qpos[self.qa];speed=self.matrix@self.d.qvel[self.va]
        force=np.clip(self.kp*controls+self.bias[:,0]+self.bias[:,1]*length+self.bias[:,2]*speed,self.caps[:,0],self.caps[:,1])
        if not np.isfinite(force).all():raise ValueError('Nonfinite motor command')
        self.last_t=float(t);self.calls+=1
        return force,dict(stage=self.controller.stage,stance_solver=self.controller.solver_status,
            stance_solver_failures=self.controller.solver_failures,clock_s=float(t),native_mirror_steps=0)
