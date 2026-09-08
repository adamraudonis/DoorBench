"""Pinned official H1 locomotion actor and an unchanged-plant motor adapter.

Observation/action semantics adapted from Unitree Robotics' BSD-3-Clause
unitree_rl_gym deploy/deploy_mujoco/deploy_mujoco.py and configs/h1.yaml at
276801e46c5d433564f24658bac64f254b7d2d4b. See
``docs/licenses/UNITREE_RL_GYM_LICENSE.txt`` for the retained copyright/license.
DoorBench modifications: strict provenance/shapes, named joint mapping, native
force/control bounds, and separation of actor observations from simulator state.
"""
from pathlib import Path
import hashlib
import numpy as np

UPSTREAM_URL = 'https://github.com/unitreerobotics/unitree_rl_gym.git'
UPSTREAM_REVISION = '276801e46c5d433564f24658bac64f254b7d2d4b'
POLICY_RELATIVE_PATH = 'deploy/pre_train/h1/motion.pt'
POLICY_SHA256 = '44a0fbceb81f3877833ae9a398d039bea1759cb0d3c8188181013885f70589eb'
JOINT_NAMES = tuple(side+'_'+joint for side in ('left','right') for joint in
                    ('hip_yaw','hip_roll','hip_pitch','knee','ankle'))
DEFAULT_ANGLES = np.array([0.,0.,-.1,.3,-.2]*2)
KP = np.array([150.,150.,150.,200.,40.]*2)
KD = np.array([2.,2.,2.,4.,2.]*2)
CONTROL_PERIOD = .02


def finite_vector(value, count, name):
    value=np.asarray(value,dtype=float)
    if value.shape!=(count,) or not np.isfinite(value).all():
        raise ValueError(f'{name} must contain {count} finite values')
    return value


class H1WalkingPolicy:
    """41-input, 10-output proprioceptive actor; never receives a plant object.

Inputs are leg encoders, body-frame angular velocity and projected gravity,
requested body-frame velocity, and the controller's clock. It does not observe
root position, linear velocity, targets/doors, images, touch, or contact labels.
A waypoint planner that supplies commands may still be privileged; this class
alone does not establish a sensor-only door controller.
    """
    def __init__(self, checkpoint):
        import torch
        checkpoint=Path(checkpoint)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=POLICY_SHA256:
            raise ValueError('Expected the pinned official H1 checkpoint; refusing other robot/weights')
        torch.set_num_threads(1)
        self.policy=torch.jit.load(str(checkpoint),map_location='cpu').eval()
        self.previous_action=np.zeros(10,dtype=np.float32)
        self.last_observation=None

    def reset(self):
        self.previous_action[:]=0.;self.last_observation=None

    def step(self, joint_position, joint_velocity, angular_velocity, gravity_body,
             command, elapsed_seconds):
        import torch
        q=finite_vector(joint_position,10,'joint_position')
        dq=finite_vector(joint_velocity,10,'joint_velocity')
        gyro=finite_vector(angular_velocity,3,'angular_velocity')
        gravity=finite_vector(gravity_body,3,'gravity_body')
        velocity=finite_vector(command,3,'command')
        if not np.isfinite(elapsed_seconds) or elapsed_seconds<0:
            raise ValueError('elapsed_seconds must be finite and nonnegative')
        if not .9<=np.linalg.norm(gravity)<=1.1:
            raise ValueError('gravity_body must be a normalized projected direction')
        phase=elapsed_seconds%.8/.8
        obs=np.r_[gyro*.25,gravity,velocity*[2.,2.,.25],q-DEFAULT_ANGLES,dq*.05,
                  self.previous_action,np.sin(2*np.pi*phase),np.cos(2*np.pi*phase)].astype(np.float32)
        with torch.inference_mode():
            action=self.policy(torch.from_numpy(obs).unsqueeze(0)).numpy().reshape(-1)
        action=finite_vector(action,10,'policy output')
        self.previous_action=action.astype(np.float32)
        self.last_observation=obs.copy()
        return DEFAULT_ANGLES+.25*action

    @staticmethod
    def torques(target, joint_position, joint_velocity):
        return KP*(finite_vector(target,10,'target')-finite_vector(joint_position,10,'joint_position'))-KD*finite_vector(joint_velocity,10,'joint_velocity')


class NativeH1MotorAdapter:
    """Map desired leg torques to the *unchanged* audited native servo motors.

The affine servo's inverse is evaluated every physics step. Physical model,
actuator gain/bias arrays, force limits, joint limits and root are never changed.
For Isaac's torque-mode adapter, use ``bounded_torques`` with the same caps.
    """
    def __init__(self, model, *, prefix='robot/'):
        import mujoco
        self.model=model
        self.joints=np.array([model.joint(prefix+n).id for n in JOINT_NAMES])
        self.actuators=np.array([model.actuator(prefix+n).id for n in JOINT_NAMES])
        self.qadr=model.jnt_qposadr[self.joints].copy()
        self.vadr=model.jnt_dofadr[self.joints].copy()
        for jid,aid in zip(self.joints,self.actuators):
            if model.jnt_type[jid]!=mujoco.mjtJoint.mjJNT_HINGE or model.actuator_trntype[aid]!=mujoco.mjtTrn.mjTRN_JOINT or model.actuator_trnid[aid,0]!=jid or not np.allclose(model.actuator_gear[aid],[1.,0.,0.,0.,0.,0.]) or not model.actuator_forcelimited[aid] or not model.actuator_ctrllimited[aid]:
                raise ValueError('H1 motor mapping/transmission or bounds differ from audited structure')
        self.force_limits=model.actuator_forcerange[self.actuators].copy()
        self.control_limits=model.actuator_ctrlrange[self.actuators].copy()
        self.gain=model.actuator_gainprm[self.actuators,0].copy()
        self.bias=model.actuator_biasprm[self.actuators,:3].copy()
        if np.any(self.gain<=0):raise ValueError('Native adapter requires positive affine servo gain')

    def bounded_torques(self, target, q, dq):
        return np.clip(H1WalkingPolicy.torques(target,q,dq),self.force_limits[:,0],self.force_limits[:,1])

    def command(self, data, target):
        torque=self.bounded_torques(target,data.qpos[self.qadr],data.qvel[self.vadr])
        bias=self.bias[:,0]+self.bias[:,1]*data.actuator_length[self.actuators]+self.bias[:,2]*data.actuator_velocity[self.actuators]
        return np.clip((torque-bias)/self.gain,self.control_limits[:,0],self.control_limits[:,1])
