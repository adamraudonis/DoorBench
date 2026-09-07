"""DoorBench plant with separate sensor observations and privileged diagnostics.

Policy code should receive only copied observation arrays and return an action.
The evaluator owns this object and never gives a student its plant reference.
"""

import numpy as np
import mujoco

from doorbench.benchmark.env import DoorEnv

OBSERVATION_KEYS = frozenset({"joint_position", "joint_velocity", "gravity_body",
    "imu_gyro", "imu_accelerometer", "tactile", "previous_action", "rgb_left", "rgb_right"})


class DexterousDoorEnv:
    def __init__(self, door_dir, robot_xml, audit, *, image_size=128, frame_skip=10):
        self.plant = DoorEnv(str(door_dir), robot_xml=str(robot_xml),
                             robot_base_body="robot/pelvis")
        self.m, self.d = self.plant.m, self.plant.d
        self.m.vis.global_.offwidth = max(960, image_size)
        self.m.vis.global_.offheight = max(720, image_size)
        self.audit = audit
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.renderer = None
        self.actuators = np.array([i for i in range(self.m.nu)
                                 if self.m.actuator(i).name.startswith("robot/")])
        if len(self.actuators) != 61:
            raise ValueError("Robot attachment changed the actuator count")
        self.joints = np.array([i for i in range(self.m.njnt)
            if self.m.joint(i).name.startswith("robot/") and self.m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE])
        self.qadr = self.m.jnt_qposadr[self.joints]
        self.vadr = self.m.jnt_dofadr[self.joints]
        self.root_joint = self.m.joint("robot/free_base").id
        self.root_qadr = int(self.m.jnt_qposadr[self.root_joint])
        self.root_vadr = int(self.m.jnt_dofadr[self.root_joint])
        self.pelvis = self.m.body("robot/pelvis").id
        self.tactile_sensor_ids = {i for i in range(self.m.nsensor)
            if self.m.sensor(i).name.startswith("robot/") and self.m.sensor(i).name.endswith("_touch")}
        self.tactile_indices = np.concatenate([
            np.arange(self.m.sensor_adr[i], self.m.sensor_adr[i] + self.m.sensor_dim[i])
            for i in range(self.m.nsensor)
            if self.m.sensor(i).name.startswith("robot/") and self.m.sensor(i).name.endswith("_touch")])
        self.previous_action = np.zeros(len(self.actuators), dtype=np.float32)
        self.low = self.m.actuator_ctrlrange[self.actuators, 0].copy()
        self.high = self.m.actuator_ctrlrange[self.actuators, 1].copy()

    def reset(self, seed=0, *, randomize=True, images=True):
        self.plant.reset(scenario="open_and_traverse", seed=seed, randomize=randomize)
        for name, value in self.audit["nominal_joint_positions"].items():
            j = self.m.joint("robot/" + name).id
            self.d.qpos[self.m.jnt_qposadr[j]] = value
        self.d.qpos[self.root_qadr + 2] = self.audit["free_root_height"]
        # Scenario yaw is atan2(dy, dx), matching H1's +x-forward convention.
        if not randomize:
            start = self.plant.scenario()["start"]
            self.d.qpos[self.root_qadr:self.root_qadr + 2] = start["center"][:2]
            yaw = start["yaw"]
            self.d.qpos[self.root_qadr + 3:self.root_qadr + 7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        self.d.ctrl[self.actuators] = 0
        for i in self.actuators:
            if self.m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                j = self.m.actuator_trnid[i, 0]
                self.d.ctrl[i] = self.d.qpos[self.m.jnt_qposadr[j]]
        mujoco.mj_forward(self.m, self.d)
        self.previous_action = self.normalize(self.d.ctrl[self.actuators]).astype(np.float32)
        return self.observe(images=images)

    def normalize(self, control):
        return 2 * (np.asarray(control) - self.low) / (self.high - self.low) - 1

    def denormalize(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.low.shape or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain 61 finite normalized actuator commands")
        return self.low + (np.clip(action, -1, 1) + 1) * .5 * (self.high - self.low)

    def observe(self, *, images=True):
        def sensor(name):
            return self.d.sensor("robot/" + name).data.copy().astype(np.float32)
        obs = {
            "joint_position": self.d.qpos[self.qadr].astype(np.float32),
            "joint_velocity": self.d.qvel[self.vadr].astype(np.float32),
            "gravity_body": (self.d.xmat[self.pelvis].reshape(3, 3).T @ [0., 0., -1.]).astype(np.float32),
            "imu_gyro": sensor("imu_gyro"), "imu_accelerometer": sensor("imu_accelerometer"),
            "tactile": np.clip(self.d.sensordata[self.tactile_indices], -100., 100.).astype(np.float32),
            "previous_action": self.previous_action.copy(),
        }
        if images:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.m, height=self.image_size, width=self.image_size)
            options = mujoco.MjvOption()
            options.sitegroup[:] = 0
            for side in ("left", "right"):
                self.renderer.update_scene(self.d, camera=f"robot/{side}_eye_camera", scene_option=options)
                self.hide_sensor_overlays(self.renderer.scene)
                obs[f"rgb_{side}"] = self.renderer.render().copy()
        if set(obs) - OBSERVATION_KEYS:
            raise AssertionError("Unexpected policy observation")
        return obs

    def hide_sensor_overlays(self, scene):
        # touch_grid adds debug boxes even when sites are hidden. Hide only its
        # visual decorations; never modify sensor values or collision geometry.
        for geom in scene.geoms[:scene.ngeom]:
            if (geom.objtype == mujoco.mjtObj.mjOBJ_UNKNOWN and
                    geom.category == mujoco.mjtCatBit.mjCAT_DECOR and
                    geom.objid in self.tactile_sensor_ids):
                geom.rgba[3] = 0.

    def step(self, action, *, images=True):
        control = self.denormalize(action)
        self.previous_action = np.clip(action, -1, 1).astype(np.float32)
        for _ in range(self.frame_skip):
            self.d.ctrl[self.actuators] = control
            self.plant.step()
        return self.observe(images=images)

    def diagnostics(self):
        up = self.d.xmat[self.m.body("robot/torso_link").id].reshape(3, 3)[:, 2]
        warnings=sum(int(self.d.warning[w].number) for w in (mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC,mujoco.mjtWarning.mjWARN_BADCTRL))
        return {"sim_time_s": float(self.d.time), "door_q": float(self.plant._door_q()),
                "root_height_m": float(self.d.qpos[self.root_qadr + 2]),
                "torso_tilt_deg": float(np.rad2deg(np.arccos(np.clip(up[2], -1., 1.)))),
                "contact_count": self.d.ncon, "finite": all(bool(np.all(np.isfinite(x))) for x in
                    (self.d.qpos,self.d.qvel,self.d.ctrl,self.d.sensordata)),
                "numerical_warnings":warnings,
                "external_wrench_max": float(np.max(np.abs(self.d.xfrc_applied))),
                "applied_generalized_force_max": float(np.max(np.abs(self.d.qfrc_applied)))}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
