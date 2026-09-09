"""Finite own-sensor capture during an actual native robot rollout.

Reads the live plant but never steps it or commands it. Encoder/camera samples
use the refreshed post-step state. MuJoCo IMU and tactile values belong to the
preceding dynamics epoch and retain that timestamp. No simulator geometry,
gravity oracle, contact identities or teacher phase enters an actor packet.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import mujoco
from .sensor_contract import ActorObservationBuilder, SENSOR_KEYS


class NativeSensorCapture:
    def __init__(self, sim, motors, layout, output, *, control_source='privileged_teacher'):
        if control_source not in ('privileged_teacher', 'sensor_actor'):
            raise ValueError('Declare teacher or sensor actor explicitly')
        self.control_source = control_source
        self.initial_recorded = False
        self.sim = sim
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        names=[sim.m.joint(int(j)).name.removeprefix('robot/') for j in sim.joints]
        actions=[sim.m.actuator(int(i)).name.removeprefix('robot/') for i in sim.actuators]
        if names!=layout['joint_order'] or actions!=layout['action_order'] or actions!=[a['name'] for a in motors['actuators']]:
            raise ValueError('Native sensor and original actuator order differ')
        self.caps=np.array([a['force_range'] for a in motors['actuators']],float)
        self.builder=ActorObservationBuilder(joint_count=len(names),action_count=len(actions),
            tactile_dimension=len(sim.tactile_indices),image_shape=(sim.image_size,sim.image_size,3))
        if layout['tactile_dimension']!=len(sim.tactile_indices):raise ValueError('Native tactile layout differs')
        self.layout=dict(layout,capture_clock_profile='native-preintegration-inertial-tactile-v1')
        (self.output/'layout.json').write_text(json.dumps(self.layout,indent=2)+'\n')
        self.rows=[];self.chunks=[];self.frames=[];self.frame_times=[];self.count=0;self.last_time=None
        self.numeric_keys=[k for k in SENSOR_KEYS if not k.startswith('rgb_')]+['previous_action','sensor_time_s','sensor_valid']
        self._report(False)

    def initial_packet(self):
        """Observe the actual reset; do not invent initial IMU/touch history."""
        if self.count or self.initial_recorded or self.sim.d.time != 0:
            raise ValueError('Initial observation requires the untouched episode reset')
        mujoco.mj_camlight(self.sim.m, self.sim.d)
        observation = self.sim.observe(images=True)
        for key in ('joint_position', 'joint_velocity', 'rgb_left', 'rgb_right'):
            self.builder.push(key, observation[key], capture_s=0.)
        packet = self.builder.observe(now_s=0., previous_action=np.zeros(61))
        np.savez_compressed(self.output/'actor-initial-decision.npz', **packet, time_s=np.asarray(0.))
        self.initial_recorded = True
        return packet

    def capture(self, *, start_s, end_s, actual_forces):
        if self.last_time is not None and abs(start_s-self.last_time)>1e-8:
            raise ValueError('Native sensor capture cannot skip a physical interval')
        if abs(self.sim.d.time-end_s)>1e-8 or abs(end_s-start_s-self.sim.m.opt.timestep)>1e-8:
            raise ValueError('Require the actual completed native interval')
        images=self.count%20==0
        if images:mujoco.mj_camlight(self.sim.m,self.sim.d)
        observation=self.sim.observe(images=images)
        for key in SENSOR_KEYS:
            if key not in observation:continue
            capture_time=end_s if key.startswith(('joint_','rgb_')) else start_s
            self.builder.push(key,observation[key],capture_s=capture_time,available_s=end_s)
        if images:
            self.frames.append({k:observation[k] for k in ('rgb_left','rgb_right')});self.frame_times.append(end_s)
        forces=np.asarray(actual_forces,float)
        if forces.shape!=(61,) or not np.isfinite(forces).all() or np.any(forces<self.caps[:,0]-1e-5) or np.any(forces>self.caps[:,1]+1e-5):
            raise ValueError('Require actual original-capped delivered motor forces')
        previous_action=2*(forces-self.caps[:,0])/(self.caps[:,1]-self.caps[:,0])-1
        packet=self.builder.observe(now_s=end_s,previous_action=previous_action)
        self.rows.append(dict(time_s=end_s,**{k:packet[k] for k in self.numeric_keys}))
        self.count+=1;self.last_time=end_s
        if len(self.rows)>=250:self.flush()
        return packet

    def flush(self):
        if not self.rows:return
        name=f'sensors-{len(self.chunks):05d}.npz';path=self.output/name
        arrays={k:np.asarray([row[k] for row in self.rows]) for k in ['time_s',*self.numeric_keys]}
        with path.with_suffix('.writing').open('wb') as f:np.savez_compressed(f,**arrays)
        path.with_suffix('.writing').replace(path)
        self.chunks.append(dict(file=name,rows=len(self.rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.rows=[];self._report(False)

    def _report(self,complete):
        result=dict(schema='doorbench.native-sensor-capture.v1',capture_complete=complete,
            control_source=self.control_source,scope=__doc__,samples=self.count,last_time_s=self.last_time,
            chunks=self.chunks,camera_frames=len(self.frame_times),camera_stride_steps=20,
            startup_observation_recorded=self.initial_recorded,
            camera_pose_refresh='mj_camlight after current-state kinematics, before render',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            layout_sha256=hashlib.sha256((self.output/'layout.json').read_bytes()).hexdigest(),
            rgb_sha256=hashlib.sha256((self.output/'actor-rgb.npz').read_bytes()).hexdigest() if complete and self.frames else None,
            limitation='Recorded sensors only; task, sensor and training qualification require separate audits')
        tmp=self.output/'report.writing';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(self.output/'report.json')

    def finish(self,*,complete):
        self.flush()
        if self.frames:
            np.savez_compressed(self.output/'actor-rgb.npz',time_s=np.asarray(self.frame_times),
                **{k:np.asarray([f[k] for f in self.frames]) for k in ('rgb_left','rgb_right')})
            from PIL import Image
            for i in sorted({0,len(self.frames)//4,len(self.frames)//2,len(self.frames)-1}):
                for key in ('rgb_left','rgb_right'):Image.fromarray(self.frames[i][key]).save(self.output/f'{key}-{i:05d}.png')
        self._report(bool(complete and self.count))
