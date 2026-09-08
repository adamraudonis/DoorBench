"""Local-distal-touch preload and progression over a fixed joint press route.

The controller sees no object IDs, lever angle, world pose or active plant.
Local palmar force projections are control signals, not anatomical grasp labels.
"""
import numpy as np
from .sensor_contract import validate_actor_packet


class SensorDistalTouchController:
    def __init__(self,arm_controller,operation_schedule,sensor_layout):
        self.arm=arm_controller;self.schedule=operation_schedule;self.shapes=self.arm.shapes
        if self.schedule.start_s!=19. or self.schedule.press_seconds!=8.:raise ValueError('This version preserves the qualified19s acquisition and8s press')
        if sensor_layout['channel_order']!=['z','x','y'] or sensor_layout['layout']!='sensor order, then channel, vertical bin, horizontal bin':raise ValueError('Unexpected tactile channel order')
        native=self.arm.balance.m
        actual_touch=[native.sensor(i).name for i in range(native.nsensor) if native.sensor(i).name.endswith('_touch')]
        if [r['name'] for r in sensor_layout['sensors']]!=actual_touch or any(r['dimension']!=native.sensor(r['name']).dim[0] for r in sensor_layout['sensors']):raise ValueError('Tactile offsets differ from the admitted robot')
        self.digits=('ff','mf','rf','lf','th');self.slices={};offset=0
        for row in sensor_layout['sensors']:
            dim=row['dimension']
            for digit in self.digits:
                if row['name']=='rh_'+digit+'distal_touch':
                    if row['body_name']!='rh_'+digit+'distal' or (row['width'],row['height'],dim)!=(4,2,24) or row['fov_degrees']!=[180.,90.] or row['gamma']!=0:
                        raise ValueError('Expected calibrated four-by-two distal tactile grids')
                    wanted=[0.,0.,.02 if digit=='th' else .017]
                    if not np.allclose(row['position_body_m'],wanted,atol=1e-12,rtol=0) or not np.allclose(row['quaternion_xyzw_body'],[-2**-.5,0,0,2**-.5],atol=1e-12,rtol=0):raise ValueError('Distal palmar frame calibration mismatch')
                    if digit in self.slices:raise ValueError('Duplicate distal sensor')
                    self.slices[digit]=slice(offset,offset+dim)
            offset+=dim
        if set(self.slices)!=set(self.digits) or offset!=self.shapes['tactile'][0]:raise ValueError('Complete distal tactile layout required')
        self.target=np.array([2.,2.,2.,2.,3.]);self.minimum=np.array([.8,.8,.8,.8,1.2])
        self.maximum=np.array([.06,.06,.06,.06,.08]);self.dt=.002;self.tau=.02
        self.integral_gain=.015;self.maximum_rate=.03;self.preload_seconds=2.;self.required_stable_seconds=.1
        self.press_joint_indices=[self.arm.joint_names.index(n) for n in self.schedule.press_names]
        self.reset_episode()

    @property
    def last_force(self):return self.arm.last_force
    @property
    def last_info(self):return dict(self.info)
    @property
    def goal_names(self):return self.arm.goal_names
    @property
    def joint_names(self):return self.arm.joint_names
    @property
    def action_names(self):return self.arm.action_names
    @property
    def caps(self):return self.arm.caps

    def reset_episode(self):
        self.arm.reset_episode();self.last_time=None;self.failed_reason=None
        self.filtered=np.zeros(5);self.delta=np.zeros(5);self.virtual_s=19.;self.good_since=None;self.finished_at=None;self.info={}

    def local_distal_loads(self,packet):
        # Sensor -Z faces body -Y; the inner hemisphere occupies columns1/2.
        # Positive sensor-Z is the force into a distal palmar surface. Finite
        # bins cannot establish counterpart identity or the original axial pad
        # band; those remain independent evaluator checks.
        return np.array([max(0.,float(packet['tactile'][self.slices[d]].reshape(3,2,4)[0,:,1:3].sum())) for d in self.digits])

    def closure_integration_mask(self,raw):
        """Legacy profile integrates every bounded closure coordinate."""
        return np.ones(5,dtype=bool)

    def progression_contact_ready(self,raw):
        """Legacy profile uses its filtered load and encoder gates below."""
        return True

    def force(self,packet,*,now_s):
        if self.failed_reason is not None:raise RuntimeError('Touch controller requires reset_episode: '+self.failed_reason)
        try:
            if isinstance(now_s,(bool,np.bool_)):raise ValueError('Clock must be numeric seconds, not boolean')
            validate_actor_packet(packet,self.shapes,61);now=float(now_s)
            if not np.isfinite(now) or now<0 or (self.last_time is None and now!=0.) or (self.last_time is not None and abs(now-self.last_time-self.dt)>1e-8):raise ValueError('Exact2ms episode clock required')
            if now<19.-1e-9:
                goals=self.schedule.goals(now);raw=np.zeros(5);stage='acquisition';ready=False;tracking=True
            else:
                if not packet['sensor_valid'][4] or not 0<=now-packet['sensor_time_s'][4]<=.006+1e-9:raise ValueError('Fresh local distal touch is required')
                raw=self.local_distal_loads(packet)
                if self.last_time<19.-1e-9:self.filtered=raw.copy()
                else:self.filtered+=(-np.expm1(-self.dt/self.tau))*(raw-self.filtered)
                self.delta=np.clip(self.delta+self.closure_integration_mask(raw)*self.dt*np.clip(self.integral_gain*(self.target-self.filtered),-self.maximum_rate,self.maximum_rate),0.,self.maximum)
                prior=self.schedule.goals(self.virtual_s)
                tracking=bool(max(abs(packet['joint_position'][i]-prior[n]) for n,i in zip(self.schedule.press_names,self.press_joint_indices))<.03)
                ready=bool(np.all(self.filtered>=self.minimum) and tracking and self.progression_contact_ready(raw))
                if ready:
                    if self.good_since is None:self.good_since=now
                else:self.good_since=None
                if now>=19.+self.preload_seconds and self.good_since is not None and now-self.good_since>=self.required_stable_seconds:
                    self.virtual_s=min(27.,self.virtual_s+self.dt)
                if self.virtual_s>=27.-1e-9 and self.finished_at is None:self.finished_at=now
                goals=self.schedule.goals(self.virtual_s)
                for digit,delta in zip(self.digits,self.delta):
                    if digit=='th':goals['rh_THJ1']+=delta
                    else:
                        for joint in ('1','2'):goals['rh_'+digit.upper()+'J'+joint]+=.5*delta
                stage='preload' if self.virtual_s==19. else 'press' if self.finished_at is None else 'hold'
            force,base=self.arm.force(packet,now_s=now,joint_goals=goals)
            self.last_time=now
            self.info=dict(base,high_level_controller='local_distal_touch_preload_v1',high_level_source='instance-specific joint route with local tactile feedback',
                touch_stage=stage,distal_projected_force_N=raw.tolist(),filtered_distal_projected_force_N=self.filtered.tolist(),
                finger_motor_closure_offsets_rad=self.delta.tolist(),virtual_press_clock_s=self.virtual_s,
                local_touch_progression_ready=ready,encoder_press_tracking_ready=tracking,press_completed_clock_s=self.finished_at,
                touch_scope='calibrated distal palmar hemisphere projection only; no counterpart or anatomical ground truth')
            return force,dict(self.info)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc);raise
