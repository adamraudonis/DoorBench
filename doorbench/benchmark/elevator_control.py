"""Measured interlock sequencing for a stationary elevator landing fixture.

The call button requests service. Hook position establishes release. Every
mechanical action uses a bounded native motor/cam force; power loss zeros those
inputs without modifying joints, contacts, equalities or native state.
"""
import numpy as np


class ElevatorControl:
    def __init__(self,env):
        self.env=env;m,d=env.m,env.d
        self.assembly=env.meta['elevator_interlocks'];self.rows=[]
        for r in self.assembly['leaves']:
            qa={k:int(m.jnt_qposadr[env._jid(r[k])]) for k in ('joint','hook_joint','cam_joint')}
            va={k:int(m.jnt_dofadr[env._jid(r[k])]) for k in ('joint','cam_joint')}
            self.rows.append((r,qa,va,m.actuator(r['leaf']+'_drive').id))
        self.buttons=[(int(m.jnt_qposadr[env._jid(r['joint'])]),.8*r['travel_m'])
                      for r in env.meta['wall_switches'] if r['kind']=='call_button']
        if not self.buttons:raise ValueError('Elevator needs its actual call switch')
        act=env.spec['kinematics']['actuator']
        self.open_speed=float(act['open_speed_m_s']);self.close_speed=float(act['close_speed_m_s'])
        self.hold=float(act['hold_open_s']);self.targets=[float(d.qpos[q['joint']]) for _,q,_,_ in self.rows]
        self.state='release_open' if env._was_open else 'idle'
        self.pressed=False;self.opened_at=self.last_presence=float(d.time);self.stall_since=None
        self.supply_lost=False

    def step(self,base_xy,t):
        e=self.env;m,d=e.m,e.d;dt=float(m.opt.timestep)
        for _,_,_,aid in self.rows:d.ctrl[aid]=0.
        pressed=any(float(d.qpos[q])>=threshold for q,threshold in self.buttons)
        request=pressed and not self.pressed;self.pressed=pressed
        if e.elevator_power is False:
            self.supply_lost=True;self.state='idle';return
        if self.supply_lost:
            # Restoration supplies power but is not an unrequested call.
            self.supply_lost=False;self.targets=[float(d.qpos[q['joint']]) for _,q,_,_ in self.rows]
        seated=all(abs(float(d.qpos[q['joint']]))<r['seated_m'] for r,q,_,_ in self.rows)
        released=all(float(d.qpos[q['hook_joint']])>=r['released_angle_rad'] for r,q,_,_ in self.rows)
        if released:e.tracker.L.lock_released=True
        if request and self.state=='idle':
            self.state='seat' if max(abs(float(d.qpos[q['joint']])) for _,q,_,_ in self.rows)<.01 else 'release_open'
        lane=float(d.xpos[m.body(self.rows[0][0]['leaf']).id,1])
        xy=np.asarray(base_xy,float)
        present=bool(xy.shape==(2,) and np.isfinite(xy).all() and abs(xy[0])<e.spec['opening']['width']/2+.30 and abs(xy[1]-lane)<.50)
        if present:self.last_presence=t
        if self.state=='idle':return
        if self.state=='seat' and seated:self.state='release'
        if self.state in ('release','release_open') and released:
            self.state='opening';self.targets=[float(d.qpos[q['joint']]) for _,q,_,_ in self.rows]
        if self.state=='opening' and all(float(d.qpos[q['joint']])>=r['stroke_m']-.003 for r,q,_,_ in self.rows):
            self.state='hold';self.opened_at=t
        if self.state=='hold' and t-max(self.opened_at,self.last_presence)>=self.hold:self.state='closing'
        if self.state=='closing':
            # Ideal doorway occupancy sensor plus measured drive overload.
            # This is not a certified light curtain or safety control.
            stalled=any(float(d.actuator_force[aid])<-108. and float(d.qvel[v['joint']])>-.015 and float(d.qpos[q['joint']])>.02
                        for _,q,v,aid in self.rows)
            self.stall_since=(self.stall_since if self.stall_since is not None else t) if stalled else None
            if present or request or (self.stall_since is not None and t-self.stall_since>.20):
                self.state='opening';self.targets=[float(d.qpos[q['joint']]) for _,q,_,_ in self.rows];self.stall_since=None
            elif seated and all(abs(float(d.qvel[v['joint']]))<.03 for _,_,v,_ in self.rows):self.state='locking'
        if self.state=='locking' and all(abs(float(d.qpos[q['hook_joint']]))<.015 and abs(float(d.qpos[q['cam_joint']]))<.001 for _,q,_,_ in self.rows):
            self.state='idle';return
        cam_on=self.state in ('release','release_open','opening','hold','closing')
        for i,(r,q,v,aid) in enumerate(self.rows):
            if cam_on:d.qfrc_applied[v['cam_joint']]+=r['max_cam_force_N']
            if self.state in ('seat','release','locking'):self.targets[i]=-.020
            elif self.state=='opening':self.targets[i]+=float(np.clip(r['stroke_m']+.020-self.targets[i],0.,self.open_speed*dt))
            elif self.state=='hold':self.targets[i]=r['stroke_m']+.020
            elif self.state=='closing':self.targets[i]+=float(np.clip(-.020-self.targets[i],-self.close_speed*dt,0.))
            # release_open keeps the current position while its hook lifts.
            d.ctrl[aid]=float(np.clip(400.*(self.targets[i]-d.qpos[q['joint']])-60.*d.qvel[v['joint']],-135.,135.))
