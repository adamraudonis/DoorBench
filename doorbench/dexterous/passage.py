"""Privileged Door55 passage guard and bounded waypoint steering.

Geometry is queried on separate unstepped data. A static corridor screen cannot
certify a future gait: the caller must still reject every physical-step contact
or limit failure and require whole-body clearance plus a quiet terminal state.
"""
import mujoco
import numpy as np
from .locomotion_approach import WaypointApproach


class RobotBounds:
    """Conservative collision-mesh bounds with cached local corners only."""
    def __init__(self,m):
        self.geoms=np.array([g for g in range(m.ngeom) if (m.geom_contype[g]|m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name.startswith('robot/')])
        if not len(self.geoms):raise ValueError('Expected robot collision geometry')
        box=m.geom_aabb[self.geoms]
        signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        self.corners=box[:,None,:3]+signs[None]*box[:,None,3:]
    def __call__(self,d):
        rotations=d.geom_xmat[self.geoms].reshape(-1,3,3)
        points=np.einsum('nkj,nij->nki',self.corners,rotations)+d.geom_xpos[self.geoms,None,:]
        if not np.isfinite(points).all():raise ValueError('Nonfinite actual robot collision geometry')
        return points.min(axis=(0,1)),points.max(axis=(0,1))


def robot_bounds(m,d):
    return RobotBounds(m)(d)


def screen_translation(sim,goal,*,templates=(),margin=.003,samples=61):
    """Check actual articulation shape on a static measured-door XY corridor."""
    m=sim.m;active=sim.d;d=mujoco.MjData(m);goal=np.asarray(goal,float)
    if goal.shape!=(2,) or not np.isfinite(goal).all() or samples<31 or margin<.003:raise ValueError('Invalid bounded passage screen')
    start=active.qpos.copy();root=sim.root_qadr;robot=[g for g in range(m.ngeom) if (m.geom_contype[g]|m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name.startswith('robot/')]
    env=[g for g in range(m.ngeom) if (m.geom_contype[g]|m.geom_conaffinity[g]) and not m.body(m.geom_bodyid[g]).name.startswith('robot/') and m.geom(g).name!='floor']
    pairs=[(g,h) for g in robot for h in env if (m.geom_contype[g]&m.geom_conaffinity[h]) or (m.geom_contype[h]&m.geom_conaffinity[g])]
    poses=[start]
    for template in templates:
        template=np.asarray(template,float)
        if template.shape!=start.shape or not np.isfinite(template).all():raise ValueError('Malformed measured gait template')
        q=start.copy();q[sim.qadr]=template[sim.qadr]
        q[root+2:root+7]=template[root+2:root+7]
        poses.append(q)
    minimum=.05;bad=[]
    for template_index,pose in enumerate(poses):
        for u in np.linspace(0,1,samples):
            d.qpos[:]=pose;d.qpos[root:root+2]=start[root:root+2]*(1-u)+goal*u;mujoco.mj_kinematics(m,d)
            for g,h in pairs:
                if np.linalg.norm(d.geom_xpos[g]-d.geom_xpos[h])-m.geom_rbound[g]-m.geom_rbound[h]>.05:continue
                gap=float(mujoco.mj_geomDistance(m,d,g,h,.05,None));minimum=min(minimum,gap)
                if gap<margin:
                    bad.append(dict(template=template_index,fraction=float(u),robot_geom=m.geom(g).name,environment_geom=m.geom(h).name,gap_m=gap))
                    break
            if bad:break
        if bad:break
    leaf=float(active.qpos[m.jnt_qposadr[m.joint('leaf_hinge').id]])
    return dict(passed=not bad and leaf>=1.57,measured_leaf_rad=leaf,start_xy=start[root:root+2].tolist(),goal_xy=goal.tolist(),minimum_gap_m=minimum,margin_m=margin,templates=len(poses),samples_per_template=samples,failures=bad,scope='Actual measured door and collision shape; static paths, no future gait or passage proof')


class PassageWaypoints:
    def __init__(self,sim,base,*,start_time=34.,target_x=.065,target_y=1.05):
        self.sim=sim;self.base=base;self.start_time=start_time;self.target_x=target_x;self.target_y=target_y
        self.stage='post-opening component';self.guide=None;self.guide_start=None;self.current_command=np.zeros(3);self.amplitude=0.;self.screens=[];self.events=[];self.templates=[];self.done=False;self.blocked=False;self.started=None
    def command(self,t):
        s=self.sim;m,d=s.m,s.d;root=s.root_qadr;v=s.root_vadr
        if self.base.walk_started is not None and 1.<t-self.base.walk_started<3.4 and len(self.templates)<8:
            if not self.templates or t-self.last_template>.3:self.templates.append(d.qpos.copy());self.last_template=t
        if t<self.start_time:return None,None
        if self.guide is None and not self.blocked:
            goal=np.array([self.target_x,d.qpos[root+1]])
            screen=screen_translation(s,goal,templates=self.templates);self.screens.append(screen)
            if not screen['passed']:self.blocked=True;self.stage='alignment geometry blocked';return np.zeros(3),0.
            self.guide=WaypointApproach(goal,np.pi/2,brake_velocity_window=.8,max_speed=.2);self.guide_start=t;self.started=t;self.stage='align'
        if self.stage=='traverse' and float(d.qpos[m.jnt_qposadr[m.joint('leaf_hinge').id]])<1.57:
            self.blocked=True;self.stage='measured aperture decreased'
        if self.blocked:return np.zeros(3),0.
        if self.done:return np.zeros(3),0.
        rotation=d.xmat[s.pelvis].reshape(3,3);yaw=np.arctan2(rotation[1,0],rotation[0,0]);clock=t-self.base.walk_started
        self.current_command,self.amplitude,_=self.guide.step(d.qpos[root:root+2],yaw,d.qvel[v:v+2],clock)
        self.amplitude*=min(1.,(t-self.guide_start)/.4)
        if self.guide.stop_time is not None and clock-self.guide.stop_time>=3. and np.linalg.norm(d.qvel[v:v+2])<.03:
            if self.stage=='align':
                goal=np.array([self.target_x,self.target_y]);screen=screen_translation(s,goal,templates=self.templates);self.screens.append(screen)
                if not screen['passed']:self.blocked=True;self.stage='passage geometry blocked';return np.zeros(3),0.
                self.events.append(dict(completed='alignment',time_s=t,root=d.qpos[root:root+7].tolist()));self.guide=WaypointApproach(goal,np.pi/2,brake_velocity_window=.8,max_speed=.25);self.guide_start=t;self.stage='traverse';self.current_command[:]=0.;self.amplitude=0.
            else:
                lo,hi=robot_bounds(m,d)
                if lo[1]>.2:self.done=True;self.stage='quiet beyond doorway';self.events.append(dict(completed='traversal',time_s=t,minimum_body_y_m=float(lo[1])))
        return self.current_command.copy(),float(self.amplitude)
