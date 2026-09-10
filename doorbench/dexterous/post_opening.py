"""Initialized, privileged post-opening planning and original-capped motor control.

Planning owns a separate unstepped MjData. Physical feet stay free; the stance
QP returns motor forces and inserts no contacts or constraints into the plant.
This first component supports only its screened Door55 terminal, not passage.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .grasp_verification import scalar_transmission_matrix
from .locomotion_manipulation import LandedFootStanceController
from .locomotion import H1WalkingPolicy, DEFAULT_ANGLES


def smooth(u):
    u=float(np.clip(u,0.,1.))
    return u**3*(10+u*(-15+6*u))


def compact_posture(model, start, reset, *, inward_roll=0.,finger_profile='original-v1'):
    if not np.isfinite(inward_roll) or not 0<=inward_roll<=.1:raise ValueError('Invalid bounded inward shoulder roll')
    if finger_profile not in ('original-v1','relaxed-v3'):raise ValueError('Unknown stow finger profile')
    q=np.array(start,copy=True)
    values={n:v for n,v in reset['joints'].items() if not any(t in n for t in ('hip','knee','ankle'))}
    for hand in ('lh','rh'):
        for digit in ('FF','MF','RF','LF'):
            for j,v in ((3,.35),(2,.8),(1,.8)):values[f'{hand}_{digit}J{j}']=v
        for j,v in ((5,.5),(4,.7),(3,0),(2,.5),(1,.5)):values[f'{hand}_THJ{j}']=v
        values[f'{hand}_LFJ5']=.12
        if finger_profile=='relaxed-v3':
            values[f'{hand}_LFJ5']=0.
            for digit in ('FF','MF','RF','LF'):
                for j,v in ((4,0.),(3,.2),(2,.4),(1,.4)):values[f'{hand}_{digit}J{j}']=v
    for side,sign in (('left',-1),('right',1)):values[side+'_shoulder_roll']=sign*inward_roll
    for name,value in values.items():q[model.jnt_qposadr[model.joint('robot/'+name).id]]=value
    return q


def screen_state(m,d,q):
    """Unstepped whole-scene collision and exact authored joint/anatomy screen."""
    q=np.asarray(q,float)
    if q.shape!=(m.nq,) or not np.isfinite(q).all():raise ValueError("Malformed planning pose")
    d.qpos[:]=q;d.qvel[:]=0.;mujoco.mj_forward(m,d)
    collision=[]
    for c in d.contact[:d.ncon]:
        bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
        geoms=[m.geom(int(g)).name for g in c.geom]
        if not any(n.startswith('robot/') for n in bodies):continue
        if 'floor' in geoms and any(n.endswith('_ankle_link') for n in bodies):continue
        if c.dist<-.003:collision.append(dict(bodies=bodies,depth_m=-float(c.dist)))
    violations=[]
    for j in range(m.njnt):
        if not m.jnt_limited[j]:continue
        value=q[m.jnt_qposadr[j]];gap=max(m.jnt_range[j,0]-value,value-m.jnt_range[j,1],0.)
        if gap>.02:violations.append(dict(joint=m.joint(j).name,violation_rad=float(gap)))
    loopbacks=[]
    for hand in ('lh','rh'):
        for digit in ('FF','MF','RF','LF'):
            vals=[q[m.jnt_qposadr[m.joint(f'robot/{hand}_{digit}J{i}').id]] for i in (1,2)]
            loopbacks.append(float(vals[0]-vals[1]))
    return dict(passed=not collision and not violations and max(loopbacks)<=.02,
                collisions=collision[:8],joint_violations=violations,
                maximum_loopback_violation_rad=max(0.,*loopbacks))


def plan_stow(sim,reset,*,retreat_m=.14,samples=101,inward_roll=0.,retreat_normal_world=None):
    """Retreat along measured leaf outward normal, then each arm, then waist.

    The loaded side is chosen from the measured contact normal. No geometry or
    root state is changed in the active simulator. Checks retain <=3 mm native
    soft-contact tolerance, and do not predict moving-door dynamics.
    """
    if samples<51 or not 0<retreat_m<=.2:raise ValueError('Invalid bounded route resolution')
    m=sim.m;d=mujoco.MjData(m);start=sim.d.qpos.copy();d.qpos[:]=start;mujoco.mj_forward(m,d)
    target=compact_posture(m,start,reset,inward_roll=inward_roll);palm=m.site('robot/lh_palm_touch').id
    position=d.site_xpos[palm].copy();rotation=d.site_xmat[palm].reshape(3,3).copy()
    normals=[]
    for c in d.contact[:d.ncon]:
        bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
        if 'leaf' in bodies and any(n.startswith('robot/lh_') for n in bodies):
            normals.append((1 if bodies[1].startswith('robot/lh_') else -1)*c.frame[:3])
    if retreat_normal_world is None:
        if not normals:raise ValueError('Expected attained left-hand panel contact')
        normal=np.mean(normals,axis=0);normal/=np.linalg.norm(normal)
    else:
        normal=np.asarray(retreat_normal_world,float)
        if normal.shape!=(3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal),1.,atol=1e-6,rtol=0):
            raise ValueError('Expected a normalized measured outward release direction')
    names=['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1']
    js=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[js]
    lo=m.jnt_range[js,0]+.015;hi=m.jnt_range[js,1]-.015
    path=[];stage=[];bad=[];q=start.copy()
    for u in np.linspace(0,1,samples):
        goal=position+normal*retreat_m*u;previous=q[qa].copy()
        def residual(x):
            d.qpos[:]=q;d.qpos[qa]=x;mujoco.mj_kinematics(m,d)
            return np.r_[100*(d.site_xpos[palm]-goal),5*Rotation.from_matrix(rotation@d.site_xmat[palm].reshape(3,3).T).as_rotvec(),.05*(x-previous)]
        fit=least_squares(residual,np.clip(previous,lo,hi),bounds=(lo,hi),max_nfev=90)
        q[qa]=fit.x
        audit=screen_state(m,d,q)
        if not audit['passed'] or np.linalg.norm(fit.fun[:3])>.1:bad.append(dict(stage='release',fraction=float(u),audit=audit,position_error_m=float(np.linalg.norm(fit.fun[:3])/100)))
        path.append(q.copy());stage.append('release')
    for name in ('left','right','torso'):
        before=q.copy();destination=q.copy()
        for j in sim.joints:
            n=m.joint(j).name.removeprefix('robot/')
            if n==name or n.startswith(name+'_') or (name=='left' and n.startswith('lh_')) or (name=='right' and n.startswith('rh_')):destination[m.jnt_qposadr[j]]=target[m.jnt_qposadr[j]]
        for u in np.linspace(0,1,samples):
            q=before*(1-u)+destination*u;audit=screen_state(m,d,q)
            if not audit['passed']:bad.append(dict(stage=name,fraction=float(u),audit=audit))
            path.append(q.copy());stage.append(name)
    return dict(passed=not bad,path_qpos=np.array(path).tolist(),stage=stage,
                bad_samples=bad[:30],bad_sample_count=len(bad),sample_count=len(path),
                retreat_normal_world=normal.tolist(),retreat_distance_m=retreat_m,inward_shoulder_roll_rad=inward_roll,
                scope='Static measured terminal leaf/root; no dynamic or passage proof')


class StowRiseController:
    """Privileged reference controller, never a sensor-only learned policy."""
    def __init__(self,sim,motors,plan,*,rise=True,phase_seconds=4.,arm_gain=10.,checkpoint=None):
        if not plan['passed']:raise ValueError('Refuse failed geometric route')
        if rise and checkpoint is None:raise ValueError('Rise requires the audited H1 gait checkpoint')
        if not 2<=phase_seconds<=10 or not 1<=arm_gain<=10:raise ValueError('Invalid bounded reference schedule')
        self.sim=sim;m,d=sim.m,sim.d;self.names=motors['joint_names']
        self.joints=np.array([m.joint('robot/'+n).id for n in self.names]);self.qa=m.jnt_qposadr[self.joints];self.va=m.jnt_dofadr[self.joints]
        self.act=np.array([m.actuator('robot/'+a['name']).id for a in motors['actuators']]);self.motor_names=[a['name'] for a in motors['actuators']]
        self.matrix=scalar_transmission_matrix(m,self.act,self.joints);self.caps=np.array([a['force_range'] for a in motors['actuators']])
        self.kp=np.array([a['kp'] for a in motors['actuators']]);self.bias=np.array([a['bias'] for a in motors['actuators']]);self.ctrl_range=np.array([a['control_range'] for a in motors['actuators']])
        self.arm=np.array([n=='torso' or any(n.startswith(s) for s in ('left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','lh_A_WRJ','rh_A_WRJ')) for n in self.motor_names])
        self.extra_gain=self.arm*(arm_gain-1)
        self.damping=np.array([20. if n=='torso' else .8 if 'WRJ' in n else 10. if a else 0. for n,a in zip(self.motor_names,self.arm)])
        self.plan={name:np.array([q for q,s in zip(plan['path_qpos'],plan['stage']) if s==name])[:,self.qa] for name in ('release','left','right','torso')}
        self.phase=0;self.phases=('release','left','right','torso','rise','hold');self.phase_start=0.;self.phase_seconds=phase_seconds;self.clear_since=None;self.rise=rise
        sim.stance_solver_settings={'max_iter':100000,'rho':.001,'adaptive_rho_interval':25}
        sim.external_generalized_force=np.zeros(m.nv)
        self.stance=LandedFootStanceController(sim);self.initial_height=float(self.stance.target_root[2]);self.stance_force=d.ctrl[self.stance.act].copy();self.solver_failures=0
        self.last_target=self.plan['release'][0];self.tick=0;self.events=[];self.info={}
        self.policy=H1WalkingPolicy(checkpoint) if checkpoint is not None and rise else None
        self.walk_target=DEFAULT_ANGLES.copy();self.walk_started=None

    def force(self,t,*,left_contacts,left_load,walking_command=None,walking_amplitude=None,hand_forces=None):
        if walking_command is not None:
            walking_command=np.asarray(walking_command,float)
            if walking_command.shape!=(3,) or not np.isfinite(walking_command).all() or np.any(abs(walking_command)>[.3,.15,.4]) or walking_amplitude is None or not 0<=walking_amplitude<=1:raise ValueError('Invalid bounded walking command')
        m,d=self.sim.m,self.sim.d;name=self.phases[self.phase];elapsed=t-self.phase_start
        if left_contacts==0 and left_load<.1:
            if self.clear_since is None:self.clear_since=t
        else:self.clear_since=None
        if self.phase<4:
            path=self.plan[name];u=smooth(elapsed/self.phase_seconds);coordinate=u*(len(path)-1);i=min(int(coordinate),len(path)-2);f=coordinate-i;target=path[i]*(1-f)+path[i+1]*f
            self.last_target=target
            error=float(np.max(abs((self.matrix@target-d.actuator_length[self.act])[self.arm])))
            if elapsed>=self.phase_seconds and error<.12 and (name!='release' or self.clear_since is not None and t-self.clear_since>=.2):
                self.events.append(dict(completed=name,time_s=t,motor_position_error_rad=error,left_contacts=left_contacts,left_load_N=left_load));self.phase+=1;self.phase_start=t
        else:
            target=self.last_target;error=float(np.max(abs((self.matrix@target-d.actuator_length[self.act])[self.arm])))
        if self.phases[self.phase]=='rise' and self.rise:
            u=smooth((t-self.phase_start)/6.)
            self.stance.target_root[2]=self.initial_height+(1.0-self.initial_height)*u
            if t-self.phase_start>=7. and d.qpos[self.sim.root_qadr+2]>.985 and np.linalg.norm(d.qvel[self.sim.root_vadr:self.sim.root_vadr+3])<.04:
                self.events.append(dict(completed='rise',time_s=t));self.phase=5;self.phase_start=t
        elif self.phases[self.phase]=='rise':self.phase=5;self.phase_start=t
        target_length=np.clip(self.matrix@target,self.ctrl_range[:,0],self.ctrl_range[:,1]);length=d.actuator_length[self.act];speed=d.actuator_velocity[self.act]
        force=self.kp*target_length+self.bias[:,0]+self.bias[:,1]*length+self.bias[:,2]*speed+self.kp*self.extra_gain*(target_length-length)-self.damping*speed
        for local in np.flatnonzero(self.arm):
            aid=self.act[local]
            if m.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:force[local]+=d.qfrc_bias[m.jnt_dofadr[m.actuator_trnid[aid,0]]]
        if self.tick%5==0 and self.walk_started is None:
            # Match the portable body teacher: measured hand loads about body
            # origins, not solver-generated robot joint-limit impulses.
            external=self.sim.external_generalized_force;external[:]=0.
            jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
            if hand_forces is None:
                raise ValueError('Stance requires separately captured actual-interval hand forces')
            for body_name,world in hand_forces.items():
                if not body_name.startswith(('lh_','rh_')):raise ValueError('Unexpected hand-load body')
                world=np.asarray(world,float)
                if world.shape!=(3,) or not np.isfinite(world).all():raise ValueError('Malformed actual hand force')
                body=m.body('robot/'+body_name).id
                mujoco.mj_jacBody(m,d,jp,jr,body);external+=jp.T@world
            proposed,status=self.stance.command()
            if proposed is None:self.solver_failures+=1
            else:self.stance_force=proposed
            self.stance_status=status
        if self.stance_force is not None:force[self.stance.local]=self.stance_force
        if self.phase==5 and self.policy is not None:
            if self.walk_started is None:self.walk_started=t;self.policy.reset()
            q=d.qpos[self.stance.qa];v=d.qvel[self.stance.v[6:]]
            if self.tick%10==0:
                self.walk_target=self.policy.step(q,v,d.qvel[self.sim.root_vadr+3:self.sim.root_vadr+6],d.xmat[self.sim.pelvis].reshape(3,3).T@np.array([0.,0.,-1.]),np.zeros(3) if walking_command is None else walking_command,t-self.walk_started,phase_amplitude=max(0.,min(1.,4.4-(t-self.walk_started))) if walking_amplitude is None else walking_amplitude)
            force[self.stance.local]=self.policy.torques(self.walk_target,q,v)
        self.tick+=1;force=np.clip(force,self.caps[:,0],self.caps[:,1])
        if not np.isfinite(force).all():raise ValueError('Nonfinite motor force')
        self.info=dict(phase='walking stabilizer' if self.walk_started is not None else self.phases[self.phase],phase_elapsed_s=t-self.phase_start,arm_motor_error_rad=error,stance_status=self.stance_status,left_contacts=left_contacts,left_load_N=left_load)
        return force,self.info.copy()
