"""Original MIT adult reference rig. Geometry and limits are declared approximations.

Forward is +Y, anatomical left -X, Z up; angles are radians. The default
pelvis height is .94m and neutral head top 1.68m. This is a kinematic model,
not a calibrated biomechanics or humanoid actuator model.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
from .humanoid import JOINTS

DIMENSIONS = {'upper_arm_m':.30,'forearm_m':.28,'thigh_m':.43,'shin_m':.43,
              'hip_half_width_m':.105,'shoulder_half_width_m':.18,
              'foot_width_m':.10,'foot_length_m':.26,'ankle_above_sole_m':.055,
              'neutral_head_top_above_pelvis_m':.74}
TARGET_SITES = {'pelvis':'pelvis','chest':'chest','head':'head',
                'left_hand':'wrist_l','right_hand':'wrist_r',
                'left_elbow':'elbow_l','right_elbow':'elbow_r',
                'left_foot':'ankle_l','right_foot':'ankle_r'}


def _numbers(x): return ' '.join(f'{float(v):.12g}' for v in x)


def rig_xml(root_pos=(0.,-1.,.94), root_yaw=0.):
    """Return independent MJCF; no third-party meshes, skeletons, or motion data."""
    root_pos=np.asarray(root_pos,float)
    if root_pos.shape!=(3,) or not np.isfinite(root_pos).all() or not math.isfinite(root_yaw):
        raise ValueError('root_pos must contain three finite values and root_yaw must be finite')
    xml=ET.Element('mujoco',model='doorbench_original_adult')
    ET.SubElement(xml,'compiler',angle='radian',autolimits='true')
    world=ET.SubElement(xml,'worldbody')
    def body(parent,name,pos,mass,ipos=(0,0,0)):
        b=ET.SubElement(parent,'body',name='actor_'+name,pos=_numbers(pos))
        ET.SubElement(b,'inertial',pos=_numbers(ipos),mass=str(mass),diaginertia=_numbers([max(.001,mass*.012)]*3))
        return b
    def hinge(b,name,axis,bounds):
        ET.SubElement(b,'joint',name='actor_'+name,type='hinge',axis=_numbers(axis),range=_numbers(bounds),limited='true',damping='1')
    def site(b,name,pos=(0,0,0)):
        ET.SubElement(b,'site',name='actor_site_'+name,pos=_numbers(pos),size='.008',rgba='1 .6 .1 1')
    def geom(b,name,kind,size,**kw):
        ET.SubElement(b,'geom',name='actor_geom_'+name,type=kind,size=_numbers(size),rgba='.08 .35 .42 1',contype='1',conaffinity='1',**kw)
    def capsule(b,name,a,c,r): geom(b,name,'capsule',[r],fromto=_numbers([*a,*c]))
    pelvis=body(world,'pelvis',root_pos,9.)
    pelvis.set('quat',_numbers([math.cos(root_yaw/2),0,0,math.sin(root_yaw/2)]))
    ET.SubElement(pelvis,'freejoint',name='actor_root')
    capsule(pelvis,'pelvis',(0,0,-.03),(0,0,.04),.105);site(pelvis,'pelvis')
    chest=body(pelvis,'chest',(0,0,.35),20.,(0,0,-.12))
    for name,axis,bounds in [('pitch',(1,0,0),(-.65,.65)),('roll',(0,1,0),(-.4,.4)),('yaw',(0,0,1),(-.8,.8))]:hinge(chest,'spine_'+name,axis,bounds)
    for joint in chest.findall('joint'):joint.set('pos','0 0 -.30')
    capsule(chest,'torso',(0,0,-.19),(0,0,-.025),.12);site(chest,'chest')
    neck=body(chest,'neck',(0,0,.16),1.)
    hinge(neck,'neck_yaw',(0,0,1),(-1.,1.));hinge(neck,'neck_pitch',(1,0,0),(-.5,.5))
    capsule(neck,'neck',(0,0,-.12),(0,0,.04),.04);site(neck,'neck')
    head=body(neck,'head',(0,0,.13),4.5)
    geom(head,'head','sphere',[.10]);site(head,'head')
    for side,sign in [('l',-1),('r',1)]:
        upper=body(chest,'shoulder_'+side,(sign*.18,0,.06),2.,(0,0,-.15))
        for name,axis,bounds in [('pitch',(1,0,0),(-1.2,2.7)),('roll',(0,1,0),(-2.3,2.3)),('yaw',(0,0,1),(-1.6,1.6))]:hinge(upper,'shoulder_'+side+'_'+name,axis,bounds)
        capsule(upper,'upper_arm_'+side,(0,0,-.025),(0,0,-.275),.043);site(upper,'shoulder_'+side)
        fore=body(upper,'elbow_'+side,(0,0,-.30),1.2,(0,0,-.14))
        hinge(fore,'elbow_'+side,(1,0,0),(0,2.65))
        capsule(fore,'forearm_'+side,(0,0,-.02),(0,0,-.26),.036);site(fore,'elbow_'+side)
        hand=body(fore,'wrist_'+side,(0,0,-.28),.4)
        for name,axis,bounds in [('pitch',(1,0,0),(-1.1,1.1)),('roll',(0,1,0),(-.65,.65)),('yaw',(0,0,1),(-1.4,1.4))]:hinge(hand,'wrist_'+side+'_'+name,axis,bounds)
        geom(hand,'hand_'+side,'sphere',[.035]);site(hand,'wrist_'+side)
        thigh=body(pelvis,'hip_'+side,(sign*.105,0,-.06),8.,(0,0,-.215))
        for name,axis,bounds in [('pitch',(1,0,0),(-.7,1.8)),('roll',(0,1,0),(-.65,.65)),('yaw',(0,0,1),(-.8,.8))]:hinge(thigh,'hip_'+side+'_'+name,axis,bounds)
        capsule(thigh,'thigh_'+side,(0,0,-.04),(0,0,-.39),.065);site(thigh,'hip_'+side)
        shin=body(thigh,'knee_'+side,(0,0,-.43),3.5,(0,0,-.215))
        hinge(shin,'knee_'+side,(1,0,0),(-2.7,0))
        capsule(shin,'shin_'+side,(0,0,-.035),(0,0,-.395),.047);site(shin,'knee_'+side)
        foot=body(shin,'ankle_'+side,(0,0,-.43),1.,(0,.04,-.0275))
        hinge(foot,'ankle_'+side+'_pitch',(1,0,0),(-.8,.8));hinge(foot,'ankle_'+side+'_roll',(0,1,0),(-.45,.45))
        geom(foot,'foot_'+side,'box',[.05,.13,.0275],pos='0 .04 -.0275')
        site(foot,'ankle_'+side);site(foot,'sole_'+side,(0,.04,-.055))
    return ET.tostring(xml,encoding='unicode')


@dataclass
class CombinedRig:
    model: object
    home_qpos: np.ndarray
    native_model: object
    native_qpos_indices: np.ndarray
    native_dof_indices: np.ndarray
    actor_qpos_indices: np.ndarray
    actor_dof_indices: np.ndarray
    actor_body_id: int


def combine_with_door(door_dir, *, root_pos=(0.,-1.,.94), root_yaw=0., native_qpos=None):
    """Compile a private scene, preserving native joints/assets through MjSpec attach."""
    import mujoco
    source=Path(door_dir).resolve()/'door.xml'
    native=mujoco.MjModel.from_xml_path(str(source))
    spec=mujoco.MjSpec.from_file(str(source))
    child=mujoco.MjSpec.from_string(rig_xml(root_pos,root_yaw))
    frame=spec.worldbody.add_frame(name='actor_attach')
    spec.attach(child,frame=frame,prefix="",suffix="")
    model=spec.compile()
    if any(model.joint(i).name.startswith('actor_') for i in range(native.njnt)):
        raise ValueError('Source joint ordering changed during rig attachment')
    q_indices=[];v_indices=[]
    for i in range(native.njnt):
        original=native.joint(i);copied=model.joint(original.name)
        qwidth={mujoco.mjtJoint.mjJNT_FREE:7,mujoco.mjtJoint.mjJNT_BALL:4}.get(original.type[0],1)
        vwidth={mujoco.mjtJoint.mjJNT_FREE:6,mujoco.mjtJoint.mjJNT_BALL:3}.get(original.type[0],1)
        q_indices.extend(range(int(copied.qposadr[0]),int(copied.qposadr[0])+qwidth))
        v_indices.extend(range(int(copied.dofadr[0]),int(copied.dofadr[0])+vwidth))
    qi=np.array(q_indices,int);vi=np.array(v_indices,int)
    aq=np.setdiff1d(np.arange(model.nq),qi);av=np.setdiff1d(np.arange(model.nv),vi)
    home=model.qpos0.copy()
    native_qpos=native.qpos0 if native_qpos is None else np.asarray(native_qpos,float)
    if native_qpos.shape!=(native.nq,) or not np.isfinite(native_qpos).all():raise ValueError(f'native_qpos must have {native.nq} finite coordinates')
    home[qi]=native_qpos
    # Symmetric bent-knee stance keeps both rigid soles on z=0 at .94m pelvis.
    alpha=math.acos(float(np.clip((float(root_pos[2])-.06-.055)/.86,-1,1)))
    for side in ('l','r'):
        for name,value in [('hip_'+side+'_pitch',alpha),('knee_'+side,-2*alpha),('ankle_'+side+'_pitch',alpha)]:
            home[int(model.joint('actor_'+name).qposadr[0])]=value
        home[int(model.joint('actor_shoulder_'+side+'_roll').qposadr[0])]=.16 if side=='l' else -.16
        home[int(model.joint('actor_elbow_'+side).qposadr[0])]=.15
    return CombinedRig(model,home,native,qi,vi,aq,av,int(model.body('actor_pelvis').id))
