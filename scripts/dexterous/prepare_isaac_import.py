#!/usr/bin/env python3
"""Prepare an importer-compatible copy; keep the audited native model untouched.

Touch-grid plugins are simulator-specific. Their sites/geometry remain available
for the PhysX contact adapter; this file alone does not implement tactile sensing.
Actuator forces are implemented explicitly from the compiled native motor contract.
"""
import argparse
import json
import re
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from doorbench.dexterous.isaac_materials import native_contact_contract


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--robot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();spec=mujoco.MjSpec.from_file(str(a.robot));m=spec.compile()
    tree=ET.parse(a.robot);root=tree.getroot()
    for tag in ('sensor','extension','keyframe','visual','actuator'):
        node=root.find(tag)
        if node is not None:root.remove(node)
    if np.any(m.tendon_stiffness) or np.any(m.tendon_damping) or np.any(m.tendon_limited):
        raise ValueError('This motor-transmission adapter does not drop passive tendon mechanics')
    tendon=root.find('tendon')
    if tendon is not None:root.remove(tendon)
    # Shadow's unbounded zero-stiffness tendons are actuator transmissions, not
    # springs. The motor matrix below preserves their shared distal forces;
    # importing them would make Isaac invent a stiffness of 1 for each tendon.
    for node in root.findall('./worldbody//joint'):
        j=m.joint(node.attrib['name']).id;v=m.jnt_dofadr[j]
        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            node.attrib.clear()
            node.set('name',m.joint(j).name)
            node.set('type','free')
            continue
        if int(m.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError('Unsupported non-scalar robot joint')
        node.set('type','hinge' if int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_HINGE) else 'slide')
        for key,value in {'axis':m.jnt_axis[j],'pos':m.jnt_pos[j],'range':m.jnt_range[j]}.items():
            node.set(key,' '.join(format(float(x),'.12g') for x in value))
        node.set('limited','true' if m.jnt_limited[j] else 'false')
        node.set('damping',str(float(m.dof_damping[v])))
        node.set('armature',str(float(m.dof_armature[v])))
        node.set('frictionloss',str(float(m.dof_frictionloss[v])))
    # The 5.1 importer does not resolve every nested MJCF default correctly.
    geoms=root.findall('./worldbody//geom')
    assert len(geoms)==len(list(spec.geoms))
    for i,(node,g) in enumerate(zip(geoms,spec.geoms)):
        node.set('name',g.name or f'geom_{i}')
        node.set('type',str(g.type).split('mjGEOM_')[-1].lower())
        node.set('contype',str(g.contype));node.set('conaffinity',str(g.conaffinity))
        node.set('friction',' '.join(str(float(x)) for x in g.friction))
        node.set('rgba',' '.join(str(float(x)) for x in g.rgba))
    # Sites belong to the separate FK/tactile definition, never collision bodies.
    for parent in root.iter():
        for child in list(parent):
            if child.tag=='site':parent.remove(child)
    # These are compiled values, so importer mass inference cannot change the robot.
    for body in root.findall('.//body'):
        bid=m.body(body.attrib['name']).id
        if m.body_mass[bid]<=0:continue
        inertial=body.find('inertial')
        if inertial is None:inertial=ET.SubElement(body,'inertial')
        inertial.attrib.clear()
        for key,value in {'mass':[m.body_mass[bid]],'pos':m.body_ipos[bid],
                          'quat':m.body_iquat[bid],'diaginertia':m.body_inertia[bid]}.items():
            inertial.set(key,' '.join(format(float(x),'.12g') for x in value))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    mesh_dir=a.output.parent/'meshes';mesh_dir.mkdir(exist_ok=True)
    # Isaac 5.1 converts OBJ files concurrently. Two mesh declarations pointing
    # at the same filename race on a shared *_tmp USD layer (e.g. both thumbs).
    # Give every mesh declaration its own basename, with unchanged file bytes.
    for node in root.findall('./asset/mesh'):
        source=Path(node.attrib['file'])
        if not source.is_absolute():source=(a.robot.parent/source).resolve()
        name=re.sub(r'[^A-Za-z0-9_-]','_',node.attrib['name'])+source.suffix
        destination=mesh_dir/name
        shutil.copy2(source,destination)
        node.set('file','meshes/'+name)
    ET.indent(tree);tree.write(a.output,encoding='unicode')
    data={'mass_kg':float(m.body_mass.sum()),'free_root':True,'contact_material':native_contact_contract(m),
          'sensor_status':'native touch-grid removed; PhysX contact adapter required',
          'joint_names':[m.joint(j).name for j in range(m.njnt) if int(m.jnt_type[j])!=int(mujoco.mjtJoint.mjJNT_FREE)],
          'actuators':[]}
    for i in range(m.nu):
        trn=int(m.actuator_trntype[i]);tid=int(m.actuator_trnid[i,0])
        if trn==int(mujoco.mjtTrn.mjTRN_JOINT):terms={m.joint(tid).name:float(m.actuator_gear[i,0])}
        elif trn==int(mujoco.mjtTrn.mjTRN_TENDON):
            terms={m.joint(int(m.wrap_objid[k])).name:float(m.wrap_prm[k]) for k in range(m.tendon_adr[tid],m.tendon_adr[tid]+m.tendon_num[tid])}
        else:raise ValueError('Unsupported actuator transmission')
        data['actuators'].append(dict(name=m.actuator(i).name,terms=terms,
             kp=float(m.actuator_gainprm[i,0]),bias=m.actuator_biasprm[i,:3].tolist(),
             force_range=m.actuator_forcerange[i].tolist(),control_range=m.actuator_ctrlrange[i].tolist()))
    data['passive']={m.joint(j).name:dict(damping=float(m.dof_damping[m.jnt_dofadr[j]]),
        armature=float(m.dof_armature[m.jnt_dofadr[j]]),friction=float(m.dof_frictionloss[m.jnt_dofadr[j]]),
        stiffness=float(m.jnt_stiffness[j]),springref=float(m.qpos_spring[m.jnt_qposadr[j]]))
        for j in range(m.njnt) if int(m.jnt_type[j])!=int(mujoco.mjtJoint.mjJNT_FREE)}
    a.output.with_suffix('.motors.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'joints':len(data['joint_names']),'actuators':len(data['actuators']),'mass_kg':data['mass_kg']}))

if __name__=='__main__':main()
