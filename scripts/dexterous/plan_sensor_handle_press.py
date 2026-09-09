#!/usr/bin/env python3
"""Offline instance-specific joint press plan from an attained acquisition.

This script may use actual archived world geometry. Its output is only a static
named-joint schedule; it is never an inference-time FK/contact controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from doorbench.dexterous.environment import DexterousDoorEnv
from scripts.dexterous.probe_sensor_acquisition_balance import intended_contact_geometry


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acquisition',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--independent-audit',type=Path,required=True,help='Hash-bound native-independent.json for the attained acquisition')
    p.add_argument('--handle-angle',type=float,default=.85);a=p.parse_args()
    if a.output.exists():p.error('Fresh planner output required')
    if not np.isfinite(a.handle_angle) or not 0<a.handle_angle<=.87:p.error('Original handle travel required')
    a.output.mkdir(parents=True)
    source=a.acquisition;report=json.loads((source/'report.json').read_text());prov=json.loads((source/'provenance.json').read_text())
    from doorbench.dexterous.press_plan_binding import acquisition_binding,verify_acquisition_audit
    if not report['passed']:raise ValueError('Actual acquired-state qualification required')
    verify_acquisition_audit(source,json.loads(a.independent_audit.read_text()))
    ref=json.loads((source/'reference.json').read_text());robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door']);door=door if door.is_dir() else door.parent
    if sha(robot)!=prov['robot_xml_sha256'] or sha(door/'door.xml')!=prov['door_xml_sha256']:raise ValueError('Planning plant source differs')
    archive=np.load(source/'trajectory.npz');terminal=archive['qpos'][-1].copy()
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()),frame_skip=1)
    m=sim.m;d=mujoco.MjData(m);d.qpos[:]=terminal;mujoco.mj_kinematics(m,d)
    names=ref['workspace_fit']['joint_names'];ids=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[ids]
    arm0=d.qpos[qa].copy();source_names=ref['acquisition']['joint_names'];nominal=np.asarray(ref['acquisition']['path_qpos'][-1])
    nominal_arm=nominal[[source_names.index(n) for n in names]];offset=nominal_arm-arm0
    bounds=np.array([np.maximum(m.jnt_range[ids,0],m.jnt_range[ids,0]-offset),np.minimum(m.jnt_range[ids,1],m.jnt_range[ids,1]-offset)])
    palm=m.site('robot/rh_palm_touch').id;handle=m.body('leaf_handle').id;lever=m.geom('leaf_handle_lever_col_n').id
    hj=m.jnt_qposadr[m.joint('leaf_handle_hinge').id];bj=m.jnt_qposadr[m.joint('leaf_latch_bolt_slide').id]
    R0=d.xmat[handle].reshape(3,3).copy();p0=d.xpos[handle].copy()
    relative_p=R0.T@(d.site_xpos[palm]-p0);relative_R=R0.T@d.site_xmat[palm].reshape(3,3)
    start_angle=float(d.qpos[hj]);rows=[];screen=[];last=np.clip(arm0,*bounds)
    for index,angle in enumerate(np.linspace(start_angle,a.handle_angle,121)):
        d.qpos[:]=terminal;d.qpos[hj]=angle;d.qpos[bj]=.014598*angle;mujoco.mj_kinematics(m,d)
        R=d.xmat[handle].reshape(3,3);desired_p=d.xpos[handle]+R@relative_p;desired_R=R@relative_R
        def residual(q):
            d.qpos[qa]=q;mujoco.mj_kinematics(m,d)
            return np.r_[100*(d.site_xpos[palm]-desired_p),10*Rotation.from_matrix(desired_R@d.site_xmat[palm].reshape(3,3).T).as_rotvec(),.001*(q-last)]
        if index==0:solved=arm0.copy()
        else:solved=least_squares(residual,last,bounds=bounds,max_nfev=300,ftol=1e-11,xtol=1e-11,gtol=1e-11).x
        residual(solved);mujoco.mj_collision(m,d)
        pos_error=float(np.linalg.norm(d.site_xpos[palm]-desired_p));rot_error=float(Rotation.from_matrix(desired_R@d.site_xmat[palm].reshape(3,3).T).magnitude())
        bad=[];penetration=0.
        for c in d.contact[:d.ncon]:
            bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom];geoms=[m.geom(g).name for g in c.geom]
            if c.dist<=0 and any(n.startswith(('robot/rh_','robot/lh_')) for n in bodies) and not intended_contact_geometry(m,d,c,lever):bad.append(dict(reason='unintended hand contact',bodies=bodies,distance_m=float(c.dist)))
            if any(n.startswith('robot/') for n in bodies) and not ('floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)):
                penetration=max(penetration,-float(c.dist))
        command=solved+offset
        if index==0:command=nominal_arm.copy()
        valid=pos_error<.0001 and rot_error<.001 and not bad and penetration<=.003 and np.all(command>=m.jnt_range[ids,0]) and np.all(command<=m.jnt_range[ids,1])
        rows.append(command.tolist());screen.append(dict(index=index,handle_angle_rad=float(angle),palm_error_m=pos_error,orientation_error_rad=rot_error,max_nonfoot_penetration_m=penetration,passed=bool(valid),bad_contacts=bad))
        last=solved.copy()
    result=dict(schema='doorbench.offline-sensor-press-plan.v2',acquisition_binding=acquisition_binding(prov),
        independent_acquisition_audit_sha256=sha(a.independent_audit),passed=all(r['passed'] for r in screen),
        scope='Offline geometry-derived static motor goals; no runtime root/door/geometry access',joint_names=names,path_qpos=rows,press_seconds=8.,settle_seconds=3.,
        acquisition_seconds=19.,duration_s=30.,requested_handle_angle_rad=a.handle_angle,source_terminal_time_s=float(archive['time'][-1]),
        source_acquisition_provenance_sha256=sha(source/'provenance.json'),source_trajectory_sha256=sha(source/'trajectory.npz'),source_reference_sha256=sha(source/'reference.json'),
        robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),planner_source_sha256=sha(__file__),
        fixed_attained_root=terminal[sim.root_qadr:sim.root_qadr+7].tolist(),attained_arm_angles=arm0.tolist(),constant_tracking_offset=offset.tolist(),screen=screen)
    (a.output/'plan.json').write_text(json.dumps(result,indent=2)+'\n');sim.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('path_qpos','screen')},indent=2));print('bad samples',[(r['index'],r['palm_error_m'],r['orientation_error_rad'],r['bad_contacts']) for r in screen if not r['passed']][:10])
if __name__=='__main__':main()
