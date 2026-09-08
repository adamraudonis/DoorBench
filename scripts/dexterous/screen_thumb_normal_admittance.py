#!/usr/bin/env python3
"""Detached signed-gap and anatomy screen of the source23s thumb admittance.

The kinematic response assumes target velocity is realized. Constant synthetic
taxel loads do not predict physical forces or qualify a loaded grasp.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import scalar_transmission_matrix
from doorbench.dexterous.thumb_normal_admittance import RobotThumbNormalAdmittance,PROFILE,NAMES
from scripts.dexterous.probe_sensor_touch_operation import intended_contact_geometry


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('trial','profile','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():parser.error('Fresh screen directory required')
    a.output.mkdir(parents=True)
    prov=json.loads((a.trial/'provenance.json').read_text());robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door'])
    if sha(robot)!=prov['robot_xml_sha256'] or sha(door/'door.xml')!=prov['door_xml_sha256']:raise ValueError('Actual authored source changed')
    profile=json.loads(a.profile.read_text());index=11500
    with np.load(a.trial/'trajectory.npz') as z:base=z['qpos'][index].copy();q=base[10:].copy()
    with np.load(a.trial/'actor-inputs.npz') as z:touch=z['tactile'][index].copy()
    with gzip.open(a.trial/'controller.jsonl.gz','rt') as f:
        for i,line in enumerate(f):
            if i==index:info=json.loads(line);break
    with gzip.open(a.trial/'physics.jsonl.gz','rt') as f:
        for i,line in enumerate(f):
            if i==index:physical=json.loads(line);break
    patches=[c for c in physical['pad_grasp']['contacts'] if c['digit']=='th' and c['pad_qualified'] and c['normal_force_N']>0]
    if not patches:raise ValueError('Actual source23s qualified thumb material point required')
    weights=np.array([c['normal_force_N'] for c in patches]);weights/=weights.sum()
    contact_material_point=sum(w*np.array(c['body_position_m']) for w,c in zip(weights,patches))
    goals=dict(zip(info['goal_joint_names'],info['goal_joint_position_rad']))
    layout=json.loads((a.trial/'sensor-layout.json').read_text());offset=0
    for sensor in layout['sensors']:
        if sensor['name']=='rh_thdistal_touch':grid=touch[offset:offset+sensor['dimension']].reshape(3,2,4)
        offset+=sensor['dimension']
    channels=grid[:,:,1:3].sum(axis=(1,2)).astype(float);vector=channels[[1,2,0]];load=float(channels[0])
    rm=mujoco.MjModel.from_xml_path(str(robot));names=[rm.joint(i).name for i in range(1,rm.njnt)];actions=[rm.actuator(i).name for i in range(rm.nu)]
    M=scalar_transmission_matrix(rm,np.arange(61),np.arange(1,rm.njnt))
    calc=RobotThumbNormalAdmittance(rm,names,actions,M,profile);pos,R,jp,jr=calc.geometry(q);B,p=calc.basis(q)
    direction=R@(vector/np.linalg.norm(vector));coeff=np.linalg.lstsq(jp@B,direction,rcond=None)[0];delta=B@coeff;delta*=.001/np.max(abs(delta))
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));m=sim.m;d=mujoco.MjData(m)
    qa=np.array([m.joint('robot/'+name).qposadr[0] for name in NAMES]);body=m.body('robot/rh_thdistal').id;lever=m.geom('leaf_handle_lever_col_n').id
    geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and ((m.geom_contype[g]&m.geom_conaffinity[lever]) or (m.geom_contype[lever]&m.geom_conaffinity[g]))]
    fixed=np.setdiff1d(np.arange(m.nq),qa);site_ids=[m.site('robot/rh_'+digit+'distal_touch').id for digit in ('ff','mf','rf','lf')]
    palm=m.site('robot/rh_palm_touch').id;d.qpos[:]=base;mujoco.mj_kinematics(m,d);fixed_sites=d.site_xpos[site_ids+[palm]].copy()
    def geometry(angles):
        d.qpos[:]=base;d.qpos[qa]=angles;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
        gap=min(float(mujoco.mj_geomDistance(m,d,g,lever,.05,None)) for g in geoms);bad=[];overlaps=0
        for c in d.contact[:d.ncon]:
            bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
            if c.dist>0 or not any(n.startswith(('robot/rh_','robot/lh_')) for n in bodies):continue
            if not intended_contact_geometry(m,d,c,lever):bad.append('unqualified overlapping hand surface:'+','.join(bodies))
            if c.dist<-.003:bad.append('hand penetration above3mm')
            if lever in c.geom and body in [m.geom_bodyid[g] for g in c.geom]:overlaps+=1
        changed=float(np.max(abs(d.qpos[fixed]-base[fixed])));site_change=float(np.max(abs(d.site_xpos[site_ids+[palm]]-fixed_sites)))
        normal=-d.site_xmat[m.site('robot/rh_thdistal_touch').id].reshape(3,3)[:,2]
        finger_dot=max(float(normal@(-d.site_xmat[s].reshape(3,3)[:,2])) for s in site_ids)
        material_world=d.xpos[body]+d.xmat[body].reshape(3,3)@contact_material_point
        lever_axis=d.geom_xmat[lever].reshape(3,3)[:,2]
        axial_clearance=float(m.geom_size[lever,1]-abs((material_world-d.geom_xpos[lever])@lever_axis))
        return dict(gap_m=gap,failed_contacts=bad,geometric_thumb_overlaps=overlaps,
                    maximum_non_thumb_coordinate_change=changed,maximum_other_finger_palm_site_change_m=site_change,
                    source_material_point_axial_clearance_m=axial_clearance,
                    maximum_thumb_finger_calibrated_normal_dot=finger_dot)
    differential=[dict(sign=s,delta_rad=(s*delta).tolist(),**geometry(base[qa]+s*delta)) for s in (-1,0,1)]
    trials=[]
    for requested_load in (3.,6.):
        c=RobotThumbNormalAdmittance(rm,names,actions,M,profile);current=q.copy()
        c.update(goals,current,vector,load,now_s=23.);samples=[];error=None
        for step in range(1001):
            if step:
                try:
                    target,meta=c.update(goals,current,vector*(requested_load/load),requested_load,now_s=23.+step*.002)
                    current[c.columns]+=np.array(meta['thumb_admittance_goal_velocity_rad_s'])*.002
                except Exception as exc:error=type(exc).__name__+': '+str(exc);break
            screen=geometry(current[c.columns])
            samples.append(dict(time_s=23.+step*.002,joint_qpos=current[c.columns].tolist(),**screen))
        checks=dict(complete=error is None and len(samples)==1001,
                    all_overlapping_hand_patches_qualified=all(not r['failed_contacts'] for r in samples),
                    other_digits_palm_and_entire_scene_fixed=all(r['maximum_non_thumb_coordinate_change']==0. and r['maximum_other_finger_palm_site_change_m']<1e-12 for r in samples),
                    thumb_remains_opposite=all(r['maximum_thumb_finger_calibrated_normal_dot']<-.2 for r in samples),
                    material_point_stays_inside_lever_ends=all(r['source_material_point_axial_clearance_m']>=.001 for r in samples),
                    actual_lower_margin_screen=bool(np.min(current[c.columns]-c.limits[:,0])>=.01),
                    calculators_never_stepped=c.d.time==0. and c.mapper.d.time==0. and d.time==0.)
        result=dict(synthetic_constant_local_force_N=requested_load,passed=all(checks.values()),checks=checks,error=error,
                    initial_gap_m=samples[0]['gap_m'],final_gap_m=samples[-1]['gap_m'],
                    maximum_gap_m=max(r['gap_m'] for r in samples),final_thumb_angles_rad=current[c.columns].tolist(),samples=samples)
        result['minimum_source_material_axial_clearance_m']=min(r['source_material_point_axial_clearance_m'] for r in samples)
        (a.output/f'kinematic-{requested_load:g}N.json').write_text(json.dumps(result,indent=2)+'\n');trials.append({k:v for k,v in result.items() if k!='samples'})
    checks=dict(retained_position_rank=int(np.linalg.matrix_rank(jp@B))==3,
                effort_tangent_exact=float(np.max(abs(p@(calc.gain[:,None]*B[3:]))))<1e-12,
                positive_relief_gap_derivative=differential[2]['gap_m']>differential[1]['gap_m']>differential[0]['gap_m'],
                differential_surface_and_side=all(not r['failed_contacts'] and r['maximum_thumb_finger_calibrated_normal_dot']<-.2 and r['source_material_point_axial_clearance_m']>=.001 for r in differential),
                bounded_kinematic_cases=all(r['passed'] for r in trials))
    report=dict(schema='doorbench.thumb-normal-admittance-screen.v1',scope=__doc__,passed=all(checks.values()),checks=checks,
        profile=profile,profile_sha256=sha(a.profile),robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),
        source_trial_provenance_sha256=sha(a.trial/'provenance.json'),source_state_index=index,source_time_s=23.,
        source_local_thumb_force_xyz_N=vector.tolist(),source_local_palmar_force_N=load,position_basis=B.tolist(),
        position_jacobian_singular_values=np.linalg.svd(jp@B,compute_uv=False).tolist(),differential=differential,trials=trials,
        input_sha256={n:sha(a.trial/n) for n in ('provenance.json','trajectory.npz','controller.jsonl.gz','physics.jsonl.gz','actor-inputs.npz','reference.json','calibration.json','schedule.json','motors.json')},
        controller_sources_sha256={n:sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous'/n) for n in ('thumb_normal_admittance.py','sensor_thumb_admittance.py')},
        screen_source_sha256=sha(__file__))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');sim.close();print(json.dumps(report,indent=2))


if __name__=='__main__':main()
