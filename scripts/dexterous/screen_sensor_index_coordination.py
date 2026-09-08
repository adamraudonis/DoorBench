#!/usr/bin/env python3
"""Detached attained-contact diagnosis and bounded index-coordination screen."""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_archive import unpacked
from scripts.dexterous.probe_sensor_acquisition_balance import intended_contact_geometry


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('attained','acquisition','press-plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--index-proximal-bound',type=float,default=.02)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh diagnosis/screen receipt required')
    if not np.isfinite(a.index_proximal_bound) or not 0<a.index_proximal_bound<=.04:p.error('Small explicit proximal angle required')
    source=a.attained;prov=json.loads((source/'provenance.json').read_text())
    robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door']);door=door if door.is_dir() else door.parent
    if sha(robot)!=prov['robot_xml_sha256'] or sha(door/'door.xml')!=prov['door_xml_sha256']:raise ValueError('Changed original plant')
    with gzip.open(source/'physics.jsonl.gz','rt') as stream:
        for line in stream:row=json.loads(line)
    manifest=json.loads((source/'actual-transitions/manifest.json').read_text())
    last=source/'actual-transitions'/manifest['chunks'][-1]['file']
    if sha(last)!=manifest['chunks'][-1]['sha256']:raise ValueError('Changed actual contact archive')
    with np.load(last) as z:raw=list(unpacked({key:z[key] for key in z.files}))[-1]
    actual_contacts=[c for c in row['pad_grasp']['contacts'] if c['digit']=='ff' and c['normal_force_N']>0]
    if not actual_contacts or not all(c['pad_qualified'] for c in actual_contacts):raise ValueError('Actual index contact must already satisfy anatomy')
    if abs(raw['interval_end_s']-row['sim_time_s'])>1e-9:raise ValueError('Actual contact and geometry epochs differ')
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()),frame_skip=1)
    m=sim.m;d=mujoco.MjData(m);q=np.asarray(raw['qpos_before']);d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
    lever=m.geom('leaf_handle_lever_col_n').id;body=m.body('robot/rh_ffdistal').id
    ff=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and (m.geom_contype[g] or m.geom_conaffinity[g])]
    ff=[g for g in ff if (m.geom_contype[g]&m.geom_conaffinity[lever]) or (m.geom_contype[lever]&m.geom_conaffinity[g])]
    if not ff:raise ValueError('Receiving-compatible index collision shapes required')
    qa=lambda name:int(m.jnt_qposadr[m.joint('robot/'+name).id])
    def distance():
        return min(float(mujoco.mj_geomDistance(m,d,g,lever,1.,None)) for g in ff)
    baseline_distance=distance();weights=np.array([c['normal_force_N'] for c in actual_contacts]);weights/=weights.sum()
    local=np.average([c['body_position_m'] for c in actual_contacts],axis=0,weights=weights)
    normal=np.average([c['hand_outward_normal_body'] for c in actual_contacts],axis=0,weights=weights);normal/=np.linalg.norm(normal)
    world0=d.xpos[body]+d.xmat[body].reshape(3,3)@local;world_normal=d.xmat[body].reshape(3,3)@normal
    def contact_failures():
        failures=[];depth=0.
        for c in d.contact[:d.ncon]:
            bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom];geoms=[m.geom(g).name for g in c.geom]
            if c.dist<=0 and any(n.startswith(('robot/rh_','robot/lh_')) for n in bodies) and not intended_contact_geometry(m,d,c,lever):
                failures.append(dict(reason='unintended hand surface',bodies=bodies,distance_m=float(c.dist)))
            if any(n.startswith('robot/') for n in bodies) and not ('floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)):
                depth=max(depth,-float(c.dist))
                if c.dist<-.003:failures.append(dict(reason='nonfoot penetration',bodies=bodies,distance_m=float(c.dist)))
        for j in range(m.njnt):
            if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:continue
            value=d.qpos[m.jnt_qposadr[j]]
            if m.jnt_limited[j] and max(m.jnt_range[j,0]-value,value-m.jnt_range[j,1],0.)>.02:
                failures.append(dict(reason='actual individual joint bound',joint=m.joint(j).name))
        for side,digit in itertools.product(('lh','rh'),('FF','MF','RF','LF')):
            if d.qpos[qa(side+'_'+digit+'J1')]-d.qpos[qa(side+'_'+digit+'J2')]>.02:
                failures.append(dict(reason='actual passive loopback',hand=side,digit=digit))
        return failures,depth
    sensitivities={}
    for name,changes in {'proximal_FFJ3':{'rh_FFJ3':1.},'coupled_FFJ0':{'rh_FFJ1':.5,'rh_FFJ2':.5},'abduction_FFJ4':{'rh_FFJ4':1.}}.items():
        eps=.0001;d.qpos[:]=q
        for joint,scale in changes.items():d.qpos[qa(joint)]+=eps*scale
        mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
        position=d.xpos[body]+d.xmat[body].reshape(3,3)@local
        sensitivities[name]=dict(joint_coordinate=changes,distance_derivative_m_per_rad=(distance()-baseline_distance)/eps,
            attained_pad_motion_toward_lever_m_per_rad=float((position-world0)@world_normal/eps),
            actual_no_step=True)
    actual_samples=[]
    for offset in np.linspace(0.,a.index_proximal_bound,101):
        d.qpos[:]=q;d.qpos[qa('rh_FFJ3')]+=offset;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
        bad,depth=contact_failures();actual_samples.append(dict(offset_rad=float(offset),index_distance_m=distance(),max_nonfoot_penetration_m=depth,failures=bad))
    plan=json.loads(a.press_plan.read_text());acq=np.load(a.acquisition/'trajectory.npz')['qpos'][-1]
    if not plan['passed'] or plan['source_trajectory_sha256']!=sha(a.acquisition/'trajectory.npz') or plan['robot_xml_sha256']!=sha(robot) or plan['door_xml_sha256']!=sha(door/'door.xml'):raise ValueError('Original attained acquisition and press plan required')
    armqa=[qa(n) for n in plan['joint_names']];hj=int(m.jnt_qposadr[m.joint('leaf_handle_hinge').id]);bj=int(m.jnt_qposadr[m.joint('leaf_latch_bolt_slide').id])
    maximum_depth=0.;count=0;bad_samples=[]
    # Sample four proximal levels throughout every rigid press pose and every
    # previously declared five-digit closure corner. This is not a continuous
    # volume proof or a physical force result.
    for i,arm in enumerate(plan['path_qpos']):
        for choice in itertools.product((0,1),repeat=5):
            for proximal in np.linspace(0.,a.index_proximal_bound,4):
                d.qpos[:]=acq;d.qpos[armqa]=np.asarray(arm)-plan['constant_tracking_offset'];d.qpos[hj]=plan['screen'][i]['handle_angle_rad'];d.qpos[bj]=.014598*d.qpos[hj]
                for digit,flag in zip(('FF','MF','RF','LF','TH'),choice):
                    for j in (('1',) if digit=='TH' else ('1','2')):d.qpos[qa('rh_'+digit+'J'+j)]+=flag*(.08 if digit=='TH' else .03)
                d.qpos[qa('rh_FFJ3')]+=proximal;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
                bad,depth=contact_failures();maximum_depth=max(maximum_depth,depth);count+=1
                if bad:bad_samples.append(dict(press_index=i,offset_corner=choice,index_proximal_offset_rad=float(proximal),failures=bad))
    result=dict(schema='doorbench.offline-index-coordination.v1',passed=not bad_samples and not any(r['failures'] for r in actual_samples),
        scope='Detached archived actual contact diagnosis and sampled hypothetical coordination geometry; no stepped dynamics, runtime oracle, or loaded-grasp qualification',
        attained_contact_interval_s=[raw['interval_start_s'],raw['interval_end_s']],attained_qpos_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        actual_loaded_index_contacts=actual_contacts,force_weighted_index_contact_body_m=local.tolist(),force_weighted_index_outward_normal_body=normal.tolist(),
        actual_index_minimum_gap_m=baseline_distance,coordinate_sensitivities=sensitivities,
        index_proximal_bound_rad=a.index_proximal_bound,actual_attained_samples=actual_samples,
        nominal_press_samples=count,nominal_press_max_nonfoot_penetration_m=maximum_depth,nominal_press_bad_samples=bad_samples,
        robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),press_plan_sha256=sha(a.press_plan),
        inputs_sha256={str(path):sha(path) for path in [source/'provenance.json',source/'physics.jsonl.gz',source/'actual-transitions/manifest.json',last,a.acquisition/'trajectory.npz']},
        screen_source_sha256=sha(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');sim.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('actual_attained_samples','nominal_press_bad_samples')},indent=2))
    print('First nominal failures',bad_samples[:1])


if __name__=='__main__':main()
