#!/usr/bin/env python3
"""Attained load-direction diagnosis and full nominal palm-shift envelope."""
import argparse,gzip,hashlib,itertools,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_archive import unpacked
from doorbench.dexterous.robot_palm_translation import RobotPalmTranslation,ARM_NAMES
from scripts.dexterous.probe_sensor_acquisition_balance import intended_contact_geometry


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('attained','acquisition','press-plan','index-screen','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh detached screen receipt required')
    prov=json.loads((a.attained/'provenance.json').read_text());robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door']);door=door if door.is_dir() else door.parent
    if sha(robot)!=prov['robot_xml_sha256'] or sha(door/'door.xml')!=prov['door_xml_sha256']:raise ValueError('Original plant changed')
    index_screen=json.loads(a.index_screen.read_text());plan=json.loads(a.press_plan.read_text())
    if index_screen['passed'] is not True or index_screen['index_proximal_bound_rad']!=.01 or index_screen['press_plan_sha256']!=sha(a.press_plan):raise ValueError('Original index envelope required')
    if plan['source_trajectory_sha256']!=sha(a.acquisition/'trajectory.npz'):raise ValueError('Original attained acquisition required')
    manifest=json.loads((a.attained/'actual-transitions/manifest.json').read_text());last=a.attained/'actual-transitions'/manifest['chunks'][-1]['file']
    if sha(last)!=manifest['chunks'][-1]['sha256']:raise ValueError('Actual contact archive changed')
    with np.load(last) as z:raw=list(unpacked({k:z[k] for k in z.files}))[-1]
    with gzip.open(a.attained/'physics.jsonl.gz','rt') as f:
        for line in f:row=json.loads(line)
    if abs(row['sim_time_s']-raw['interval_end_s'])>1e-9:raise ValueError('Contact/geometry epochs differ')
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()),frame_skip=1);m=sim.m;d=mujoco.MjData(m)
    own=mujoco.MjModel.from_xml_path(str(robot));names=[own.joint(i).name for i in range(1,own.njnt)];translator=RobotPalmTranslation(own,names)
    qa=np.array([m.jnt_qposadr[m.joint('robot/'+n).id] for n in names]);armqa=np.array([m.jnt_qposadr[m.joint('robot/'+n).id] for n in ARM_NAMES])
    lever=m.geom('leaf_handle_lever_col_n').id
    def evaluate(q,offset):
        d.qpos[:]=q
        goals,info=translator.goals(dict(zip(ARM_NAMES,q[armqa])),q[qa],float(offset))
        d.qpos[armqa]=[goals[n] for n in ARM_NAMES];mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
        bad=[];depth=0.
        for c in d.contact[:d.ncon]:
            bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom];geoms=[m.geom(g).name for g in c.geom]
            if c.dist<=0 and any(n.startswith(('robot/rh_','robot/lh_')) for n in bodies) and not intended_contact_geometry(m,d,c,lever):bad.append(dict(reason='unintended hand surface',bodies=bodies,distance_m=float(c.dist)))
            if any(n.startswith('robot/') for n in bodies) and not ('floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)):
                depth=max(depth,-float(c.dist))
                if c.dist<-.003:bad.append(dict(reason='nonfoot penetration',bodies=bodies,distance_m=float(c.dist)))
        # The translator itself strictly bounds every commanded arm angle;
        # retained actual finger splits keep their original independent audit.
        for side,digit in itertools.product(('lh','rh'),('FF','MF','RF','LF')):
            a1=int(m.jnt_qposadr[m.joint('robot/'+side+'_'+digit+'J1').id]);a2=int(m.jnt_qposadr[m.joint('robot/'+side+'_'+digit+'J2').id])
            if d.qpos[a1]-d.qpos[a2]>.02:bad.append(dict(reason='actual passive loopback',hand=side,digit=digit))
        return dict(offset_m=float(offset),maximum_nonfoot_penetration_m=depth,failures=bad,kinematics=info)
    actual=np.asarray(raw['qpos_before']);d.qpos[:]=actual;mujoco.mj_kinematics(m,d)
    direction=np.mean([d.xmat[m.body('robot/rh_'+f+'distal').id].reshape(3,3)@np.array([0.,-1.,0.]) for f in ('ff','mf','rf','lf')],axis=0);direction/=np.linalg.norm(direction)
    palmR=d.site_xmat[m.site('robot/rh_palm_touch').id].reshape(3,3).copy();opposition={}
    for digit in ('ff','mf','rf','lf','th'):
        contacts=[c for c in row['pad_grasp']['contacts'] if c['digit']==digit and c['normal_force_N']>0 and c['pad_qualified']]
        if not contacts:raise ValueError('Actual original opposed distal grasp required for direction diagnosis')
        weights=np.array([c['normal_force_N'] for c in contacts]);weights/=weights.sum()
        normal=np.average([c['hand_outward_normal_body'] for c in contacts],axis=0,weights=weights)
        R=d.xmat[m.body('robot/rh_'+digit+'distal').id].reshape(3,3)
        opposition[digit]=dict(motion_dot_actual_outward_normal=float(direction@(R@normal)),actual_normal_force_N=sum(c['normal_force_N'] for c in contacts))
    if not all(opposition[f]['motion_dot_actual_outward_normal']>.8 for f in ('ff','mf','rf','lf')) or opposition['th']['motion_dot_actual_outward_normal']>-.8:raise ValueError('Own-robot direction does not redistribute actual opposed contact as hypothesized')
    actual_samples=[evaluate(actual,x) for x in np.linspace(0,.00025,51)]
    acq=np.load(a.acquisition/'trajectory.npz')['qpos'][-1];hj=int(m.jnt_qposadr[m.joint('leaf_handle_hinge').id]);bj=int(m.jnt_qposadr[m.joint('leaf_latch_bolt_slide').id])
    count=0;bad=[];maximum_depth=0.;max_joint_correction=0.;maximum_pos_error=0.;maximum_rot_error=0.
    for i,arm in enumerate(plan['path_qpos']):
        for corner in itertools.product((0,1),repeat=6):
            q=acq.copy();q[armqa]=np.asarray(arm)-plan['constant_tracking_offset'];q[hj]=plan['screen'][i]['handle_angle_rad'];q[bj]=.014598*q[hj]
            for digit,flag in zip(('FF','MF','RF','LF','TH'),corner[:5]):
                for j in (('1',) if digit=='TH' else ('1','2')):q[m.jnt_qposadr[m.joint('robot/rh_'+digit+'J'+j).id]]+=flag*(.08 if digit=='TH' else .03)
            q[m.jnt_qposadr[m.joint('robot/rh_FFJ3').id]]+=corner[5]*.01
            for offset in (0.,.000125,.00025):
                try:r=evaluate(q,offset)
                except ValueError as exc:r=dict(offset_m=offset,maximum_nonfoot_penetration_m=0.,failures=[dict(reason=str(exc))],kinematics={})
                count+=1;maximum_depth=max(maximum_depth,r['maximum_nonfoot_penetration_m'])
                max_joint_correction=max(max_joint_correction,r['kinematics'].get('maximum_joint_correction_rad',0.));maximum_pos_error=max(maximum_pos_error,r['kinematics'].get('position_error_m',0.));maximum_rot_error=max(maximum_rot_error,r['kinematics'].get('orientation_error_rad',0.))
                if r['failures']:bad.append(dict(press_index=i,offset_corner=corner,**r))
    result=dict(schema='doorbench.offline-palm-translation.v1',passed=not bad and not any(r['failures'] for r in actual_samples),
        scope='Actual interval contact-direction diagnosis and sampled unstepped geometry; not loaded force or continuous-volume qualification',
        actual_contact_interval_s=[raw['interval_start_s'],raw['interval_end_s']],actual_load_direction=opposition,direction_palm_local=(palmR.T@direction).tolist(),
        maximum_palm_translation_m=.00025,actual_attained_samples=actual_samples,nominal_samples=count,nominal_bad_samples=bad,
        maximum_nonfoot_penetration_m=maximum_depth,maximum_joint_correction_rad=max_joint_correction,
        maximum_position_error_m=maximum_pos_error,maximum_orientation_error_rad=maximum_rot_error,
        robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),press_plan_sha256=sha(a.press_plan),index_screen_sha256=sha(a.index_screen),
        inputs_sha256={str(path):sha(path) for path in [a.attained/'provenance.json',a.attained/'physics.jsonl.gz',a.attained/'actual-transitions/manifest.json',last,a.acquisition/'trajectory.npz']},
        source_sha256=sha(__file__),translator_source_sha256=sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous/robot_palm_translation.py'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');sim.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('actual_attained_samples','nominal_bad_samples')},indent=2));print('First failures',bad[:1])


if __name__=='__main__':main()
