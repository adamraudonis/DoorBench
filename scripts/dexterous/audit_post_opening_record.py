#!/usr/bin/env python3
"""Read-only continuity, motor and frame audit of an actual-transition archive."""
import argparse,gzip,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',required=True,type=Path);a=p.parse_args()
    manifest=json.loads((a.trial/'inputs.json').read_text());report=json.loads((a.trial/'report.json').read_text());args=report['arguments']
    for name in ('robot','motors','initial_trajectory','door.xml'):
        receipt=manifest['files'][name]
        if hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()!=receipt['sha256']:raise ValueError('Changed input: '+name)
    robot=Path(args['robot']);s=DexterousDoorEnv(Path(args['door']),robot,json.loads(robot.with_suffix('.audit.json').read_text()));s.reset(images=False,randomize=False)
    m=s.m;planning=mujoco.MjData(m);motors=json.loads(Path(args['motors']).read_text());ids=np.array([m.actuator('robot/'+x['name']).id for x in motors['actuators']]);caps=np.array([x['force_range'] for x in motors['actuators']])
    initial=np.load(args['initial_trajectory']);prior_q=initial['terminal_qpos'];prior_v=initial['terminal_qvel'];previous_end=0.
    count=0;max_force_error=0.;max_geometry_error=0.;geometry_samples=0;max_overshoot=0.;pairs={}
    with gzip.open(a.trial/'actual-transitions.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);q=np.array(row['qpos_before']);v=np.array(row['qvel_before']);end=float(row['interval_end_s']);start=float(row['interval_start_s'])
            if abs(start-previous_end)>1e-8 or abs(end-start-m.opt.timestep)>1e-8 or row['geometry_time_s']!=start:raise ValueError('Broken actual transition clock')
            if not np.array_equal(q,prior_q) or not np.array_equal(v,prior_v):raise ValueError('State discontinuity or runtime reset')
            forces=np.array(row['actuator_force'])[ids];control=np.array(row['controls'])[ids]
            if not np.isfinite(np.r_[q,v,forces,control]).all():raise ValueError('Nonfinite state/action')
            max_force_error=max(max_force_error,float(np.max(abs(forces-control))))
            max_overshoot=max(max_overshoot,float(np.max(np.maximum(forces-caps[:,1],caps[:,0]-forces))))
            if count%100==0:
                planning.qpos[:]=q;mujoco.mj_kinematics(m,planning);b=np.array(row['body_ids'],int)
                errors=[np.max(abs(planning.xpos[b]-row['body_positions_world_m'])),np.max(abs(planning.xmat[b].reshape(-1,3,3)-row['body_rotations_world']))]
                max_geometry_error=max(max_geometry_error,*map(float,errors));geometry_samples+=1
            for c in row['contacts']:
                names=[m.body(b).name for b in c['body']]
                robot_sides=[n.startswith('robot/') for n in names]
                if sum(robot_sides)!=1:continue
                robot_name=names[robot_sides.index(True)]
                if robot_name.endswith('_ankle_link'):continue
                if c['distance_m']>0 and c['wrench_contact_frame'][0]<=0:continue
                key=' | '.join([m.geom(g).name or names[i] for i,g in enumerate(c['geom'])])
                record=pairs.setdefault(key,dict(intervals=0,max_normal_force_N=0.,max_penetration_m=0.,first_time_s=start,last_time_s=start))
                record['intervals']+=1;record['max_normal_force_N']=max(record['max_normal_force_N'],float(c['wrench_contact_frame'][0]));record['max_penetration_m']=max(record['max_penetration_m'],-float(c['distance_m']));record['last_time_s']=start
            prior_q=np.array(row['qpos_after']);prior_v=np.array(row['qvel_after']);previous_end=end;count+=1
    final=np.load(a.trial/'trajectory.npz')
    checks=dict(complete=count==round(report['expected_duration_s']/m.opt.timestep),final_state_matches=np.array_equal(prior_q,final['terminal_qpos']) and np.array_equal(prior_v,final['terminal_qvel']),force_delivery=max_force_error<1e-5,motor_caps=max_overshoot<1e-5,independent_geometry_matches=max_geometry_error<1e-9)
    out=dict(passed=all(checks.values()),checks=checks,actual_transitions=count,geometry_samples=geometry_samples,geometry_sample_stride=100,maximum_geometry_error=max_geometry_error,maximum_motor_delivery_error_Nm=max_force_error,maximum_force_cap_excess_Nm=max_overshoot,nonfoot_environment_contacts=pairs,scope='Read-only archive continuity and sampled FK replay; no physics steps or controller execution',files={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.trial/'actual-transitions.jsonl.gz',Path(__file__))})
    (a.trial/'independent-archive-audit.json').write_text(json.dumps(out,indent=2)+'\n');s.close();print(json.dumps(out))
    return 0 if out['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
