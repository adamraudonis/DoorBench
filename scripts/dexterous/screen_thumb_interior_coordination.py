#!/usr/bin/env python3
"""Thumb-only geometric relief from actual pre-stop grasp states.

The palm, four fingers, root and door stay fixed in a detached calculator.
This is a candidate configuration/path screen, never a loaded grasp result.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from scipy.optimize import least_squares
from doorbench.dexterous.environment import DexterousDoorEnv
from scripts.dexterous.probe_sensor_touch_operation import intended_contact_geometry


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('trial','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh complete screen directory required')
    a.output.mkdir(parents=True)
    prov=json.loads((a.trial/'provenance.json').read_text());robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door'])
    if sha(robot)!=prov['robot_xml_sha256'] or sha(door/'door.xml')!=prov['door_xml_sha256']:raise ValueError('Actual source geometry changed')
    with np.load(a.trial/'trajectory.npz') as z:states=z['qpos'].copy()
    with gzip.open(a.trial/'physics.jsonl.gz','rt') as f:physical=[json.loads(line) for line in f]
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));m=sim.m;d=mujoco.MjData(m)
    names=['rh_THJ'+str(i) for i in (5,4,3,2,1)];joints=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[joints]
    body=m.body('robot/rh_thdistal').id;lever=m.geom('leaf_handle_lever_col_n').id;palm=m.site('robot/rh_palm_touch').id
    geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and ((m.geom_contype[g]&m.geom_conaffinity[lever]) or (m.geom_contype[lever]&m.geom_conaffinity[g]))]
    results=[]
    for time_s in (23.,23.5,24.,24.4):
        index=round(time_s/.002);base=states[index].copy();q0=base[qa].copy();d.qpos[:]=base;mujoco.mj_kinematics(m,d)
        fixed_palm=d.site_xpos[palm].copy();fixed_indices=np.setdiff1d(np.arange(m.nq),qa)
        fixed_sites=[m.site('robot/rh_'+digit+'distal_touch').id for digit in ('ff','mf','rf','lf')]
        fixed_fingers=d.site_xpos[fixed_sites].copy()
        center=d.geom_xpos[lever].copy();axis=d.geom_xmat[lever].reshape(3,3)[:,2].copy();half=m.geom_size[lever,1]
        actual=[c for c in physical[index]['pad_grasp']['contacts'] if c['digit']=='th' and c['pad_qualified'] and c['normal_force_N']>0]
        if not actual:raise ValueError('Actual pre-stop qualified thumb required')
        loads=np.array([c['normal_force_N'] for c in actual]);loads/=loads.sum()
        local_point=sum(w*np.array(c['body_position_m']) for w,c in zip(loads,actual))
        local_normal=sum(w*np.array(c['hand_outward_normal_body']) for w,c in zip(loads,actual));local_normal/=np.linalg.norm(local_normal)
        initial_normal=d.xmat[body].reshape(3,3)@local_normal
        initial_axial=float((d.xpos[body]+d.xmat[body].reshape(3,3)@local_point-center)@axis)
        desired_axial=float(np.clip(initial_axial,-half+.003,half-.003))
        lower=np.maximum(m.jnt_range[joints,0]+.01,q0-.12);upper=np.minimum(m.jnt_range[joints,1]-.01,q0+.12)
        lower[3]=max(lower[3],m.jnt_range[joints[3],0]+.025)
        def geometry(q):
            d.qpos[:]=base;d.qpos[qa]=q;mujoco.mj_kinematics(m,d)
            distances=[float(mujoco.mj_geomDistance(m,d,g,lever,.05,None)) for g in geoms]
            R=d.xmat[body].reshape(3,3);point=d.xpos[body]+R@local_point
            return min(distances),float((point-center)@axis),R@local_normal
        initial_gap=geometry(q0)[0]
        def residual(q):
            gap,axial,normal=geometry(q)
            return np.r_[1000*(gap+.00003),200*(axial-desired_axial),5*(normal-initial_normal),.05*(q-q0)]
        fit=least_squares(residual,np.clip(q0,lower+1e-10,upper-1e-10),bounds=(lower,upper),max_nfev=350,ftol=1e-12,xtol=1e-12,gtol=1e-12)
        goal=fit.x.copy();end=geometry(goal);bad=[];path=[];max_pen=0.;minimum_thumb_overlap_count=100000;max_palm_error=0.;max_normal_turn=0.;max_fixed_q=max_finger_motion=0.
        for step,s in enumerate(np.linspace(0.,1.,1001)):
            blend=s**3*(10+s*(-15+6*s));q=q0+(goal-q0)*blend
            gap,axial,normal=geometry(q);mujoco.mj_collision(m,d);failures=[];thumb=0
            if np.any(q<m.jnt_range[joints,0]) or np.any(q>m.jnt_range[joints,1]):failures.append('authored thumb joint bound')
            for c in d.contact[:d.ncon]:
                bn=[m.body(m.geom_bodyid[g]).name for g in c.geom]
                if not any(n.startswith(('robot/rh_','robot/lh_')) for n in bn) or c.dist>0:continue
                max_pen=max(max_pen,-float(c.dist))
                if not intended_contact_geometry(m,d,c,lever):failures.append('unintended hand surface:'+','.join(bn))
                if c.dist<-.003:failures.append('hand penetration above3mm')
                if lever in c.geom and body in [m.geom_bodyid[g] for g in c.geom]:thumb+=1
            if thumb==0 or gap>0:failures.append('thumb detached geometric path')
            max_palm_error=max(max_palm_error,float(np.linalg.norm(d.site_xpos[palm]-fixed_palm)))
            max_fixed_q=max(max_fixed_q,float(np.max(abs(d.qpos[fixed_indices]-base[fixed_indices]))))
            max_finger_motion=max(max_finger_motion,float(np.max(np.linalg.norm(d.site_xpos[fixed_sites]-fixed_fingers,axis=1))))
            max_normal_turn=max(max_normal_turn,float(np.arccos(np.clip(normal@initial_normal,-1,1))))
            minimum_thumb_overlap_count=min(minimum_thumb_overlap_count,thumb)
            if failures:bad.append(dict(sample=step,fraction=float(s),failures=sorted(set(failures)),gap_m=gap,axial_m=axial))
            path.append(q.tolist())
        endpoint_margin=goal-m.jnt_range[joints,0]
        checks=dict(solver_succeeded=bool(fit.success),target_gap_relieved=bool(-.00005<=end[0]<=0 and end[0]>initial_gap),
            th2_interior_margin=bool(endpoint_margin[3]>=.025-1e-9),all_thumb_interior=bool(np.all(goal>=m.jnt_range[joints,0]+.01-1e-9) and np.all(goal<=m.jnt_range[joints,1]-.01+1e-9)),
            complete_connected_path=not bad,palm_and_four_fingers_fixed=max_palm_error<1e-12 and max_fixed_q==0. and max_finger_motion<1e-12,
            bounded_goal_adjustment=bool(np.max(abs(goal-q0))<=.12+1e-9),calculator_never_stepped=d.time==0.)
        result=dict(time_s=time_s,passed=all(checks.values()),checks=checks,source_interval_start_s=physical[index]['contact_interval_start_s'],
            joint_names=names,actual_initial_qpos=q0.tolist(),candidate_qpos=goal.tolist(),delta_rad=(goal-q0).tolist(),
            initial_gap_m=initial_gap,candidate_gap_m=end[0],candidate_axial_m=end[1],minimum_axial_clearance_m=half-abs(end[1]),
            th2_interior_margin_rad=float(endpoint_margin[3]),maximum_hand_penetration_m=max_pen,
            maximum_palm_motion_m=max_palm_error,maximum_thumb_normal_turn_rad=max_normal_turn,
            maximum_non_thumb_coordinate_change=max_fixed_q,maximum_four_finger_site_motion_m=max_finger_motion,
            minimum_geometric_thumb_contacts=minimum_thumb_overlap_count,solver_evaluations=fit.nfev,solver_cost=fit.cost,
            failed_path_samples=bad,path_qpos=path,
            target_construction='relieve signed overlap toward30um, retain actual thumb normal and axial band; THJ2>=lower+25mrad; all other thumb margins>=10mrad',
            geometry_is_not_a_load_measurement=True)
        (a.output/f'candidate-{index:05d}.json').write_text(json.dumps(result,indent=2)+'\n');results.append({k:v for k,v in result.items() if k not in ('path_qpos','failed_path_samples')})
    report=dict(scope=__doc__,results=results,passed_candidates=sum(r['passed'] for r in results),total_candidates=len(results),
        constraints='Only five thumb joints change; actual palm, four other fingers, torso, pelvis, feet and door fixed; original collision surfaces and bounds',
        source_sha256=sha(__file__),robot_xml_sha256=sha(robot),door_xml_sha256=sha(door/'door.xml'),
        inputs_sha256={n:sha(a.trial/n) for n in ['provenance.json','trajectory.npz','physics.jsonl.gz']})
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');sim.close();print(json.dumps(report,indent=2))

if __name__=='__main__':main()
