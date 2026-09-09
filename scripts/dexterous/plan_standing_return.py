#!/usr/bin/env python3
"""Plan and independently screen lever return from a fully qualified transfer."""
import argparse,hashlib,json,subprocess,sys,shutil
from pathlib import Path
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_archive import unpacked
from doorbench.dexterous.whole_body_return_planner import iter_whole_body_return


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();trial=a.source_run
    checks=[trial/n for n in ('report.json','independent-pad-audit.json','independent-whole-handle-audit.json')]
    if not all(json.loads(f.read_text())['passed'] for f in checks):raise ValueError('Runtime, pad and full handle source audits must pass')
    manifest=json.loads((trial/'manifest.json').read_text());cfg=manifest['configuration'];robot=Path(cfg['robot']);door=Path(cfg['door'])
    if sha(robot)!=manifest['inputs']['robot']['sha256'] or sha(door/'door.xml')!=manifest['inputs']['door']['door.xml']:raise ValueError('Exact recorded model required')
    archive=json.loads((trial/'raw-transitions/manifest.json').read_text())
    if not archive['complete']:raise ValueError('Complete source archive required')
    chunk=archive['chunks'][-1];path=trial/'raw-transitions'/chunk['file']
    if sha(path)!=chunk['sha256']:raise ValueError('Changed source dynamics')
    with np.load(path) as z:row=list(unpacked({k:z[k] for k in z.files}))[-1]
    state={k:row[v] for k,v in [('qpos','qpos_before'),('qvel','qvel_before'),('time_s','interval_start_s'),('geometry_time_s','geometry_time_s'),('body_ids','body_ids'),('body_positions_world_m','body_positions_world_m'),('body_rotations_world','body_rotations_world')]};state['observation_scope']='archived-native-contact-bodies'
    a.output.mkdir(parents=True,exist_ok=False);frozen=a.output/'return-planner-source.py';shutil.copy2(__file__,frozen)
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));rows=[]
    try:
        for r in iter_whole_body_return(sim.m,**state):
            r['geometry_passed']=bool(not r['forbidden_collisions'] and max(r['right_position_error_m'],r['left_position_error_m'],*r['foot_position_errors_m'])<=.001 and max(r['right_rotation_error_rad'],r['left_rotation_error_rad'],*r['foot_rotation_errors_rad'])<=.01 and r['torso_tilt_deg']<=4)
            rows.append(r);(a.output/'latest.json').write_text(json.dumps(r));print(json.dumps({k:r[k] for k in ('progress','geometry_passed','torso_tilt_deg','right_position_error_m')}),flush=True)
            if not r['geometry_passed']:break
    finally:sim.close()
    report=dict(passed=len(rows)==41 and all(r['geometry_passed'] for r in rows),scope='Unstepped measured-state candidate, not physical return qualification',physics_steps=0,rows=rows,source_state=state,source_sha256={str(f):sha(f) for f in [robot,door/'door.xml',path,frozen,*checks]})
    plan=a.output/'report.json';plan.write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:return 1
    audit=a.output/'dense-audit.json';subprocess.run([sys.executable,str(Path(__file__).with_name('audit_standing_return_route.py')),'--robot',str(robot),'--door',str(door),'--plan',str(plan),'--output',str(audit)],check=True)
    if not json.loads(audit.read_text())['passed']:return 1
    config=dict(schema='doorbench.standing-return.v1',plan_path=str(plan),plan_sha256=sha(plan),audit_path=str(audit),audit_sha256=sha(audit),robot_xml_sha256=sha(robot),robot_path=str(robot),door_path=str(door/'door.xml'))
    (a.output/'return.json').write_text(json.dumps(config,indent=2)+'\n');return 0


if __name__=='__main__':raise SystemExit(main())
