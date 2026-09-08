#!/usr/bin/env python3
"""Freeze the23s geometric candidate into named goals and audit source/rates.

No physical simulation or runtime scene input is created by this preparer.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np
from doorbench.dexterous.scripted_thumb_coordination import FIXED,THUMB_NAMES,ScriptedThumbGoalCoordinator


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('trial','candidate','profile','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.profile.exists() or a.output.exists():p.error('Fresh profile and receipt required')
    candidate=json.loads(a.candidate.read_text());parent=json.loads((a.candidate.parent/'report.json').read_text());prov=json.loads((a.trial/'provenance.json').read_text())
    if candidate.get('passed') is not True or candidate['time_s']!=23. or candidate['joint_names']!=list(THUMB_NAMES):raise ValueError('Exactly the passed23s thumb-only candidate required')
    if parent['inputs_sha256']!={n:sha(a.trial/n) for n in ('provenance.json','trajectory.npz','physics.jsonl.gz')}:raise ValueError('Candidate source trajectory changed')
    robot=Path(prov['parameters']['robot'])
    if sha(robot)!=prov['robot_xml_sha256'] or parent['robot_xml_sha256']!=sha(robot):raise ValueError('Candidate robot changed')
    with gzip.open(a.trial/'controller.jsonl.gz','rt') as f:info=[json.loads(line) for line in f]
    start=dict(zip(info[11500]['goal_joint_names'],info[11500]['goal_joint_position_rad']))
    profile=dict(FIXED,source_goal_at_start={n:float(start[n]) for n in THUMB_NAMES},terminal_goal=dict(zip(THUMB_NAMES,candidate['candidate_qpos'])),
        robot_xml_sha256=sha(robot),source_candidate_sha256=sha(a.candidate),source_trial_provenance_sha256=sha(a.trial/'provenance.json'))
    schedule=ScriptedThumbGoalCoordinator(profile);m=mujoco.MjModel.from_xml_path(str(robot));goals=[];prefix=True;other=True;limit=True;time=[]
    for i,r in enumerate(info):
        old=dict(zip(r['goal_joint_names'],r['goal_joint_position_rad']));new,_=schedule.apply(old,now_s=i*.002)
        if i<11500:prefix &= new==old
        other &= all(new[n]==old[n] for n in old if n not in THUMB_NAMES)
        limit &= all(m.jnt_range[m.joint(n).id,0]<=v<=m.jnt_range[m.joint(n).id,1] for n,v in new.items())
        goals.append([new[n] for n in THUMB_NAMES]);time.append(i*.002)
    goals=np.array(goals);velocity=np.diff(goals,axis=0)/.002;acceleration=np.diff(velocity,axis=0)/.002
    speed=float(abs(velocity).max());accel=float(abs(acceleration[11499:]).max())
    checks=dict(candidate_source_identity=True,first23s_goals_unchanged=bool(prefix),other25goals_unchanged=bool(other),
        original_joint_limits=bool(limit),existing_goal_slew=speed<=1.5,
        post23_maximum_thumb_goal_speed=float(abs(velocity[11499:]).max())<=.08,
        post23_maximum_thumb_goal_acceleration=accel<=.3,
        final_goals_equal_frozen_candidate=bool(np.array_equal(goals[-1],candidate['candidate_qpos'])))
    a.profile.parent.mkdir(parents=True,exist_ok=True);a.profile.write_text(json.dumps(profile,indent=2)+'\n')
    result=dict(scope=__doc__,passed=all(checks.values()),checks=checks,maximum_thumb_goal_speed_radps=speed,
        maximum_post23_thumb_goal_speed_radps=float(abs(velocity[11499:]).max()),maximum_post23_thumb_goal_acceleration_radps2=accel,
        maximum_source_goal_change_rad=max(abs(profile['terminal_goal'][n]-profile['source_goal_at_start'][n]) for n in THUMB_NAMES),
        profile_sha256=sha(a.profile),source_candidate_sha256=sha(a.candidate),source_candidate_report_sha256=sha(a.candidate.parent/'report.json'),
        robot_xml_sha256=sha(robot),source_trial_provenance_sha256=sha(a.trial/'provenance.json'),
        controller_source_sha256=sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous/scripted_thumb_coordination.py'),
        source_sha256=sha(__file__),profile=profile,
        caution='This screens requested goals on the retained source history; the actual force-controlled thumb need not follow the nominal geometric path.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
