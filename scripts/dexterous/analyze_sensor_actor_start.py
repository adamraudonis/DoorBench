"""Compare an actual failed actor start with the qualified teacher reset."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('actor','teacher','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();read=lambda p,n:json.loads((p/n).read_text())
    model=read(args.actor,'motor-contract.json');names=[a['name'] for a in model['actuators']]
    actor=np.load(args.actor/'acquisition-physics.npz',allow_pickle=False)
    teacher=np.load(args.teacher/'acquisition-physics.npz',allow_pickle=False)
    first=np.load(args.actor/'sensors/actor-initial-decision.npz',allow_pickle=False)
    ar=read(args.actor,'acquisition-reset.json');tr=read(args.teacher,'acquisition-reset.json')
    difference=actor['motor_forces'][0]-teacher['motor_forces'][0]
    ticks=[]
    for target in (.002,.01,.02,.05,.1,.2,.3,.4,float(actor['time_s'][-1])):
        i=int(np.argmin(abs(actor['time_s']-target)));j=int(np.argmin(abs(teacher['time_s']-actor['time_s'][i])))
        ticks.append(dict(time_s=float(actor['time_s'][i]),actor_tilt_deg=float(actor['torso_tilt_deg'][i]),
            teacher_tilt_deg=float(teacher['torso_tilt_deg'][j]),actor_height_m=float(actor['root'][i,2]),
            joint_difference_norm_rad=float(np.linalg.norm(actor['joints'][i]-teacher['joints'][j]))))
    crossings={str(bound):float(actor['time_s'][np.flatnonzero(actor['torso_tilt_deg']>=bound)[0]])
        for bound in (12,45) if np.any(actor['torso_tilt_deg']>=bound)}
    files=('report.json','acquisition-reset.json','acquisition-physics.npz','motor-contract.json',
        'sensors/actor-initial-decision.npz','sensors/report.json')
    report=dict(scope='Actual closed-loop sensor actor failure; independent initial-command diagnostic, no reset replay or teacher force application',
        actor_run=str(args.actor),teacher_run=str(args.teacher),actor_report=read(args.actor,'report.json'),
        exact_reset_root_match=bool(np.array_equal(ar['root'],tr['root'])),exact_reset_joint_match=bool(np.array_equal(ar['joints'],tr['joints'])),
        initial_sensor_valid=first['sensor_valid'].tolist(),initial_sensor_times_s=first['sensor_time_s'].tolist(),
        initial_previous_action_zero=bool(np.all(first['previous_action']==0)),
        initial_recorded_force_matches_first_delivered_force=bool(np.array_equal(first['motor_forces'],actor['motor_forces'][0])),
        initial_largest_force_errors=[dict(motor=names[i],actor_Nm=float(actor['motor_forces'][0,i]),teacher_Nm=float(teacher['motor_forces'][0,i]),error_Nm=float(difference[i])) for i in np.argsort(-abs(difference))[:12]],
        sampled_progress=ticks,tilt_bound_first_crossings_s=crossings,
        actor_files_sha256={name:hashlib.sha256((args.actor/name).read_bytes()).hexdigest() for name in files},
        limitations=['Same-time teacher force comparisons after the first tick are not corrective labels for the divergent actor state.',
            'The all-invalid first observation is retained; whether it was supervised depends on this checkpoint training provenance.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(reset_match=report['exact_reset_root_match'] and report['exact_reset_joint_match'],crossings=crossings,first_errors=report['initial_largest_force_errors'][:2])))


if __name__=='__main__':main()
