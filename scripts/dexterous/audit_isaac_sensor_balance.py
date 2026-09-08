"""Independently verify a completed or failed Isaac stationary, arm reach or scripted acquisition archive.

Recomputes every step's floor support and hand contact count from archived raw
normal-contact slots. No inference, physics step, source mutation or GPU action.
"""
import argparse,gzip,hashlib,json,re
from pathlib import Path
import numpy as np
from doorbench.dexterous.sensor_balance_runtime import evaluate_sensor_balance,BALANCE_PROTOCOL
from doorbench.dexterous.passive_evidence import audit_joint_passive_evidence


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def recompute_contacts(layout,row):
    paths=layout['sensor_paths'];filters=np.asarray(layout['filter_paths']);capacity=layout['capacity']
    if type(capacity) is not int or capacity<=0 or len(paths)!=len(set(paths)) or filters.ndim!=2 or filters.shape[0]!=len(paths):
        raise ValueError('Malformed original contact layout')
    names=[p.rsplit('/',1)[-1] for p in paths];feet_names=['left_ankle_link','right_ankle_link']
    if any(names.count(n)!=1 for n in feet_names):raise ValueError('Actual ankle sensor rows required')
    feet=np.zeros(2);hands=0;occupied=set()
    for c in row['contacts']:
        i,j,slot=c['sensor'],c['filter'],c['slot']
        if (any(type(x) is not int for x in (i,j,slot)) or not 0<=i<len(paths) or not 0<=j<filters.shape[1] or
                not 0<=slot<capacity or slot in occupied):raise ValueError('Duplicate or invalid occupied contact slot')
        occupied.add(slot);force=float(c['force_N']);gap=float(c['distance_m']);normal=np.asarray(c['normal'],float);position=np.asarray(c['position'],float)
        if (normal.shape!=(3,) or position.shape!=(3,) or not np.isfinite(np.r_[force,gap,normal,position]).all() or
                force<0 or not np.isclose(np.linalg.norm(normal),1.,atol=1e-5,rtol=0)):
            raise ValueError('Invalid actual normal contact')
        if names[i] in feet_names and filters[i,j].rsplit('/',1)[-1]=='floor':feet[feet_names.index(names[i])]+=force*normal[2]
        if names[i].startswith(('rh_','lh_')):hands+=int(gap<=0 or force>1e-8)
    if len(occupied)>=capacity:raise ValueError('Potentially truncated occupied contact archive')
    return feet,hands


def recompute_unintended_contacts(layout,row):
    """Independent all-hand counterpart reduction for the explicit Door55 grasp."""
    recompute_contacts(layout,row)  # Validate every raw slot/vector first.
    bad=0
    for p in row['contacts']:
        body=layout['sensor_paths'][p['sensor']].rsplit('/',1)[-1]
        if not body.startswith(('rh_','lh_')) or not (p['distance_m']<=0 or p['force_N']>1e-8):continue
        digit=re.fullmatch(r'rh_(ff|mf|rf|lf|th)(knuckle|proximal|middle|distal|base|hub|metacarpal)',body)
        counterpart=layout['filter_paths'][p['sensor']][p['filter']]
        bad+=int(digit is None or counterpart!='/World/Door/Articulation/leaf_handle')
    return bad


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--robot',type=Path,help='Robot-only XML required for independent reach FK');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep prior independent audits')
    run=a.run
    with gzip.open(run/'balance-steps.json.gz','rt') as f:steps=json.load(f)
    with gzip.open(run/'balance-contacts.jsonl.gz','rt') as f:contacts=[json.loads(line) for line in f]
    layout=json.loads((run/'balance-contact-layout.json').read_text());declared=json.loads((run/'balance-report.json').read_text())
    errors=[];load_error=0.;hand_error=0;unintended_error=0
    if len(contacts)!=len(steps):errors.append('Actual contact and step archives differ in length')
    for i,(r,c) in enumerate(zip(steps,contacts)):
        try:
            if (abs(c['interval_end_s']-r['time_s'])>1e-8 or
                    abs(c['interval_start_s']-(r['time_s']-.002))>1e-8):raise ValueError('Actual contact interval mismatch')
            feet,hands=recompute_contacts(layout,c)
            load_error=max(load_error,float(abs(feet-np.asarray(r['foot_floor_loads'])).max()))
            hand_error=max(hand_error,abs(hands-r['hand_contact_count']))
            if declared.get('schema')=='doorbench.sensor-scripted-acquisition-19s.v1':
                unintended=recompute_unintended_contacts(layout,c)
                unintended_error=max(unintended_error,abs(unintended-r['unintended_hand_contact_count']))
        except (ValueError,KeyError,TypeError) as e:errors.append(f'Contact row{i}: {e}')
    names=['balance-steps.json.gz','balance-contacts.jsonl.gz','balance-contact-layout.json','balance-report.json','sensor-balance-calibration.json','configuration.json','provenance.json','motor-contract.json','sensors/layout.json']
    if declared.get('schema')==BALANCE_PROTOCOL:
        scored=evaluate_sensor_balance(steps,declared.get('original_physics_checks'))
    else:
        from doorbench.dexterous.sensor_arm_balance_runtime import evaluate_sensor_arm_balance,ARM_PROTOCOL
        if declared.get('schema')==ARM_PROTOCOL:
            names+=['balance-arm-reset.json','balance-arm-schedule.json']
            scored=evaluate_sensor_arm_balance(steps,declared.get('original_physics_checks'),
                initial_arm_joint_position=json.loads((run/'balance-arm-reset.json').read_text()),schedule=run/'balance-arm-schedule.json')
        elif declared.get('schema')=='doorbench.sensor-scripted-reach-11s.v1':
            from doorbench.dexterous.sensor_reach_evaluation import evaluate_sensor_reach_balance
            if a.robot is None:raise ValueError('Reach audit requires the original robot-only XML for FK')
            names+=['balance-reach-reset.json','balance-reach-protocol.json','balance-reach-route.json']
            reset=json.loads((run/'balance-reach-reset.json').read_text())
            scored=evaluate_sensor_reach_balance(steps,declared.get('original_physics_checks'),
                robot_xml=a.robot,motors=json.loads((run/'motor-contract.json').read_text()),
                initial_root13_actororigin=reset['root13_actororigin'],initial_joint_position=reset['joint_position'],
                calibration=run/'sensor-balance-calibration.json',protocol=run/'balance-reach-protocol.json',joint_route=run/'balance-reach-route.json')
        elif declared.get('schema')=='doorbench.sensor-scripted-acquisition-19s.v1':
            from doorbench.dexterous.sensor_acquisition_evaluation import evaluate_sensor_acquisition_balance
            if a.robot is None:raise ValueError('Acquisition audit requires the original robot-only XML for FK')
            names+=['balance-acquisition-reset.json','balance-acquisition-protocol.json','balance-acquisition-route.json']
            reset=json.loads((run/'balance-acquisition-reset.json').read_text())
            scored=evaluate_sensor_acquisition_balance(steps,declared.get('original_physics_checks'),
                robot_xml=a.robot,motors=json.loads((run/'motor-contract.json').read_text()),
                initial_root13_actororigin=reset['root13_actororigin'],initial_joint_position=reset['joint_position'],
                initial_hand_contact_count=reset['hand_contact_count'],initial_door_position=reset['initial_door_position'],
                calibration=run/'sensor-balance-calibration.json',protocol=run/'balance-acquisition-protocol.json',joint_route=run/'balance-acquisition-route.json')
        else:
            # An exception-prefix report has no completed qualification schema.
            # Preserve that failure instead of crashing or inventing a pass.
            scored=dict(passed=False,checks={});errors.append('No completed supported balance qualification report')
    passive=audit_joint_passive_evidence(run,steps,declared)
    errors.extend('Passive evidence: '+error for error in passive['errors'])
    reproduced=scored['checks']==declared.get('checks') and scored['passed']==declared.get('passed')
    report=dict(scope=__doc__,run=str(run),verification_passed=bool(reproduced and not errors and load_error<1e-8 and hand_error==0 and unintended_error==0),
        actual_balance_trial_passed=scored['passed'],actual_protocol=declared.get('schema'),recomputed_checks=scored['checks'],declared_report_reproduced=reproduced,
        passive_evidence=passive,actual_trial_with_passive_evidence_passed=bool(scored['passed'] and passive['verification_passed']),
        steps=len(steps),contact_intervals=len(contacts),maximum_floor_load_reconstruction_error_N=load_error,
        maximum_hand_contact_count_error=hand_error,maximum_unintended_hand_contact_count_error=unintended_error,errors=errors,source_sha256={n:sha(run/n) for n in names},
        audit_source_sha256=sha(__file__),physics_steps_in_audit=0,limitation='Verifying a failed report never changes the actual failure to a pass. This remains the declared balance component, not a door task.')
    if declared.get('schema')==BALANCE_PROTOCOL:report['actual_stationary_trial_passed']=scored['passed']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('verification_passed','actual_balance_trial_passed','actual_protocol','steps','maximum_floor_load_reconstruction_error_N','maximum_hand_contact_count_error','errors')}),flush=True)
if __name__=='__main__':main()
