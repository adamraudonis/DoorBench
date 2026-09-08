"""Compare two actual 11-second reach captures in named motor/split coordinates.

No engine, policy inference or active plant writes. Matching static inputs and
controller-source hashes support a controlled adapter comparison; this does not
establish contact or grasp success from a contact-free reach.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(legacy, dry):
    legacy, dry = Path(legacy), Path(dry)
    identical_names = ['motor-contract.json','balance-reach-reset.json','balance-reach-protocol.json',
        'balance-reach-route.json','sensor-balance-calibration.json','sensors/layout.json']
    identical_names += ['source-'+n+'.py' for n in ('sensor_reach_balance','locomotion_manipulation',
        'sensor_reach_evaluation','sensor_contract','sensor_balance_runtime','sensor_actor','sensor_balance',
        'motor_contract_identity','stance','sensor_reach_runtime','reach_balance_schedule')]
    matched = {n:sha(legacy/n) for n in identical_names}
    if any(sha(dry/n) != h for n,h in matched.items()):
        raise ValueError('Frozen reset, route, robot calibration or controller source differs')
    physical_inputs = {}
    for key in ('robot_usd','door_usd','sensor_balance_robot','sensor_layout','reference'):
        values=[]
        for run in (legacy,dry):
            config=json.loads((run/'configuration.json').read_text())
            provenance=json.loads((run/'provenance.json').read_text())
            values.append(provenance['files'][config['args'][key]])
        if values[0] != values[1]:
            raise ValueError('Pre-step physical/reference input hash differs: '+key)
        physical_inputs[key]=values[0]
    names = json.loads((legacy/'motor-contract.json').read_text())['joint_names']
    if len(names)!=69 or len(set(names))!=69:
        raise ValueError('Require original named 69-joint contract')
    indices = {digit:[names.index('rh_'+digit+j) for j in ('J1','J2')] for digit in ('FF','MF','RF','LF')}
    histories = {}; summaries = {}; hashes = {}
    for label,run in [('legacy',legacy),('dry',dry)]:
        with gzip.open(run/'balance-steps.json.gz','rt') as f:rows=json.load(f)
        reset=json.loads((run/'balance-reach-reset.json').read_text())
        report=json.loads((run/'balance-report.json').read_text())
        config=json.loads((run/'configuration.json').read_text())
        selected=config['args'].get('joint_passive_profile','legacy-tanh-v1')
        if selected != ('legacy-tanh-v1' if label=='legacy' else 'backend-dry-v2'):
            raise ValueError('Comparison requires original legacy and explicit backend-dry profiles')
        if (len(rows)!=5500 or not np.allclose([r['time_s'] for r in rows],np.arange(1,5501)*.002,atol=1e-8,rtol=0) or
                report['schema']!='doorbench.sensor-scripted-reach-11s.v1'):
            raise ValueError('Require complete matched 11-second physical step clocks')
        q=np.array([[r['actual_joint_position'][n] for n in names] for r in rows])
        initial=np.array([reset['joint_position'][n] for n in names])
        desired=[]
        for r in rows:
            target=initial.copy();info=r['controller_info'];gn=info['goal_joint_names'];gv=info['goal_joint_position_rad']
            if len(gn)!=30 or len(set(gn))!=30 or len(gv)!=30:
                raise ValueError('Require complete applied nominal joint goals')
            target[[names.index(n) for n in gn]]=gv;desired.append(target)
        desired=np.array(desired)
        if q.shape!=(5500,69) or not np.isfinite(np.r_[q,desired]).all():
            raise ValueError('Nonfinite actual/nominal joint trajectory')
        history=np.r_[initial[None],q];histories[label]=(history,np.r_[initial[None],desired])
        splits={};sums={};hold_drift={}
        for digit,(j1,j2) in indices.items():
            splits[digit]=float(np.max(abs(q[:,j1]-q[:,j2]-desired[:,j1]+desired[:,j2])))
            sums[digit]=float(np.max(abs(q[:,j1]+q[:,j2]-desired[:,j1]-desired[:,j2])))
            hold_drift[digit]=float((history[500,j1]-history[500,j2])-(initial[j1]-initial[j2]))
        summaries[label]=dict(profile=selected,reported_trial_passed=report['passed'],
            maximum_nominal_passive_split_error_rad=splits,maximum_pair_motor_sum_error_rad=sums,
            first_second_actual_split_drift_rad=hold_drift,maximum_nominal_joint_error_rad=float(np.max(abs(q-desired))),
            total_hand_contacts=sum(r['hand_contact_count'] for r in rows),
            actual_initial_joint_max_error_rad=float(np.max(abs(initial-histories['legacy'][0][0]))))
        hashes[label]={n:sha(run/n) for n in ('balance-steps.json.gz','balance-report.json','configuration.json')}
    if not np.array_equal(histories['legacy'][1],histories['dry'][1]):
        raise ValueError('Actual applied nominal joint goal histories differ')
    samples=[]
    for time in (0.,.002,1.,4.,8.,11.):
        i=round(time/.002);entry=dict(time_s=time,pairs={})
        for digit,pair in indices.items():
            values={label:data[0][i,pair] for label,data in histories.items()}
            entry['pairs'][digit]=dict(legacy_J1_J2_rad=values['legacy'].tolist(),dry_J1_J2_rad=values['dry'].tolist(),
                dry_minus_legacy_sum_rad=float(np.sum(values['dry'])-np.sum(values['legacy'])),
                dry_minus_legacy_split_rad=float(np.diff(values['legacy'])[0]-np.diff(values['dry'])[0]))
        samples.append(entry)
    return dict(schema='doorbench.recorded-reach-profile-comparison.v1',scope=__doc__,
        legacy=str(legacy),dry=str(dry),matched_static_and_controller_sha256=matched,
        matched_physical_input_sha256=physical_inputs,source_sha256=hashes,
        exact_applied_goal_histories_match=True,steps_per_run=5500,physics_steps_in_audit=0,
        summaries=summaries,samples=samples,audit_source_sha256=sha(__file__),
        limitation='Passive split is compared from actual named measurements. Both runs are contact-free; reduced drift cannot demonstrate a successful grasp or identical engine compliance.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('legacy','dry','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve earlier comparison evidence')
    result=compare(a.legacy,a.dry);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summaries'],indent=2))


if __name__=='__main__':main()
