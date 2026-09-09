#!/usr/bin/env python3
"""Compare two completed physical grasp trials without converting failures to passes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Retain prior comparisons')
    reports=[];arrays=[];sources=[];configurations=[];parameters=[]
    for run in (a.baseline,a.candidate):
        task=json.loads((run/'balance-report.json').read_text())
        physical=json.loads((run/'independent-audit.json').read_text())
        gyro=json.loads((run/'independent-gyro-audit.json').read_text())
        if physical.get('verification_passed') is not True or gyro.get('verification_passed') is not True:
            raise ValueError('Require independent reproduction of each actual result')
        if task['physics_steps']!=9500 or task['duration_s']!=19.:
            raise ValueError('Require complete matched nineteen-second trials')
        with np.load(run/'acquisition-physics.npz',allow_pickle=False) as z:
            arrays.append({k:z[k].copy() for k in z.files})
        reports.append(dict(passed=task['passed'],checks=task['checks'],
            longest_opposed_hold_s=task['strongest_qualified_hold_s'],
            final_half_second_minimum_pad_force_N=task['final_half_second_minimum_pad_force_N'],
            invalid_loaded_patches=task['invalid_loaded_patch_count'],
            maximum_motor_coordinate_tracking_error_rad=task['maximum_motor_coordinate_tracking_error_rad'],
            palm_endpoint_error_m=task['palm_endpoint_error_m']))
        sources.append({n:digest(run/n) for n in ('balance-report.json','independent-audit.json','independent-gyro-audit.json',
            'acquisition-physics.npz','balance-steps.json.gz','motor-contract.json','balance-acquisition-route.json',
            'balance-acquisition-protocol.json','configuration.json','sensors/layout.json')})
        configurations.append(json.loads((run/'configuration.json').read_text())['args'])
        parameters.append(json.loads((run/'balance-acquisition-protocol.json').read_text())['parameters'])
    excluded={'feedback_profile','tactile_reflex_profile','scope'}
    if ({k:v for k,v in parameters[0].items() if k not in excluded} !=
            {k:v for k,v in parameters[1].items() if k not in excluded} or
            parameters[1].get('tactile_reflex_profile') not in ('four-finger-preload-v1','four-finger-preload-v2','four-finger-preload-v3')):
        raise ValueError('Different nominal acquisition protocols')
    same_inputs={k:sources[0][k]==sources[1][k] for k in ('motor-contract.json','balance-acquisition-route.json','sensors/layout.json')}
    same_config={k:configurations[0][k]==configurations[1][k] for k in
        ('seconds','grasp_profile','joint_passive_profile','sensor_gyro_profile','robot_usd','door_usd')}
    if not all(same_inputs.values()) or not all(same_config.values()):
        raise ValueError('Physical embodiment, sensors or protocol settings changed')
    if any(not np.allclose(z['time_s'],np.arange(1,9501)*.002,atol=1e-8,rtol=0) for z in arrays):
        raise ValueError('Incomplete physical clocks')
    prefix=arrays[0]['time_s']<=17.+1e-8
    comparison={}
    for name in ('root','joints','motor_forces','door','door_velocity'):
        x,y=(z[name][prefix] for z in arrays)
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid matched actual records')
        comparison[name]=dict(exactly_equal=bool(np.array_equal(x,y)),maximum_absolute_error=float(np.max(abs(x-y))))
    with gzip.open(a.candidate/'balance-steps.json.gz','rt') as f:steps=json.load(f)
    offsets={d:[] for d in ('ff','mf','rf','lf')};active=0
    for row in steps:
        reflex=row['controller_info']['tactile_grasp_reflex'];active+=int(reflex['active'])
        for d in offsets:offsets[d].append(reflex['coupled_motor_preload_rad'][d])
    result=dict(scope=__doc__,baseline=reports[0],candidate=reports[1],
        same_frozen_inputs=same_inputs,same_runtime_settings=same_config,
        pre_reflex_physical_intervals=int(prefix.sum()),pre_reflex_comparison=comparison,
        candidate_active_reflex_intervals=active,
        candidate_maximum_coupled_preload_rad={d:max(v) for d,v in offsets.items()},
        source_reports=sources,script_sha256=digest(__file__),
        limitation='Same-start component comparison only; no opening, traversal, learned policy or robustness claim')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('baseline','candidate','candidate_maximum_coupled_preload_rad')}))


if __name__=='__main__':main()
