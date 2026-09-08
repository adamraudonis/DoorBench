"""Replay actual Isaac balance decisions; no simulated state or teacher input.

Decision zero uses the saved cold packet. Decision i>0 uses actual sensor row
i-1 and its exact causal camera frame, never the observation produced by i.
This checks command reproducibility, not a second physical balance trial.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from doorbench.dexterous.sensor_actor import ActorDimensions, prepare_actor_packet
from doorbench.dexterous.sensor_balance_runtime import SensorBalanceRuntime
from doorbench.dexterous.sensor_contract import SENSOR_KEYS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def decision_packet(index,initial,numeric,rgb,dimensions):
    if not 0<=index<len(numeric['time_s']):raise IndexError('Decision outside actual episode')
    expected=set(dimensions.shapes)|{'previous_action','sensor_time_s','sensor_valid'}
    if set(initial)!=expected|{'time_s','motor_forces'} or float(initial['time_s'])!=0.:
        raise ValueError('Missing exact initial decision')
    if set(numeric)!=(expected-{'rgb_left','rgb_right'})|{'time_s'}:
        raise ValueError('Unexpected recorded numeric fields')
    if index==0:
        packet={k:initial[k].copy() for k in expected};time=0.
    else:
        time=float(numeric['time_s'][index-1])
        packet={k:v[index-1].copy() for k,v in numeric.items() if k!='time_s'}
        for key in ('rgb_left','rgb_right'):
            stream=SENSOR_KEYS.index(key);capture=packet['sensor_time_s'][stream]
            frame=int(np.searchsorted(rgb['time_s'],capture+1e-9,side='right')-1)
            if not packet['sensor_valid'][stream]:packet[key]=np.zeros(dimensions.shapes[key],np.uint8)
            elif frame<0 or abs(rgb['time_s'][frame]-capture)>1e-8:
                raise ValueError('Missing causal camera frame')
            else:packet[key]=rgb[key][frame].copy()
    if abs(time-index*.002)>1e-8:raise ValueError('Actual decision clock is not continuous2ms')
    prepare_actor_packet(packet,time,dimensions)
    return packet,time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','robot','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scripted-arms',action='store_true',help='Explicitly replay the separate six-second arm runtime')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve earlier replay evidence')
    run=a.run;motors=json.loads((run/'motor-contract.json').read_text());layout=json.loads((run/'sensors/layout.json').read_text())
    if a.scripted_arms:
        from doorbench.dexterous.sensor_arm_balance_runtime import SensorArmBalanceRuntime
        controller=SensorArmBalanceRuntime(a.robot,motors,layout,run/'sensor-balance-calibration.json',run/'balance-arm-schedule.json')
    else:controller=SensorBalanceRuntime(a.robot,motors,layout,run/'sensor-balance-calibration.json')
    controller.reset_episode()
    dimensions=ActorDimensions(tactile=layout['tactile_dimension'])
    initial=arrays(run/'sensors/actor-initial-decision.npz');numeric=arrays(run/'sensors/actor-sensors.npz')
    rgb=arrays(run/'sensors/actor-rgb.npz');physical=arrays(run/'acquisition-physics.npz')
    n=len(numeric['time_s']);caps=np.array([m['force_range'][1] for m in motors['actuators']]);errors=[];failure=None
    if (physical['motor_forces'].shape!=(n,61) or not np.allclose(numeric['time_s'],physical['time_s'],atol=1e-8,rtol=0)
            or not np.allclose(numeric['time_s'],np.arange(1,n+1)*.002,atol=1e-8,rtol=0)):
        raise ValueError('Physical commands and sensor endpoints differ')
    normalization_error=float(abs(numeric['previous_action']-physical['motor_forces']/caps).max())
    initial_error=float(abs(initial['motor_forces']-physical['motor_forces'][0]).max())
    for i in range(n):
        try:
            packet,time=decision_packet(i,initial,numeric,rgb,dimensions)
            force=controller.force(packet,time)
            errors.append(float(abs(force-physical['motor_forces'][i]).max()))
        except Exception as exc:
            failure=dict(decision=i,time_s=i*.002,error=type(exc).__name__+': '+str(exc));break
    names=['sensor-balance-calibration.json','motor-contract.json','sensors/layout.json','sensors/actor-initial-decision.npz',
        'sensors/actor-sensors.npz','sensors/actor-rgb.npz','acquisition-physics.npz','balance-report.json','configuration.json','provenance.json']
    if a.scripted_arms:names.append('balance-arm-schedule.json')
    maximum=max(errors,default=None)
    report=dict(scope=__doc__,run=str(run),replay_passed=failure is None and maximum is not None and maximum<1e-5 and normalization_error<2e-7 and initial_error<1e-8,
        decisions=n,replayed_decisions=len(errors),maximum_motor_force_error_Nm=maximum,maximum_normalized_action_error=normalization_error,
        initial_force_record_error_Nm=initial_error,failure=failure,physics_steps_in_replay=0,teacher_calls=0,
        calculator_time_s=controller.last_info.get('calculator_time_s'),actual_balance_trial_passed=json.loads((run/'balance-report.json').read_text()).get('passed') is True,
        runtime_mode='scripted_arm_balance' if a.scripted_arms else 'stationary_balance',
        source_sha256={name:sha(run/name) for name in names},robot_sha256=sha(a.robot),script_sha256=sha(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('replay_passed','decisions','replayed_decisions','maximum_motor_force_error_Nm','failure')}),flush=True)
if __name__=='__main__':main()
