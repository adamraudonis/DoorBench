"""Replay actual native reach sensor packets through the portable runtime.

This never steps physics or qualifies a new trajectory. Actual root/door/reset
measurements are not loaded by this command; only recorded packets and the
frozen joint-only route enter inference. Actual61 forces are comparison data.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from doorbench.dexterous.sensor_reach_runtime import SensorReachBalanceRuntime


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','robot','calibration','protocol','joint-route','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous replay evidence')
    run=a.run;motors=json.loads((run/'motors.json').read_text());layout=json.loads((run/'sensor-layout.json').read_text())
    c=SensorReachBalanceRuntime(a.robot,motors,layout,a.calibration,a.protocol,a.joint_route);c.reset_episode()
    packets=np.load(run/'actor-inputs.npz',allow_pickle=False);trajectory=np.load(run/'trajectory.npz',allow_pickle=False)
    actual=trajectory['force'];errors=[];failure=None
    for i in range(len(actual)):
        try:
            packet={k:packets[k][i].copy() for k in packets.files}
            packet.update(rgb_left=np.zeros((128,128,3),np.uint8),rgb_right=np.zeros((128,128,3),np.uint8))
            if np.any(packet['sensor_valid'][-2:]):raise ValueError('Native replay requires recorded invalid RGB')
            force=c.force(packet,i*.002);errors.append(float(abs(force-actual[i]).max()))
            if i%500==0:print(json.dumps(dict(decision=i,time_s=i*.002,max_error_Nm=max(errors))),flush=True)
        except Exception as exc:failure=dict(decision=i,time_s=i*.002,error=type(exc).__name__+': '+str(exc));break
    result=dict(scope=__doc__,replay_passed=failure is None and len(errors)==5500 and max(errors)<1e-7,
        decisions=len(actual),replayed_decisions=len(errors),maximum_force_error_Nm=max(errors,default=None),failure=failure,
        physics_steps=0,calculator_time_s=c.last_info.get('calculator_time_s'),robot_sha256=sha(a.robot),calibration_sha256=sha(a.calibration),
        protocol_sha256=sha(a.protocol),joint_route_sha256=sha(a.joint_route),script_sha256=sha(__file__),
        runtime_sha256=sha(Path(__file__).resolve().parents[2]/'doorbench/dexterous/sensor_reach_runtime.py'),
        source_run=str(run),source_sha256={n:sha(run/n) for n in ('report.json','provenance.json','actor-inputs.npz','trajectory.npz','motors.json','sensor-layout.json')})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('replay_passed','replayed_decisions','maximum_force_error_Nm','failure')}),flush=True)
if __name__=='__main__':main()
