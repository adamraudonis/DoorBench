#!/usr/bin/env python3
"""Replay recorded angles through proposed references only; no physics or success claim."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from doorbench.dexterous.coupled_release_geometry import sha
from doorbench.dexterous.coupled_release_reference import CoupledReleaseReference
from doorbench.dexterous.json_record_stream import iter_json_object_array
from doorbench.dexterous.release_source_admission import admit_release_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--recorded-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh replay diagnostic required')
    source_dir=a.output.with_name(a.output.stem+'-source');source_dir.mkdir(exist_ok=False)
    import doorbench.dexterous.coupled_release_geometry as geometry_source
    import doorbench.dexterous.coupled_release_reference as reference_source
    import doorbench.dexterous.coupled_release_motion as motion_source
    frozen_inputs={}
    for module in (__file__,geometry_source.__file__,reference_source.__file__,motion_source.__file__):
        frozen=source_dir/Path(module).name;shutil.copy2(module,frozen);frozen_inputs[str(frozen.resolve())]=sha(frozen)
    config=json.loads(a.config.read_text());source=Path(config['source_run']);envelope=json.loads(Path(config['coupled_envelope_path']).read_text())
    admission=admit_release_source(source,profile=config['grasp_profile'],contact_audit_name=config['contact_audit_name'],measured_rest=True)
    with np.load(source/'trajectory.npz') as z:initial=z['terminal_qpos'].copy();epoch=float(z['terminal_time_s'])
    reference=CoupledReleaseReference(config,admission,initial,envelope['duration_s']);g=reference.geometry;m,d=g.m,g.d
    report=json.loads((a.recorded_run/'report.json').read_text());start=report['standing_withdrawal']['started_s']
    if start!=epoch:raise ValueError('Recorded experiment must share exact release source epoch')
    clocks={}
    with gzip.open(a.recorded_run/'controller-steps.json.gz','rt') as f:
        for row in iter_json_object_array(f):
            if 'withdrawal_clock_s' in row:clocks[round(row['time_s'],9)]=row['withdrawal_clock_s']
    archive=a.recorded_run/'raw-transitions';manifest=json.loads((archive/'manifest.json').read_text())
    if not manifest['complete']:raise ValueError('Complete recorded dynamics required for a replay diagnostic')
    count=0;first=None;last=None;max_clock_error=0.;failure=None;max_motion={};inputs={**frozen_inputs,str(a.config.resolve()):sha(a.config),str((archive/'manifest.json').resolve()):sha(archive/'manifest.json')}
    for chunk in manifest['chunks']:
        if chunk['interval_end_s']<=epoch+1e-8:continue
        path=archive/chunk['file']
        if sha(path)!=chunk['sha256']:raise ValueError('Recorded chunk changed')
        inputs[str(path.resolve())]=chunk['sha256']
        with np.load(path) as rows:
            for t,q in zip(rows['interval_start_s'],rows['qpos_before']):
                t=float(t)
                if t<epoch-1e-8:continue
                elapsed=t-epoch;clock_key=round(t,9)
                if clock_key not in clocks:raise ValueError('Every replayed interval must retain actual recorded reference clock')
                if first is None:
                    first=t
                    if not np.array_equal(q,initial):raise ValueError('Replay must begin at the exact qualified source')
                d.qpos[:]=q;mujoco.mj_kinematics(m,d)
                leaf=np.r_[d.xpos[g.leaf],d.xquat[g.leaf]];handle=np.r_[d.xpos[g.handle],d.xquat[g.handle]]
                angles=dict(leaf=float(q[g.lq]),operator=float(q[g.oq]),latch=float(q[g.bq]))
                try:
                    result=reference.update(t,elapsed,angles,leaf,handle)
                    error=abs(result[-1]['coupled_reference_release_clock_s']-clocks[clock_key]);max_clock_error=max(max_clock_error,error)
                    if error>1e-10:raise ValueError('Recorded clock and once-warped candidate clock differ')
                except ValueError as exc:
                    failure=dict(time_s=t,elapsed_s=elapsed,angles=angles,recorded_withdrawal_clock_s=clocks[clock_key],error=str(exc),reference_qpos=d.qpos.tolist(),last_accepted_info=reference.info)
                    break
                count+=1;last=t
                for key,value in result[-1]['coupled_reference_motion'].items():
                    if key!='limited':max_motion[key]=max(max_motion.get(key,0.),value)
        if failure is not None:break
    result=dict(schema='doorbench.coupled-release-reference-replay.v1',passed=failure is None and count==round(envelope['duration_s']/.002),scope=__doc__,physics_steps=0,physical_admission=False,recorded_run=str(a.recorded_run.resolve()),samples=count,first_capture_s=first,last_accepted_s=last,maximum_recorded_clock_error_s=max_clock_error,maximum_nominal_reference_motion=max_motion,failure=failure,input_sha256=inputs)
    a.output.write_text(json.dumps(result,indent=2)+'\n');g.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','failure')},indent=2))
    if failure:print(json.dumps({k:v for k,v in failure.items() if k not in ('reference_qpos','last_accepted_info')},indent=2))


if __name__=='__main__':main()
