#!/usr/bin/env python3
"""Isolated RunPod allocation and deadline guard for dexterous experiments.

Does not read or overwrite the legacy shared pod state. Never prints credentials.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Explicit per-allocation journals let a replacement node coexist with a stopped
# node whose persistent volume still contains evidence. Child guards inherit it.
STATE = Path(os.environ.get('DOORBENCH_POD_STATE', str(Path.home() / '.runpod/doorbench_dexterous_pod.json'))).expanduser()

def api():
    path = Path(__file__).resolve().parents[1] / 'runpod_pod.py'
    spec = importlib.util.spec_from_file_location('doorbench_pod_api', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.STATE = STATE
    return module

def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['create','status','guard','terminate'])
    p.add_argument('--hours', type=float, default=4.)
    p.add_argument('--gpu', choices=['NVIDIA L40S','NVIDIA RTX 6000 Ada Generation',
                                   'NVIDIA A40','NVIDIA GeForce RTX 4090'],
                   default='NVIDIA L40S', help='Explicit RTX-capable allocation; default remains L40S')
    a = p.parse_args(); rp = api()
    if a.command == 'create':
        if STATE.exists() and json.loads(STATE.read_text()).get('active'):
            raise SystemExit('An owned allocation is already active; inspect status first')
        if not 0 < a.hours <= 8:
            raise SystemExit('Initial allocation must have a deadline of at most eight hours')
        rp.DEFAULT_POD = dict(rp.DEFAULT_POD, name='doorbench-dexterous-humanoid',
                              volumeInGb=100, minVCPUPerGPU=16)
        rp.cmd_create(argparse.Namespace(gpu=a.gpu))
        state = json.loads(STATE.read_text()); state.update(active=True, deadline=time.time()+a.hours*3600)
        STATE.write_text(json.dumps(state,indent=2))
        log = STATE.with_suffix('.guard.log')
        try:
            with log.open('ab') as stream:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'guard'],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True)
            state['guard_pid'] = process.pid
            STATE.write_text(json.dumps(state,indent=2))
        except BaseException:
            rp._req('DELETE', '/pods/' + state['id'])
            raise
        print(json.dumps(state,indent=2))
    elif a.command == 'guard':
        state = json.loads(STATE.read_text()); pod_id=state['id']
        while time.time() < state['deadline']:
            time.sleep(min(30, max(.1,state['deadline']-time.time())))
            current=json.loads(STATE.read_text())
            if current.get('id') != pod_id or not current.get('active'):
                return
        # Capture the allocation ID; never terminate a subsequently created pod.
        for attempt in range(10):
            try:
                rp._req('DELETE','/pods/'+pod_id)
                state.update(active=False,terminated_at=time.time(),termination_reason='deadline')
                STATE.write_text(json.dumps(state,indent=2)); print('Deadline teardown complete',flush=True); return
            except (Exception,SystemExit) as exc:
                print(type(exc).__name__, 'during deadline teardown; retrying',flush=True); time.sleep(30)
        raise SystemExit('Deadline teardown failed after retries')
    elif a.command == 'status':
        state=json.loads(STATE.read_text()); pod=rp._req('GET','/pods/'+state['id'])
        clean={k:pod.get(k) for k in ('id','name','desiredStatus','costPerHr','publicIp','portMappings')}
        print(json.dumps(dict(clean,deadline=state.get('deadline'),guard_pid=state.get('guard_pid')),indent=2))
    else:
        state=json.loads(STATE.read_text()); rp._req('DELETE','/pods/'+state['id'])
        state.update(active=False,terminated_at=time.time());STATE.write_text(json.dumps(state,indent=2))

if __name__=='__main__':
    main()
