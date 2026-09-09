#!/usr/bin/env python3
"""Run a source-bound native prerequisite and actual Isaac standing operation.

Run on a prepared GPU node, or wait for its ready.json. Only motor-driven
privileged partial opening is tested; this is not a learned policy benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','ready','reference','output','work'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--hold-attained-grasp',action='store_true',help='Test qualified attained finger hold in both physics backends')
    p.add_argument('--wait-for-run',type=Path,help='Wait for this earlier coordinator to finish before using the prepared node')
    p.add_argument('--isaac-timeout-seconds',type=float,default=4200.,help='Wall-clock budget including periodic evidence export')
    a=p.parse_args()
    if not 300<=a.isaac_timeout_seconds<=7200:raise ValueError('Isaac wall-clock budget must be 300..7200 seconds')
    if a.deadline_unix-time.time()<300:raise ValueError('At least five minutes of guarded runtime required')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'run.pid').write_text(str(os.getpid()))
    result={'passed':False,'scope':'Privileged standing acquisition and held partial opening; no full opening, traversal or learned actor'}
    result['hold_attained_grasp']=a.hold_attained_grasp
    hold_options=['--hold-attained-grasp'] if a.hold_attained_grasp else []
    env=dict(os.environ,PYTHONPATH=str(a.source),DOORBENCH_WORK=str(a.work),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y')
    commands=[]
    def stage(name):
        (a.output/'pipeline.json').write_text(json.dumps(dict(stage=name,report_file='coordinator-result.json',scope=result['scope'],deadline_unix=a.deadline_unix)))
        print(name,flush=True)
    def run(argv,name,maximum,allowed_codes=(0,)):
        remaining=a.deadline_unix-time.time()-300
        if remaining<30:raise TimeoutError('Guarded run budget exhausted')
        argv=list(map(str,argv));commands.append(dict(name=name,argv=argv,started_unix=time.time()))
        (a.output/'commands.json').write_text(json.dumps(commands,indent=2))
        stage(name)
        with (a.output/(name+'.log')).open('w') as log:
            process=subprocess.Popen(argv,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=min(maximum,remaining))
            except subprocess.TimeoutExpired:
                # Let the simulator close streams and export its measured prefix
                # before any process signal. An early stop remains a failed run.
                if name=='isaac-operation':
                    (Path(argv[argv.index('--output')+1])/'stop.request').write_text('Coordinator wall-clock budget exhausted\n')
                    commands[-1]['graceful_stop_requested']=True
                    try:code=process.wait(timeout=max(1.,min(240.,a.deadline_unix-time.time()-30.)))
                    except subprocess.TimeoutExpired:pass
                    else:
                        commands[-1].update(returncode=code,finished_unix=time.time())
                        (a.output/'commands.json').write_text(json.dumps(commands,indent=2))
                        if code not in allowed_codes:raise RuntimeError(name+' exited '+str(code))
                        return
                import signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        commands[-1].update(returncode=code,finished_unix=time.time())
        (a.output/'commands.json').write_text(json.dumps(commands,indent=2))
        if code not in allowed_codes:raise RuntimeError(name+' exited '+str(code))
    try:
        if a.wait_for_run is not None:
            stage('Waiting for previous recorded run to finish')
            while not (a.wait_for_run/'coordinator-result.json').exists():
                if time.time()>a.deadline_unix-2400:raise TimeoutError('Previous run left insufficient time')
                time.sleep(10)
        stage('Waiting for pinned Isaac environment and live physics proof')
        while not a.ready.exists():
            if time.time()>a.deadline_unix-2400:raise TimeoutError('Preparation left insufficient time for the physical test')
            time.sleep(10)
        ready=json.loads(a.ready.read_text())
        if ready.get('ready') is not True or ready.get('mechanics_profile')!='shadow-loopback-v2':raise ValueError('Corrected hand mechanics readiness required')
        frozen=json.loads((a.source/'source-manifest.json').read_text())
        for name,digest in frozen['files'].items():
            if sha(a.source/name)!=digest:raise ValueError('Prepared source changed: '+name)
        robot=Path(ready['native_robot']);motors=Path(ready['motor_contract']);door=Path(ready['door_usd']).parent
        if sha(robot)!=ready['native_robot_sha256']:raise ValueError('Ready robot bytes changed')
        asset=a.work/'asset-venv/bin/python';isaac=a.work/'venv/bin/python'
        screen=a.output/'screen';native=a.output/'native';layout=a.output/'sensor-layout.json'
        run([asset,a.source/'scripts/dexterous/rescreen_acquisition_reference.py','--robot',robot,'--door',door,'--reference',a.reference,'--output',screen],'geometry-screen',900)
        if json.loads((screen/'geometry-audit.json').read_text()).get('passed') is not True:raise ValueError('Destination geometry screen failed')
        reference=screen/'reference.json'
        run([asset,a.source/'scripts/dexterous/probe_acquisition_operation.py','--robot',robot,'--door',door,'--reference',reference,'--motors',motors,'--stance-profile','landed-foot-v1','--record-transitions','--index-finger-force','3','--portable-wrapper','--operator-compliance-gain','.2','--pressure-segment','distal','--grasp-offset-in-handle-m','.004','-.003','.0025','--seconds','36','--output',native,*hold_options],'native-prerequisite',1200,allowed_codes=(0,1))
        result['native_runtime_passed']=json.loads((native/'report.json').read_text()).get('passed') is True
        run([asset,a.source/'scripts/dexterous/audit_sensor_acquisition_contacts.py','--trial',native,'--output',a.output/'native-independent-audit.json'],'native-contact-audit',600)
        if not result['native_runtime_passed']:raise ValueError('Destination-native sustained hold failed')
        if json.loads((a.output/'native-independent-audit.json').read_text()).get('passed') is not True:raise ValueError('Native independent raw-contact audit failed')
        if a.hold_attained_grasp:
            started=json.loads((native/'trace.json').read_text())[-1]['teacher'].get('attained_hold_started_s')
            result['native_attained_hold_started_s']=started
            if type(started) not in (int,float) or not 0<started<=35.5:raise ValueError('Native attained hold did not activate with a qualified hold window')
        run([asset,a.source/'scripts/dexterous/export_sensor_layout.py','--robot',robot,'--output',layout],'robot-sensor-layout',120)
        inputs=[a.ready,a.reference,robot,motors,reference,screen/'geometry-audit.json',native/'report.json',layout,Path(ready['robot_usd']),Path(ready['door_usd'])]
        (a.output/'provenance.json').write_text(json.dumps(dict(source=frozen,coordinator_sha256=sha(__file__),input_sha256={str(path):sha(path) for path in inputs},scope=result['scope']),indent=2))
        trial=a.output/'trial'
        if a.deadline_unix-time.time()<a.isaac_timeout_seconds+300:raise TimeoutError('Insufficient guarded time for Isaac run and evidence export')
        run([isaac,a.source/'scripts/dexterous/isaac_opening.py','--robot-usd',ready['robot_usd'],'--door-usd',ready['door_usd'],'--motors',motors,'--reference',reference,'--sensor-layout',layout,'--reset-from-acquisition-path','--grasp-profile','distal-pad-v1','--joint-passive-profile','backend-dry-v2','--seconds','36','--output',trial,'--headless','--device','cuda:0','--record','--enable_cameras','--sensor-gyro-profile','pose-delta-angle-v1','--acquisition','--native-robot',robot,'--acquisition-stance-profile','landed-foot-v1','--acquisition-pressure-segment','distal','--acquisition-index-finger-force','3','--operate-after-acquisition','--operator-compliance-gain','.2','--operation-min-acquisition-seconds','10.6','--operation-grasp-offset-in-handle-m','.004','-.003','.0025',*hold_options],'isaac-operation',a.isaac_timeout_seconds,allowed_codes=(0,1))
        # Preserve and independently audit failures as well as successful runs.
        result['isaac_runtime_passed']=json.loads((trial/'operation-report.json').read_text())['passed']
        run([asset,a.source/'scripts/dexterous/audit_isaac_acquisition_contacts.py','--trial',trial,'--output',a.output/'isaac-independent-audit.json'],'isaac-contact-audit',600)
        audit=json.loads((a.output/'isaac-independent-audit.json').read_text())
        result['independent_audit_passed']=bool(audit['accounting_passed'] and audit['independent_raw_contact_audit_complete'])
        result['passed']=bool(result['isaac_runtime_passed'] and result['independent_audit_passed'])
        if a.hold_attained_grasp:
            latest=json.loads((trial/'latest.json').read_text());started=latest['teacher'].get('attained_hold_started_s')
            result['isaac_attained_hold_started_s']=started
            result['attained_hold_activated']=bool(type(started) in (int,float) and 0<started<=latest['time_s']-.5)
            result['passed'] &= result['attained_hold_activated']
    except Exception as exc:result['error']=type(exc).__name__+': '+str(exc)
    finally:
        result['finished_unix']=time.time()
        (a.output/'coordinator-result.json').write_text(json.dumps(result,indent=2)+'\n')
        stage('Completed: PASS' if result['passed'] else 'Completed: FAIL — inspect preserved evidence')
        print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':sys.exit(main())
