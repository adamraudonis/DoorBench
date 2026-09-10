#!/usr/bin/env python3
"""Dry-run-first renewal of only the journaled dexterous RunPod allocation.

Replacement local/remote guards must acknowledge before the journal changes or
any verified old guard is signaled. Never signal process groups or GPU jobs.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid
import pod
from arm_remote_guard import REMOTE as LEGACY_REMOTE

STATE=pod.STATE
GUARD_ROOT=Path.home()/'.runpod/doorbench-guards'

GUARD='''import json,os,sys,time,urllib.request,urllib.error
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads(p.read_text());receipt=p.with_name('status.json')
def status(phase):
    if phase=='terminated' and s.get('journal'):
        journal=Path(s['journal']);current=json.loads(journal.read_text())
        if current.get('id')==s['id'] and current.get('deadline')==s['deadline']:
            current.update(active=False,terminated_at=time.time(),termination_reason='renewed_deadline')
            temporary=journal.with_name(journal.name+'.guard.tmp');temporary.write_text(json.dumps(current,indent=2));temporary.chmod(0o600);os.replace(temporary,journal)
    receipt.write_text(json.dumps({'id':s['id'],'deadline':s['deadline'],'pid':os.getpid(),'phase':phase,'heartbeat_unix':time.time()}))
status('armed')
while time.time()<s['deadline']:
    if s.get('journal'):
        current=json.loads(Path(s['journal']).read_text())
        if current.get('id')!=s['id'] or not current.get('active'):status('allocation_changed');sys.exit(0)
    time.sleep(min(10,max(.1,s['deadline']-time.time())))
    status('armed')
for attempt in range(10):
    try:
        request=urllib.request.Request('https://rest.runpod.io/v1/pods/'+s['id'],method='DELETE',headers={'Authorization':'Bearer '+s['key'],'User-Agent':'DoorBench/1.0'})
        urllib.request.urlopen(request,timeout=30).close()
        status('terminated');break
    except urllib.error.HTTPError as exc:
        if exc.code==404:status('terminated');break
        status('teardown_retry');time.sleep(30)
    except Exception:status('teardown_retry');time.sleep(30)
else:status('teardown_failed');sys.exit(1)
'''

REMOTE_OPS='''import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
r=json.load(sys.stdin)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snapshot(pid):
    p=Path('/proc')/str(pid)
    try:
        argv=[v.decode() for v in p.joinpath('cmdline').read_bytes().rstrip(b'\\0').split(b'\\0')]
        stat=p.joinpath('stat').read_text().split(') ')[1].split()
    except (FileNotFoundError,ProcessLookupError):return None
    if stat[0]=='Z':return None
    return {'pid':pid,'argv':argv,'start':stat[19]}
def inventory():
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        current=snapshot(int(p.name))
        if not current:continue
        a=current['argv']
        legacy=len(a)==2 and a[1]=='/workspace/dex-remote-guard.py'
        modern=len(a)==3 and a[1].startswith('/workspace/.doorbench-guards/') and a[1].endswith('/guard.py') and a[2]==str(Path(a[1]).with_name('config.json'))
        if not legacy and not modern:continue
        cfg=Path('/workspace/.dex-deadline.json') if legacy else Path(a[2]);c=json.loads(cfg.read_text())
        if c['id']!=r['id'] or c['deadline']!=r['old_deadline']:raise ValueError('Remote guard ownership/deadline differs')
        expected=r['legacy_hash'] if legacy else r['guard_hash']
        if digest(a[1])!=expected:raise ValueError('Remote guard source differs')
        current.update(script_sha256=expected,deadline=c['deadline']);rows.append(current)
    return sorted(rows,key=lambda x:x['pid'])
if r['action']=='inventory':print(json.dumps(inventory()))
elif r['action']=='arm':
    base=Path('/workspace/.doorbench-guards')/r['generation'];base.mkdir(parents=True,mode=0o700,exist_ok=False);base.chmod(0o700)
    # Reuse the already private credential; never send it to stdout or argv.
    old=json.loads(Path('/workspace/.dex-deadline.json').read_text())
    if old['id']!=r['id']:raise ValueError('Remote credential belongs to another allocation')
    cfg={'id':r['id'],'deadline':r['deadline'],'key':old['key']}
    script=base/'guard.py';config=base/'config.json';script.write_text(r['program']);script.chmod(0o600);config.write_text(json.dumps(cfg));config.chmod(0o600)
    with (base/'guard.log').open('ab') as log:
        process=subprocess.Popen(['python3',str(script),str(config)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    receipt=base/'status.json';end=time.time()+10
    while time.time()<end and not receipt.exists():time.sleep(.1)
    if not receipt.exists() or process.poll() is not None:raise RuntimeError('Replacement remote guard failed to arm')
    value=json.loads(receipt.read_text());current=snapshot(process.pid)
    if not current or value['id']!=r['id'] or value['deadline']!=r['deadline'] or value['phase']!='armed':raise ValueError('Replacement remote acknowledgement differs')
    current.update(script_sha256=digest(script),deadline=value['deadline'],receipt=str(receipt));print(json.dumps(current))
elif r['action']=='verify':
    value=json.loads(Path(r['expected']['receipt']).read_text());current=snapshot(r['expected']['pid'])
    if not current or any(current[k]!=r['expected'][k] for k in ('pid','argv','start')) or digest(current['argv'][1])!=r['expected']['script_sha256'] or value['id']!=r['id'] or value['deadline']!=r['deadline'] or value['phase']!='armed' or time.time()-value['heartbeat_unix']>30:raise ValueError('New remote guard is not alive')
    print(json.dumps({'verified':True}))
elif r['action']=='stop':
    for expected in r['guards']:
        current=snapshot(expected['pid'])
        if not current or any(current[k]!=expected[k] for k in ('pid','argv','start')) or digest(current['argv'][1])!=expected['script_sha256']:raise ValueError('Old remote guard identity changed; no signal sent')
    for expected in r['guards']:os.kill(expected['pid'],signal.SIGTERM)
    end=time.time()+5
    while time.time()<end and any(snapshot(x['pid']) for x in r['guards']):time.sleep(.1)
    if any(snapshot(x['pid']) for x in r['guards']):raise RuntimeError('An old remote guard did not stop')
    print(json.dumps({'stopped':[x['pid'] for x in r['guards']]}))
else:raise ValueError('Unknown remote operation')
'''


def sha(value):return hashlib.sha256(value).hexdigest()

def validate_owned(state,pod_id,now,hours,*,max_total_hours=8.):
    if not all(math.isfinite(float(value)) for value in (now,hours,state.get('created',float('nan')),state.get('deadline',float('nan')))):raise ValueError('Finite allocation times required')
    if state.get('id')!=pod_id or state.get('active') is not True:raise ValueError('Only the active journaled allocation can be renewed')
    if not 0<hours<=3:raise ValueError('A renewal is bounded to at most three hours')
    if not math.isfinite(max_total_hours) or not 8<=max_total_hours<=24:raise ValueError('Explicit total allocation ceiling must be 8..24 hours')
    deadline=now+hours*3600
    if deadline>state['created']+max_total_hours*3600:raise ValueError('Total allocation exceeds the declared ceiling')
    if state['deadline']-now<120:raise ValueError('Too close to the existing teardown to renew safely')
    if deadline<=state['deadline']:raise ValueError('Renewal must extend the existing deadline')
    return deadline

def same_process(current,expected):
    return current is not None and all(current.get(k)==expected.get(k) for k in ('pid','argv','start','script_sha256'))

def local_snapshot(pid):
    def read(field):
        r=subprocess.run(['ps','-p',str(pid),'-o',field+'='],capture_output=True,text=True)
        return r.stdout.strip() if r.returncode==0 else ''
    command=read('command');start=read('lstart');state=read('stat')
    if not command or not start or state.startswith('Z'):return None
    argv=shlex.split(command)
    return dict(pid=pid,argv=argv,start=start,script_sha256=sha(Path(argv[1]).read_bytes()) if len(argv)>1 and Path(argv[1]).is_file() else None)

def local_inventory(state):
    rows=[];legacy_hash=sha(Path(pod.__file__).read_bytes())
    listing=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    for row in listing.splitlines():
        parts=row.strip().split(None,1)
        if len(parts)!=2:continue
        try:argv=shlex.split(parts[1])
        except ValueError:continue
        legacy=len(argv)==3 and argv[1].endswith('/scripts/dexterous/pod.py') and argv[2]=='guard'
        modern=len(argv)==3 and argv[1].startswith(str(GUARD_ROOT)+'/') and argv[1].endswith('/guard.py') and argv[2]==str(Path(argv[1]).with_name('config.json'))
        if not legacy and not modern:continue
        current=local_snapshot(int(parts[0]));expected=legacy_hash if legacy else sha(GUARD.encode())
        if not current or current['script_sha256']!=expected:raise ValueError('Existing local guard source changed')
        if modern:
            cfg=json.loads(Path(argv[2]).read_text())
            if cfg['id']!=state['id'] or cfg['deadline']!=state['deadline']:raise ValueError('Local guard ownership/deadline differs')
        current['deadline']=state['deadline'];rows.append(current)
    if state.get('guard_pid') not in [r['pid'] for r in rows]:raise ValueError('Journaled local guard identity was not verified')
    return sorted(rows,key=lambda x:x['pid'])

def remote(ssh,payload):
    result=subprocess.run(ssh+[shlex.join(['python3','-c',REMOTE_OPS])],input=json.dumps(payload),capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Remote guard operation failed: '+result.stderr[-1000:])
    return json.loads(result.stdout)

def save(path,value,mode=0o600):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as stream:os.chmod(tmp,mode);json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

def dry_run(pod_id,hours,output,*,max_total_hours=8.):
    state=json.loads(STATE.read_text());now=time.time();deadline=validate_owned(state,pod_id,now,hours,max_total_hours=max_total_hours);api=pod.api();record=api._req('GET','/pods/'+pod_id)
    if record.get('id')!=pod_id or record.get('name')!='doorbench-dexterous-humanoid':raise ValueError('API allocation identity differs')
    host='root@'+record['publicIp'];port=int(record['portMappings']['22']);key=str(api.KEY_FILE)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-i',key,'-p',str(port),host]
    old_local=local_inventory(state)
    old_remote=remote(ssh,dict(action='inventory',id=pod_id,old_deadline=state['deadline'],legacy_hash=sha(LEGACY_REMOTE.encode()),guard_hash=sha(GUARD.encode())))
    if not old_remote:raise ValueError('No existing remote guard was verified')
    plan=dict(schema='doorbench.guard-renewal.v1',dry_run=True,pod_id=pod_id,planned_at_unix=now,old_deadline=state['deadline'],new_deadline=deadline,hours=hours,max_total_hours=max_total_hours,state_sha256=sha(STATE.read_bytes()),ssh=ssh,old_local=old_local,old_remote=old_remote,generation=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime(now))+'-'+uuid.uuid4().hex[:8],guard_source_sha256=sha(GUARD.encode()),steps=['Arm fresh local guard','Arm fresh remote guard','Verify both live acknowledgements','Atomically update only owned journal','Verify and signal only inventoried old guard PIDs','Verify both replacements remain alive'])
    save(output,plan);return plan

def check_new_local(record,pod_id,deadline):
    receipt=json.loads(Path(record['receipt']).read_text())
    if not same_process(local_snapshot(record['pid']),record) or receipt['id']!=pod_id or receipt['deadline']!=deadline or receipt['phase']!='armed' or time.time()-receipt['heartbeat_unix']>30:raise ValueError('Replacement local guard is not alive')

def apply(plan_path,output):
    plan=json.loads(Path(plan_path).read_text());state=json.loads(STATE.read_text())
    if plan.get('schema')!='doorbench.guard-renewal.v1' or not plan.get('dry_run'):raise ValueError('A dry-run plan is required')
    if sha(STATE.read_bytes())!=plan['state_sha256'] or time.time()-plan['planned_at_unix']>300:raise ValueError('Plan is stale; run a new dry run')
    validate_owned(state,plan['pod_id'],plan['planned_at_unix'],plan['hours'],max_total_hours=plan.get('max_total_hours',8.))
    expected_deadline=validate_owned(state,plan['pod_id'],plan['planned_at_unix'],plan['hours'],max_total_hours=plan.get('max_total_hours',8.))
    if plan['new_deadline']!=expected_deadline or state['deadline']-time.time()<120:raise ValueError('Planned deadline changed or old guard is too close')
    record=pod.api()._req('GET','/pods/'+state['id'])
    expected_ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-i',str(pod.api().KEY_FILE),'-p',str(int(record['portMappings']['22'])),'root@'+record['publicIp']]
    if record.get('id')!=state['id'] or record.get('name')!='doorbench-dexterous-humanoid' or plan['ssh']!=expected_ssh:raise ValueError('Owned API endpoint changed')
    if plan['guard_source_sha256']!=sha(GUARD.encode()):raise ValueError('Guard source changed after dry run')
    if local_inventory(state)!=plan['old_local']:raise ValueError('Local guards changed after dry run')
    actual=remote(plan['ssh'],dict(action='inventory',id=state['id'],old_deadline=state['deadline'],legacy_hash=sha(LEGACY_REMOTE.encode()),guard_hash=sha(GUARD.encode())))
    if actual!=plan['old_remote']:raise ValueError('Remote guards changed after dry run')
    base=GUARD_ROOT/plan['generation'];base.mkdir(parents=True,mode=0o700,exist_ok=False);base.chmod(0o700)
    script=base/'guard.py';script.write_text(GUARD);script.chmod(0o600);config=base/'config.json'
    save(config,dict(id=state['id'],deadline=plan['new_deadline'],key=pod.api()._key(),journal=str(STATE)))
    with (base/'guard.log').open('ab') as log:process=subprocess.Popen([sys.executable,str(script),str(config)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    receipt=base/'status.json';end=time.time()+10
    while time.time()<end and not receipt.exists():time.sleep(.1)
    if not receipt.exists() or process.poll() is not None:raise ValueError('Replacement local guard did not acknowledge')
    new_local=local_snapshot(process.pid);new_local.update(receipt=str(receipt),deadline=plan['new_deadline'])
    new_remote=remote(plan['ssh'],dict(action='arm',id=state['id'],deadline=plan['new_deadline'],generation=plan['generation'],program=GUARD))
    # No journal update or old-process signal before these two checks succeed.
    check_new_local(new_local,state['id'],plan['new_deadline'])
    remote(plan['ssh'],dict(action='verify',id=state['id'],deadline=plan['new_deadline'],expected=new_remote))
    if sha(STATE.read_bytes())!=plan['state_sha256']:raise ValueError('Owned journal changed during renewal; old guards retained')
    updated=dict(state,deadline=plan['new_deadline'],guard_pid=new_local['pid'],remote_guard_pid=new_remote['pid'],guard_generation=plan['generation'],last_guard_renewal_unix=time.time())
    updated['guard_renewals']=state.get('guard_renewals',[])+[dict(previous_deadline=state['deadline'],deadline=plan['new_deadline'],max_total_hours=plan.get('max_total_hours',8.),local_pid=new_local['pid'],remote_pid=new_remote['pid'])]
    save(STATE,updated)
    for expected in plan['old_local']:
        if not same_process(local_snapshot(expected['pid']),expected):raise ValueError('Old local guard identity changed; no signal sent')
    stopped_remote=remote(plan['ssh'],dict(action='stop',guards=plan['old_remote']))
    for expected in plan['old_local']:
        if not same_process(local_snapshot(expected['pid']),expected):raise ValueError('Local guard identity changed immediately before signal')
        os.kill(expected['pid'],signal.SIGTERM)
    end=time.time()+5
    while time.time()<end and any(local_snapshot(x['pid']) for x in plan['old_local']):time.sleep(.1)
    if any(local_snapshot(x['pid']) for x in plan['old_local']):raise ValueError('An old local guard did not stop')
    check_new_local(new_local,state['id'],plan['new_deadline'])
    remote(plan['ssh'],dict(action='verify',id=state['id'],deadline=plan['new_deadline'],expected=new_remote))
    result=dict(renewed=True,pod_id=state['id'],previous_deadline=state['deadline'],deadline=plan['new_deadline'],max_total_hours=plan.get('max_total_hours',8.),verified_at_unix=time.time(),local_guard=new_local,remote_guard=new_remote,stopped_local=[r['pid'] for r in plan['old_local']],stopped_remote=stopped_remote['stopped'],scope='Only journaled allocation deadline and exact verified guard processes changed; no GPU process signals')
    save(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pod-id');p.add_argument('--hours',type=float,default=3.);p.add_argument('--max-total-hours',type=float,default=8.,help='Explicit finite total allocation ceiling for sustained work; default8, maximum24. Each renewal remains at most3hours.');p.add_argument('--output',type=Path,required=True);p.add_argument('--apply-plan',type=Path);a=p.parse_args()
    with STATE.with_suffix('.renewal.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.apply_plan:result=apply(a.apply_plan,a.output)
        else:
            if not a.pod_id:p.error('--pod-id is required for dry run')
            result=dry_run(a.pod_id,a.hours,a.output,max_total_hours=a.max_total_hours)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
