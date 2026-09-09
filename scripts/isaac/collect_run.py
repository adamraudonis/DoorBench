#!/usr/bin/env python3
"""Independently copy GPU evidence until exit; verify final bytes before teardown.

Run as a detached local process before launching the trial. A copied partial
snapshot is explicitly unverified; task success is never inferred by this tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

REMOTE_PROBE=r'''
import hashlib,json,os
from pathlib import Path
p=Path(INPUT['remote']);pid=None;alive=False
try:
 pid=int((p/'run.pid').read_text().strip())
 stat=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
 alive=stat[0]!='Z'
except (OSError,ValueError):pass
result=dict(exists=p.is_dir(),pid=pid,alive=alive,terminal=(p/INPUT['terminal']).is_file())
if INPUT['manifest'] and result['terminal'] and pid and not alive:
 result['files']={}
 for f in sorted(p.rglob('*')):
  if f.is_symlink():raise ValueError('Archive may not silently follow remote symlinks')
  if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith(('.writing','.tmp','.pyc')):
   result['files'][str(f.relative_to(p))]=dict(bytes=f.stat().st_size,sha256=hashlib.file_digest(f.open('rb'),'sha256').hexdigest())
print(json.dumps(result))
'''


def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.writing')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def verify_local(root,manifest):
    if not manifest:raise ValueError('Empty final evidence manifest')
    for name,expected in manifest.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe evidence path')
        p=Path(root)/relative
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=expected['bytes']:
            raise ValueError('Missing or changed evidence: '+name)
        if hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=expected['sha256']:
            raise ValueError('Evidence hash differs: '+name)
    return len(manifest)


def collect(a):
    if not a.host or a.host.startswith('-') or any(c.isspace() for c in a.host):raise ValueError('Invalid SSH host')
    if not Path(a.remote).is_absolute() or not 1<=a.port<=65535:raise ValueError('Absolute remote path and valid port required')
    if not time.time()<a.deadline or not 5<=a.interval<=60:raise ValueError('Future bounded deadline and 5..60s interval required')
    target=a.destination.resolve();target.mkdir(parents=True,exist_ok=False)
    receipt=target.with_name(target.name+'-collector.json');log=target.with_name(target.name+'-collector.log')
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-i',str(a.key.expanduser()),'-p',str(a.port),a.host]
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    state=dict(schema='doorbench.evidence-collector.v1',pid=os.getpid(),remote=a.remote,
        destination=str(target),deadline_unix=a.deadline,started_unix=time.time(),
        final_bytes_verified=False,task_success_not_inferred=True,copies=0)
    def save(**changes):
        state.update(changes,heartbeat_unix=time.time());atomic_json(receipt,state)
    def probe(manifest=False):
        code='INPUT='+repr(dict(remote=a.remote,terminal=a.terminal,manifest=manifest))+'\n'+REMOTE_PROBE
        r=subprocess.run(ssh+['python3','-'],input=code,text=True,capture_output=True,timeout=90,check=True)
        return json.loads(r.stdout)
    save(status='waiting_for_run')
    while not stopped and time.time()<a.deadline:
        try:
            before=probe(manifest=True)
            if before['exists']:
                with log.open('ab') as stream:
                    subprocess.run(['rsync','-az','--timeout=30','--exclude=*.writing','--exclude=*.tmp',
                        '--exclude=*.pyc','--exclude=__pycache__','-e',shlex.join(ssh[:-1]),
                        a.host+':'+shlex.quote(a.remote.rstrip('/')+'/'),str(target)+'/'],
                        stdout=stream,stderr=stream,timeout=180,check=True)
                save(status='partial_snapshot',copies=state['copies']+1,last_copy_unix=time.time(),
                    remote_pid=before['pid'],remote_alive=before['alive'],last_error=None)
                if before.get('files'):
                    after=probe(manifest=True)
                    if before!=after:raise ValueError('Remote evidence changed during final transfer')
                    count=verify_local(target,after['files'])
                    atomic_json(target.with_name(target.name+'-remote-manifest.json'),after)
                    save(status='verified_final_archive',final_bytes_verified=True,verified_files=count)
                    return
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            save(status='retrying_partial_archive',last_error=type(error).__name__+': '+str(error))
        # Short sleeps keep SIGTERM responsive; the detached collector does not
        # hold an assistant tool invocation open.
        until=min(a.deadline,time.time()+a.interval)
        while not stopped and time.time()<until:time.sleep(min(1.,until-time.time()))
    save(status='incomplete_stopped' if stopped else 'incomplete_deadline',final_bytes_verified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--key',type=Path,required=True);p.add_argument('--remote',required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--terminal',default='balance-report.json');p.add_argument('--interval',type=float,default=30)
    p.add_argument('--detach',action='store_true');a=p.parse_args()
    if a.detach:
        argv=[arg for arg in sys.argv[1:] if arg!='--detach']
        log=a.destination.with_name(a.destination.name+'-collector-process.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('ab') as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*argv],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
        print(json.dumps(dict(collector_pid=child.pid,log=str(log))))
    else:collect(a)


if __name__=='__main__':main()
