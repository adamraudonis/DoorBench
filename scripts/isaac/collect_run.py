#!/usr/bin/env python3
"""Independently copy GPU evidence until exit; verify final bytes before teardown.

Run as a detached local process before launching the trial. A copied partial
snapshot is explicitly unverified; task success is never inferred by this tool.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import shutil

REMOTE_PROBE=r'''
import hashlib,json,os
from pathlib import Path
p=Path(INPUT['remote']);pid=None;alive=False
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
try:
 pid=int((p/INPUT['pid_file']).read_text().strip())
 stat=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
 alive=stat[0]!='Z'
except (OSError,ValueError):pass
result=dict(exists=p.is_dir(),pid=pid,alive=alive,terminal=(p/INPUT['terminal']).is_file(),transfer_bytes=0)
final=INPUT['manifest'] and result['terminal'] and pid and not alive
if final:result['files']={}
if result['exists']:
 for f in sorted(p.rglob('*')):
  if f.is_symlink():raise ValueError('Archive may not silently follow remote symlinks')
  if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith(('.writing','.tmp','.pending','.pyc')) and '.tmp.' not in f.name and '.writing.' not in f.name:
   size=f.stat().st_size;result['transfer_bytes']+=size
   if final:result['files'][str(f.relative_to(p))]=dict(bytes=size,sha256=digest(f))
print(json.dumps(result))
'''


def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.writing')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def final_transfer_options(manifest):
    # Same-size progress rewrites can share an mtime during a live run.
    # Final evidence must refresh bytes even when rsync's quick check agrees.
    return ['--checksum'] if manifest else []


def verify_local(root,manifest):
    if not manifest:raise ValueError('Empty final evidence manifest')
    for name,expected in manifest.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe evidence path')
        p=Path(root)/relative
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=expected['bytes']:
            raise ValueError('Missing or changed evidence: '+name)
        with p.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected['sha256']:
                raise ValueError('Evidence hash differs: '+name)
    return len(manifest)


def collect(a):
    minimum_free_mib=getattr(a,'minimum_free_mib',10240)
    if type(minimum_free_mib) is not int or not 0<=minimum_free_mib<=16384:raise ValueError('Explicit storage reserve must be 0..16384 MiB')
    if not a.host or a.host.startswith('-') or any(c.isspace() for c in a.host):raise ValueError('Invalid SSH host')
    if not Path(a.remote).is_absolute() or not 1<=a.port<=65535:raise ValueError('Absolute remote path and valid port required')
    if not time.time()<a.deadline or not 5<=a.interval<=60:raise ValueError('Future bounded deadline and 5..60s interval required')
    target=a.destination.resolve()
    receipt=target.with_name(target.name+'-collector.json');log=target.with_name(target.name+'-collector.log')
    target.parent.mkdir(parents=True,exist_ok=True)
    lock=target.with_name(target.name+'-collector.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.resume:
        old=json.loads(receipt.read_text())
        if old['remote']!=a.remote or old['destination']!=str(target) or old.get('final_bytes_verified'):
            raise ValueError('Resume requires the same unfinished evidence archive')
        target.mkdir(exist_ok=True)
    else:target.mkdir(parents=True,exist_ok=False)
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
        code='INPUT='+repr(dict(remote=a.remote,terminal=a.terminal,pid_file=a.pid_file,manifest=manifest))+'\n'+REMOTE_PROBE
        r=subprocess.run(ssh+['python3','-'],input=code,text=True,capture_output=True,timeout=90,check=True)
        return json.loads(r.stdout)
    save(status='waiting_for_run')
    while not stopped and time.time()<a.deadline:
        try:
            before=probe(manifest=True)
            free_bytes=shutil.disk_usage(target).free
            # Reserve space for a full atomic replacement, not just a free-space
            # threshold before an arbitrarily large copy. Conservative for deltas.
            required_bytes=minimum_free_mib*1024**2+before.get("transfer_bytes",0)
            enough_space=free_bytes>=required_bytes
            if before['exists'] and not enough_space:
                save(status='waiting_for_storage',free_bytes=free_bytes,minimum_free_bytes=minimum_free_mib*1024**2,required_bytes=required_bytes,
                     storage_waits=state.get('storage_waits',0)+1,last_error='Local evidence reserve unavailable; remote experiment is not stopped')
            if before['exists'] and enough_space:
                # Aggregate retention is separate from the host free-space guard.
                sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
                from doorbench.dexterous.storage_budget import check_retained_budget
                archive_root=next((p for p in target.parents if p.name=='DoorBench-runs'),target.parent)
                check_retained_budget([archive_root],incoming_bytes=before.get('transfer_bytes',0))
                with log.open('ab') as stream:
                    subprocess.run(['rsync','-az',*final_transfer_options(before.get('files')),'--timeout=30','--exclude=*.writing','--exclude=*.tmp',
                        '--exclude=*.tmp.*','--exclude=*.writing.*','--exclude=*.pending','--exclude=*.pyc','--exclude=__pycache__','-e',shlex.join(ssh[:-1]),
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
    p.add_argument('--pid-file',default='run.pid',help='Process receipt relative to the copied directory')
    p.add_argument('--minimum-free-mib',type=int,default=10240,help='Keep this free-space reserve PLUS room for the full incoming snapshot; does not stop the remote experiment')
    p.add_argument('--resume',action='store_true',help='Resume the same unfinished archive after its worker exits')
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
