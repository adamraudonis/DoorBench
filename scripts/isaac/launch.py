#!/usr/bin/env python3
"""One-command Isaac preparation, owned allocation, live status and reusable receipt.

Uses only Python's standard library locally. Secrets stay in environment/config.
A second invocation attaches to an active preparation instead of duplicating it.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
import webbrowser

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from doorbench.dexterous.isaac_readiness import mechanics_profile,cache_identity,ready_directory,robot_filename
DEFAULT=ROOT/'configs/isaac/runtime.json'
LOCAL=ROOT/'out/isaac-launch'


def load_pod():
    sys.path.insert(0,str(ROOT/'scripts/dexterous'))
    import pod
    return pod


def run(args,**kw):
    return subprocess.run([str(x) for x in args],check=True,**kw)


def source_bundle(output):
    # Explicit source allowlist; never package assets, credentials or outputs.
    allow=('doorbench/','scripts/','isaaclab/','configs/')
    names=subprocess.check_output(['git','ls-files','--cached'],cwd=ROOT,text=True).splitlines()
    files=sorted(set(n for n in names if n.startswith(allow) or n in ('pyproject.toml','README.md','LICENSE')))
    files=[n for n in files if (ROOT/n).is_file() and not (ROOT/n).is_symlink() and '/__pycache__/' not in n and not n.endswith('.pyc')]
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    with tarfile.open(output,'w:gz') as archive:
        for n in files:archive.add(ROOT/n,arcname=n,recursive=False)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (output.parent/'source-manifest.json').write_text(json.dumps(dict(sha256=identity,files=hashes,source_git_revision=revision),indent=2)+'\n')
    return identity


def ssh_args(host,port,key):
    if not host or host.startswith('-') or any(c.isspace() for c in host):raise ValueError('Invalid SSH host')
    return ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=10','-i',str(key),'-p',str(port),host]


def dashboard(config,port,show=True):
    config.parent.mkdir(parents=True,exist_ok=True)
    if not config.exists():config.write_text('[]\n')
    registry_id=hashlib.sha256(str(config.resolve()).encode()).hexdigest()
    for candidate in range(port,port+50):
        with socket.socket() as sock:occupied=sock.connect_ex(('127.0.0.1',candidate))==0
        if not occupied:port=candidate;break
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{candidate}/api/runs',timeout=2) as r:
                if json.load(r).get('registry_id')==registry_id:port=candidate;break
        except Exception:pass
    else:raise RuntimeError('No free local dashboard port')
    if occupied:
        # Do not terminate or replace unrelated local services.
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/runs',timeout=2) as r:assert 'runs' in json.load(r)
        except Exception:raise RuntimeError(f'Port {port} belongs to another app; use --dashboard-port')
    else:
        with (LOCAL/'dashboard.log').open('ab') as log:
            subprocess.Popen([sys.executable,str(ROOT/'scripts/gpu_dashboard/server.py'),'--config',str(config),'--port',str(port)],stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
    if show:webbrowser.open(f'http://127.0.0.1:{port}/')
    return port


def prepare(a,cfg):
    profile=mechanics_profile(cfg.get('mechanics_profile','upstream-v1'))
    if a.standing_operation and (profile!='shadow-loopback-v2' or cfg.get('door')!='db0055_swing_single'):
        raise ValueError('Standing operation requires corrected Shadow hands and the explicitly qualified Door55 configuration')
    session=LOCAL/a.session;session.mkdir(parents=True,exist_ok=True)
    (session/'run.pid').write_text(str(os.getpid()))
    pipeline=dict(stage='Connecting to GPU',scope='Isaac environment preparation, not a policy score',completion_marker='ISAAC_ENVIRONMENT_READY')
    (session/'pipeline.json').write_text(json.dumps(pipeline))
    registry=LOCAL/'runs.json'
    def register(host=None,port=None,key=None,results=None):
        cmd=[sys.executable,ROOT/'scripts/gpu_dashboard/server.py','--config',registry,'--register-only',
             '--id','isaac-ready-'+a.session,'--name',cfg['name'],'--results',results or session]
        if host:cmd+=['--ssh-host',host,'--ssh-port',port,'--ssh-key',key]
        run(cmd)
    register()
    pod=None
    if a.host:
        host=a.host;port=a.port;key=Path(a.key).expanduser();deadline=None;pod_id=None
    else:
        pod=load_pod()
        state=json.loads(pod.STATE.read_text()) if pod.STATE.exists() else {}
        if not state.get('active'):
            run([sys.executable,ROOT/'scripts/dexterous/pod.py','create','--hours',a.hours or cfg['hours']])
            state=json.loads(pod.STATE.read_text())
        deadline=state['deadline'];pod_id=state['id']
        if deadline-time.time()<1800:raise RuntimeError('Owned pod has less than 30 minutes before teardown; finish/archive it before creating the next allocation')
        print('Waiting for owned pod '+pod_id,flush=True)
        end=min(deadline,time.time()+900)
        while time.time()<end:
            record=pod.api()._req('GET','/pods/'+pod_id)
            port=(record.get('portMappings') or {}).get('22');ip=record.get('publicIp')
            if ip and port:
                host='root@'+ip;key=pod.api().KEY_FILE
                if subprocess.run(ssh_args(host,port,key)+['true'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:break
            time.sleep(10)
        else:raise RuntimeError('GPU SSH did not become ready before the launch deadline')
        run([sys.executable,ROOT/'scripts/dexterous/arm_remote_guard.py'])
    ssh=ssh_args(host,port,key)
    work=a.work or cfg['work']
    if not work.startswith('/') or '\n' in work:raise ValueError('Remote work must be an absolute path')
    digest=source_bundle(session/'source.tar.gz')
    identity=cache_identity(digest,profile)
    remote=work+'/doorbench-ready/'+profile+'/'+identity[:16]
    remote_status=remote+'/out/launch'
    run(ssh+[shlex.join(['mkdir','-p',remote,remote_status])])
    # An unfinished process in this exact source checkout is attached, never restarted.
    active=subprocess.run(ssh+[f'test -f {shlex.quote(remote_status+"/run.pid")} && kill -0 "$(cat {shlex.quote(remote_status+"/run.pid")})" 2>/dev/null'],stdout=subprocess.DEVNULL).returncode==0
    if not active:
        with (session/'source.tar.gz').open('rb') as source:
            run(ssh+[shlex.join(['tar','--no-same-owner','-xzf','-','-C',remote])],stdin=source)
        pipeline.update(stage='Preparing pinned runtime and checking live physics',deadline_unix=deadline,
                        pod_id=pod_id,source_sha256=digest,cache_identity=identity,mechanics_profile=profile,started_at_unix=time.time(),hourly_cost_usd=record.get('costPerHr') if pod else None)
        run(ssh+[shlex.join(['tee',remote+'/source-manifest.json'])],input=(session/'source-manifest.json').read_text(),text=True,stdout=subprocess.DEVNULL)
        run(ssh+[shlex.join(['tee',remote_status+'/pipeline.json'])],input=json.dumps(pipeline),text=True,stdout=subprocess.DEVNULL)
        script=f'''cd {shlex.quote(remote)}
export DOORBENCH_WORK={shlex.quote(work)}
export DOORBENCH_MECHANICS_PROFILE={shlex.quote(profile)}
export DOORBENCH_GENERATE_IDS={shlex.quote(cfg['door'])}
nohup bash scripts/isaac/prepare.sh > {shlex.quote(remote_status+'/run.log')} 2>&1 < /dev/null &
echo $! > {shlex.quote(remote_status+'/run.pid')}
'''
        run(ssh+['bash -s'],input=script,text=True)
    # A process independent of this launcher/assistant preserves partial output
    # and verifies final bytes. Existing-cluster mode only bounds collection;
    # it does not grant permission to tear down that cluster.
    collector_deadline=(deadline-60 if deadline is not None else
                        time.time()+3600*(a.hours or cfg['hours']))
    collector_argv=[sys.executable,str(ROOT/'scripts/isaac/collect_run.py'),
        '--host',host,'--port',str(port),'--key',str(key),
        '--remote',remote+'/out','--destination',str(session/'remote-evidence'),
        '--deadline',str(collector_deadline),'--pid-file','launch/run.pid',
        '--terminal','isaac-ready/'+profile+'/ready.json','--detach']
    collection=subprocess.run(collector_argv,text=True,capture_output=True,check=True)
    collector=json.loads(collection.stdout)
    register(host,port,key,remote_status)
    receipt=dict(host=host,port=port,key=str(key),source_sha256=digest,remote=remote,status=remote_status,pod_id=pod_id,deadline_unix=deadline,
                 connect_command=shlex.join(ssh),mechanics_profile=profile,cache_identity=identity,ready_receipt=str(ready_directory(remote,profile)/'ready.json'),native_robot=str(ready_directory(remote,profile)/robot_filename(profile)))
    receipt['evidence_collector']=collector
    receipt['evidence_directory']=str(session/'remote-evidence')
    (session/'connection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if a.standing_operation:
        receipt['standing_operation']=dispatch_standing_operation(receipt,session,registry,a.session,work,collector_deadline)
        (session/'connection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Preparation running. Run Center shows all stages and errors.',flush=True)
    print('Connection details: '+str(session/'connection.json'),flush=True)


def dispatch_standing_operation(receipt,session,registry,session_name,work,deadline):
    """Attach one bounded physical test to this exact frozen preparation."""
    if receipt['mechanics_profile']!='shadow-loopback-v2':raise ValueError('Corrected hand mechanics required')
    if deadline-time.time()<2400:raise ValueError('At least 40 minutes required for the standing operation pipeline')
    remote=receipt['remote'];output=remote+'/out/standing-operation'
    ssh=ssh_args(receipt['host'],receipt['port'],receipt['key'])
    argv=['python3',remote+'/scripts/isaac/run_standing_operation.py','--source',remote,
        '--ready',receipt['ready_receipt'],'--reference',remote+'/configs/dexterous/door55-standing-acquisition-v1.json',
        '--output',output,'--work',work,'--deadline-unix',str(deadline-60)]
    # A complete/failed/in-progress output is retained, never silently replaced.
    # Background within a script so SSH closes without retaining its stdout pipe.
    script=f'''if test ! -e {shlex.quote(output)}; then
nohup {shlex.join(argv)} > {shlex.quote(remote+'/out/standing-operation-coordinator.log')} 2>&1 < /dev/null &
fi
'''
    run(ssh+['bash -s'],input=script,text=True)
    collector_argv=[sys.executable,str(ROOT/'scripts/isaac/collect_run.py'),'--host',receipt['host'],
        '--port',str(receipt['port']),'--key',receipt['key'],'--remote',output,
        '--destination',str(session/'standing-operation-evidence'),'--deadline',str(deadline),
        '--pid-file','run.pid','--terminal','coordinator-result.json','--detach']
    collector=json.loads(subprocess.run(collector_argv,text=True,capture_output=True,check=True).stdout)
    run([sys.executable,ROOT/'scripts/gpu_dashboard/server.py','--config',registry,'--register-only',
        '--id','standing-operation-'+session_name,'--name','Door55: native prerequisite and actual Isaac standing operation',
        '--results',output,'--ssh-host',receipt['host'],'--ssh-port',receipt['port'],'--ssh-key',receipt['key']])
    return dict(remote=output,deadline_unix=deadline-60,evidence_collector=collector,
        local_evidence=str(session/'standing-operation-evidence'),scope='Privileged partial opening qualification; not traversal or a learned policy')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT)
    p.add_argument('--host',help='Use an existing SSH GPU node; no RunPod API calls or teardown')
    p.add_argument('--port',type=int,default=22);p.add_argument('--key',default='~/.ssh/runpod_doorbench')
    p.add_argument('--work');p.add_argument('--hours',type=float);p.add_argument('--dashboard-port',type=int)
    p.add_argument('--no-browser',action='store_true');p.add_argument('--foreground',action='store_true')
    p.add_argument('--standing-operation',action='store_true',help='After preparation, rescreen the versioned Door55 seed, qualify native contact, and test actual Isaac operation')
    p.add_argument('--session',default=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
    a=p.parse_args();cfg=json.loads(a.config.read_text());mechanics_profile(cfg.get('mechanics_profile','upstream-v1'));LOCAL.mkdir(parents=True,exist_ok=True)
    a.dashboard_port=dashboard(LOCAL/'runs.json',a.dashboard_port or cfg['dashboard_port'],not a.no_browser)
    if a.foreground:
        lock=(LOCAL/'launch.lock').open('w')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another launch is already preparing this environment. See Run Center.');return
        try:prepare(a,cfg)
        except BaseException as exc:
            print('PREPARATION_FAILED: '+str(exc),flush=True)
            raise
    else:
        session=LOCAL/a.session;session.mkdir(parents=True,exist_ok=True)
        with (session/'run.log').open('ab') as log:
            subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--foreground','--no-browser','--session',a.session,'--dashboard-port',str(a.dashboard_port)],
                             stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        print(f'Run Center: http://127.0.0.1:{a.dashboard_port or cfg["dashboard_port"]}/')
        print('Launch log: '+str(session/'run.log'))

if __name__=='__main__':main()
