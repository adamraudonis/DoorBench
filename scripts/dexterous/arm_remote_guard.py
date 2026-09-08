#!/usr/bin/env python3
"""Arm an allocation-specific remote deadline without logging its API credential."""
import json
from pathlib import Path
import shlex
import subprocess
import pod

REMOTE = '''import json,time,urllib.request,urllib.error
from pathlib import Path
s=json.loads(Path('/workspace/.dex-deadline.json').read_text())
while time.time()<s['deadline']:time.sleep(min(30,max(.1,s['deadline']-time.time())))
for attempt in range(10):
    try:
        request=urllib.request.Request('https://rest.runpod.io/v1/pods/'+s['id'],method='DELETE',headers={'Authorization':'Bearer '+s['key'],'User-Agent':'DoorBench/1.0'})
        urllib.request.urlopen(request,timeout=30).close()
        break
    except urllib.error.HTTPError as exc:
        if exc.code==404:break
        time.sleep(30)
    except Exception:time.sleep(30)
'''


def main():
    state=json.loads(pod.STATE.read_text());api=pod.api()
    if not state.get('active'):raise SystemExit('No active owned allocation')
    record=api._req('GET','/pods/'+state['id'])
    ip=record.get('publicIp');port=(record.get('portMappings') or {}).get('22')
    if not ip or not port:raise SystemExit('SSH is not ready')
    state.update(ip=ip,port=port);pod.STATE.write_text(json.dumps(state,indent=2)+'\n')
    ssh=['ssh','-i',str(api.KEY_FILE),'-p',str(port),'-o','StrictHostKeyChecking=accept-new','root@'+ip]
    private={'id':state['id'],'deadline':state['deadline'],'key':api._key()}
    subprocess.run(ssh+['umask 077; cat > /workspace/.dex-deadline.json'],input=json.dumps(private),text=True,check=True)
    subprocess.run(ssh+['cat > /workspace/dex-remote-guard.py'],input=REMOTE,text=True,check=True)
    subprocess.run(ssh+['nohup python3 /workspace/dex-remote-guard.py > /workspace/dex-guard.log 2>&1 < /dev/null &'],check=True)
    print(json.dumps({'pod':state['id'],'remote_deadline_armed':True,'deadline':state['deadline']}))


if __name__=='__main__':main()
