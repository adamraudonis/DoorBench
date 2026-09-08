#!/usr/bin/env python3
"""Backup deadline guard. Config arrives privately over SSH, never via argv/logs."""
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

def main():
    path=Path(sys.argv[1]);config=json.loads(path.read_text())
    if path.stat().st_mode & 0o077:
        raise SystemExit('Guard configuration must be private')
    while time.time()<config['deadline']:
        time.sleep(min(30,max(.1,config['deadline']-time.time())))
    request=urllib.request.Request('https://rest.runpod.io/v1/pods/'+config['id'],method='DELETE',
        headers={'Authorization':'Bearer '+config['key'],'User-Agent':'DoorBench/deadline-guard'})
    for _ in range(20):
        try:
            with urllib.request.urlopen(request,timeout=30) as response:
                response.read()
            path.unlink(missing_ok=True)
            return
        except urllib.error.HTTPError as e:
            if e.code==404:return
        except Exception:
            pass
        print('Teardown attempt did not complete; retrying',flush=True);time.sleep(30)
    raise SystemExit('Backup teardown exhausted retries')

if __name__=='__main__':main()
