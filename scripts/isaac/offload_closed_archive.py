#!/usr/bin/env python3
"""Copy a closed archive to a verified persistent RunPod mount before local eviction.

Requires a specific volume identity. Never deletes the volume or unique bytes;
optional local eviction happens only after independent remote and local hashes
match, and leaves a retrieval receipt beside the former local archive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


REMOTE = r'''
import hashlib,json
from pathlib import Path
p=Path(INPUT['path']);volume=INPUT['volume']
mounts=[line.split() for line in Path('/proc/mounts').read_text().splitlines()]
matches=[row for row in mounts if row[1]=='/workspace' and row[0].endswith('/networkvolumes/'+volume)]
if len(matches)!=1:raise ValueError('Expected persistent network volume is not mounted')
if not p.is_relative_to('/workspace/archive') or '..' in p.parts:raise ValueError('Archive destination outside owned prefix')
if INPUT['mode']=='prepare':
 p.parent.mkdir(parents=True,exist_ok=True)
 print(json.dumps(dict(mount=matches[0][0],existing=p.exists())))
else:
 if p.is_symlink() or not p.is_file():raise ValueError('Expected regular remote archive')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 print(json.dumps(dict(mount=matches[0][0],bytes=p.stat().st_size,sha256=h.hexdigest())))
'''


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--key',type=Path,required=True);p.add_argument('--volume-id',required=True)
    p.add_argument('--region',required=True)
    p.add_argument('--receipt-directory',type=Path,help='Persistent local retrieval index, required when evicting')
    p.add_argument('--evict-local',action='store_true')
    a=p.parse_args();src=a.source.absolute()
    if a.evict_local and a.receipt_directory is None:p.error('Local eviction requires a persistent receipt directory')
    if src.is_symlink() or not src.is_file():p.error('Closed regular source file required')
    if a.host.startswith('-') or any(c.isspace() for c in a.host) or not 1<=a.port<=65535:p.error('Invalid SSH endpoint')
    receipt=src.with_name(src.name+'.remote.json')
    if receipt.exists():p.error('A retrieval receipt already exists; inspect it before retrying')
    stat=src.stat();expected=dict(bytes=stat.st_size,sha256=digest(src))
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-i',str(a.key),'-p',str(a.port),a.host]
    def remote(mode):
        code='INPUT='+repr(dict(mode=mode,path=a.destination,volume=a.volume_id))+'\n'+REMOTE
        result=subprocess.run(ssh+['python3','-'],input=code,text=True,capture_output=True,timeout=180,check=True)
        return json.loads(result.stdout)
    before=remote('prepare')
    if before['existing']:
        found=remote('verify')
        if any(found[k]!=v for k,v in expected.items()):raise ValueError('Refuse to overwrite different remote evidence')
    else:
        subprocess.run(['rsync','-a','--no-owner','--no-group','--timeout=60','-e',shlex.join(ssh[:-1]),
                        str(src),a.host+':'+shlex.quote(a.destination)],check=True)
    actual=remote('verify')
    if any(actual[k]!=v for k,v in expected.items()):raise ValueError('Remote archive hash/size mismatch')
    if src.stat().st_mtime_ns!=stat.st_mtime_ns or src.stat().st_size!=stat.st_size or digest(src)!=expected['sha256']:
        raise ValueError('Source changed while copying; local evidence retained')
    value=dict(schema='doorbench.persistent-archive.v1',source=str(src),destination=a.destination,
               volume_id=a.volume_id,region=a.region,verified_unix=time.time(),**expected,
               remote_mount=actual['mount'],local_evicted=False,
               retrieval='Attach this network volume to a new pod; copy destination to source and verify sha256 before using it.')
    receipt.write_text(json.dumps(value,indent=2)+'\n')
    central=None
    if a.receipt_directory:
        a.receipt_directory.mkdir(parents=True,exist_ok=True)
        central=a.receipt_directory/(hashlib.sha256(str(src).encode()).hexdigest()+'.json')
        if central.exists():raise ValueError('Persistent retrieval receipt already exists; local archive retained')
        central.write_text(json.dumps(value,indent=2)+'\n')
    if a.evict_local:
        src.unlink();value['local_evicted']=True
        temp=receipt.with_name(receipt.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(receipt)
        central.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
