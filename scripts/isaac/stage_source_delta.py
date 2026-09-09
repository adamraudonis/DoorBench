#!/usr/bin/env python3
"""Stage a new immutable source tree from a verified base and a streamed delta."""
import argparse,hashlib,json,os,shutil,tarfile
from pathlib import Path,PurePosixPath


def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or str(p)!=name:raise ValueError('Unsafe source member')
    return p


def stage(base,destination,stream):
    base=Path(base);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    received=[]
    with tarfile.open(fileobj=stream,mode='r|gz') as archive:
        for member in archive:
            relative=safe_name(member.name)
            if not member.isfile() or member.name in received:raise ValueError('Only unique regular source files are accepted')
            path=destination/relative;path.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,path.open('xb') as target:shutil.copyfileobj(source,target)
            path.chmod(member.mode&0o777);os.utime(path,(member.mtime,member.mtime));received.append(member.name)
    manifest=json.loads((destination/'source-manifest.json').read_text());files=manifest['files']
    identity=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    if identity!=manifest['sha256']:raise ValueError('Source identity differs from manifest')
    if set(received)-set(files)-{'source-manifest.json'}:raise ValueError('Delta contains unlisted source files')
    copied=0
    for name,digest in files.items():
        relative=safe_name(name);target=destination/relative
        if not target.exists():
            source=base/relative
            if source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError('Base source changed or required delta is missing: '+name)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);copied+=1
        if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Staged source differs: '+name)
    return dict(source=str(destination),source_sha256=identity,copied_files=copied,transferred_files=len(received),all_source_hashes_verified=True)


if __name__=='__main__':
    import sys
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--destination',required=True);args=parser.parse_args()
    print(json.dumps(stage(args.base,args.destination,sys.stdin.buffer)))
