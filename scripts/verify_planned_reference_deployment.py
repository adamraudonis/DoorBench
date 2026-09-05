#!/usr/bin/env python3
"""Verify an immutable MotionLab index before a GitHub Pages build.

deploy/planned-references.json is populated from the actual publication receipt:
schema_version=1, release, web_index_url, web_index_sha256, native_manifest_sha256.
The URL must use a full Hub commit and the matching experimental release path.
No credentials, mutable tags, source fallback, or fabricated release URL are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_INDEX_BYTES=4*1024*1024
HASH=re.compile(r'[a-f0-9]{64}')
RELEASE=re.compile(r'planned-[A-Za-z0-9][A-Za-z0-9._-]*')
URL=re.compile(r'https://huggingface\.co/datasets/[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*/resolve/(?P<commit>[a-f0-9]{40})/experimental/planned-reference/(?P<release>planned-[A-Za-z0-9][A-Za-z0-9._-]*)/web/index\.json')


def require(ok,message):
    if not ok:raise ValueError(message)


def json_value(raw):
    def pairs(items):
        obj={}
        for key,value in items:
            require(key not in obj,f'Duplicate JSON field: {key}');obj[key]=value
        return obj
    def constant(_):raise ValueError('JSON must contain finite numbers')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def validate_config(config):
    require(isinstance(config,dict) and set(config)=={'schema_version','release','web_index_url','web_index_sha256','native_manifest_sha256'},'Deployment config fields must be schema_version, release, web_index_url, web_index_sha256, native_manifest_sha256')
    require(type(config['schema_version']) is int and config['schema_version']==1,'Unsupported planned deployment schema version')
    require(isinstance(config['release'],str) and RELEASE.fullmatch(config['release']),'Invalid experimental release version')
    match=URL.fullmatch(config['web_index_url']) if isinstance(config['web_index_url'],str) else None
    require(match is not None,'MotionLab URL must be an HTTPS Hugging Face dataset resolve URL pinned to a full 40-character commit')
    require(match['release']==config['release'],'Deployment release version differs from the URL path')
    for name in ['web_index_sha256','native_manifest_sha256']:
        require(isinstance(config[name],str) and HASH.fullmatch(config[name]),f'Invalid {name}')
    return dict(config)


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        target=urlsplit(newurl)
        require(target.scheme=='https' and target.username is None and target.password is None,'Motion index redirect must remain HTTPS without credentials')
        return super().redirect_request(request,fp,code,msg,headers,newurl)


def download_index(url):
    request=Request(url,headers={'User-Agent':'DoorBench-verified-Pages-build/1'})
    with build_opener(HTTPSRedirects()).open(request,timeout=30) as response:
        declared=response.headers.get('Content-Length')
        require(declared is None or int(declared)<=MAX_INDEX_BYTES,'Motion index exceeds 4 MB deployment limit')
        raw=response.read(MAX_INDEX_BYTES+1)
    require(len(raw)<=MAX_INDEX_BYTES,'Motion index exceeds 4 MB deployment limit')
    return raw


def descriptor(value):
    require(isinstance(value,dict),'Invalid motion artifact descriptor')
    path=value.get('path')
    require(isinstance(path,str) and re.fullmatch(r'(clips|audits)/[A-Za-z0-9_./-]+',path) and
            PurePosixPath(path).as_posix()==path and '..' not in PurePosixPath(path).parts,'Unsafe motion artifact path')
    require(isinstance(value.get('sha256'),str) and HASH.fullmatch(value['sha256']) and type(value.get('bytes')) is int and value['bytes']>0,'Invalid motion artifact checksum/size')


def validate_index(raw,config):
    require(isinstance(raw,bytes) and len(raw)<=MAX_INDEX_BYTES,'Motion index exceeds 4 MB deployment limit')
    require(hashlib.sha256(raw).hexdigest()==config['web_index_sha256'],'Motion index SHA-256 differs from the deployment receipt')
    index=json_value(raw)
    require(isinstance(index,dict) and index.get('schema')=='doorbench.planned-reference-web-index.v1','Unsupported MotionLab index schema version')
    require(index.get('manifest_sha256')==config['native_manifest_sha256'],'Motion index uses a different native dataset manifest')
    rows=index.get('doors');require(isinstance(rows,list) and len(rows)==1000,'Motion index must contain all 1000 door statuses')
    seen=set()
    for row in rows:
        require(isinstance(row,dict),'Invalid motion status row')
        door=row.get('door_id');require(isinstance(door,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',door) and door not in seen,'Unsafe or duplicate motion door ID');seen.add(door)
        require(row.get('status') in ['accepted_kinematic','rejected','unresolved'],'Unsupported motion status')
        require(isinstance(row.get('family'),str),'Missing motion family')
        clip=row.get('clip')
        require((isinstance(clip,dict) if row['status']=='accepted_kinematic' else clip is None),'Only accepted motions may be playable')
        audits=row.get('audits');require(isinstance(audits,dict),'Missing motion audit inventory')
        for item in audits.values():descriptor(item)
        if clip is not None:
            descriptor(clip)
            require(row.get('source_scenario') in ['open_and_traverse','unlock_and_traverse','locked_recognize'],'Accepted clip lacks a supported bound source scenario')
            require(clip['path'].startswith('clips/') and clip['path'].endswith('.json.gz'),'Invalid compressed motion path')
            require(isinstance(row.get('identity_sha256'),str) and HASH.fullmatch(row['identity_sha256']) and isinstance(clip.get('json_sha256'),str) and HASH.fullmatch(clip['json_sha256']),'Missing accepted clip identity/checksum')
            require(type(clip.get('frames')) is int and 2<=clip['frames']<=100000 and clip['bytes']<=64*1024*1024,'Accepted clip exceeds browser frame or byte limits')
            require(type(clip.get('duration')) in (int,float) and math.isfinite(clip['duration']) and clip['duration']>0,'Invalid motion duration')
    require(index.get('counts')==dict(Counter(row['status'] for row in rows)),'Motion index status counts disagree')
    return index


def verify_deployment(config_path,site_assets_path,*,index_bytes=None,assets_manifest=None):
    config=validate_config(json_value(Path(config_path).read_bytes()))
    site=json_value(Path(site_assets_path).read_bytes())
    require(site.get('schema_version')==1 and site.get('dataset_manifest_sha256')==config['native_manifest_sha256'],'Planned motion native manifest differs from the pinned site asset release')
    if assets_manifest is not None:
        require(hashlib.sha256(Path(assets_manifest).read_bytes()).hexdigest()==config['native_manifest_sha256'],'Restored site assets differ from the planned motion native manifest')
    raw=download_index(config['web_index_url']) if index_bytes is None else index_bytes
    index=validate_index(raw,config)
    return config,index,raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='deploy/planned-references.json')
    parser.add_argument('--site-assets',default='deploy/site-assets.json')
    parser.add_argument('--index-file',help='Reuse previously downloaded exact bytes instead of making another network request')
    parser.add_argument('--index-out',help='Write verified exact index bytes for the post-restore check')
    parser.add_argument('--assets-manifest',help='Also check the actual restored assets/manifest.json')
    parser.add_argument('--github-env',help='Append VITE_PLANNED_REFERENCE_INDEX only after every check passes')
    args=parser.parse_args()
    raw=Path(args.index_file).read_bytes() if args.index_file else None
    config,index,raw=verify_deployment(args.config,args.site_assets,index_bytes=raw,assets_manifest=args.assets_manifest)
    if args.index_out:
        target=Path(args.index_out);target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as stream:stream.write(raw);temporary=stream.name
        os.replace(temporary,target)
    if args.github_env:
        with Path(args.github_env).open('a') as stream:stream.write('VITE_PLANNED_REFERENCE_INDEX='+config['web_index_url']+'\n')
    print(json.dumps({'verified':True,'release':config['release'],'web_index_sha256':config['web_index_sha256'],
                      'native_manifest_sha256':config['native_manifest_sha256'],'doors':len(index['doors']),'counts':index['counts']}))


if __name__=='__main__':main()
