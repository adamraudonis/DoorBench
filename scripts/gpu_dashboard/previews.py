"""Bounded read-only frame collector, executable remotely over SSH stdin."""
import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import struct
import subprocess
import sys
import threading
import time
import zlib

VIEWS={'wide':re.compile(r'frame-([0-9]{5})\.png\Z'),'hand':re.compile(r'hand-frame-([0-9]{5})\.png\Z')}
MAX_BYTES=4*1024*1024


def valid_png(data):
    """Reject partial/non-PNG/oversized output before giving it to a browser."""
    if not 57<=len(data)<=MAX_BYTES or data[:8]!=b'\x89PNG\r\n\x1a\n':return False
    offset=8;seen_header=False;seen_pixels=False
    while offset+12<=len(data):
        size=struct.unpack('>I',data[offset:offset+4])[0];kind=data[offset+4:offset+8];end=offset+12+size
        if end>len(data) or zlib.crc32(data[offset+4:end-4])&0xffffffff!=struct.unpack('>I',data[end-4:end])[0]:return False
        if not seen_header:
            if kind!=b'IHDR' or size!=13:return False
            width,height=struct.unpack('>II',data[offset+8:offset+16])
            if not 0<width<=4096 or not 0<height<=4096 or width*height>16777216:return False
            seen_header=True
        elif kind==b'IHDR':return False
        if kind==b'IDAT':seen_pixels=True
        if kind==b'IEND':return seen_pixels and size==0 and end==len(data)
        offset=end
    return False


def collect(directory):
    root=Path(directory).resolve(strict=True)
    if not root.is_dir():raise ValueError('Unavailable run directory')
    candidates={view:[] for view in VIEWS}
    with os.scandir(root) as entries:
        for i,entry in enumerate(entries):
            if i>=10000:raise ValueError('Run directory exceeds bounded preview scan')
            for view,pattern in VIEWS.items():
                if pattern.fullmatch(entry.name) and entry.is_file(follow_symlinks=False):candidates[view].append(entry.name)
    result={}
    for view,names in candidates.items():
        result[view]=None
        for name in sorted(names,reverse=True)[:3]:
            try:
                fd=os.open(root/name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
                with os.fdopen(fd,'rb') as stream:
                    before=os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_size>MAX_BYTES:continue
                    data=stream.read(MAX_BYTES+1);after=os.fstat(stream.fileno())
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or not valid_png(data):continue
                result[view]=dict(frame=name,modified_at=after.st_mtime,sha256=hashlib.sha256(data).hexdigest(),png_base64=base64.b64encode(data).decode())
                break
            except OSError:continue
    return result


def validate_result(result):
    if type(result) is not dict or set(result)!=set(VIEWS):raise ValueError('Malformed preview response')
    for view,item in result.items():
        if item is None:continue
        if type(item) is not dict or set(item)!={'frame','modified_at','sha256','png_base64'} or not VIEWS[view].fullmatch(item['frame']):raise ValueError('Unexpected preview file')
        if type(item['png_base64']) is not str or len(item['png_base64'])>4*((MAX_BYTES+2)//3):raise ValueError('Preview too large')
        data=base64.b64decode(item['png_base64'],validate=True)
        if not valid_png(data) or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Invalid preview image')
        if type(item['modified_at']) not in (int,float) or not 0<item['modified_at']<1e12:raise ValueError('Invalid capture timestamp')
    return result


def fetch(run):
    if not run.get('ssh_host'):return collect(run['results'])
    host=run['ssh_host']
    if type(host) is not str or host.startswith('-'):raise ValueError('Invalid SSH host')
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p',str(run.get('ssh_port',22))]
    if run.get('ssh_key'):command+=['-i',str(Path(run['ssh_key']).expanduser())]
    command += [host,shlex.join(['python3','-',run['results']])]
    response=subprocess.run(command,input=Path(__file__).read_text(),capture_output=True,text=True,timeout=12)
    if response.returncode:raise RuntimeError('Preview connection unavailable')
    return validate_result(json.loads(response.stdout))


class PreviewCache:
    """On-demand only; status monitor workers never wait on preview reads."""
    def __init__(self,*,fetcher=None,clock=time.time,ttl=10.,limit=4):
        self.fetcher=fetcher or fetch;self.clock=clock;self.ttl=ttl;self.limit=limit
        self.pool=ThreadPoolExecutor(max_workers=2);self.lock=threading.Lock();self.entries=OrderedDict()

    def get(self,run):
        # Identity includes the server-side registration, not just a reusable ID.
        key=json.dumps(run,sort_keys=True);now=self.clock()
        with self.lock:
            item=self.entries.get(key)
            if item is None:
                if len(self.entries)>=self.limit:
                    removable=next((k for k,v in self.entries.items() if v['future'] is None or v['future'].done()),None)
                    if removable is None:return dict(views={v:None for v in VIEWS},pending=True,error=None,checked_at=None)
                    del self.entries[removable]
                item=dict(views={v:None for v in VIEWS},future=None,checked_at=None,error=None,stale=False)
                self.entries[key]=item
            self.entries.move_to_end(key)
            if item['future'] is not None and item['future'].done():
                try:
                    fresh=validate_result(item['future'].result())
                    # Keep an earlier complete image when a new write is partial.
                    for view,value in fresh.items():
                        if value is not None:item['views'][view]=value
                    item['stale']=any(fresh[v] is None and item['views'][v] is not None for v in VIEWS)
                    item['error']=None
                except Exception:
                    item['stale']=True;item['error']='Preview unavailable; showing the last complete capture if available.'
                item['future']=None;item['checked_at']=now
            if item['future'] is None and (item['checked_at'] is None or now-item['checked_at']>=self.ttl):
                item['future']=self.pool.submit(self.fetcher,dict(run))
            return dict(views=dict(item['views']),pending=item['future'] is not None,error=item['error'],stale=item['stale'],checked_at=item['checked_at'])


def registered_run(config,query):
    parsed=__import__('urllib.parse',fromlist=['parse_qs']).parse_qs(query,keep_blank_values=True,strict_parsing=True)
    if set(parsed)!={'run'} or len(parsed['run'])!=1 or not 0<len(parsed['run'][0])<=128:raise ValueError('Expected one registered run')
    runs=json.loads(Path(config).read_text());matches=[r for r in runs if r.get('id')==parsed['run'][0]]
    if len(matches)!=1:raise ValueError('Run is not registered')
    return matches[0]


if __name__=='__main__':
    try:print(json.dumps(collect(sys.argv[1]),allow_nan=False))
    except Exception:sys.exit(1)
