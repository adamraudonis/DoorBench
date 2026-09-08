import base64,hashlib,importlib.util,json,struct,threading,time,zlib
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('previews',ROOT/'scripts/gpu_dashboard/previews.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def png():
 def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
 return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b'\0\xff\x80\0'))+chunk(b'IEND',b'')


def test_exact_frames_only_and_incomplete_newest_falls_back(tmp_path):
 image=png();assert m.valid_png(image)
 (tmp_path/'frame-00001.png').write_bytes(image);(tmp_path/'frame-00002.png').write_bytes(image[:-6]);(tmp_path/'hand-frame-00004.png').write_bytes(image)
 for name in ('secret.png','frame-000003.png','frame-99999.png.bak'):(tmp_path/name).write_bytes(image)
 result=m.collect(tmp_path);assert result['wide']['frame']=='frame-00001.png' and result['hand']['frame']=='hand-frame-00004.png'
 assert base64.b64decode(result['wide']['png_base64'])==image
 assert m.validate_result(result)==result


@pytest.mark.parametrize('kind',['symlink','directory','html','bad_crc','trailing','large'])
def test_special_or_invalid_files_never_escape_as_images(tmp_path,kind):
 p=tmp_path/'frame-00001.png';image=png()
 if kind=='symlink':
  other=tmp_path.parent/'outside.png';other.write_bytes(image);p.symlink_to(other)
 elif kind=='directory':p.mkdir()
 elif kind=='html':p.write_text('<script>private data</script>')
 elif kind=='bad_crc':p.write_bytes(image[:-1]+b'x')
 elif kind=='trailing':p.write_bytes(image+b'secret')
 else:p.write_bytes(image+b'\0'*m.MAX_BYTES)
 assert m.collect(tmp_path)=={'wide':None,'hand':None}


def test_remote_command_quotes_registered_root_and_hides_transport_details(monkeypatch):
 seen={}
 def run(args,**kw):seen.update(args=args,kw=kw);return SimpleNamespace(returncode=0,stdout='{"wide":null,"hand":null}')
 monkeypatch.setattr(m.subprocess,'run',run)
 path='/work/results; cat /secret';assert m.fetch(dict(results=path,ssh_host='root@example',ssh_key='/private/credential'))=={'wide':None,'hand':None}
 assert m.shlex.split(seen['args'][-1])==['python3','-',path]
 assert '/private/credential' not in seen['kw']['input'] and seen['kw']['timeout']==12


def test_only_registered_run_identity_is_accepted(tmp_path):
 config=tmp_path/'runs.json';run=dict(id='known',results='/private/results',ssh_key='/private/key');config.write_text(json.dumps([run]))
 assert m.registered_run(config,'run=known')==run
 for query in ('run=../secret','run=known&file=/secret','run=known&run=known','view=hand','run='):
  with pytest.raises(ValueError):m.registered_run(config,query)


def test_http_preview_boundary_retains_host_guard_and_never_returns_registry(tmp_path):
 import http.client,socket,subprocess,sys
 (tmp_path/'frame-00001.png').write_bytes(png())
 config=tmp_path/'runs.json';config.write_text(json.dumps([dict(id='known',results=str(tmp_path),ssh_key='/private/not-for-browser')]))
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/gpu_dashboard/server.py'),'--config',str(config),'--port',str(port)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 def request(path,host=None):
  conn=http.client.HTTPConnection('127.0.0.1',port,timeout=2)
  try:
   conn.request('GET',path,headers={'Host':host or f'127.0.0.1:{port}'});r=conn.getresponse();return r.status,dict(r.getheaders()),r.read()
  finally:conn.close()
 try:
  for _ in range(100):
   try:status,headers,body=request('/api/previews?run=known');break
   except ConnectionRefusedError:time.sleep(.02)
  else:raise AssertionError('Dashboard failed to start')
  assert status==200 and headers['Cache-Control']=='no-store' and headers['X-Content-Type-Options']=='nosniff'
  assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
  for path in ('/api/previews?run=unknown','/api/previews?run=known&file=/secret','/api/previews?run=known&run=known','/api/previews?path=/secret','/frame-00001.png'):
   assert request(path)[0]==404
  assert request('/api/previews?run=known','untrusted.example')[0]==403
  for _ in range(100):
   status,_,body=request('/api/previews?run=known');result=json.loads(body)
   if not result['pending']:break
   time.sleep(.01)
  assert result['views']['wide']['frame']=='frame-00001.png'
  assert str(tmp_path).encode() not in body and b'/private' not in body and b'ssh_key' not in body
 finally:proc.terminate();proc.wait(timeout=5)


def finished(cache,run):
 for _ in range(100):
  r=cache.get(run)
  if not r['pending']:return r
  time.sleep(.005)
 raise AssertionError('Preview worker did not finish')


def test_cache_fetches_only_requested_run_deduplicates_and_keeps_last_good(tmp_path):
 (tmp_path/'frame-00001.png').write_bytes(png());release=threading.Event();calls=[];clock=[1.];bad=[False]
 def fetch(run):
  calls.append(run['id']);release.wait(1)
  if bad[0]:raise RuntimeError('/private/key transport error')
  return m.collect(tmp_path)
 cache=m.PreviewCache(fetcher=fetch,clock=lambda:clock[0]);run=dict(id='selected',results=str(tmp_path))
 try:
  for _ in range(10):cache.get(run)
  release.set();r=finished(cache,run);assert calls==['selected'] and r['views']['wide']
  for _ in range(10):cache.get(run)
  assert calls==['selected']
  clock[0]+=11;bad[0]=True;r=finished(cache,run)
  assert r['stale'] and r['views']['wide'] and '/private' not in json.dumps(r) and len(calls)==2
  r=cache.get(dict(run,results='/different'));assert r['views']['wide'] is None
 finally:cache.pool.shutdown(wait=True)
