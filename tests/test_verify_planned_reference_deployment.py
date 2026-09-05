"""The Pages build must never consume a moving or unverified MotionLab index."""
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import Request

import pytest

from scripts import verify_planned_reference_deployment as deploy


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value)+'\n').encode()
def write(path,value):Path(path).write_bytes(encoded(value))


@pytest.fixture
def deployment(tmp_path):
    manifest=encoded({'doors':list(range(1000))});(tmp_path/'manifest.json').write_bytes(manifest)
    native=sha(manifest)
    index={'schema':'doorbench.planned-reference-web-index.v1','manifest_sha256':native,
           'counts':{'unresolved':1000},'doors':[{'door_id':f'db{i:04}_swing_single','family':'swing_single','status':'unresolved','reason':'No accepted complete motion','clip':None,'audits':{}} for i in range(1,1001)]}
    raw=encoded(index)
    config={'schema_version':1,'release':'planned-fixture-v1','web_index_url':'https://huggingface.co/datasets/example/DoorBench/resolve/'+'a'*40+'/experimental/planned-reference/planned-fixture-v1/web/index.json','web_index_sha256':sha(raw),'native_manifest_sha256':native}
    write(tmp_path/'config.json',config);write(tmp_path/'site.json',{'schema_version':1,'dataset_manifest_sha256':native});(tmp_path/'index.json').write_bytes(raw)
    return tmp_path,config,index


def test_verified_config_matches_pinned_url_index_and_actual_restored_native(deployment,monkeypatch):
    root,config,index=deployment;seen=[]
    def download(url):seen.append(url);return (root/'index.json').read_bytes()
    monkeypatch.setattr(deploy,'download_index',download)
    actual,parsed,raw=deploy.verify_deployment(root/'config.json',root/'site.json',assets_manifest=root/'manifest.json')
    assert actual==config and parsed==index and sha(raw)==config['web_index_sha256']
    assert seen==[config['web_index_url']]


@pytest.mark.parametrize('change',['wrong_hash','wrong_native','index_version','coverage','duplicate','nonaccepted_clip','count','traversal'])
def test_wrong_or_stale_index_is_rejected_even_when_self_consistent(deployment,change):
    root,config,index=deployment
    if change=='wrong_hash':config['web_index_sha256']='0'*64
    if change=='wrong_native':index['manifest_sha256']='0'*64
    if change=='index_version':index['schema']='doorbench.planned-reference-web-index.v2'
    if change=='coverage':index['doors'].pop()
    if change=='duplicate':index['doors'][1]=index['doors'][0]
    if change=='nonaccepted_clip':index['doors'][0]['clip']={}
    if change=='count':index['counts']['unresolved']=999
    if change=='traversal':index['doors'][0]['audits']={'result.json':{'path':'audits/../outside.json','sha256':'b'*64,'bytes':1}}
    raw=encoded(index)
    if change!='wrong_hash':config['web_index_sha256']=sha(raw)
    write(root/'config.json',config)
    with pytest.raises(ValueError):deploy.verify_deployment(root/'config.json',root/'site.json',index_bytes=raw)


@pytest.mark.parametrize('change',['schema','bool_schema','release','tag','http','query','fragment','credentials','host','malformed_hash','extra','missing'])
def test_malformed_config_or_moving_url_is_rejected_before_network(deployment,change,monkeypatch):
    root,config,_=deployment
    if change=='schema':config['schema_version']=2
    if change=='bool_schema':config['schema_version']=True
    if change=='release':config['release']='planned-other-v1'
    if change=='tag':config['web_index_url']=config['web_index_url'].replace('a'*40,'main')
    if change=='http':config['web_index_url']=config['web_index_url'].replace('https:','http:')
    if change=='query':config['web_index_url']+='?download=true'
    if change=='fragment':config['web_index_url']+='#other'
    if change=='credentials':config['web_index_url']=config['web_index_url'].replace('https://','https://secret@')
    if change=='host':config['web_index_url']=config['web_index_url'].replace('huggingface.co','huggingface.co.example.test')
    if change=='malformed_hash':config['native_manifest_sha256']='bad'
    if change=='extra':config['token']='must not be needed'
    if change=='missing':config.pop('release')
    write(root/'config.json',config)
    monkeypatch.setattr(deploy,'download_index',lambda _:pytest.fail('Invalid config reached network'))
    with pytest.raises(ValueError):deploy.verify_deployment(root/'config.json',root/'site.json')


def test_duplicate_fields_and_nonfinite_json_are_rejected():
    with pytest.raises(ValueError,match='Duplicate'):deploy.json_value('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError,match='finite'):deploy.json_value('{"bytes":NaN}')


@pytest.mark.parametrize('actual',['site_pin','restored_bytes'])
def test_native_site_pin_and_actual_restored_manifest_must_both_match(deployment,actual):
    root,_,_=deployment
    if actual=='site_pin':write(root/'site.json',{'schema_version':1,'dataset_manifest_sha256':'0'*64})
    else:(root/'manifest.json').write_bytes(b'changed restored assets')
    with pytest.raises(ValueError,match='manifest'):
        deploy.verify_deployment(root/'config.json',root/'site.json',index_bytes=(root/'index.json').read_bytes(),assets_manifest=root/'manifest.json')


def test_build_environment_is_written_only_after_complete_verification(deployment,monkeypatch):
    root,config,_=deployment;env=root/'github-env';env.write_text('EXISTING=safe\n')
    argv=['verify','--config',str(root/'config.json'),'--site-assets',str(root/'site.json'),'--index-file',str(root/'index.json'),'--index-out',str(root/'verified-index.json'),'--github-env',str(env)]
    monkeypatch.setattr(sys,'argv',argv);deploy.main()
    assert env.read_text()=='EXISTING=safe\nVITE_PLANNED_REFERENCE_INDEX='+config['web_index_url']+'\n'
    assert (root/'verified-index.json').read_bytes()==(root/'index.json').read_bytes()
    before=env.read_bytes();config['web_index_sha256']='0'*64;write(root/'config.json',config)
    with pytest.raises(ValueError,match='SHA-256'):deploy.main()
    assert env.read_bytes()==before


def test_network_download_is_bounded_and_redirects_cannot_downgrade(monkeypatch):
    class Response(io.BytesIO):
        headers={}
    class Opener:
        def open(self,*args,**kwargs):return Response(b'x'*(deploy.MAX_INDEX_BYTES+1))
    monkeypatch.setattr(deploy,'build_opener',lambda *_:Opener())
    with pytest.raises(ValueError,match='4 MB'):deploy.download_index('https://huggingface.co/test')
    with pytest.raises(ValueError,match='HTTPS'):
        deploy.HTTPSRedirects().redirect_request(Request('https://huggingface.co/test'),None,302,'',{},'http://example.test/data')


def test_valid_accepted_descriptor_is_preserved_without_fetching_the_trajectory(deployment):
    root,config,index=deployment;row=index['doors'][0]
    row.update(status='accepted_kinematic',source_scenario='open_and_traverse',identity_sha256='b'*64,clip={'path':'clips/db0001.'+'c'*64+'.json.gz','sha256':'c'*64,'json_sha256':'d'*64,'bytes':20,'frames':2,'duration':1.})
    index['counts']={'accepted_kinematic':1,'unresolved':999};raw=encoded(index);config['web_index_sha256']=sha(raw)
    assert deploy.validate_index(raw,config)['doors'][0]['clip']==row['clip']
