#!/usr/bin/env python3
"""Run every eligible catalogue fixture; retain errors in the denominator.
Fresh processes isolate batches. A failed batch's missing trials get one isolated
retry, so a malformed asset cannot silently erase its neighbors from the score.
"""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',required=True);p.add_argument('--out',required=True);p.add_argument('--batch',type=int,default=16);p.add_argument('--timeout',type=float,default=1200);p.add_argument('--hero',action='store_true');a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];assets=Path(a.assets).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);suite=json.loads((assets/'demo-suite.json').read_text());cases=suite['cases'];rows={};start=datetime.now(timezone.utc).isoformat()
def stamp():
 d={'scope':suite['scope'],'started_at_utc':start,'updated_at_utc':datetime.now(timezone.utc).isoformat(),'complete':len(rows)==len(cases),'eligible_doors':len(cases),'attempted':len(rows),'successes':sum(r.get('success',False)for r in rows.values()),'errors':sum(bool(r.get('simulator_error'))for r in rows.values()),'excluded':suite['excluded'],'per_door':rows,'suite_sha256':hashlib.sha256((assets/'demo-suite.json').read_bytes()).hexdigest()}
 temp=out/'results.tmp';temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(out/'results.json');return d

def batch(ids,folder,video=False):
 folder.mkdir(parents=True,exist_ok=False);cmd=[sys.executable,str(ROOT/'scripts/isaaclab/grid_g1.py'),'--headless','--device','cuda:0','--assets',str(assets),'--out',str(folder),'--batch-doors',*ids]
 if video:cmd.append('--video')
 with(folder/'run.log').open('w')as log:
  try:proc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout);error=None if proc.returncode==0 else f'process_exit_{proc.returncode}'
  except subprocess.TimeoutExpired:error='batch_timeout'
 text=(folder/'run.log').read_text(errors='replace');error=error or(None if 'GRID_COMPLETE'in text and 'Traceback (most recent call last)'not in text else 'missing_native_completion')
 results={}
 for id in ids:
  p=folder/(id+'.json')
  if p.exists():
   r=json.loads(p.read_text());r['evidence_directory']=str(folder.relative_to(out))
   if error:r.update(success=False,simulator_error=error)
   if r.get('failure_reason')=='nonfinite_native_state':r['simulator_error']='nonfinite_native_state'
   results[id]=r
 return results,error

for offset in range(0,len(cases),a.batch):
 group=cases[offset:offset+a.batch];ids=[]
 for c in group:
  if c.get('export_error'):rows[c['id']]={'success':False,'simulator_error':'export_error','detail':c['export_error']}
  else:ids.append(c['id'])
 if ids:
  results,error=batch(ids,out/f'batch-{offset:04d}')
  for id in ids:
   if id not in results or results[id].get('simulator_error'):
    retry,e=batch([id],out/f'retry-{id}');r=retry.get(id,{'door_id':id,'success':False,'simulator_error':e or 'missing_receipt'});r['initial_batch_error']=error;results[id]=r
  rows.update(results)
 print('CATALOGUE_PROGRESS',json.dumps({k:v for k,v in stamp().items()if k in ('attempted','successes','errors','eligible_doors')}),flush=True)
final=stamp();final['completed_at_utc']=datetime.now(timezone.utc).isoformat();(out/'results.json').write_text(json.dumps(final,indent=2)+'\n')
if a.hero:
 passing=[c for c in cases if rows[c['id']].get('success')];chosen=[];families=set()
 for c in passing:
  if c['family']not in families:chosen.append(c['id']);families.add(c['family'])
 chosen=(chosen+[c['id']for c in passing if c['id']not in chosen])[:16]
 if len(chosen)==16:
  hero,error=batch(chosen,out/'hero',True);(out/'hero-selection.json').write_text(json.dumps({'scope':'Sixteen successful catalogue cases selected for illustration; rerun together, not a random sample','ids':chosen,'rerun_error':error,'rerun_successes':sum(r.get('success',False)for r in hero.values())},indent=2)+'\n')
 else:(out/'hero-selection.json').write_text(json.dumps({'error':'Fewer than sixteen successful distinct cases; do not imply sixteen successful traversals','successes':len(passing)})+'\n')
print('CATALOGUE_COMPLETE',flush=True)
