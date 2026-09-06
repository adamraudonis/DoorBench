#!/usr/bin/env python3
"""Prepare a closed-start locomotion diagnostic across the non-pet catalogue.

This deliberately uses a uniform upright traversal task, not each door's assigned
manipulation/recognition scenario. Failed exports remain in the inventory.
"""
import argparse,json,hashlib,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from doorbench.spec import generate_all
from doorbench.build import export_door
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];excluded=[]
for s in generate_all():
 if s['family']=='pet_door':excluded.append({'id':s['id'],'reason':'supplementary_pet_collection'});continue
 row={'id':s['id'],'family':s['family'],'task':'uniform_closed_start_upright_traversal','initial_open_fraction':0.,'description':'Closed-start locomotion diagnostic; not the assigned core benchmark task.','assigned_task':s['task']}
 try:
  x=export_door(s,str(out/'doors'),str(out/'hardware'),formats=('json','usd'))
  errors={k:v for k,v in x['files'].items()if isinstance(v,str)and v.startswith('ERROR:')}
  if errors:raise RuntimeError(str(errors))
  row['source_sha256']={f:hashlib.sha256((out/'doors'/s['id']/f).read_bytes()).hexdigest()for f in ('spec.json','model.json','door_rl.usda')}
 except Exception:row['export_error']=traceback.format_exc()
 rows.append(row);print(s['id'],row.get('export_error','prepared').splitlines()[-1],flush=True)
 (out/'demo-suite.json').write_text(json.dumps({'scope':'Uniform closed-start canonical USD locomotion diagnostic; not core benchmark task success','cases':rows,'excluded':excluded},indent=2)+'\n')
