"""Detached read-only diagnostic: compare actual producer and JSON arithmetic."""
import argparse, gzip, hashlib, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=a.run/'balance-steps.json.gz';rows=json.load(gzip.open(source,'rt'))
if isinstance(rows,dict):rows=rows['rows']
metrics=[];exact=True;empty=0;max_load=0
for i,row in enumerate(rows):
 e=row['pad_evidence'];pairs=e['handle_pair_forces_world_N'];groups={name:[] for name in pairs}
 for c in e['contacts']:groups[c['body']].append(c)
 old=producer=0.;worst=None
 for body,patches in groups.items():
  f=np.asarray([c['normal_force_N'] for c in patches],float)
  n=np.asarray([c['normal'] for c in patches],float).reshape(-1,3)
  v=np.asarray(pairs[body],float)
  exact=exact and all(np.array_equal(x,x.astype(np.float32).astype(float)) for x in (f,n,v))
  sequential=np.zeros(3)
  for force,normal in zip(f,n):sequential+=force*normal
  norm64=float(np.linalg.norm(sequential-v))
  vectors=f.astype(np.float32)[:,None]*n.astype(np.float32)
  total32=vectors.sum(axis=0);difference32=total32-v.astype(np.float32)
  norm32=float(np.linalg.norm(difference32))
  old=max(old,norm64);producer=max(producer,norm32)
  if norm64>1e-8 and (worst is None or norm64>worst['float64_error_N']):
   worst=dict(body=body,forces=f.tolist(),normals=n.tolist(),pair=v.tolist(),sum32=total32.tolist(),sum64=sequential.tolist(),float64_error_N=norm64,float32_error_N=norm32)
 declared=e['normal_pair_force_consistency_error_N'];gap64=abs(old-declared);gap32=abs(producer-declared)
 max_load=max(max_load,max((c['normal_force_N'] for c in e['contacts']),default=0))
 metrics.append(dict(index=i,time_s=row['time_s'],declared_error_N=declared,float64_error_N=old,float32_error_N=producer,old_declared_disagreement_N=gap64,producer_declared_disagreement_N=gap32,worst=worst))
bad=[m for m in metrics if m['old_declared_disagreement_N']>1e-8 or m['float64_error_N']>1e-3]
r=dict(schema='doorbench.raw-pair-arithmetic-diagnostic.v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),original_report_sha256=hashlib.sha256((a.run/'balance-report.json').read_bytes()).hexdigest(),rows=len(rows),all_source_force_normal_matrix_values_exact_float32=exact,original_mismatch_count=len(bad),first_original_mismatch=bad[0] if bad else None,max_float64_error_N=max(m['float64_error_N'] for m in metrics),max_float32_error_N=max(m['float32_error_N'] for m in metrics),max_declared_error_N=max(m['declared_error_N'] for m in metrics),max_old_declared_disagreement_N=max(m['old_declared_disagreement_N'] for m in metrics),max_producer_declared_disagreement_N=max(m['producer_declared_disagreement_N'] for m in metrics),high_precision_force_balance_failures=sum(m['float64_error_N']>1e-3 for m in metrics),maximum_normal_patch_force_N=max_load,worst_samples=sorted(metrics,key=lambda m:m['old_declared_disagreement_N'],reverse=True)[:3],scope='Read-only detached arithmetic diagnostic; no task qualification or change to original report')
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('worst_samples','first_original_mismatch')}))
