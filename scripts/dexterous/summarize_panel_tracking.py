"""Extract bounded-size tracking diagnostics from a completed panel replay. No physics.

The plan must be the segment active throughout the requested sample interval.
"""
import argparse
import gzip,json,pathlib,hashlib,numpy as np
from scipy.spatial.transform import Rotation
from scripts.dexterous.stress_panel_tracking_error import pose
from doorbench.dexterous.json_record_stream import iter_json_object_array
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--trial',type=pathlib.Path,required=True);ap.add_argument('--plan',type=pathlib.Path,required=True);ap.add_argument('--start',type=float,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
root=a.trial;p=json.load(open(a.plan));rq=p['root_qpos_address'];qa=np.array(p['joint_qpos_addresses']);rows=[];next_time=a.start
with gzip.open(root/'controller-steps.json.gz','rt') as f:
 for row in iter_json_object_array(f):
  if row['time_s']<next_time or 'panel_progress' not in row:continue
  rows.append({k:row.get(k) for k in ['time_s','panel_progress','panel_reference_aperture_rad','left_panel_load_N','support_load_target_N','left_tracking_error_m']});next_time=row['time_s']+.25
manifest=json.load(open(root/'raw-transitions/manifest.json'));result=[];ci=-1;data=None
for row in rows:
 t=row['time_s'];index=next(i for i,c in enumerate(manifest['chunks']) if c['interval_start_s']<=t<c['interval_end_s'])
 if index!=ci:
  if data is not None:data.close()
  c=manifest['chunks'][index];path=root/'raw-transitions'/c['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256'];data=np.load(path);ci=index
 k=int(np.argmin(abs(data['interval_start_s']-t)));assert abs(data['interval_start_s'][k]-t)<1e-8
 actual=data['qpos_before'][k];planned=pose(p,row['panel_progress']);dr=Rotation.from_quat(actual[rq:rq+7][[4,5,6,3]])*Rotation.from_quat(planned[rq:rq+7][[4,5,6,3]]).inv()
 row.update(root_rotation_error_rad=float(dr.magnitude()),root_error_rotvec_rad=dr.as_rotvec().tolist(),root_translation_error_m=(actual[rq:rq+3]-planned[rq:rq+3]).tolist(),joint_errors_rad=(actual[qa]-planned[qa]).tolist(),raw_chunk_sha256=c['sha256']);result.append(row)
if data is not None:data.close()
summary={'source_plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'sample_start_s':a.start,'scope':'Recorded controller-reference versus raw attained pose sampled at approximately0.25s. No new physics or inference of contact forces. Unlisted finger coordinates are not covered.','samples':len(result),'joint_names':p['joint_names'],'maximum_root_rotation_error':max(result,key=lambda r:r['root_rotation_error_rad']),'first':result[0],'last':result[-1]}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(summary=summary,rows=result),indent=2)+'\n');a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ['samples']}));print('maximum root rotation',summary['maximum_root_rotation_error']['root_rotation_error_rad'],'at',summary['maximum_root_rotation_error']['time_s'])
