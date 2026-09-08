#!/usr/bin/env python3
"""Render and measure actual archived palm/finger support; never solve forces."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--time',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 run=a.run.resolve();sys.path.insert(0,str(run.with_name(run.name+'-source')))
 from doorbench.dexterous.environment import DexterousDoorEnv
 cfg=json.loads((run/'manifest.json').read_text())['configuration'];robot=Path(cfg['robot']);sim=DexterousDoorEnv(cfg['door'],robot,json.loads(robot.with_suffix('.audit.json').read_text()));m,d=sim.m,sim.d
 archive=json.loads((run/'raw-transitions/manifest.json').read_text());chunk=next(c for c in archive['chunks'] if c['interval_start_s']-1e-8<=a.time<c['interval_end_s']-1e-8);source=run/'raw-transitions'/chunk['file']
 if hashlib.file_digest(source.open('rb'),'sha256').hexdigest()!=chunk['sha256']:raise ValueError('Changed actual contact archive')
 with np.load(source,allow_pickle=False) as raw:
  i=int(np.argmin(abs(raw['interval_start_s']-a.time)));d.qpos[:]=raw['qpos_before'][i];d.qvel[:]=raw['qvel_before'][i];start,end=raw['contact_offsets'][i:i+2]
  contacts={k:raw[k][start:end].copy() for k in raw.files if k.startswith('contact_') and k!='contact_offsets'}
  time=float(raw['interval_start_s'][i]);intervalend=float(raw['interval_end_s'][i]);bs,be=raw['body_offsets'][i:i+2];bodyids=raw['body_ids'][bs:be];bodyp=raw['body_positions_world_m'][bs:be];bodyr=raw['body_rotations_world'][bs:be]
 mujoco.mj_kinematics(m,d);mujoco.mj_camlight(m,d)
 error=max(float(np.max(abs(d.xpos[bodyids]-bodyp))),float(np.max(abs(d.xmat[bodyids].reshape(-1,3,3)-bodyr))))
 if error>1e-9:raise ValueError('Body FK and actual contact frame disagree')
 leaf=m.body('leaf').id;lp=d.xpos[leaf].copy();lr=d.xmat[leaf].reshape(3,3).copy();normal=lr[:,1]
 palm=m.site('robot/lh_palm_touch').id;pp=d.site_xpos[palm].copy();pr=d.site_xmat[palm].reshape(3,3).copy()
 rows=[]
 # The raw archive uses plural packed keys; each contact normal points toward
 # its second body. Read the actual interval wrench, never mj_forward forces.
 for j,(g0,g1) in enumerate(contacts['contact_geom']):
  names=[m.body(m.geom_bodyid[int(g)]).name for g in (g0,g1)]
  hands=[k for k,name in enumerate(names) if name.startswith('robot/lh_')]
  if len(hands)!=1 or all(name.startswith('robot/') for name in names):continue
  k=hands[0];body=m.geom_bodyid[int((g0,g1)[k])];point=contacts['contact_position_world_m'][j];frame=contacts['contact_frame_world'][j];wrench=contacts['contact_wrench_contact_frame'][j]
  force=(1 if k==1 else -1)*(frame.T@wrench[:3]);load=max(0.,-float(force@normal))
  if wrench[0]<=.05:continue
  rows.append(dict(body=names[k],world_position_m=point.tolist(),palm_local_position_m=(pr.T@(point-pp)).tolist(),body_local_position_m=(d.xmat[body].reshape(3,3).T@(point-d.xpos[body])).tolist(),leaf_local_position_m=(lr.T@(point-lp)).tolist(),world_force_on_hand_N=force.tolist(),normal_load_N=load,raw_normal_force_N=float(wrench[0])))
 clouds={}
 for g in range(m.ngeom):
  body=m.body(m.geom_bodyid[g]).name
  if not body.startswith('robot/lh_') or not m.geom_contype[g]:continue
  mesh=m.geom_dataid[g];rotation=d.geom_xmat[g].reshape(3,3);kind=int(m.geom_type[g]);size=m.geom_size[g]
  if kind==int(mujoco.mjtGeom.mjGEOM_MESH) and mesh>=0:
   begin=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh];local=m.mesh_vert[begin:begin+count]
  elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):
   import itertools
   local=np.array(list(itertools.product((-1.,1.),repeat=3)))*size
  elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
   axis=rotation.T@normal;radial=axis[:2]/max(np.linalg.norm(axis[:2]),1e-12)
   support=np.r_[radial*size[0],np.sign(axis[2])*size[1]];local=np.array([support,-support])
  else:raise ValueError('Unsupported original hand collision primitive')
  v=local@rotation.T+d.geom_xpos[g]
  clouds.setdefault(body,[]).append((v-lp)@lr)
 geometry=[]
 for body,parts in clouds.items():
  v=np.vstack(parts);index=int(np.argmax(v[:,1]));geometry.append(dict(body=body,maximum_leaf_normal_coordinate_m=float(v[index,1]),support_vertex_leaf_m=v[index].tolist(),normal_span_m=[float(v[:,1].min()),float(v[:,1].max())]))
 result=dict(scope='Actual preceding dynamics interval loads with exact matching pre-integration pose; render-only colors/visibility change no physics.',interval_s=[time,intervalend],source_sha256=chunk['sha256'],maximum_actual_body_frame_error=error,palm_site_leaf_m=(lr.T@(pp-lp)).tolist(),palm_site_rotation_leaf=(lr.T@pr).tolist(),palm_normal_alignment_error_deg=float(np.degrees(np.arccos(np.clip((-pr[:,2])@normal,-1,1)))),loaded_contacts=rows,collision_supports=geometry)
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 # Render exact collision meshes only. A translucent leaf reveals the actual
 # contact interface without hiding the fingers behind an opaque slab.
 for g in range(m.ngeom):
  name=m.body(m.geom_bodyid[g]).name;m.geom_matid[g]=-1
  if name.startswith('robot/lh_') and m.geom_contype[g]:m.geom_rgba[g]=[.95,.68,.2,1] if name=='robot/lh_palm' else [.65,.22,.75,1] if 'lf' in name else [.15,.55,.85,1]
  elif name=='leaf' and m.geom_contype[g]:m.geom_rgba[g]=[.6,.7,.78,.18]
  else:m.geom_rgba[g,3]=0.
 options=mujoco.MjvOption();options.sitegroup[:]=0;options.geomgroup[:]=1
 leafangle=np.degrees(float(d.qpos[m.jnt_qposadr[m.joint('leaf_hinge').id]]))
 canvas=Image.new('RGB',(1800,1000),'#202630')
 with mujoco.Renderer(m,height=500,width=600) as renderer:
  for index,(az,el) in enumerate(((30,-10),(70,-10),(110,-10),(-30,10),(-70,10),(-110,10))):
   camera=mujoco.MjvCamera();camera.lookat[:]=pp;camera.distance=.3;camera.azimuth=leafangle+az;camera.elevation=el
   renderer.update_scene(d,camera=camera,scene_option=options);sim.hide_sensor_overlays(renderer.scene)
   for row in rows:
    if renderer.scene.ngeom>=renderer.scene.maxgeom:break
    geom=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.0012,0.,0.]),np.array(row['world_position_m']),np.eye(3).ravel(),np.array([.1,1.,.1,1.]));renderer.scene.ngeom+=1
   im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,600,42),fill='#202630');draw.text((8,7),f'Actual {time:.3f}s / collision meshes / view {az:+} degrees',fill='white');draw.text((8,23),'Gold palm; purple little finger; green measured loaded contacts',fill='white');im.save(a.output/f'view-{index+1}.png');canvas.paste(im,((index%3)*600,(index//3)*500))
 canvas.save(a.output/'contact-sheet.png');sim.close();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
