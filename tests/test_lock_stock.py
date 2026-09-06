"""Independent stock/sweep checks bypass parent-child contact filtering."""
from copy import deepcopy
from functools import lru_cache
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import pytest
from doorbench.build import build_model
from doorbench.export.mjcf import build_mjcf
from doorbench.geometry import common as C
from doorbench.geometry.lock_stock import cut_stock,geom_bounds
from doorbench.ir import Body,ALL_TIERS,QUAT_ID
from doorbench.spec import generate_all

@lru_cache(None)
def specs():return {s['id']:s for s in generate_all()}

def native(ir,tier='full'):
 assets={g.mesh_name+'.obj':g.mesh.export(file_type='obj',include_normals=False,include_texture=False).encode() for b in ir.bodies for g in b.geoms if g.type=='mesh'}
 return mujoco.MjModel.from_xml_string(ET.tostring(build_mjcf(ir,tier=tier,mesh_dir_rel=''),encoding='unicode'),assets)


def test_cut_preserves_only_actual_remaining_stock_and_mass():
 b=Body('leaf',None);g=C.box('slab',(0,0,0),(1,.1,1),'wood',500,mass=100,semantic='leaf');b.geoms=[g]
 report=cut_stock(b,(-.2,-.2,-.3),(.2,.2,.3),'hole')
 assert report['removed_geometry_volume_m3']==pytest.approx(.4*.2*.6)
 assert sum(x.mass_override for x in b.geoms)==pytest.approx(100*(1-.4*.2*.6/.8))
 for x in b.geoms:
  a,c=geom_bounds(x)
  assert np.any(np.minimum(c,[.2,.2,.3])-np.maximum(a,[-.2,-.2,-.3])<=1e-12)


def test_reject_unsupported_intersecting_stock_and_invalid_cut():
 b=Body('leaf',None);g=C.cyl('solid_bar',(0,0,0),.1,.4,'steel',semantic='leaf');b.geoms=[g]
 with pytest.raises(ValueError,match='non-box stock'):cut_stock(b,[-.2]*3,[.2]*3,'bad')
 with pytest.raises(ValueError,match='Invalid stock'):cut_stock(b,[0]*3,[0]*3,'bad')


@pytest.mark.parametrize('id',('db0002_swing_single','db0011_automatic_swing','db0016_swing_single','db0035_swing_single','db0418_swing_single','db0122_swing_single','db0304_swing_single'))
@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_native_bolt_sweep_has_stock_and_guide_clearance(id,tier):
 ir=build_model(deepcopy(specs()[id]));m=native(ir,tier);d=mujoco.MjData(m)
 json.dumps(ir.meta)  # metadata must be strict JSON-native values
 for r in ir.meta.get('lock_stock',[]):
  body=ir.body(r['bolt_body'])
  if tier not in body.tiers:continue
  j=m.joint(body.joint.name).id if body.joint else None
  for value in np.linspace(*body.joint.range,17) if j is not None else [0.]:
   d.qpos[:]=m.qpos0
   if j is not None:d.qpos[m.jnt_qposadr[j]]=value
   mujoco.mj_forward(m,d)
   for bolt in r['bolt_geoms']:
    a=m.geom(bolt).id
    for g in ir.body(r['leaf_body']).geoms:
     if not g.collision or tier not in g.tiers or g.semantic not in ('leaf','glass','lock'):continue
     b=m.geom(g.name).id
     if np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])>m.geom_rbound[a]+m.geom_rbound[b]+.001:continue
     gap=mujoco.mj_geomDistance(m,d,a,b,.003,None)
     assert gap>=-1e-6,(id,tier,bolt,g.name,value,gap)


def test_missing_cut_reintroduces_parent_filtered_overlap():
 ir=build_model(deepcopy(specs()['db0016_swing_single']));r=next(r for r in ir.meta['lock_stock'] if r['bolt_body'].endswith('deadbolt'));leaf=ir.body(r['leaf_body'])
 bolt=ir.body(r['bolt_body']);g=bolt.geoms[0]
 leaf.geoms.append(C.box('filled_mortise',tuple(np.asarray(bolt.pos)+g.pos),tuple(np.asarray(g.size)*.8),'mat_leaf',semantic='leaf'))
 m=native(ir);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 a=m.geom(r['bolt_geoms'][0]).id;b=m.geom('filled_mortise').id
 # Native GJK can report0 for containment; exact AABB overlap is decisive here.
 ha=np.abs(d.geom_xmat[a].reshape(3,3))@m.geom_size[a];hb=np.abs(d.geom_xmat[b].reshape(3,3))@m.geom_size[b]
 overlap=np.minimum(d.geom_xpos[a]+ha,d.geom_xpos[b]+hb)-np.maximum(d.geom_xpos[a]-ha,d.geom_xpos[b]-hb)
 assert min(overlap)>.001
 assert not any(set(c.geom)=={a,b} for c in d.contact)  # native filtering hides it


def test_thin_glass_has_real_clamped_cartridge_and_surface_thumb_grips():
 ir=build_model(deepcopy(specs()['db0016_swing_single']));r=next(r for r in ir.meta['lock_stock'] if r['bolt_body'].endswith('deadbolt'))
 assert r['thin_stock_edge_cartridge'] and len(r['mount_geoms'])==8
 assert r['stock_cut']['removed_geoms'] and r['spindle_bore']['removed_geoms']
 body=ir.body(r['bolt_body']+'_thumbturn');paddle=next(g for g in body.geoms if g.name.endswith('thumbturn_col'))
 for site in body.sites:
  from doorbench.ir import quat_to_mat
  local=np.abs(quat_to_mat(paddle.quat).T@(np.asarray(site.pos)-paddle.pos))/paddle.size
  assert np.max(local)==pytest.approx(1.) and np.all(local<=1.+1e-9)
 assert not any(set(pair)=={r['bolt_body'],body.name} for pair in ir.contact_excludes)


@pytest.mark.parametrize('id',('db0016_swing_single','db0035_swing_single','db0304_swing_single','db0418_swing_single'))
def test_thumbturn_native_retract_and_throw_cycles_keep_real_contacts(id):
 ir=build_model(deepcopy(specs()[id]));records=[r for r in ir.meta['lock_stock'] if r.get('thumbturn_grip_sites')]
 if not records:pytest.skip('Authored key-only state has no thumbturn input')
 m=native(ir);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 for r in records:
  turn=ir.body(r['bolt_body']+'_thumbturn');j=m.joint(turn.joint.name).id;q=m.jnt_qposadr[j];v=m.jnt_dofadr[j]
  bolt=m.joint(ir.body(r['bolt_body']).joint.name).id;bq=m.jnt_qposadr[bolt];throw=m.jnt_range[bolt,1]
  for target in [0.,m.jnt_range[j,1],0.,m.jnt_range[j,1]]:
   for _ in range(round(2./m.opt.timestep)):
    d.qfrc_applied[:]=0;d.qfrc_applied[v]=np.clip(5*(target-d.qpos[q])-.2*d.qvel[v],-1.2,1.2)
    mujoco.mj_step(m,d)
    assert np.isfinite(d.qpos).all() and not any(d.warning.number)
   expected=throw*target/m.jnt_range[j,1]
   assert abs(d.qpos[bq]-expected)<.0015,(id,target,d.qpos[bq],expected)


@pytest.mark.parametrize('id',('db0011_automatic_swing','db0016_swing_single','db0304_swing_single'))
def test_direct_stock_gate_checks_housing_and_rejects_refilled_guide(id):
 from doorbench.lock_stock_qa import run_lock_stock_qa
 ir=build_model(deepcopy(specs()[id]));m=native(ir)
 result=run_lock_stock_qa(m,ir.meta)
 assert result['ok'],result
 row=ir.meta['lock_stock'][0];bolt=m.geom(row['bolt_geoms'][0]).id;guide=m.geom(row['guide_geoms'][0]).id
 d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
 parent=m.geom_bodyid[guide]
 m.geom_pos[guide]=d.xmat[parent].reshape(3,3).T@(d.geom_xpos[bolt]-d.xpos[parent])
 m.geom_size[guide]=[.02,.01,.02]
 result=run_lock_stock_qa(m,ir.meta)
 assert not result['ok']
 assert any(row['guide_geoms'][0] in failure.get('geoms',[]) for failure in result['failures'])


def test_internal_guide_gap_is_positive_and_does_not_exempt_world(monkeypatch):
 from doorbench.clearance import Clearance,RUN_MIN
 c=object.__new__(Clearance);c.meta={'lock_stock':[{'bolt_geoms':['bolt'],'guide_geoms':['guide']}]}
 c.strip_bearing_pairs=set();c.run_allow=[];c.sem={'bolt':'lock','guide':'lock','leaf':'leaf','wall':'wall'};c.run_min=RUN_MIN
 assert c.required_gap('bolt','guide')==.00075
 assert c.required_gap('guide','bolt')==.00075
 assert c.required_gap('leaf','wall')==RUN_MIN


@pytest.mark.parametrize('door,cut',(
 ('db0063_swing_single','plate_spindle_bores'),
 ('db0201_swing_single','plate_spindle_bores'),
 ('db0912_swing_single','plate_spindle_bores'),
 ('db0086_swing_single','keypad_spindle_socket'),
 ('db0182_swing_single','keypad_spindle_socket')))
def test_prepared_operator_plate_or_socket_has_real_rotating_spindle_gap(door,cut):
 from doorbench.lock_stock_qa import run_lock_stock_qa
 ir=build_model(deepcopy(specs()[door]));row=next(r for r in ir.meta['lock_stock'] if r.get(cut,{}).get('removed_geoms'))
 assert row[cut]['removed_geometry_volume_m3']>0
 m=native(ir);result=run_lock_stock_qa(m,ir.meta)
 assert result['ok'],result['failures']
 assert any(r['part']=='thumbturn' and r['minimum_gap_m']>=.0005-1e-5 for r in result['measurements'])
 # Refill the prepared bore with the original plate's material envelope.
 lo=np.asarray(row[cut]['lower']);hi=np.asarray(row[cut]['upper'])
 ir.body(row['leaf_body']).geoms.append(C.box('refilled_spindle_stock',tuple((lo+hi)/2),tuple((hi-lo)/2),'steel',7900,semantic='operator'))
 result=run_lock_stock_qa(native(ir),ir.meta)
 assert not result['ok']
 assert any('refilled_spindle_stock' in r.get('geoms',[]) for r in result['failures'])


def test_every_latch_coupling_respects_its_real_operator_endpoint():
 checked=0
 for spec in specs().values():
  ir=build_model(deepcopy(spec));joints={b.joint.name:b.joint for b in ir.bodies if b.joint}
  for tendon in ir.tendons:
   if len(tendon.sites)!=2 or not all(name in joints for name,_ in tendon.sites):continue
   (bn,bc),(hn,hc)=tendon.sites;bolt,driver=joints[bn],joints[hn]
   if bolt.role!='latch' or bc!=1 or hc>=0 or not bolt.range or not driver.range:continue
   assert -hc*driver.range[1]<=bolt.range[1]+1e-10,(spec['id'],tendon.name,driver.range,bolt.range)
   checked+=1
  for row in ir.meta.get('lock_stock',[]):
   c=row.get('handle_coupling')
   if not c:continue
   driver=joints[c['joint']];bolt=joints[row['name']+'_slide']
   assert driver.range is not None
   assert c['bolt_m_per_joint_unit']*driver.range[1]<=bolt.range[1]+1e-10,(spec['id'],c,driver.range,bolt.range)
   coupling=next(t for t in ir.tendons if t.name==row['name']+'_coupling')
   assert dict(coupling.sites)[c['joint']]==-c['bolt_m_per_joint_unit']
 assert checked>=490


def test_long_travel_handle_retracts_without_driving_bolt_through_guide_back():
 ir=build_model(deepcopy(specs()['db0767_swing_single']));row=next(r for r in ir.meta['lock_stock'] if r.get('handle_coupling'))
 m=native(ir);d=mujoco.MjData(m);c=row['handle_coupling'];j=m.joint(c['joint']).id;b=m.joint(row['name']+'_slide').id
 q,v=m.jnt_qposadr[j],m.jnt_dofadr[j];bq=m.jnt_qposadr[b];target=m.jnt_range[j,1]
 assert target>.7
 for _ in range(round(2/m.opt.timestep)):
  d.qfrc_applied[:]=0;d.qfrc_applied[v]=np.clip(10*(target-d.qpos[q])-.5*d.qvel[v],-2,2)
  mujoco.mj_step(m,d)
  assert np.isfinite(d.qpos).all() and not any(d.warning.number)
 assert abs(d.qpos[bq]-m.jnt_range[b,1])<.0015
