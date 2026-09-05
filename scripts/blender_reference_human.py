"""Original DoorBench integration: invoke external MPFB's CC0 asset services.
Run with Blender in an isolated profile; MPFB GPL tool sources remain external.
This constructs a skinned asset, not a claim about motion/contact/dynamics.
"""
import bpy, addon_utils, pathlib, math, json, hashlib, sys, bmesh, argparse, os, tempfile
from mathutils import Matrix, Vector
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--toolcache',type=pathlib.Path,required=True)
ap.add_argument('--out',type=pathlib.Path,required=True)
ap.add_argument('--no-render',action='store_true')
a=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
if not os.environ.get('BLENDER_USER_RESOURCES'):
 raise RuntimeError('Use an isolated BLENDER_USER_RESOURCES directory for this external build tool.')
TOOL=a.toolcache.resolve();OUT=a.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
# A hash-correct ZIP does not prove that an existing extracted tree is intact.
# Use a fresh bytecode location so stale local .pyc files cannot bypass source verification.
bytecode_cache=tempfile.TemporaryDirectory(prefix='doorbench-mpfb-bytecode-')
sys.pycache_prefix=bytecode_cache.name
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from setup_human_reference import verify_toolcache, PACKAGES
verify_toolcache(TOOL)

ASSETS=TOOL/'system-assets'; COMMIT='80919fa4682335c41847f761a4d79dcad4124732'
bpy.context.preferences.extensions.repos.new(name='DoorBench isolated MPFB',module='doorbench_mpfb',custom_directory=str(TOOL/'extension-repo'))
addon_utils.enable('bl_ext.doorbench_mpfb.mpfb',default_set=True,persistent=False)
from bl_ext.doorbench_mpfb.mpfb.services.humanservice import HumanService
from bl_ext.doorbench_mpfb.mpfb.services.targetservice import TargetService
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
macro=TargetService.get_default_macro_info_dict(); macro.update(gender=1.0,age=.5,muscle=.5,weight=.5)
# Measure the actual evaluated skin surface rather than helper cages.
def vertices(obj):
 ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()); m=ev.to_mesh()
 try:return [obj.matrix_world@v.co for v in m.vertices]
 finally:ev.to_mesh_clear()
def bounds(obj):
 pts=vertices(obj);return [[min(p[i] for p in pts),max(p[i] for p in pts)] for i in range(3)]
human=HumanService.create_human(macro_detail_dict=macro); bpy.context.view_layer.update()
height=bounds(human)[2]; actual=height[1]-height[0]
bpy.data.objects.remove(human,do_unlink=True)
human=HumanService.create_human(macro_detail_dict=macro,scale=.1*1.75/actual); human.name='Human.skin'
rig=HumanService.add_builtin_rig(human,'default',import_weights=True); rig.name='Human.rig'; rig.data.name='Human.default163'
skin=ASSETS/'skins/young_caucasian_male2/young_caucasian_male2.mhmat'
HumanService.set_character_skin(str(skin),human,skin_type='ENHANCED_SSS')
selection=[('eyes','high-poly','Eyes'),('eyebrows','eyebrow001','Eyebrows'),('eyelashes','eyelashes01','Eyelashes'),('teeth','teeth_base','Teeth'),('tongue','tongue01','Tongue'),('hair','short02','Hair'),('clothes','male_casualsuit01','Clothes'),('clothes','shoes01','Clothes')]
objects=[human]; inputs=[skin]
for sub,name,kind in selection:
 path=ASSETS/sub/name/(name+'.mhclo'); print('ADDING',path,flush=True)
 obj=HumanService.add_mhclo_asset(str(path),human,asset_type=kind,subdiv_levels=1,material_type='MAKESKIN')
 objects.append(obj);inputs.append(path)
# Bake only the chosen macro shape and masks. Retain armature deformation and
# subdivision as ordinary Blender modifiers; no running addon is needed later.
bpy.ops.object.select_all(action='DESELECT')
for obj in objects:
 bpy.context.view_layer.objects.active=obj;obj.select_set(True)
 if obj.data.shape_keys:bpy.ops.object.shape_key_remove(all=True,apply_mix=True)
 for mod in list(obj.modifiers):
  if mod.type=='MASK':bpy.ops.object.modifier_apply(modifier=mod.name)
 if obj==human and not any(m.type=='SUBSURF' for m in obj.modifiers):
  mod=obj.modifiers.new('Skin surface','SUBSURF');mod.levels=1;mod.render_levels=2
 for p in obj.data.polygons:p.use_smooth=True
 obj.select_set(False)
# MPFB faces -Y with anatomical left +X. Rotate geometry and rest data together
# into DoorBench's +Y-forward/-X-left frame, keeping object transforms identity.
rotation=Matrix.Rotation(math.pi,4,'Z');rig.data.transform(rotation)
for obj in objects:obj.data.transform(rotation)
bpy.context.view_layer.update()
sole=min(v.z for obj in objects for v in vertices(obj))
shift=Matrix.Translation((0,0,-sole));rig.data.transform(shift)
for obj in objects:obj.data.transform(shift)
# The stock footwear contains calf-length socks that intersect these fitted
# jeans. Remove their fully covered upper section, retaining shoe/ankle mesh.
shoes=next(obj for obj in objects if obj.name=='Human.shoes01')
bm=bmesh.new();bm.from_mesh(shoes.data)
bmesh.ops.delete(bm,geom=[v for v in bm.verts if v.co.z>.115],context='VERTS')
bm.to_mesh(shoes.data);bm.free();shoes.data.update()
bpy.context.view_layer.update()
# All package textures are packed; external asset paths are provenance only.
bpy.ops.file.pack_all()
for image in bpy.data.images:
 if image.source=='FILE':
  if not image.packed_file:raise RuntimeError('Unpacked texture: '+image.name)
rig['asset_license']='CC0-1.0';rig['source']='MakeHuman MPFB2 '+COMMIT
rig['coordinate_convention']='meters; Z up; +Y forward; anatomical left -X; shoe soles Z=0'
rig['motion_status']='Unanimated skinned human asset. No motion or dynamics approval.'
collection=bpy.data.collections.new('Human');bpy.context.scene.collection.children.link(collection)
for obj in [rig]+objects:
 for c in list(obj.users_collection):c.objects.unlink(obj)
 collection.objects.link(obj)
# Capture rest metadata before a separate preview-only relaxed pose.
def arr(m):return [[float(v) for v in row] for row in m]
bones=[]
for b in rig.data.bones:
 bones.append({'name':b.name,'parent':b.parent.name if b.parent else None,'head':list(b.head_local),'tail':list(b.tail_local),'matrix_local':arr(b.matrix_local),'use_deform':b.use_deform})
meshmeta=[]
for obj in objects:
 deform={b.name for b in rig.data.bones if b.use_deform};groups={g.index:g.name for g in obj.vertex_groups}
 totals=[sum(g.weight for g in v.groups if groups[g.group] in deform) for v in obj.data.vertices]
 meshmeta.append({'name':obj.name,'vertices':len(obj.data.vertices),'faces':len(obj.data.polygons),'armature_modifiers':[m.object.name for m in obj.modifiers if m.type=='ARMATURE' and m.object],'unweighted_vertices':sum(x<1e-6 for x in totals),'deform_weight_sum_min':min(totals,default=0),'deform_weight_sum_max':max(totals,default=0),'bounds_world':bounds(obj)})
metadata={'schema':'doorbench.cc0-human-rig.v1','name':'Neutral adult / MPFB default163','unit':'m','up_axis':'Z','forward_axis':'+Y','left_axis':'-X','target_barefoot_height_m':1.75,'macro_parameters':macro,'clothing_fit_adjustments':['Removed stock shoes01 sock vertices above world Z=0.115m, fully hidden by fitted jeans; no body or rig dimensions changed.'],'rig_object':rig.name,'rest_world_matrix':arr(rig.matrix_world),'bones':bones,'meshes':meshmeta,'anatomical_landmarks':{'head':'head','left_palm':'wrist.L','right_palm':'wrist.R','left_foot':'foot.L','right_foot':'foot.R','left_toes':['toe1-1.L','toe2-1.L','toe3-1.L','toe4-1.L','toe5-1.L'],'right_toes':['toe1-1.R','toe2-1.R','toe3-1.R','toe4-1.R','toe5-1.R']},'notes':['Palm names identify skeletal wrist-to-hand segments, not contact points. Derive palm surface frames from skinned vertices.','163-bone rest hierarchy has authored deformation weights, fingers, individual toes and facial bones.','No collision shapes, mass/inertial model, motor controller, or motion validation is supplied by this asset.','Core CC0 assets are historically authored meshes/textures, not a photogrammetric human scan.']}
(OUT/'rig.json').write_text(json.dumps(metadata,indent=2)+'\n')
# Save the canonical unposed asset without stage objects.
bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'human-neutral.blend'))
# Relax the arms in a preview copy, retaining the original rest rig unchanged.
for side,x in [('L',-.23),('R',.23)]:
 p=rig.pose.bones['upperarm01.'+side];r=p.bone.matrix_local.to_quaternion()
 target=Vector((x,0,-.973)).normalized();q=(p.bone.tail_local-p.bone.head_local).normalized().rotation_difference(target)
 p.rotation_mode='QUATERNION';p.rotation_quaternion=r.inverted()@q@r
bpy.context.view_layer.update()
scene=bpy.context.scene;scene.unit_settings.system='METRIC';scene.render.engine='CYCLES';scene.cycles.samples=96;scene.cycles.use_denoising=True
try:
 prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='METAL';prefs.get_devices()
 for d in prefs.devices:d.use=d.type=='METAL'
 scene.cycles.device='GPU'
except Exception as e:print('CPU fallback:',repr(e))
scene.render.resolution_x=1000;scene.render.resolution_y=1400;scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=0
world=bpy.data.worlds.new('Neutral studio');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(.6,.65,.72,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.22;scene.world=world
mat=bpy.data.materials.new('Studio warm grey');mat.use_nodes=True;bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(.19,.21,.23,1);bs.inputs['Roughness'].default_value=.78
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.003));bpy.context.object.name='Studio.floor';bpy.context.object.data.materials.append(mat)
def area(name,pos,energy,size,color):
 d=bpy.data.lights.new(name,'AREA');d.energy=energy;d.shape='DISK';d.size=size;d.color=color;o=bpy.data.objects.new(name,d);scene.collection.objects.link(o);o.location=pos;o.rotation_euler=(Vector((0,0,1))-o.location).to_track_quat('-Z','Y').to_euler()
area('Key softbox',(-2.8,3.6,4.1),500,3,(1,.93,.87));area('Fill softbox',(3.0,1.0,2.7),190,2.5,(.86,.93,1));area('Rim',(-1.3,-2.3,3.4),330,2.0,(1,.96,.92))
cam=bpy.data.cameras.new('Preview.camera');co=bpy.data.objects.new('Preview.camera',cam);scene.collection.objects.link(co);co.location=(2.7,5.6,2.5);co.rotation_euler=(Vector((0,0,.91))-co.location).to_track_quat('-Z','Y').to_euler();cam.type='ORTHO';cam.ortho_scale=2.18;cam.lens=70;scene.camera=co
scene.render.image_settings.file_format='PNG';scene.render.filepath=str(OUT/'neutral-full-body.png');bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'human-preview.blend'))
if not a.no_render:bpy.ops.render.render(write_still=True)
# Provenance captures all selected source files, package hashes and output bytes.
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
used=set()
for p in inputs:
 for f in p.parent.iterdir():
  if f.is_file():used.add(f)
for rel in ['data/3dobjs/base.obj','data/rigs/standard/rig.default.json','data/rigs/standard/weights.default.json']:
 used.add(TOOL/'source'/('mpfb2-'+COMMIT)/'src/mpfb'/rel)
manifest={'schema':'doorbench.human-asset-provenance.v1','asset_license':'CC0-1.0','tool_license':'GPL-3.0-or-later; external build tool, not relicensed or copied into repository code','source_commit':COMMIT,'license_urls':['https://github.com/makehumancommunity/mpfb2/blob/'+COMMIT+'/LICENSE.md','https://github.com/makehumancommunity/mpfb2/blob/'+COMMIT+'/LICENSE.ASSETS.md','https://static.makehumancommunity.org/assets/assetpacks/makehuman_system_assets.html'],'downloads':[{'file':str(TOOL/name),'url':url,'bytes':(TOOL/name).stat().st_size,'sha256':digest} for name,url,digest in PACKAGES],'selected_inputs':[{'path':str(p.relative_to(TOOL)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(used)],'construction_script':{'path':str(pathlib.Path(__file__).resolve()),'sha256':sha(pathlib.Path(__file__))},'blender_version':bpy.app.version_string,'outputs':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name in ({'human-neutral.blend','human-preview.blend','rig.json'} | ({'neutral-full-body.png'} if not a.no_render else set()))],'review_status':'Asset construction and static preview only; no human motion ground truth or physical approval.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('COMPLETE',json.dumps({'bones':len(bones),'meshes':meshmeta,'outputs':manifest['outputs']}),flush=True)
