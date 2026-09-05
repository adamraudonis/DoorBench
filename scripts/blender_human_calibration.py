"""Create an explicit target calibration pose, preserving the MPFB rest rig.
This is an asset calibration action, not motion synthesis or source retargeting.
"""
import bpy,json,pathlib,math,hashlib,sys,argparse
from mathutils import Matrix,Vector
REPO=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from doorbench.human_reference.bvh import read_bvh
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from setup_human_reference import BVH_SHA256
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source',type=pathlib.Path,required=True)
ap.add_argument('--out',type=pathlib.Path,required=True)
ap.add_argument('--no-render',action='store_true')
a=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
OUT=a.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
rig=bpy.data.objects['Human.rig']
# Bind metadata to the actual loaded scene before modifying its calibration pose.
rig_path=OUT/'rig.json';rig_metadata=json.loads(rig_path.read_text())
manifest=json.loads((OUT/'manifest.json').read_text())
outputs={item['path']:item for item in manifest['outputs']}
loaded=pathlib.Path(bpy.data.filepath).resolve()
for file in [loaded,rig_path]:
 if file.name not in outputs or hashlib.sha256(file.read_bytes()).hexdigest()!=outputs[file.name]['sha256']:
  raise ValueError('Loaded human scene/rig metadata differs from asset manifest: '+str(file))
def same_values(a,b,tolerance=1e-7):
 return len(a)==len(b) and all(abs(float(x)-float(y))<=tolerance for x,y in zip(a,b))
def same_matrix(a,b):
 return len(a)==len(b) and all(same_values(x,y) for x,y in zip(a,b))
if rig_metadata.get('schema')!='doorbench.cc0-human-rig.v1' or rig_metadata.get('rig_object')!=rig.name or not same_matrix(rig.matrix_world,rig_metadata['rest_world_matrix']):
 raise ValueError('Loaded canonical rig identity/world transform differs from rig.json')
expected={bone['name']:bone for bone in rig_metadata['bones']}
if len(expected)!=len(rig_metadata['bones']) or set(expected)!=set(rig.data.bones.keys()):
 raise ValueError('Loaded canonical bone names differ from rig.json')
for bone in rig.data.bones:
 e=expected[bone.name]
 if e['parent']!=(bone.parent.name if bone.parent else None) or e['use_deform']!=bone.use_deform or not same_matrix(bone.matrix_local,e['matrix_local']) or not same_values(bone.head_local,e['head']) or not same_values(bone.tail_local,e['tail']):
  raise ValueError('Loaded canonical rest bone differs from rig.json: '+bone.name)
meshes={obj.name:obj for obj in bpy.data.collections['Human'].objects if obj.type=='MESH'}
if set(meshes)!={mesh['name'] for mesh in rig_metadata['meshes']}:
 raise ValueError('Loaded canonical mesh names differ from rig.json')
for m in rig_metadata['meshes']:
 obj=meshes[m['name']]
 if len(obj.data.vertices)!=m['vertices'] or len(obj.data.polygons)!=m['faces'] or [mod.object.name for mod in obj.modifiers if mod.type=='ARMATURE' and mod.object]!=m['armature_modifiers']:
  raise ValueError('Loaded canonical mesh topology/rig binding differs from rig.json: '+obj.name)
# Run from the studio blend but clear its preview-only pose.
rig.animation_data_clear()
for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update()
source_path=a.source.resolve()
source={'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest()}
if source['source_sha256']!=BVH_SHA256:
 raise ValueError('This calibration supports only the pinned CeTI d02/o03/run01 sample; source SHA differs')
capture=read_bvh(source_path)
if len(capture.values)<2 or not (capture.values[0]==capture.values[1]).all():
 raise ValueError('Expected the two identical leading CeTI calibration rows')
joints={}
for bone in capture.joints:
 if not bone.end_site:joints[bone.name]={'offset_cm':bone.offset}
 else:joints[capture.joints[bone.parent].name]['end_site_offset_cm']=bone.offset
B=Matrix(((-1,0,0),(0,0,1),(0,1,0)))
def orient(name,direction):
 p=rig.pose.bones[name];before=p.matrix.copy();head=before.translation.copy()
 current=before.to_3x3()@Vector((0,1,0));r=current.normalized().rotation_difference(Vector(direction).normalized()).to_matrix().to_4x4()
 p.matrix=Matrix.Translation(head)@r@Matrix.Translation(-head)@before
 bpy.context.view_layer.update()
def basis(forward,radial):
 a=Vector(forward).normalized();b=Vector(radial);b=(b-a*b.dot(a)).normalized();c=a.cross(b).normalized()
 return Matrix((a,b,c)).transposed()
for side,sign,source_side in [('L',-1,'Left'),('R',1,'Right')]:
 # Match the source's straight calibration legs without changing hip width,
 # bone lengths, or the foot's anatomically neutral world orientation.
 neutral_foot_rotation=rig.pose.bones['foot.'+side].matrix.to_3x3().copy()
 upper_direction=B@Vector(joints['Character1_'+source_side+'Leg']['offset_cm'])
 lower_direction=B@Vector(joints['Character1_'+source_side+'Foot']['offset_cm'])
 for part in ['upperleg01','upperleg02']:orient(part+'.'+side,upper_direction)
 for part in ['lowerleg01','lowerleg02']:orient(part+'.'+side,lower_direction)
 foot=rig.pose.bones['foot.'+side];new_foot_head=foot.head.copy()
 foot.matrix=Matrix.Translation(new_foot_head)@neutral_foot_rotation.to_4x4()
 bpy.context.view_layer.update()
 lateral=Vector((sign,0,0))
 for part in ['upperarm01','upperarm02','lowerarm01','lowerarm02']:orient(part+'.'+side,lateral)
 wrist=rig.pose.bones['wrist.'+side];middle=rig.pose.bones['finger3-1.'+side];index=rig.pose.bones['finger2-1.'+side];pinky=rig.pose.bones['finger5-1.'+side]
 old=basis(middle.head-wrist.head,index.head-pinky.head)
 prefix='Character1_'+source_side+'Hand'
 f=B@Vector(joints[prefix+'Middle1']['offset_cm']);radial=B@(Vector(joints[prefix+'Index1']['offset_cm'])-Vector(joints[prefix+'Pinky1']['offset_cm']))
 new=basis(f,radial);R=(new@old.transposed()).to_4x4();head=wrist.head.copy()
 wrist.matrix=Matrix.Translation(head)@R@Matrix.Translation(-head)@wrist.matrix
 bpy.context.view_layer.update()
 # Source fingers have two measured/retargeted joints. Target extra segments
 # share calibration direction only; no independent source measurement implied.
 for number,finger in [(1,'Thumb'),(2,'Index'),(3,'Middle'),(4,'Ring'),(5,'Pinky')]:
  d1=B@Vector(joints[prefix+finger+'2']['offset_cm']);d2=B@Vector(joints[prefix+finger+'2']['end_site_offset_cm'])
  for seg,d in [(1,d1),(2,d1),(3,d2)]:orient(f'finger{number}-{seg}.{side}',d)
# Key every local channel at frame 1, so the exact calibration is portable.
for p in rig.pose.bones:
 p.rotation_mode='QUATERNION'
 p.keyframe_insert('location',frame=1,group=p.name);p.keyframe_insert('rotation_quaternion',frame=1,group=p.name);p.keyframe_insert('scale',frame=1,group=p.name)
rig.animation_data.action.name='Calibration.TPose.CeTI_d02'
bpy.context.scene.frame_start=bpy.context.scene.frame_end=1;bpy.context.scene.frame_set(1);bpy.context.view_layer.update()
def matrix(m):return [[float(v) for v in row] for row in m]
checks={}
for side,sign in [('L',-1),('R',1)]:
 for part in ['upperarm01','upperarm02','lowerarm01','lowerarm02']:
  p=rig.pose.bones[part+'.'+side];v=(p.tail-p.head).normalized();err=math.degrees(v.angle(Vector((sign,0,0))))
  assert err<.02,(p.name,err);checks[p.name+'_lateral_error_deg']=err
for side,source_side in [('L','Left'),('R','Right')]:
 for part,key in [('upperleg01','Leg'),('upperleg02','Leg'),('lowerleg01','Foot'),('lowerleg02','Foot')]:
  p=rig.pose.bones[part+'.'+side];expected=(B@Vector(joints['Character1_'+source_side+key]['offset_cm'])).normalized();err=math.degrees((p.tail-p.head).normalized().angle(expected))
  assert err<.02,(p.name,err);checks[p.name+'_source_direction_error_deg']=err
 foot=rig.pose.bones['foot.'+side];angle=math.degrees(foot.matrix.to_quaternion().rotation_difference(foot.bone.matrix_local.to_quaternion()).angle)
 assert angle<.02;checks['foot.'+side+'_neutral_orientation_error_deg']=angle
records=[]
for p in rig.pose.bones:
 records.append({'name':p.name,'parent':p.parent.name if p.parent else None,'head':list(p.head),'tail':list(p.tail),'matrix_world':matrix(rig.matrix_world@p.matrix),'matrix_armature':matrix(p.matrix),'matrix_basis':matrix(p.matrix_basis),'rest_matrix_local':matrix(p.bone.matrix_local),'rotation_delta_armature':matrix(p.matrix.to_3x3()@p.bone.matrix_local.to_3x3().inverted())})
result={'schema':'doorbench.human-calibration-pose.v1','source_bvh_sha256':source['source_sha256'],'source_calibration_frames':[0,1],'source_attribution':{'dataset':'CeTI-Age-Kinematics v2','authors':['Loreen Pogrzeba','Evelyn Muschter','Simon Hanisch','Veronica Y. P. Wardhani','Thorsten Strufe','Frank H.P. Fitzek','Shu-Chen Li'],'doi':'10.6084/m9.figshare.26983645.v2','license':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/','modifications':'Converted source calibration offset directions into target metres/Z-up/+Y-forward frame; applied target-only bone rotations without changing target lengths/rest data. No recorded dynamic frame is included.'},'target_rig_json_sha256':hashlib.sha256((OUT/'rig.json').read_bytes()).hexdigest(),'action':rig.animation_data.action.name,'frame':1,'unit':'m','up_axis':'Z','forward_axis':'+Y','left_axis':'-X','bones':records,'checks':checks,'scope':['Arms straight lateral; hand plane and finger directions derived from the source calibration offsets.','Legs align to source calibration directions at fixed target hip width/lengths; foot global orientation stays anatomical-neutral. Spine/head retain authored rest. Full transforms are supplied; never assume equal source and target rest rotations.','Target bone lengths and original rest matrices are unchanged. Source root height and limb lengths are not copied.','Target third finger segment has no independent corresponding source measurement.','Calibration action only; no source motion retargeting, contact, force, balance or naturalness approval.']}
(OUT/'tpose-calibration.json').write_text(json.dumps(result,indent=2)+'\n')
scene=bpy.context.scene;scene.camera.location=(0,5.6,2.1);scene.camera.rotation_euler=(Vector((0,0,.92))-scene.camera.location).to_track_quat('-Z','Y').to_euler();scene.camera.data.ortho_scale=2.13
# Ground the studio against the posed shoes; preserve the rig's calibrated
# root location and expose the resulting floor plane in the metadata.
shoe=bpy.data.objects['Human.shoes01'].evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=shoe.to_mesh()
try:floor_z=min((shoe.matrix_world@v.co).z for v in mesh.vertices)
finally:shoe.to_mesh_clear()
bpy.data.objects['Studio.floor'].location.z=floor_z-.003
result['calibration_floor_z_m']=floor_z
(OUT/'tpose-calibration.json').write_text(json.dumps(result,indent=2)+'\n')
scene.render.resolution_x=1400;scene.render.resolution_y=1400;scene.render.filepath=str(OUT/'tpose-calibration.png')
scene.cycles.samples=64
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'human-tpose-calibration.blend'))
if not a.no_render:bpy.ops.render.render(write_still=True)
print('CALIBRATION',json.dumps(checks))
