import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { buildScene, geomMesh, type BuiltScene } from './scene';
import type { GeomJ, ModelJ } from './types';
import { frameAt } from './referenceMotion';

export interface WebFile {path:string;sha256:string;bytes:number;json_sha256?:string}
export const SOURCE_SCENARIOS=['open_and_traverse','unlock_and_traverse','locked_recognize'] as const;
export type SourceScenario=typeof SOURCE_SCENARIOS[number];
export interface MotionEntry {door_id:string;family:string;status:string;source_scenario?:SourceScenario|null;reason?:string|null;reason_code?:string|null;failure_counts?:Record<string,number>|null;identity_sha256?:string;clip:(WebFile&{duration:number;frames:number})|null;audits:Record<string,WebFile>}
export interface MotionIndex {schema:string;snapshot_id:string;updated_at:string;manifest_sha256:string;scope:string;doors:MotionEntry[];counts:Record<string,number>}
export interface ActorGeom {name:string;body_name:string;type:'box'|'sphere'|'capsule'|'cylinder';size:number[];pos:number[];quat_wxyz:number[]}
interface BodyFrames {body_names:string[];poses:number[][]}
export interface PlannedClip {schema:string;door_id:string;status:string;source_scenario:SourceScenario;scope:string;identity_sha256:string;source_sha256:Record<string,string>;native_resources_sha256:Record<string,string>;duration:number;times:number[];native_time:number[];phases:string[];foot_contact:number[][];hand_contact:number[][];native:BodyFrames;actor:BodyFrames&{geometries:ActorGeom[]}}
export function motionTaskLabel(scenario:SourceScenario|null|undefined) {return scenario==='locked_recognize'?'Locked-door check':scenario==='open_and_traverse'||scenario==='unlock_and_traverse'?'Traversal reference':'Source task unavailable';}
export function motionTaskDetail(scenario:SourceScenario|null|undefined) {
  if(scenario==='locked_recognize')return 'This reference checks a locked door and does not traverse it. Recognition is declared by the source benchmark; it is not independently demonstrated by the actor.';
  if(scenario==='unlock_and_traverse')return 'Source scenario: unlock and traverse. Door motion is prescribed from the source recording; actor unlocking and mechanism operation are not independently certified.';
  if(scenario==='open_and_traverse')return 'Source scenario: open and traverse. Door motion is prescribed from the source recording; this is an actor route reference, not evidence of causal humanoid operation.';
  return 'No bound source scenario is available for this attempt.';
}
const HASH=/^[a-f0-9]{64}$/;
const MAX_PACKED=64*1024*1024,MAX_DECODED=256*1024*1024,MAX_SOURCE=32*1024*1024;
const finite=(row:unknown,n:number):row is number[]=>Array.isArray(row)&&row.length===n&&row.every(x=>typeof x==='number'&&Number.isFinite(x));
function require(ok:unknown,message:string):asserts ok {if(!ok)throw Error(message);}
export const sha256=async(bytes:ArrayBuffer)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(v=>v.toString(16).padStart(2,'0')).join('');
export function artifactURL(path:string,indexURL:string) {
  require(/^(clips|audits)\/[A-Za-z0-9_./-]+$/.test(path)&&!path.split('/').includes('..')&&!path.includes('//'),'Unsafe motion artifact path');
  return new URL(path,indexURL).href;
}
export function validateMotionIndex(value:MotionIndex) {
  require(value?.schema==='doorbench.planned-reference-web-index.v1'&&Array.isArray(value.doors)&&HASH.test(value.manifest_sha256),'Unsupported motion index');
  const ids=new Set<string>();
  for(const row of value.doors) {
    require(/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(row.door_id)&&!ids.has(row.door_id),'Invalid or duplicate motion door ID');ids.add(row.door_id);
    require(['accepted_kinematic','rejected','unresolved'].includes(row.status)&&typeof row.family==='string','Invalid motion status');
    require(row.source_scenario==null||SOURCE_SCENARIOS.includes(row.source_scenario),'Unsupported bound source scenario');
    require(row.status==='accepted_kinematic'?!!row.clip:!row.clip,'Only accepted clips may be playable');
    if(row.clip){require(SOURCE_SCENARIOS.includes(row.source_scenario as SourceScenario),'Accepted clip lacks a bound source scenario');require(HASH.test(row.identity_sha256??'')&&HASH.test(row.clip.json_sha256??'')&&Number.isSafeInteger(row.clip.frames)&&row.clip.frames>=2&&row.clip.frames<=100000&&Number.isFinite(row.clip.duration)&&row.clip.duration>0&&row.clip.bytes<=MAX_PACKED,'Invalid accepted clip identity or size');}
    for(const f of [...Object.values(row.audits??{}),...(row.clip?[row.clip]:[])]) {
      require(HASH.test(f.sha256)&&Number.isSafeInteger(f.bytes)&&f.bytes>0,'Invalid artifact checksum');artifactURL(f.path,'https://example.test/index.json');
    }
  }
  return value;
}
export function validatePlannedClip(c:PlannedClip,entry:MotionEntry) {
  require(c?.schema==='doorbench.planned-reference-web.v1'&&c.door_id===entry.door_id&&c.status==='accepted_kinematic'&&entry.status==='accepted_kinematic'&&HASH.test(c.identity_sha256)&&c.identity_sha256===entry.identity_sha256,'Clip identity or accepted status mismatch');
  require(SOURCE_SCENARIOS.includes(c.source_scenario)&&c.source_scenario===entry.source_scenario,'Clip/index source scenario mismatch');
  const n=c.times?.length;
  require(n>=2&&n<=100000&&c.times[0]===0&&c.times.every((t,i)=>Number.isFinite(t)&&(i===0||t>c.times[i-1]))&&Math.abs(c.times[n-1]-c.duration)<1e-6,'Invalid motion timeline');
  require(n===entry.clip?.frames&&c.duration===entry.clip.duration,'Index/clip duration or frame count mismatch');
  require(c.native_time?.length===n&&c.native_time.every((t,i)=>Number.isFinite(t)&&t>=0&&(i===0||t>=c.native_time[i-1])),'Invalid source time');
  require(c.phases?.length===n&&c.phases.every(p=>typeof p==='string'),'Invalid motion phases');
  for(const contact of [c.foot_contact,c.hand_contact])require(contact?.length===n&&contact.every(r=>finite(r,2)&&r.every(v=>v===0||v===1)),'Invalid contact labels');
  for(const group of [c.native,c.actor]) {
    require(Array.isArray(group?.body_names)&&group.body_names.length>0&&new Set(group.body_names).size===group.body_names.length&&group.body_names.every(x=>typeof x==='string'),'Invalid body names');
    require(n*group.body_names.length*7<=16_000_000,'Motion exceeds browser pose memory limit');
    require(group.poses?.length===n&&group.poses.every(row=>finite(row,group.body_names.length*7)),'Invalid body pose dimensions');
    for(const row of group.poses)for(let k=0;k<group.body_names.length;k++)require(Math.abs(Math.hypot(...row.slice(k*7+3,k*7+7))-1)<1e-5,'Nonunit body quaternion');
  }
  require(c.actor.body_names.length===16&&Array.isArray(c.actor.geometries)&&c.actor.geometries.length>0&&c.actor.geometries.length<=1024,'Unsupported articulated actor');
  const geoms=new Set<string>();
  for(const g of c.actor.geometries) {
    require(typeof g.name==='string'&&!geoms.has(g.name)&&c.actor.body_names.includes(g.body_name),'Duplicate geometry or unknown body');geoms.add(g.name);
    require(['box','sphere','capsule','cylinder'].includes(g.type)&&finite(g.size,3)&&finite(g.pos,3)&&finite(g.quat_wxyz,4)&&Math.abs(Math.hypot(...g.quat_wxyz)-1)<1e-5,'Invalid actor geometry');
    require(g.size[0]>0&&(g.type==='sphere'||g.size[1]>0)&&(g.type!=='box'||g.size[2]>0),'Invalid actor primitive size');
  }
  require(JSON.stringify(Object.keys(c.source_sha256??{}).sort())===JSON.stringify(['door.xml','model.json','spec.json'])&&Object.values(c.source_sha256).every(x=>HASH.test(x)),'Missing source bindings');
  return c;
}
async function boundedBytes(stream:ReadableStream<Uint8Array>|null,limit:number) {
  require(stream,'Empty download body');const reader=stream.getReader(),chunks:Uint8Array<ArrayBuffer>[]= [];let size=0;
  try{while(true){const {done,value}=await reader.read();if(done)break;size+=value.byteLength;require(size<=limit,'Download exceeds browser memory limit');chunks.push(new Uint8Array(value));}}
  catch(e){await reader.cancel();throw e;}finally{reader.releaseLock();}
  return new Blob(chunks).arrayBuffer();
}
export async function fetchPlannedClip(entry:MotionEntry,indexURL:string,signal?:AbortSignal) {
  require(entry.status==='accepted_kinematic'&&entry.clip,'This attempt is not accepted for playback');
  const response=await fetch(artifactURL(entry.clip.path,indexURL),{signal});require(response.ok,`Motion download failed (${response.status})`);
  const bytes=await boundedBytes(response.body,MAX_DECODED);const digest=await sha256(bytes);let raw:ArrayBuffer;
  if(digest===entry.clip.sha256){require(bytes.byteLength===entry.clip.bytes&&bytes.byteLength<=MAX_PACKED,'Motion compressed size mismatch');raw=await boundedBytes(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')),MAX_DECODED);}
  else {require(digest===entry.clip.json_sha256,'Motion file checksum mismatch');raw=bytes;} // Hosts may decode Content-Encoding.
  require(await sha256(raw)===entry.clip.json_sha256,'Decoded motion checksum mismatch');
  const clip=validatePlannedClip(JSON.parse(new TextDecoder().decode(raw)),entry);
  const files=new Map<string,ArrayBuffer>();
  await Promise.all([...Object.entries(clip.source_sha256).map(([name,hash])=>[`doors/${entry.door_id}/${name}`,hash]),...Object.entries(clip.native_resources_sha256??{})].map(async([path,hash])=>{
    require(/^(doors\/[A-Za-z0-9_.-]+\/(model.json|spec.json|door.xml)|hardware\/[A-Za-z0-9_.-]+\.obj)$/.test(path)&&HASH.test(hash),'Unsafe source resource');
    const r=await fetch(`./assets/${path}`,{signal,cache:'no-cache'});require(r.ok,`Cannot verify ${path}`);const data=await boundedBytes(r.body,MAX_SOURCE);require(await sha256(data)===hash,`Motion does not match served ${path}`);files.set(path,data);
  }));
  const model=JSON.parse(new TextDecoder().decode(files.get(`doors/${entry.door_id}/model.json`)!)) as ModelJ;
  for(const body of model.bodies)require(clip.native.body_names.includes(body.name)||(body.name==='world_env'&&clip.native.body_names.includes('world')),'Snapshot lacks a source body');
  for(const body of model.bodies)for(const geom of body.geoms)if(geom.visual&&geom.type==='mesh')require(files.has(`hardware/${geom.mesh_name}.obj`),'Snapshot lacks verified mesh bytes');
  return {clip,model,files};
}

/** Build primitives with the shared scene, then parse hardware from verified bytes.
 * This bypasses the catalogue OBJ cache so it cannot supply stale mesh geometry. */
export async function buildVerifiedDoor(model:ModelJ,files:Map<string,ArrayBuffer>) {
  const bare={...model,bodies:model.bodies.map(b=>({...b,geoms:b.geoms.filter(g=>g.type!=='mesh')}))};
  const built=await buildScene(bare,{showEnv:true});const materials:THREE.Material[]=[];const loader=new OBJLoader();
  try {
    for(const body of model.bodies)for(const geom of body.geoms)if(geom.visual&&geom.type==='mesh') {
      const bytes=files.get(`hardware/${geom.mesh_name}.obj`);require(bytes,'Missing verified hardware');
      const group=loader.parse(new TextDecoder().decode(bytes));const source=model.materials[geom.material];const color=source?.rgba??[.7,.7,.7,1];
      const transparent=(source?.transparent??false)||color[3]<.99;
      const material=new THREE.MeshStandardMaterial({color:new THREE.Color(...color.slice(0,3) as [number,number,number]),roughness:source?.roughness??.6,metalness:source?.metallic??0,
        transparent,opacity:transparent?Math.min(color[3],.55):1,side:transparent?THREE.DoubleSide:THREE.FrontSide,depthWrite:!transparent});
      if(source?.emissive)material.emissive.fromArray(source.emissive);materials.push(material);
      group.name=geom.name;group.position.fromArray(geom.pos);group.quaternion.set(geom.quat[1],geom.quat[2],geom.quat[3],geom.quat[0]);
      let count=0;group.traverse(o=>{if(o instanceof THREE.Mesh){o.material=material;o.castShadow=true;o.receiveShadow=true;o.userData={semantic:geom.semantic};count++;}});require(count>0,'Empty verified mesh');
      (built.bodies.get(body.name)!.userData.container as THREE.Object3D).add(group);
    }
    built.root.updateMatrixWorld(true);
    built.root.traverse(o=>{if(o instanceof THREE.Mesh&&!['floor','wall'].includes(o.userData.semantic)){o.geometry.computeBoundingBox();built.bounds.union(o.geometry.boundingBox!.clone().applyMatrix4(o.matrixWorld));}});
    const dispose=built.dispose;built.dispose=()=>{dispose();materials.forEach(m=>m.dispose());};return built;
  }catch(e){built.dispose();materials.forEach(m=>m.dispose());throw e;}
}

export function applyBodyWorld(container:THREE.Object3D,position:THREE.Vector3,quaternion:THREE.Quaternion) {
  container.parent?.updateWorldMatrix(true,false);
  const world=new THREE.Matrix4().compose(position,quaternion,new THREE.Vector3(1,1,1));
  if(container.parent)world.premultiply(container.parent.matrixWorld.clone().invert());
  world.decompose(container.position,container.quaternion,container.scale);container.updateMatrixWorld(true);
}
export function buildPlannedPlayer(clip:PlannedClip,built:BuiltScene) {
  const group=new THREE.Group();group.name='planned_actor';const bodies=new Map(clip.actor.body_names.map(name=>[name,new THREE.Group()]));
  const material=new THREE.MeshStandardMaterial({color:0x4b797d,roughness:.6,metalness:.08});
  bodies.forEach((body,name)=>{body.name=name;group.add(body);});
  for(const g of clip.actor.geometries) {
    const object=geomMesh({...g,quat:g.quat_wxyz,semantic:'actor',visual:true,collision:false} as unknown as GeomJ,material);
    require(object,'Unsupported actor geometry');bodies.get(g.body_name)!.add(object);
  }
  const targets=[...built.bodies.entries()].map(([name,node])=>{
    const mapped=name==='world_env'&&!clip.native.body_names.includes(name)?'world':name;
    const index=clip.native.body_names.indexOf(mapped);require(index>=0,`Missing native body ${name}`);
    const container=node.userData.container as THREE.Object3D;let depth=0;for(let p=container.parent;p;p=p.parent)depth++;
    return {container,index,depth};
  }).sort((a,b)=>a.depth-b.depth);
  const position=new THREE.Vector3(),other=new THREE.Vector3(),quat=new THREE.Quaternion(),qOther=new THREE.Quaternion();
  const pose=(rows:number[][],index:number,a:number,b:number,s:number)=>{
    const offset=index*7,first=rows[a],second=rows[b];position.fromArray(first,offset).lerp(other.fromArray(second,offset),s);
    quat.set(first[offset+4],first[offset+5],first[offset+6],first[offset+3]);qOther.set(second[offset+4],second[offset+5],second[offset+6],second[offset+3]);quat.slerp(qOther,s);
  };
  const bounds=built.bounds.clone();for(const row of clip.actor.poses)for(let k=0;k<clip.actor.body_names.length;k++)bounds.expandByPoint(new THREE.Vector3().fromArray(row,k*7));bounds.expandByScalar(.2);
  function setTime(t:number) {
    const [a,b,s]=frameAt(clip.times,t);
    for(const target of targets){pose(clip.native.poses,target.index,a,b,s);applyBodyWorld(target.container,position,quat);}
    clip.actor.body_names.forEach((name,k)=>{pose(clip.actor.poses,k,a,b,s);const node=bodies.get(name)!;node.position.copy(position);node.quaternion.copy(quat);});
    group.updateMatrixWorld(true);return {frame:a,phase:clip.phases[a],sourceTime:clip.native_time[a]+(clip.native_time[b]-clip.native_time[a])*s};
  }
  setTime(0);
  return {group,bounds,setTime,dispose:()=>{group.traverse(o=>{if(o instanceof THREE.Mesh)o.geometry.dispose();});material.dispose();}};
}
