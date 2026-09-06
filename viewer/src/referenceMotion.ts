import { isPetDoorId, PET_COLLECTION_REASON } from "./collections";
import * as THREE from 'three';
import type { BuiltScene } from './scene';
import type { NativeCableFrame } from './cableRouting';
import {applyBodyWorld} from './nativeBodyPose';

export interface ReferenceClip {
  schema: string; door_id: string; scenario: string; duration: number; lead_in_s: number;
  joint_names: string[]; avatar_joint_names: string[]; avatar_bones: number[][];
  times: number[]; door_q: number[][]; avatar: number[][]; targets: number[][];
  hand_active: number[]; hand_error_m: number[]; phases: string[];
  outcome: {success: boolean; outcome: string; sim_time: number};
  max_hand_error_m: number; unreachable_frames: number;
  source_sha256: Record<string,string>;
  native_cables?: NativeCableFrame[];
  native?: {body_names:string[];poses?:number[][];joint_types?:string[];qpos_widths?:number[];qvel_widths?:number[]};
  oracle_contacts?: {site:string|null; joint:string|null; position:number[]|null; force_N?:number[]; torque_Nm?:number[]}[][];
}
export const isNativeReference=(clip:Pick<ReferenceClip,'schema'>)=>clip.schema==='doorbench.native-motion.v1';
export function frameAt(times: number[], t: number): [number,number,number] {
  let lo=0,hi=times.length-1;
  while(lo<hi) { const mid=Math.ceil((lo+hi)/2); if(times[mid]<=t) lo=mid; else hi=mid-1; }
  const next=Math.min(lo+1,times.length-1), span=times[next]-times[lo];
  return [lo,next,span>0?Math.max(0,Math.min(1,(t-times[lo])/span)):0];
}
export function validateClip(c: ReferenceClip,id:string) {
  const n=c.times?.length;
  const native=isNativeReference(c);
  if(native) {
    const poses=c.native?.poses;
    if(poses&&(poses.length!==n||poses.some(row=>row.length!==7*c.native!.body_names.length||row.some(v=>!Number.isFinite(v))))) throw Error('Invalid native body timeline');
    if(c.native?.joint_types?.includes('mjJNT_FREE')&&!poses) throw Error('Free roots require native body poses');
    if(c.oracle_contacts?.length!==n||c.oracle_contacts.some(row=>!Array.isArray(row)||row.some(p=>
      (p.position!==null&&(p.position?.length!==3||p.position.some(v=>!Number.isFinite(v))))||
      (p.force_N!==undefined&&(p.force_N.length!==3||p.force_N.some(v=>!Number.isFinite(v))))||
      (p.torque_Nm!==undefined&&(p.torque_Nm.length!==3||p.torque_Nm.some(v=>!Number.isFinite(v))))))) throw Error('Invalid native contact timeline');
    // Normalize for the shared timeline without adding a synthetic actor to the recording.
    c={...c,lead_in_s:0,avatar_joint_names:[],avatar_bones:[],avatar:Array.from({length:n},()=>[]),
      hand_active:Array(n).fill(0),hand_error_m:Array(n).fill(0),max_hand_error_m:0,unreachable_frames:0};
  }
  const skeleton=['pelvis','chest','neck','head','shoulder_l','elbow_l','wrist_l','shoulder_r','elbow_r','wrist_r','hip_l','knee_l','ankle_l','hip_r','knee_r','ankle_r'];
  if(!native&&JSON.stringify(c.avatar_joint_names)!==JSON.stringify(skeleton)) throw Error('Unsupported reference skeleton');
  if((!native&&c.schema!=='doorbench.reference-motion.v1')||c.door_id!==id||!n||!Number.isFinite(c.duration)||c.duration<=0) throw Error('Invalid reference recording');
  for(const field of ['door_q','avatar','targets'] as const) {
    const width=field==='door_q'?c.joint_names.length:field==='avatar'?c.avatar_joint_names.length*3:3;
    if(c[field]?.length!==n||c[field].some(row=>row.length!==width||row.some(v=>!Number.isFinite(v)))) throw Error('Invalid reference frame dimensions');
  }
  if(c.times.some((v,i)=>!Number.isFinite(v)||(i>0&&v<=c.times[i-1]))||c.phases.length!==n||c.hand_error_m.length!==n||c.hand_active.length!==n) throw Error('Invalid reference timeline');
  if(c.avatar_bones.some(b=>b.length!==2||b.some(v=>!Number.isInteger(v)||v<0||v>=c.avatar_joint_names.length))) throw Error('Invalid reference skeleton');
  if(c.hand_error_m.some(v=>!Number.isFinite(v)||v<0)||c.hand_active.some(v=>v!==0&&v!==1)||Math.abs(c.duration-c.times[n-1])>.001) throw Error('Invalid reference diagnostics');
  return c;
}
export async function fetchReference(id:string,signal?:AbortSignal): Promise<ReferenceClip> {
  if (isPetDoorId(id)) throw new Error(PET_COLLECTION_REASON);
  const r=await fetch(`./reference-motions/clips/${id}.json.gz`,{signal});
  if(!r.ok) throw Error('Reference recording unavailable for this release');
  const stream=new Blob([await r.arrayBuffer()]).stream().pipeThrough(new DecompressionStream('gzip'));
  const clip=validateClip(await new Response(stream).json(),id);
  await Promise.all(['model.json','spec.json','door.xml'].map(async name=>{
    const source=await fetch(`./assets/doors/${id}/${name}`,{signal});
    if(!source.ok) throw Error(`Cannot verify reference source ${name}`);
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await source.arrayBuffer()))).map(v=>v.toString(16).padStart(2,'0')).join('');
    if(hash!==clip.source_sha256?.[name]) throw Error(`Reference does not match current ${name}. Regenerate reference motions.`);
  }));
  return clip;
}
export function buildReferencePlayer(clip:ReferenceClip) {
  const native=isNativeReference(clip);
  const group=new THREE.Group(); group.name=native?'native_mechanism_contacts':'reference_humanoid';
  const shell=new THREE.MeshStandardMaterial({color:0x5a8e99,roughness:.42,metalness:.16});
  const jointMat=new THREE.MeshStandardMaterial({color:0xc7dae0,roughness:.38});
  const dark=new THREE.MeshStandardMaterial({color:0x253c4b,roughness:.6});
  const warning=new THREE.MeshBasicMaterial({color:0xed7046});
  const bones=clip.avatar_bones.map(([a,b])=>{
    const radius=a===0&&b===1?.14:a===1&&b===2?.07:a>=10?.055:.037;
    const mesh=new THREE.Mesh(new THREE.CylinderGeometry(radius,radius,1,12),shell);
    mesh.castShadow=true; group.add(mesh); return mesh;
  });
  const joints=clip.avatar_joint_names.map((name,i)=>{
    const mesh=new THREE.Mesh(new THREE.SphereGeometry(i===3?.108:i===0?.11:.049,16,12),i===3?shell:jointMat);
    mesh.name=name;mesh.castShadow=true; group.add(mesh);return mesh;
  });
  const feet=(native?[]:[12,15]).map(()=>{const m=new THREE.Mesh(new THREE.BoxGeometry(.105,.22,.065),dark);group.add(m);return m;});
  const contactMarkers=Array.from({length:Math.max(0,...(clip.oracle_contacts||[]).map(row=>row.length))},()=>{
    const marker=new THREE.Mesh(new THREE.SphereGeometry(.014,12,8),warning);
    const arrow=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(),.2,0xed7046,.035,.02);
    const torqueAxis=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(),.1,0x68b8ef,.025,.015);
    group.add(marker,arrow,torqueAxis);return {marker,arrow,torqueAxis};
  });
  const target=new THREE.Mesh(new THREE.SphereGeometry(.024,12,8),warning);group.add(target);
  const lineGeo=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);
  const lineMat=new THREE.LineDashedMaterial({color:0xed7046,dashSize:.045,gapSize:.025});
  const line=new THREE.Line(lineGeo,lineMat);group.add(line);
  const up=new THREE.Vector3(0,1,0), delta=new THREE.Vector3();
  let sourceScene:BuiltScene|undefined;
  let nativeTargets:{container:THREE.Object3D;index:number;depth:number}[]=[];
  const observedPosition=new THREE.Vector3(),observedQuaternion=new THREE.Quaternion();
  function setTime(t:number,built:BuiltScene) {
    const [i,j,s]=frameAt(clip.times,t), lerp=(a:number,b:number)=>a+(b-a)*s;
    built.setRecordedJoints(clip.joint_names,native||clip.native_cables?clip.door_q[i]:clip.door_q[i].map((q,k)=>lerp(q,clip.door_q[j][k])));
    if(native&&clip.native?.poses) {
      if(sourceScene!==built) {
        nativeTargets=[...built.bodies.entries()].flatMap(([name,node])=>{
          let index=clip.native!.body_names.indexOf(name);
          if(index<0&&node.userData.static)return [];
          if(index<0&&name==='world_env')index=clip.native!.body_names.indexOf('world');
          if(index<0)throw Error(`Native recording lacks moving body ${name}`);
          const container=node.userData.container as THREE.Object3D;let depth=0;
          for(let p=container.parent;p;p=p.parent)depth++;
          return [{container,index,depth}];
        }).sort((a,b)=>a.depth-b.depth);
        sourceScene=built;
      }
      const row=clip.native.poses[i];
      for(const target of nativeTargets){
        const offset=target.index*7;observedPosition.fromArray(row,offset);
        observedQuaternion.set(row[offset+4],row[offset+5],row[offset+6],row[offset+3]);
        applyBodyWorld(target.container,observedPosition,observedQuaternion);
      }
    }
    if(clip.native_cables) built.setRecordedCables(clip.native_cables[i]);
    if(native) {
      target.visible=false;line.visible=false;
      contactMarkers.forEach(({marker,arrow,torqueAxis},k)=>{
        const contact=clip.oracle_contacts?.[i][k];marker.visible=!!contact?.position;arrow.visible=false;torqueAxis.visible=false;
        if(!contact?.position)return;
        marker.position.fromArray(contact.position);arrow.position.copy(marker.position);
        if(contact.force_N) {
          const force=new THREE.Vector3().fromArray(contact.force_N),magnitude=force.length();
          if(magnitude>1e-5){arrow.visible=true;arrow.setDirection(force.multiplyScalar(1/magnitude));arrow.setLength(Math.min(.5,magnitude/300),.035,.02);}
        }
        if(contact.torque_Nm) {
          const torque=new THREE.Vector3().fromArray(contact.torque_Nm),magnitude=torque.length();
          if(magnitude>1e-6){torqueAxis.visible=true;torqueAxis.position.copy(marker.position);torqueAxis.setDirection(torque.multiplyScalar(1/magnitude));torqueAxis.setLength(Math.min(.15,magnitude*5),.025,.015);}
        }
      });
      return {phase:clip.phases[i],error:0,frame:i};
    }
    joints.forEach((m,k)=>m.position.set(...[0,1,2].map(axis=>lerp(clip.avatar[i][k*3+axis],clip.avatar[j][k*3+axis])) as [number,number,number]));
    bones.forEach((m,k)=>{const [a,b]=clip.avatar_bones[k];const p=joints[a].position,q=joints[b].position;
      delta.copy(q).sub(p);m.position.copy(p).add(q).multiplyScalar(.5);m.scale.y=delta.length();m.quaternion.setFromUnitVectors(up,delta.normalize());});
    const left=joints[4].position.clone().sub(joints[7].position).setZ(0).normalize();
    const forward=new THREE.Vector3(left.y,-left.x,0);
    feet.forEach((m,k)=>{m.position.copy(joints[k===0?12:15].position).addScaledVector(forward,.045).add(new THREE.Vector3(0,0,-.024));m.rotation.z=Math.atan2(forward.y,forward.x)-Math.PI/2;});
    target.position.set(...clip.targets[i] as [number,number,number]);
    target.visible=!!clip.hand_active[i]&&clip.hand_error_m[i]>.08&&t>=clip.lead_in_s;
    line.visible=target.visible;
    lineGeo.setFromPoints([joints[9].position,target.position]);line.computeLineDistances();
    return {phase:clip.phases[i],error:clip.hand_error_m[i],frame:i};
  }
  function dispose() {contactMarkers.forEach(({arrow})=>{group.remove(arrow);arrow.dispose();});group.traverse(o=>{if(o instanceof THREE.Mesh)o.geometry.dispose();});lineGeo.dispose();for(const m of [shell,jointMat,dark,warning,lineMat])m.dispose();}
  return {group,setTime,dispose};
}
export type ReferencePlayer=ReturnType<typeof buildReferencePlayer>;
