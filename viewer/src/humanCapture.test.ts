import {describe, expect, it} from 'bun:test';
import {sourceTime, validateAnimationClock, validateHumanReference, verifyHumanGLB, verifyHumanArtifact, verifyHumanAdjustment, humanPreviewLabel, type HumanReference} from './humanCapture';
const fixture = (): HumanReference => ({schema:'doorbench.human-capture-transfer.v1',status:'source_capture_transfer_unvalidated_interaction',action:'capture',duration_s:7.61,source_clock_offset_s:.02,source_frame_time_s:.01,source_frame_start:2,retained_frames:762,source:{sha256:'7104b8e750d5d8d35d19f52aa2d9cc721b36aa7b20022d7820952d98995a5a02'},artifacts:[{path:'animation.glb',sha256:'a'.repeat(64),bytes:128}]});
async function glb(json: unknown) {
  let text=JSON.stringify(json);while(text.length%4) text+=' ';
  const bytes=new ArrayBuffer(20+text.length), view=new DataView(bytes);
  [0x46546c67,2,bytes.byteLength,text.length,0x4e4f534a].forEach((n,i)=>view.setUint32(i*4,n,true));
  new Uint8Array(bytes,20).set(new TextEncoder().encode(text));
  const meta=fixture();meta.artifacts[0].bytes=bytes.byteLength;meta.artifacts[0].sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');return {bytes,meta};
}
const embedded=()=>({asset:{version:'2.0'},buffers:[{byteLength:0}],animations:[{name:'capture'}]});
describe('local captured human contract',()=>{
  it('keeps source offset and final frame, without stretching the clock',()=>{const m=validateHumanReference(fixture());expect(sourceTime(0,m)).toBe(.02);expect(sourceTime(7.61,m)).toBe(7.63);expect(sourceTime(99,m)).toBe(7.63);expect(()=>validateAnimationClock(7.6100002,m)).not.toThrow();});
  it.each(['duration_s','source_frame_time_s','source_clock_offset_s'])('rejects nonfinite %s',(key)=>{const m=fixture();(m as any)[key]=NaN;expect(()=>validateHumanReference(m)).toThrow();});
  it('rejects inconsistent source clocks or duplicate GLB inventory entries',()=>{const m=fixture();m.retained_frames--;expect(()=>validateHumanReference(m)).toThrow('clock');const d=fixture();d.artifacts.push(d.artifacts[0]);expect(()=>validateHumanReference(d)).toThrow('single');});
  it('never accepts an unknown stage or silently labels another capture',()=>{const m=fixture();(m as any).status='ground_truth';expect(()=>validateHumanReference(m)).toThrow('stage');const s=fixture();s.source.sha256='b'.repeat(64);expect(()=>validateHumanReference(s)).toThrow('source capture');});
  it('rejects a retimed GLB even if it has valid metadata and checksum',()=>{expect(()=>validateAnimationClock(8,fixture())).toThrow('duration');});
  it('accepts a checksum-bound self-contained binary and rejects byte corruption',async()=>{const {bytes,meta}=await glb(embedded());await verifyHumanGLB(bytes,meta);new Uint8Array(bytes)[bytes.byteLength-1]^=1;await expect(verifyHumanGLB(bytes,meta)).rejects.toThrow('checksum');});
  it.each(['buffers','images'])('rejects external %s before GLTFLoader is invoked',async(key)=>{const j={...embedded(),[key]:[{uri:'https://untrusted.test/resource'}]};const {bytes,meta}=await glb(j);await expect(verifyHumanGLB(bytes,meta)).rejects.toThrow('embed');});
  it('rejects missing or corrupted video artifacts before playback',async()=>{const {bytes,meta}=await glb(embedded());await expect(verifyHumanArtifact(bytes,meta,'normal-speed.mp4')).rejects.toThrow('required');meta.artifacts[0].path='normal-speed.mp4';await verifyHumanArtifact(bytes,meta,'normal-speed.mp4');new Uint8Array(bytes)[0]^=1;await expect(verifyHumanArtifact(bytes,meta,'normal-speed.mp4')).rejects.toThrow('checksum');});
  it('rejects mismatched or multiple actions with refreshed checksums',async()=>{const j={...embedded(),animations:[{name:'other'}]};const {bytes,meta}=await glb(j);await expect(verifyHumanGLB(bytes,meta)).rejects.toThrow('action');});
});

function fittedFixture(): HumanReference {
  const m=fixture();m.status='target_leg_contact_adjustment_candidate';m.action='CeTI_d02_o03.target_leg_contact_candidate';
  const raw='87059c7cde513ce8738918bacbdf0e544f11891a50d7353c1f59ab49435baa13', output='ae243c5a2f70b6edc8fc21e64d9987af87bf20cd8c3c69b8ea249086bf8450eb';
  m.raw_poses_sha256=raw;m.adjustment_report_sha256='c'.repeat(64);m.artifacts.push({path:'poses.npz',sha256:output,bytes:1000});
  const parts=['upperleg01','upperleg02','lowerleg01','lowerleg02','foot','toe1-1','toe1-2','toe2-1','toe2-2','toe2-3','toe3-1','toe3-2','toe3-3','toe4-1','toe4-2','toe4-3','toe5-1','toe5-2','toe5-3'];
  m.adjustment_report={schema:'doorbench.human-leg-contact-fit.v1',status:'visual_contact_candidate_requires_deformed_surface_recheck',raw_poses_sha256:raw,output_poses_sha256:output,source_bvh_sha256:m.source.sha256,unchanged_clock_pelvis_upperbody_source_arrays:true,foot_and_toe_world_rotations_unchanged:true,source_xy_translation_m:[0,0],affected_bones:['L','R'].flatMap(side=>parts.map(part=>`${part}.${side}`)),L:{maximum_ankle_correction_m:.03329797286086526},R:{maximum_ankle_correction_m:.028888644600455173}};
  return m;
}
describe('explicit fitted-leg candidate presentation',()=>{
  it('labels the authored leg adaptation separately from the raw capture',()=>{const m=validateHumanReference(fittedFixture());expect(humanPreviewLabel(m)).toBe('Captured motion · legs fitted to character');expect(humanPreviewLabel(validateHumanReference(fixture()))).toBe('Captured human · retargeted preview');});
  it('rejects a different fitted output even if it uses the same stage string',()=>{const m=fittedFixture();m.adjustment_report!.output_poses_sha256='d'.repeat(64);m.artifacts[1].sha256='d'.repeat(64);expect(()=>validateHumanReference(m)).toThrow('reviewed v2');});
  it('rejects a mismatched native pose artifact',()=>{const m=fittedFixture();m.artifacts[1].sha256='e'.repeat(64);expect(()=>validateHumanReference(m)).toThrow('pose artifact');});
  it('rejects changed preservation claims, nonleg bones and changed source timing',()=>{let m=fittedFixture();m.adjustment_report!.unchanged_clock_pelvis_upperbody_source_arrays=false;expect(()=>validateHumanReference(m)).toThrow('preserve');m=fittedFixture();m.adjustment_report!.affected_bones[0]='upperarm01.L';expect(()=>validateHumanReference(m)).toThrow('non-leg');m=fittedFixture();m.source_frame_time_s=.02;m.duration_s=15.22;m.source_clock_offset_s=.04;expect(()=>validateHumanReference(m)).toThrow('preserve');});
  it('rejects an adjusted clip mislabeled as a raw transfer',()=>{const m=fittedFixture();m.status='source_capture_transfer_unvalidated_interaction';expect(()=>validateHumanReference(m)).toThrow('cannot be labeled');});
  it('binds the embedded adjustment report to independently downloaded exact bytes',async()=>{const m=fittedFixture();const bytes=new TextEncoder().encode(JSON.stringify(m.adjustment_report)).buffer;m.adjustment_report_sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');await verifyHumanAdjustment(bytes,m);m.adjustment_report!.L.maximum_ankle_correction_m=.4;await expect(verifyHumanAdjustment(bytes,m)).rejects.toThrow('embedded');m.adjustment_report_sha256='f'.repeat(64);await expect(verifyHumanAdjustment(bytes,m)).rejects.toThrow('checksum');});
});
