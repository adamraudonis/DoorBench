/** Local preview contract. A valid transfer is not a validated human–door interaction. */
export const HUMAN_REFERENCE_BASE = '/__human-reference/';
export type HumanReference = {
  schema: 'doorbench.human-capture-transfer.v1';
  status: 'source_capture_transfer_unvalidated_interaction' | 'target_leg_contact_adjustment_candidate';
  adjustment_report_sha256?: string;
  raw_poses_sha256?: string;
  adjustment_report?: LegAdjustment;
  action: string; duration_s: number; source_clock_offset_s: number;
  source_frame_time_s: number; source_frame_start: number; retained_frames: number;
  source: {sha256: string};
  artifacts: {path: string; sha256: string; bytes: number}[];
};
export type LegAdjustment = {
  schema: 'doorbench.human-leg-contact-fit.v1';
  status: 'visual_contact_candidate_requires_deformed_surface_recheck';
  raw_poses_sha256: string; output_poses_sha256: string; source_bvh_sha256: string;
  unchanged_clock_pelvis_upperbody_source_arrays: boolean;
  foot_and_toe_world_rotations_unchanged: boolean;
  source_xy_translation_m: number[]; affected_bones: string[];
  L: {maximum_ankle_correction_m: number}; R: {maximum_ankle_correction_m: number};
};
// Explicitly reviewed local candidate. Other fitted outputs require a new review.
const FITTED_POSES = 'ae243c5a2f70b6edc8fc21e64d9987af87bf20cd8c3c69b8ea249086bf8450eb';
const RAW_POSES = '87059c7cde513ce8738918bacbdf0e544f11891a50d7353c1f59ab49435baa13';
const LEG_BONES = new Set(['L', 'R'].flatMap(side => [
  'upperleg01', 'upperleg02', 'lowerleg01', 'lowerleg02', 'foot',
  'toe1-1', 'toe1-2', ...[2, 3, 4, 5].flatMap(toe => [1, 2, 3].map(segment => `toe${toe}-${segment}`)),
].map(bone => `${bone}.${side}`)));
const sha = /^[a-f0-9]{64}$/;
export function validateHumanReference(value: unknown): HumanReference {
  const v = value as HumanReference;
  if (!v || v.schema !== 'doorbench.human-capture-transfer.v1' || !['source_capture_transfer_unvalidated_interaction', 'target_leg_contact_adjustment_candidate'].includes(v.status)) throw new Error('Unsupported human preview stage or metadata schema.');
  if (typeof v.action !== 'string' || !v.action || !sha.test(v.source?.sha256 ?? '')) throw new Error('Missing capture action or source checksum.');
  if (!Number.isFinite(v.duration_s) || v.duration_s <= 0 || v.duration_s > 600 || !Number.isFinite(v.source_frame_time_s) || v.source_frame_time_s <= 0 || v.source_frame_time_s > 1 || !Number.isInteger(v.retained_frames) || v.retained_frames < 2 || !Number.isInteger(v.source_frame_start) || v.source_frame_start < 0 || !Number.isFinite(v.source_clock_offset_s) || v.source_clock_offset_s < 0) throw new Error('Invalid original capture clock.');
  if (Math.abs(v.duration_s - (v.retained_frames - 1) * v.source_frame_time_s) > 1e-5 || Math.abs(v.source_clock_offset_s - v.source_frame_start * v.source_frame_time_s) > 1e-5) throw new Error('Capture frame count and source clock disagree.');
  const matches = Array.isArray(v.artifacts) ? v.artifacts.filter(a => a.path === 'animation.glb') : [];
  if (matches.length !== 1 || !sha.test(matches[0].sha256) || !Number.isInteger(matches[0].bytes) || matches[0].bytes <= 20 || matches[0].bytes > 256 * 1024 * 1024) throw new Error('A single checksummed animation.glb artifact is required (maximum 256 MiB).');
  if (v.source.sha256 !== '7104b8e750d5d8d35d19f52aa2d9cc721b36aa7b20022d7820952d98995a5a02') throw new Error('This preview is configured for CeTI d02 / o03 / run 01; source capture checksum differs.');
  if (v.status === 'target_leg_contact_adjustment_candidate') {
    const r = v.adjustment_report;
    if (!r || r.schema !== 'doorbench.human-leg-contact-fit.v1' || r.status !== 'visual_contact_candidate_requires_deformed_surface_recheck' || !sha.test(v.adjustment_report_sha256 ?? '')) throw new Error('Fitted-leg preview requires its explicit adjustment report and checksum.');
    if (v.action !== 'CeTI_d02_o03.target_leg_contact_candidate' || r.raw_poses_sha256 !== RAW_POSES || v.raw_poses_sha256 !== RAW_POSES || r.output_poses_sha256 !== FITTED_POSES || r.source_bvh_sha256 !== v.source.sha256) throw new Error('Fitted-leg candidate differs from the reviewed v2 pose/source binding.');
    const poses = v.artifacts.filter(a => a.path === 'poses.npz');
    if (poses.length !== 1 || poses[0].sha256 !== r.output_poses_sha256) throw new Error('Fitted-leg pose artifact differs from its adjustment report.');
    if (r.unchanged_clock_pelvis_upperbody_source_arrays !== true || r.foot_and_toe_world_rotations_unchanged !== true || !Array.isArray(r.source_xy_translation_m) || r.source_xy_translation_m.length !== 2 || r.source_xy_translation_m.some(v => v !== 0) || v.retained_frames !== 762 || v.source_frame_start !== 2 || v.source_frame_time_s !== .01) throw new Error('Fitted-leg preview must preserve the original clock, pelvis, upper body and foot rotations.');
    if (!Array.isArray(r.affected_bones) || r.affected_bones.length !== LEG_BONES.size || new Set(r.affected_bones).size !== LEG_BONES.size || r.affected_bones.some(b => !LEG_BONES.has(b))) throw new Error('Fitted-leg report includes missing, duplicate or non-leg bones.');
    if ([r.L?.maximum_ankle_correction_m, r.R?.maximum_ankle_correction_m].some(n => !Number.isFinite(n) || n < 0 || n > 1)) throw new Error('Fitted-leg report has invalid ankle correction distances.');
  } else if (v.adjustment_report !== undefined || v.action === 'CeTI_d02_o03.target_leg_contact_candidate') {
    throw new Error('An adjusted candidate cannot be labeled as a raw capture transfer.');
  }
  return v;
}
export async function verifyHumanArtifact(bytes: ArrayBuffer, metadata: HumanReference, name: 'animation.glb' | 'normal-speed.mp4'): Promise<void> {
  const matches = metadata.artifacts.filter(a => a.path === name);
  if (matches.length !== 1 || !Number.isSafeInteger(matches[0].bytes) || matches[0].bytes <= 20 || matches[0].bytes > 256 * 1024 * 1024 || !sha.test(matches[0].sha256)) throw new Error(`A checksummed ${name} artifact is required (maximum 256 MiB).`);
  const artifact = matches[0];
  if (bytes.byteLength !== artifact.bytes) throw new Error('Human preview size differs from metadata. Re-export or refresh both files.');
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
  if (digest !== artifact.sha256) throw new Error('Human preview checksum differs from metadata. Re-export or refresh both files.');
}
export async function verifyHumanGLB(bytes: ArrayBuffer, metadata: HumanReference): Promise<void> {
  await verifyHumanArtifact(bytes, metadata, 'animation.glb');
  const view = new DataView(bytes);
  if (view.getUint32(0, true) !== 0x46546c67 || view.getUint32(4, true) !== 2 || view.getUint32(8, true) !== bytes.byteLength || view.getUint32(16, true) !== 0x4e4f534a || view.getUint32(12, true) > bytes.byteLength - 20) throw new Error('Expected a self-contained glTF 2.0 binary.');
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 20, view.getUint32(12, true))));
  // Reject external resources before GLTFLoader can fetch anything beyond this local artifact.
  if (json.asset?.version !== '2.0' || [...(json.buffers ?? []), ...(json.images ?? [])].some((r: {uri?: unknown}) => r.uri !== undefined)) throw new Error('Human GLB must embed every buffer and image; external resource requests are disabled.');
  if (json.animations?.length !== 1 || json.animations[0].name !== metadata.action) throw new Error('GLB action differs from capture metadata.');
}
export function validateAnimationClock(duration: number, metadata: HumanReference): void {
  if (!Number.isFinite(duration) || Math.abs(duration - metadata.duration_s) > Math.min(.001, metadata.source_frame_time_s * .1)) throw new Error('GLB animation duration differs from the original capture clock.');
}
export function sourceTime(previewTime: number, metadata: HumanReference): number {
  return Math.min(metadata.duration_s, Math.max(0, previewTime)) + metadata.source_clock_offset_s;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  if (value !== null && typeof value === 'object') return '{' + Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, val]) => JSON.stringify(key) + ':' + canonical(val)).join(',') + '}';
  return JSON.stringify(value);
}
export async function verifyHumanAdjustment(bytes: ArrayBuffer, metadata: HumanReference): Promise<void> {
  if (metadata.status !== 'target_leg_contact_adjustment_candidate') throw new Error('Adjustment report is only supported for the explicitly fitted-leg stage.');
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
  if (digest !== metadata.adjustment_report_sha256) throw new Error('Fitted-leg report checksum differs from capture metadata.');
  const report = JSON.parse(new TextDecoder().decode(bytes));
  if (canonical(report) !== canonical(metadata.adjustment_report)) throw new Error('Fitted-leg report differs from the embedded capture provenance.');
}
export function humanPreviewLabel(metadata: HumanReference | null): string {
  return metadata?.status === 'target_leg_contact_adjustment_candidate' ? 'Captured motion · legs fitted to character' : 'Captured human · retargeted preview';
}
