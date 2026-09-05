/** Reviewer-local style observations. These never change solver acceptance. */
export const REVIEW_SCHEMA='doorbench.motion-visual-reviews.v1';
export const REVIEW_STORAGE=REVIEW_SCHEMA;
export const REVIEW_TAGS=['posture','foot_motion','reaching','timing','mechanism_semantics'] as const;
export type ReviewTag=typeof REVIEW_TAGS[number];
export type VisualStatus='unreviewed'|'pass'|'needs_work';
export interface MotionVisualReview {door_id:string;clip_sha256:string;status:VisualStatus;tags:ReviewTag[];note:string;updated_at:string}
export interface ReviewFile {schema:typeof REVIEW_SCHEMA;reviews:MotionVisualReview[]}
function require(ok:unknown,message:string):asserts ok {if(!ok)throw Error(message);}
export function validateReviewFile(value:unknown):ReviewFile {
  const file=value as ReviewFile;
  require(file?.schema===REVIEW_SCHEMA&&Array.isArray(file.reviews)&&file.reviews.length<=10000,'Unsupported review file or too many records');
  const seen=new Set<string>();
  const reviews=file.reviews.map(r=>{
    require(r&&typeof r==='object'&&Object.keys(r).sort().join(',')==='clip_sha256,door_id,note,status,tags,updated_at','Invalid review fields');
    require(typeof r.door_id==='string'&&/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(r.door_id)&&typeof r.clip_sha256==='string'&&/^[a-f0-9]{64}$/.test(r.clip_sha256),'Invalid review door or clip hash');
    const key=`${r.door_id}:${r.clip_sha256}`;require(!seen.has(key),'Duplicate review for the same clip');seen.add(key);
    require(['unreviewed','pass','needs_work'].includes(r.status),'Invalid visual review status');
    require(Array.isArray(r.tags)&&new Set(r.tags).size===r.tags.length&&r.tags.every(t=>REVIEW_TAGS.includes(t)),'Invalid visual issue tags');
    require(typeof r.note==='string'&&r.note.length<=4000,'Review note exceeds 4000 characters');
    require(typeof r.updated_at==='string'&&/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(r.updated_at)&&Number.isFinite(Date.parse(r.updated_at))&&new Date(r.updated_at).toISOString()===r.updated_at,'Invalid review date');
    return {...r,tags:[...r.tags]};
  });
  return {schema:REVIEW_SCHEMA,reviews};
}
export function parseReviewFile(text:string):ReviewFile {
  require(text.length<=2_000_000,'Review file exceeds 2 MB');
  return validateReviewFile(JSON.parse(text));
}
export function matchingReview(reviews:MotionVisualReview[],doorId:string,clipHash:string) {
  return reviews.find(r=>r.door_id===doorId&&r.clip_sha256===clipHash);
}
export function mergeReviews(existing:MotionVisualReview[],incoming:MotionVisualReview[]) {
  const map=new Map(existing.map(r=>[`${r.door_id}:${r.clip_sha256}`,r]));
  for(const r of incoming)map.set(`${r.door_id}:${r.clip_sha256}`,r);
  return validateReviewFile({schema:REVIEW_SCHEMA,reviews:[...map.values()]}).reviews;
}
export function serializeReviews(reviews:MotionVisualReview[]) {
  return JSON.stringify(validateReviewFile({schema:REVIEW_SCHEMA,reviews}),null,2)+'\n';
}
