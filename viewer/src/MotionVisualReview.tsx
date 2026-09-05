import React,{useRef,useState} from 'react';
import {matchingReview,mergeReviews,parseReviewFile,REVIEW_STORAGE,REVIEW_TAGS,serializeReviews,type MotionVisualReview,type VisualStatus} from './plannedMotionReview';

export function MotionVisualReviewPanel({doorId,clipHash}:{doorId:string;clipHash:string}) {
  const fileInput=useRef<HTMLInputElement>(null);
  const [loaded]=useState(()=>{try{return {reviews:parseReviewFile(localStorage.getItem(REVIEW_STORAGE)??'{"schema":"doorbench.motion-visual-reviews.v1","reviews":[]}').reviews,error:''};}catch(e){return {reviews:[] as MotionVisualReview[],error:`Saved review file could not be read: ${String(e)}`};}});
  const [reviews,setReviews]=useState(loaded.reviews),[message,setMessage]=useState(loaded.error);
  const saved=matchingReview(reviews,doorId,clipHash);
  const current:MotionVisualReview=saved??{door_id:doorId,clip_sha256:clipHash,status:'unreviewed',tags:[],note:'',updated_at:new Date().toISOString()};
  function store(next:MotionVisualReview[]) {
    try{localStorage.setItem(REVIEW_STORAGE,serializeReviews(next));setReviews(next);setMessage('Saved in this browser.');}
    catch(e){setMessage(`Review was not saved: ${String(e)}`);}
  }
  function update(patch:Partial<Pick<MotionVisualReview,'status'|'tags'|'note'>>) {store(mergeReviews(reviews,[{...current,...patch,updated_at:new Date().toISOString()}]));}
  function download() {
    const url=URL.createObjectURL(new Blob([serializeReviews(reviews)],{type:'application/json'})),a=document.createElement('a');
    a.href=url;a.download='doorbench-motion-visual-reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function upload(file?:File) {
    if(!file)return;
    try{if(file.size>2_000_000)throw Error('Review file exceeds 2 MB');const parsed=parseReviewFile(await file.text());const next=mergeReviews(reviews,parsed.reviews);localStorage.setItem(REVIEW_STORAGE,serializeReviews(next));setReviews(next);setMessage(`Imported ${parsed.reviews.length} clip reviews into this browser. Matching clip hashes were replaced.`);}
    catch(e){setMessage(`Import rejected: ${String(e)}`);}
  }
  return <section className="motion-visual-review" aria-label="Local visual motion review">
    <header><div><strong>Visual motion review</strong><p>Your style judgment for this exact clip. Separate from the independent kinematic checks.</p></div><div><button onClick={download}>Export notes</button><button onClick={()=>fileInput.current?.click()}>Import notes</button><input ref={fileInput} type="file" accept="application/json,.json" hidden aria-label="Import motion review JSON" onChange={e=>{void upload(e.target.files?.[0]);e.target.value='';}}/></div></header>
    {!saved&&reviews.some(r=>r.door_id===doorId)&&<p className="motion-review-stale">This clip has changed. Reviews of earlier clip hashes are retained in exports; this version is unreviewed.</p>}
    <div className="motion-review-status" role="group" aria-label="Visual review status">{([['unreviewed','Unreviewed'],['pass','Pass'],['needs_work','Needs work']] as const).map(([value,label])=><button key={value} aria-pressed={current.status===value} onClick={()=>update({status:value as VisualStatus})}>{label}</button>)}</div>
    <div className="motion-review-tags" role="group" aria-label="Visual motion issue tags">{REVIEW_TAGS.map(tag=><label key={tag}><input type="checkbox" checked={current.tags.includes(tag)} onChange={e=>update({tags:e.target.checked?[...current.tags,tag]:current.tags.filter(t=>t!==tag)})}/>{tag.replaceAll('_',' ')}</label>)}</div>
    <label className="motion-review-note">Notes<textarea aria-label="Visual motion review note" value={current.note} maxLength={4000} rows={3} placeholder="Describe the motion, phase or timestamp to revisit." onChange={e=>update({note:e.target.value})}/></label>
    <footer><small title={clipHash}>Clip {clipHash.slice(0,12)}… · local to this browser · review the full motion before marking Pass</small><span role="status">{message}</span></footer>
  </section>;
}
