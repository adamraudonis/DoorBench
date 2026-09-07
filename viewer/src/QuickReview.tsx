import React, {useEffect, useMemo, useRef, useState} from 'react';
import type {Manifest} from './types';
import {DoorView} from './DoorView';
import {datasetFor,emptyReview,loadReviews,saveReviews,makeDocument,statusOf,timestampAfter,storageKey,mergeReviews,type ReviewMap} from './reviewState';
import './QuickReview.css';
import {syncReviews} from './reviewBackend';
import {loadScreenshots, writeScreenshot, pngDataUrl, type ReviewScreenshot} from './reviewScreenshots';

export function Review({manifest}:{manifest:Manifest}) {
  const dataset=useMemo(()=>datasetFor(manifest),[manifest]);
  return <Workspace key={dataset.id} manifest={manifest}/>;
}
function Workspace({manifest}:{manifest:Manifest}) {
  const dataset=useMemo(()=>datasetFor(manifest),[manifest]);
  const doors=useMemo(()=>[...manifest.doors].sort((a,b)=>a.index-b.index),[manifest]);
  const [initial]=useState(()=>{try{return {reviews:loadReviews(localStorage,dataset),error:''};}catch(e){return {reviews:{} as ReviewMap,error:String(e)};}});
  const [reviews,setReviews]=useState(initial.reviews),[error,setError]=useState(initial.error);
  const [selected,setSelected]=useState(()=>new URLSearchParams(location.hash.split('?')[1]).get('door')||doors[0]?.id);
  const [showSaved,setShowSaved]=useState(false);
  const [screenshots,setScreenshots]=useState<ReviewScreenshot[]>([]);
  const [exporting,setExporting]=useState(false);
  const [capturing,setCapturing]=useState(0);
  const [ready,setReady]=useState(false),[saving,setSaving]=useState(false),[syncError,setSyncError]=useState('');
  const [storagePath,setStoragePath]=useState('');
  const latestReviews=useRef(reviews),syncing=useRef(false),dirty=useRef(true),pendingSave=useRef(false);
  async function flushReviews() {
    if(syncing.current)return;
    syncing.current=true;
    try {
      do {
        dirty.current=false;
        const snapshot=latestReviews.current;
        const result=await syncReviews(dataset,snapshot);
        const merged=mergeReviews(latestReviews.current,Object.values(result.reviews)).reviews;
        latestReviews.current=merged;setReviews(merged);setStoragePath(result.storage_path);
        try {saveReviews(localStorage,dataset,merged);}catch{/* Backend remains authoritative when browser storage is full. */}
        setSyncError('');setReady(true);
      } while(dirty.current);
      pendingSave.current=false;setSaving(false);
    } catch(e) {dirty.current=true;setSyncError(`Not saved to disk yet. Retrying automatically: ${String(e)}`);}
    finally {syncing.current=false;}
  }
  useEffect(()=>{
    void flushReviews();
    const timer=window.setInterval(()=>{void flushReviews();},3000);
    const leave=(e:BeforeUnloadEvent)=>{if(pendingSave.current){e.preventDefault();e.returnValue='';}};
    window.addEventListener('beforeunload',leave);
    return()=>{clearInterval(timer);window.removeEventListener('beforeunload',leave);};
  },[dataset.id]);
  useEffect(()=>{
    let active=true;
    const refresh=()=>{loadScreenshots(dataset.id).then(shots=>{if(active){setScreenshots(shots);setError('');}}).catch(e=>{if(active)setError(`Screenshot sync failed; retrying automatically: ${String(e)}`);});};
    refresh();const timer=window.setInterval(refresh,5000);
    return()=>{active=false;clearInterval(timer);};
  },[dataset.id]);
  const index=Math.max(0,doors.findIndex(d=>d.id===selected)),door=doors[index];
  const current=reviews[door?.id]||emptyReview(door?.id||'');
  const saved=doors.map(d=>reviews[d.id]||emptyReview(d.id)).filter(r=>r.notes.trim()||statusOf(r)!=='unreviewed'||screenshots.some(s=>s.door_id===r.door_id));
  function move(delta:number){setSelected(doors[(index+delta+doors.length)%doors.length]?.id);}
  function store(notes:string,good=false) {
    const row={...current,notes,updated_at:timestampAfter(current.updated_at),...(good?{flagged:false,issues:[],ratings:{appearance:'pass',physical:'pass',mechanism:'pass'}}:{flagged:!!notes.trim()})} as typeof current;
    const next={...reviews,[door.id]:row};
    latestReviews.current=next;setReviews(next);dirty.current=true;pendingSave.current=true;setSaving(true);
    try{saveReviews(localStorage,dataset,next);}catch{/* Disk save still proceeds. */}
    void flushReviews();
  }
  useEffect(()=>{if(door)history.replaceState(null,'',`#/review?door=${encodeURIComponent(door.id)}`);},[door?.id]);
  useEffect(()=>{
    const key=(e:KeyboardEvent)=>{if((e.target as HTMLElement)?.closest?.('.doorview'))return;if(e.key!=='Tab'||e.ctrlKey||e.altKey||e.metaKey||e.isComposing||!doors.length)return;e.preventDefault();(document.activeElement as HTMLElement)?.blur();move(e.shiftKey?-1:1);};
    window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);
  });
  useEffect(()=>{
    const change=(e:StorageEvent)=>{if(e.key!==storageKey(dataset))return;try{const merged=mergeReviews(latestReviews.current,Object.values(loadReviews(localStorage,dataset))).reviews;latestReviews.current=merged;setReviews(merged);dirty.current=true;void flushReviews();}catch(e){setError(String(e));}};
    window.addEventListener('storage',change);return()=>window.removeEventListener('storage',change);
  },[dataset]);
  async function attachScreenshot(blob: Blob, filename: string) {
    const doorId=door.id;
    setCapturing(n=>n+1);
    try {
      const screenshot: ReviewScreenshot={id:crypto.randomUUID(),dataset_id:dataset.id,door_id:doorId,filename,captured_at:new Date().toISOString(),image_data_url:await pngDataUrl(blob)};
      setScreenshots(await writeScreenshot(screenshot));
    } finally { setCapturing(n=>n-1); }
  }
  async function removeScreenshot(screenshot: ReviewScreenshot) {
    try { setScreenshots(await writeScreenshot(screenshot,true)); }
    catch(e) { setError(`Could not remove screenshot: ${String(e)}`); }
  }
  async function downloadPackage() {
    setExporting(true);
    try {
      const images=await Promise.all((await loadScreenshots(dataset.id)).map(async s=>({...s,image_data_url:await pngDataUrl(await (await fetch(s.image_data_url)).blob())})));
      const packageData={schema:'doorbench.feedback-package.v1',exported_at:new Date().toISOString(),review_document:makeDocument(dataset,reviews),doors:doors.filter(d=>reviews[d.id]?.notes.trim()||statusOf(reviews[d.id])!=='unreviewed'||images.some(s=>s.door_id===d.id)).map(d=>({door_id:d.id,review:reviews[d.id]||emptyReview(d.id),screenshots:images.filter(s=>s.door_id===d.id)}))};
      const url=URL.createObjectURL(new Blob([JSON.stringify(packageData,null,2)],{type:'application/json'}));
      const a=document.createElement('a');a.href=url;a.download=`doorbench-feedback-package-${new Date().toISOString().replace(/[:.]/g,'-')}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    } catch(e) { setError(`Could not export feedback: ${String(e)}`); }
    finally { setExporting(false); }
  }
  if(!door)return <p>No doors to review.</p>;
  return <section className="quick-review">
    <header className="quick-review-heading"><div><h1>Review doors</h1><p>Tab: next · Shift+Tab: previous · In 3D view, Tab moves between controls. Comments autosave.</p></div><button onClick={()=>setShowSaved(v=>!v)}>My feedback ({saved.length}){saving?" · Saving…":""}</button></header>
    {error&&<p role="alert">{error}</p>}
    {syncError&&<p role="alert">{syncError}</p>}
    <div className="quick-review-controls">
      <div className="quick-review-navigation"><button onClick={()=>move(-1)} aria-label="Previous door">←</button><span>{index+1} / {doors.length} · {door.id}</span><button onClick={()=>move(1)} aria-label="Next door">→</button>
      <button className="primary" disabled={!ready} onClick={()=>{store(current.notes,true);move(1);}}>Good · next</button></div>
      <label>Comment<textarea disabled={!ready} maxLength={10000} rows={1} aria-label="Door comment" placeholder="What needs fixing?" value={current.notes} onChange={e=>store(e.target.value)}/></label>
    </div>
    {showSaved&&<section className="quick-review-saved"><h2>My feedback</h2><p>Comments and screenshots save automatically to this computer. Just ask Codex to fix the reviewed doors.</p><small>Review folder: {storagePath}</small><br/><button className="primary" disabled={exporting||capturing>0||saving||!ready} onClick={()=>void downloadPackage()}>{exporting?"Preparing package…":"Download backup"}</button>
      <table><thead><tr><th>Door</th><th>Assessment</th><th>Comment</th><th>Screenshots</th></tr></thead><tbody>{saved.map(r=><tr key={r.door_id}><td><a href={`#/door/${r.door_id}`} target="_blank" rel="noreferrer">{r.door_id}</a></td><td>{statusOf(r)==='accepted'?'Good':'Needs review'}</td><td style={{whiteSpace:'pre-wrap'}}>{r.notes}</td><td>{screenshots.filter(s=>s.door_id===r.door_id).length}</td></tr>)}</tbody></table>
      {!saved.length&&<p>No feedback yet.</p>}
    </section>}
    {screenshots.some(s=>s.door_id===door.id)&&<div className="review-screenshot-strip" aria-label="Attached screenshots">{screenshots.filter(s=>s.door_id===door.id).map((s,i)=><div key={s.id}><a href={s.image_data_url} download={s.filename}><img src={s.image_data_url} alt={`Screenshot ${i+1} for ${s.door_id}`}/></a><button aria-label={`Remove screenshot ${i+1}`} onClick={()=>void removeScreenshot(s)}>Remove</button></div>)}</div>}
    <DoorView key={door.id} onScreenshot={attachScreenshot} manifest={manifest} id={door.id} embedded showAppearance={false} query="reference=1&autoplay=1"/>
  </section>;
}
