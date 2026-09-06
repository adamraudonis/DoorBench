import {useAppearance, AppearanceThumb} from "./Appearance";
import React, {useEffect, useMemo, useState} from 'react';
import type {Manifest} from './types';
import {DoorView} from './DoorView';
import {datasetFor,emptyReview,loadReviews,saveReviews,makeDocument,statusOf,timestampAfter,storageKey,mergeReviews,type ReviewMap} from './reviewState';
import './QuickReview.css';

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
  const [view,setView]=useState<'motion'|'blender'>('motion');
  const appearances=useAppearance();
  const index=Math.max(0,doors.findIndex(d=>d.id===selected)),door=doors[index];
  const current=reviews[door?.id]||emptyReview(door?.id||'');
  const saved=Object.values(reviews).filter(r=>r.notes.trim()||statusOf(r)!=='unreviewed');
  function move(delta:number){setSelected(doors[(index+delta+doors.length)%doors.length]?.id);}
  function store(notes:string,good=false) {
    const row={...current,notes,updated_at:timestampAfter(current.updated_at),...(good?{flagged:false,issues:[],ratings:{appearance:'pass',physical:'pass',mechanism:'pass'}}:{flagged:!!notes.trim()})} as typeof current;
    const next={...reviews,[door.id]:row};
    try{setReviews(saveReviews(localStorage,dataset,next));setError('');}catch(e){setReviews(next);setError(`Could not save locally: ${String(e)}. Download your feedback now.`);}
  }
  useEffect(()=>{if(door)history.replaceState(null,'',`#/review?door=${encodeURIComponent(door.id)}`);},[door?.id]);
  useEffect(()=>{
    const key=(e:KeyboardEvent)=>{if(e.key!=='Tab'||e.ctrlKey||e.altKey||e.metaKey||e.isComposing||!doors.length)return;e.preventDefault();(document.activeElement as HTMLElement)?.blur();move(e.shiftKey?-1:1);};
    window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);
  });
  useEffect(()=>{
    const change=(e:StorageEvent)=>{if(e.key!==storageKey(dataset))return;try{setReviews(old=>mergeReviews(old,Object.values(loadReviews(localStorage,dataset))).reviews);}catch(e){setError(String(e));}};
    window.addEventListener('storage',change);return()=>window.removeEventListener('storage',change);
  },[dataset]);
  function download(format:'json'|'txt'){
    const text=format==='json'?JSON.stringify(makeDocument(dataset,reviews),null,2):saved.map(r=>`${location.origin}${location.pathname}#/door/${r.door_id}\n${statusOf(r)==='accepted'?'Good':'Needs review'}\n${r.notes}\n`).join('\n');
    const url=URL.createObjectURL(new Blob([text],{type:format==='json'?'application/json':'text/plain'})),a=document.createElement('a');a.href=url;a.download=`doorbench-feedback-${new Date().toISOString().slice(0,10)}.${format}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  if(!door)return <p>No doors to review.</p>;
  return <section className="quick-review">
    <header><div><h1>Review doors</h1><p>Tab: next door · Shift+Tab: previous · Comments save as you type.</p></div><button onClick={()=>setShowSaved(v=>!v)}>My feedback ({saved.length})</button></header>
    {error&&<p role="alert">{error}</p>}
    <div className="quick-review-controls">
      <button onClick={()=>move(-1)} aria-label="Previous door">←</button><span>{index+1} / {doors.length} · {door.id}</span><button onClick={()=>move(1)} aria-label="Next door">→</button>
      <button className="primary" onClick={()=>{store(current.notes,true);move(1);}}>Good · next</button>
      <label>Comment<textarea aria-label="Door comment" placeholder="What needs fixing?" value={current.notes} onChange={e=>store(e.target.value)}/></label>
      <small>{error?'Not saved':statusOf(current)==='accepted'?'Marked good · saved locally':'Saved in this browser'}</small>
    </div>
    {showSaved&&<section className="quick-review-saved"><h2>My feedback</h2><p>Local to this browser and site address. Download this list to send it back.</p><button onClick={()=>download('txt')}>Download links and comments</button> <button onClick={()=>download('json')}>Download JSON backup</button>
      <table><thead><tr><th>Door</th><th>Assessment</th><th>Comment</th></tr></thead><tbody>{saved.map(r=><tr key={r.door_id}><td><a href={`#/door/${r.door_id}`} target="_blank" rel="noreferrer">{r.door_id}</a></td><td>{statusOf(r)==='accepted'?'Good':'Needs review'}</td><td style={{whiteSpace:'pre-wrap'}}>{r.notes}</td></tr>)}</tbody></table>
      {!saved.length&&<p>No feedback yet.</p>}
    </section>}
    <div className="quick-review-views"><button aria-pressed={view==='motion'} onClick={()=>setView('motion')}>Reference motion</button><button aria-pressed={view==='blender'} onClick={()=>setView('blender')}>Blender appearance</button></div>
    {view==='motion'?<DoorView key={door.id} manifest={manifest} id={door.id} embedded query="reference=1&autoplay=1"/>:<div className="quick-review-render">{(()=>{const renders=appearances?.renders.filter(r=>r.door_id===door.id&&r.image).sort((a,b)=>Number(b.quality==='photo')-Number(a.quality==='photo'))??[];return renders[0]?<><AppearanceThumb render={renders[0]} fallback="" alt={door.use_case||door.id}/><p>Saved Blender render · {renders[0].quality==='photo'?'photo':'preview'} quality · separate from live mechanism playback.</p></>:<p>No saved Blender image is available for this door.</p>;})()}</div>}
  </section>;
}
