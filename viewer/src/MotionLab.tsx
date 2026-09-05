import React,{useEffect,useMemo,useRef,useState} from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Manifest } from './types';
import { FAMILY_LABELS } from './types';
import { artifactURL,buildPlannedPlayer,buildVerifiedDoor,fetchPlannedClip,motionTaskDetail,motionTaskLabel,SOURCE_SCENARIOS,validateMotionIndex,type MotionEntry,type MotionIndex } from './plannedReferenceMotion';
import type { BuiltScene } from './scene';
import {MotionVisualReviewPanel} from './MotionVisualReview';
import './MotionLab.css';

const labels:Record<string,string>={accepted_kinematic:'Accepted',rejected:'Rejected',unresolved:'Unresolved'};
const human=(value:string)=>value.replaceAll('_',' ');
function reason(entry:MotionEntry) {
  const failures=Object.entries(entry.failure_counts??{}).sort((a,b)=>b[1]-a[1]);
  return failures.length?failures.slice(0,4).map(([key,count])=>`${human(key)} (${count.toLocaleString()})`).join(' · '):entry.reason||human(entry.reason_code??'No accepted complete motion is available in this snapshot.');
}

export function MotionLab({manifest}:{manifest:Manifest}) {
  const indexURL=useMemo(()=>new URL(import.meta.env.VITE_PLANNED_REFERENCE_INDEX||'./planned-references/index.json',window.location.href).href,[]);
  const [index,setIndex]=useState<MotionIndex|null>(null),[error,setError]=useState(''),[revision,setRevision]=useState(0);
  const [search,setSearch]=useState(''),[status,setStatus]=useState(''),[family,setFamily]=useState(''),[scenario,setScenario]=useState('');
  const queueElement=useRef<HTMLElement>(null);
  const [selected,setSelected]=useState(()=>new URLSearchParams(window.location.hash.split('?')[1]??'').get('door')??'');
  useEffect(()=>{const controller=new AbortController();setError('');
    fetch(indexURL,{signal:controller.signal,cache:'no-cache'}).then(r=>{if(!r.ok)throw Error(`Motion index unavailable (${r.status})`);return r.json();}).then(value=>{
      const next=validateMotionIndex(value);setIndex(next);setSelected(old=>next.doors.some(d=>d.door_id===old)?old:next.doors.find(d=>d.clip)?.door_id??next.doors[0]?.door_id??'');
    }).catch(e=>{if(!controller.signal.aborted)setError(String(e));});return()=>controller.abort();
  },[indexURL,revision]);
  useEffect(()=>{if(selected)window.history.replaceState(null,'',`#/motions?door=${encodeURIComponent(selected)}`);},[selected]);
  const queue=useMemo(()=>index?.doors.filter(d=>(!status||d.status===status)&&(!family||d.family===family)&&(!scenario||d.source_scenario===scenario)&&`${d.door_id} ${d.family} ${d.source_scenario??''} ${motionTaskLabel(d.source_scenario)} ${reason(d)}`.toLowerCase().includes(search.toLowerCase()))??[],[index,status,family,scenario,search]);
  useEffect(()=>{if(queue.length&&!queue.some(d=>d.door_id===selected))setSelected(queue[0].door_id);},[queue,selected]);
  useEffect(()=>{const list=queueElement.current,button=list?.querySelector<HTMLElement>('[aria-current="true"]');if(!list||!button)return;const a=button.getBoundingClientRect(),b=list.getBoundingClientRect();if(a.top<b.top)list.scrollTop+=a.top-b.top;else if(a.bottom>b.bottom)list.scrollTop+=a.bottom-b.bottom;},[selected,queue]);
  const entry=index?.doors.find(d=>d.door_id===selected),position=queue.findIndex(d=>d.door_id===selected);
  useEffect(()=>{const navigate=(e:KeyboardEvent)=>{const target=e.target as HTMLElement|null;if(e.altKey||e.ctrlKey||e.metaKey||target?.closest('input,textarea,select,[contenteditable="true"]'))return;
    if(e.defaultPrevented||e.isComposing||e.repeat)return;const next=e.key==='ArrowLeft'?position-1:e.key==='ArrowRight'?position+1:-1;
    if(next>=0&&next<queue.length){e.preventDefault();setSelected(queue[next].door_id);}};
    window.addEventListener('keydown',navigate);return()=>window.removeEventListener('keydown',navigate);
  },[queue,position]);
  const acceptedTasks=useMemo(()=>{const accepted=index?.doors.filter(d=>d.status==='accepted_kinematic')??[];return {traversal:accepted.filter(d=>d.source_scenario==='open_and_traverse'||d.source_scenario==='unlock_and_traverse').length,locked:accepted.filter(d=>d.source_scenario==='locked_recognize').length};},[index]);
  const counts=useMemo(()=>index?.doors.reduce((a,d)=>({...a,[d.status]:(a[d.status]??0)+1}),{} as Record<string,number>)??{},[index]);
  return <section className="motion-lab">
    <header className="motion-heading"><div><div className="eyebrow">Articulated reference motions</div><h1>Motion Lab</h1><p>Inspect the original adult rig through each door interaction. Explore accepted motion and see why other attempts need work.</p></div><button onClick={()=>setRevision(r=>r+1)}>Refresh status</button></header>
    <div className="motion-scope"><strong>Sampled kinematic checks only.</strong> Accepted means the saved motion passed geometry, contact, joint, timing and task-evidence checks. Task evidence uses the actor route and declared source outcome; mechanism semantics are not independently certified. This does not certify dynamic balance, causal robot control or natural appearance.</div>
    {error&&<div className="motion-error" role="alert">{error} <button onClick={()=>setRevision(r=>r+1)}>Retry</button></div>}
    {!index&&!error&&<div className="loading"><span className="loading-dot"/>Loading motion status…</div>}
    {index&&<><div className="motion-counts"><button className={!status?'selected':''} onClick={()=>setStatus('')}><strong>{index.doors.length.toLocaleString()}</strong><span>All doors</span></button>{Object.entries(labels).map(([key,label])=><button key={key} className={`motion-${key} ${status===key?'selected':''}`} onClick={()=>setStatus(status===key?'':key)}><strong>{(counts[key]??0).toLocaleString()}</strong><span>{label}</span></button>)}<small>Snapshot<br/>{new Date(index.updated_at).toLocaleString()}</small></div>
      <p className="motion-task-counts">Accepted includes <strong>{acceptedTasks.traversal} traversal references</strong> and <strong>{acceptedTasks.locked} locked-door checks</strong>.</p>
      <div className="motion-filters"><label>Search<input type="search" value={search} placeholder="Door ID or failure reason" onChange={e=>setSearch(e.target.value)}/></label><label>Door type<select value={family} onChange={e=>setFamily(e.target.value)}><option value="">All types</option>{manifest.families.map(f=><option key={f} value={f}>{FAMILY_LABELS[f]??human(f)}</option>)}</select></label><label>Source task<select aria-label="Source task" value={scenario} onChange={e=>setScenario(e.target.value)}><option value="">All source scenarios</option>{SOURCE_SCENARIOS.map(s=><option value={s} key={s}>{s==='locked_recognize'?'Locked-door check':s==='open_and_traverse'?'Open and traverse':'Unlock and traverse'}</option>)}</select></label><span>{queue.length.toLocaleString()} matching doors</span></div>
      <div className="motion-workspace"><aside ref={queueElement} className="motion-queue" aria-label="Motion attempts">{queue.map(d=><button key={d.door_id} aria-label={`${d.door_id} ${labels[d.status]??d.status}`} className={selected===d.door_id?'selected':''} aria-current={selected===d.door_id?'true':undefined} onClick={()=>setSelected(d.door_id)}><span>{d.door_id.split('_')[0].toUpperCase()}<small>{FAMILY_LABELS[d.family]??human(d.family)}</small></span><span className={`motion-badge motion-${d.status}`}>{labels[d.status]??human(d.status)}</span></button>)}{!queue.length&&<p>No doors match these filters.</p>}</aside>
        <div className="motion-main">{entry&&<><header className="motion-selected"><div><span className={`motion-badge motion-${entry.status}`}>{labels[entry.status]??human(entry.status)}</span><h2>{entry.door_id.split('_')[0].toUpperCase()} <span>{FAMILY_LABELS[entry.family]??human(entry.family)}</span></h2><div className="motion-task-kind">{motionTaskLabel(entry.source_scenario)}</div></div><div className="motion-neighbors"><button aria-label="Previous door" disabled={position<=0} onClick={()=>setSelected(queue[position-1].door_id)}>← Previous</button><button aria-label="Next door" disabled={position<0||position>=queue.length-1} onClick={()=>setSelected(queue[position+1].door_id)}>Next →</button></div></header>
          <p className="motion-task-detail">{motionTaskDetail(entry.source_scenario)}</p>
          {entry.clip?<MotionViewport key={`${entry.door_id}:${entry.clip.sha256}`} entry={entry} indexURL={indexURL}/>:<div className="motion-unavailable"><div className="motion-empty-mark">◇</div><h3>{entry.status==='rejected'?'This attempt did not pass.':'No accepted motion yet.'}</h3><p>{reason(entry)}</p><a className="button" href={`#/review?door=${encodeURIComponent(entry.door_id)}`}>Inspect the door →</a></div>}
          <div className="motion-audits"><div><strong>Evidence and source</strong><p>{entry.clip?`${entry.clip.frames.toLocaleString()} saved poses · ${(entry.clip.bytes/1048576).toFixed(1)} MB motion · every source checked before playback`:'Only accepted, verified clips are available for playback.'}</p></div><div>{Object.entries(entry.audits).map(([name,file])=><a key={name} href={artifactURL(file.path,indexURL)} target="_blank" rel="noreferrer" title={`SHA-256 ${file.sha256}`}>{name==='validation.json'?'Validation audit':name==='clip.json'?'Source manifest':'Attempt record'} ↗</a>)}<a href={`#/review?door=${encodeURIComponent(entry.door_id)}`}>Door review ↗</a></div></div>
          {entry.clip&&<MotionVisualReviewPanel doorId={entry.door_id} clipHash={entry.clip.sha256}/>}
        </>}</div></div></>}
  </section>;
}

function MotionViewport({entry,indexURL}:{entry:MotionEntry;indexURL:string}) {
  const mount=useRef<HTMLDivElement>(null),engine=useRef<{built:BuiltScene;player:ReturnType<typeof buildPlannedPlayer>;frame:()=>void}|null>(null);
  const clock=useRef({time:0,playing:false,speed:1,last:0});
  const [ready,setReady]=useState(false),[error,setError]=useState(''),[time,setTime]=useState(0),[playing,setPlaying]=useState(false),[speed,setSpeed]=useState(1),[phase,setPhase]=useState(''),[diagnostic,setDiagnostic]=useState(false);
  useEffect(()=>{const controller=new AbortController(),element=mount.current!;let animation=0,renderer:THREE.WebGLRenderer|undefined,controls:OrbitControls|undefined,observer:ResizeObserver|undefined,built:BuiltScene|undefined,player:ReturnType<typeof buildPlannedPlayer>|undefined;
    void(async()=>{
      const loaded=await fetchPlannedClip(entry,indexURL,controller.signal);if(controller.signal.aborted)return;
      built=await buildVerifiedDoor(loaded.model,loaded.files);if(controller.signal.aborted){built.dispose();return;}
      player=buildPlannedPlayer(loaded.clip,built);const scene=new THREE.Scene();scene.background=new THREE.Color(0xe8ede7);scene.add(built.root,player.group);
      const camera=new THREE.PerspectiveCamera(43,1,.01,500);camera.up.set(0,0,1);
      renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;element.appendChild(renderer.domElement);
      controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=.25;controls.maxDistance=100;
      scene.add(new THREE.HemisphereLight(0xffffff,0x73806a,2));const key=new THREE.DirectionalLight(0xfff2df,2.5);key.position.set(3,-5,7);scene.add(key);const fill=new THREE.DirectionalLight(0xcbdffe,1);fill.position.set(-3,4,3);scene.add(fill);
      function frame(){const bounds=player!.bounds,center=bounds.getCenter(new THREE.Vector3()),radius=bounds.getSize(new THREE.Vector3()).length()*.5;const distance=radius/Math.sin(THREE.MathUtils.degToRad(camera.fov)*.5)/Math.min(1,camera.aspect)*1.05;camera.position.copy(center).addScaledVector(new THREE.Vector3(.8,-1.7,.65).normalize(),distance);controls!.target.copy(center);camera.far=Math.max(100,distance*5);camera.updateProjectionMatrix();controls!.update();}
      const resize=()=>{const width=element.clientWidth,height=element.clientHeight;renderer!.setSize(width,height,false);camera.aspect=width/Math.max(height,1);camera.updateProjectionMatrix();};resize();frame();observer=new ResizeObserver(resize);observer.observe(element);
      engine.current={built,player,frame};setReady(true);setPhase(loaded.clip.phases[0]);
      if(import.meta.env.DEV)(window as any).__motionLab={scene,camera,get built(){return built;},player,clock};
      let lastPoseTime=-1;const loop=(now:number)=>{animation=requestAnimationFrame(loop);const state=clock.current,dt=state.last?Math.min(.1,(now-state.last)/1000):0;state.last=now;
        if(state.playing){state.time=Math.min(loaded.clip.duration,state.time+dt*state.speed);if(state.time>=loaded.clip.duration){state.playing=false;setPlaying(false);}}
        if(state.time!==lastPoseTime){const current=player!.setTime(state.time);setTime(state.time);setPhase(current.phase);lastPoseTime=state.time;}controls!.update();renderer!.render(scene,camera);
      };animation=requestAnimationFrame(loop);
    })().catch(e=>{if(!controller.signal.aborted)setError(String(e));});
    return()=>{controller.abort();cancelAnimationFrame(animation);observer?.disconnect();controls?.dispose();player?.dispose();built?.dispose();renderer?.dispose();renderer?.domElement.remove();engine.current=null;clock.current.playing=false;if(import.meta.env.DEV)delete(window as any).__motionLab;};
  },[entry.door_id,entry.clip?.sha256,indexURL]);
  function toggle(){const state=clock.current;if(state.time>=entry.clip!.duration)state.time=0;state.playing=!state.playing;setPlaying(state.playing);}
  return <div className="motion-player"><div className="motion-canvas" ref={mount} aria-label="Articulated motion viewport"/>{!ready&&!error&&<div className="motion-overlay">Verifying motion and source assets…</div>}{error&&<div className="motion-overlay motion-error" role="alert">Playback blocked: {error}</div>}
    <div className="motion-view-tools"><button disabled={!ready} aria-pressed={diagnostic} onClick={()=>{engine.current?.built.setDiagnostic(!diagnostic);setDiagnostic(v=>!v);}}>Brown / gold</button><button disabled={!ready} onClick={()=>engine.current?.frame()}>Frame motion</button></div>
    <div className="motion-playback"><div><button className="primary" disabled={!ready} onClick={toggle}>{playing?'Pause':'Play'}</button><button disabled={!ready} onClick={()=>{clock.current.time=0;clock.current.playing=false;setPlaying(false);setTime(0);}}>Reset</button><strong>{human(phase||'Loading')}</strong><label>Speed<select value={speed} onChange={e=>{const value=Number(e.target.value);clock.current.speed=value;setSpeed(value);}}>{[.25,.5,1,2].map(v=><option value={v} key={v}>{v}×</option>)}</select></label></div><label className="motion-scrub"><span>{time.toFixed(1)} s <span>/ {entry.clip!.duration.toFixed(1)} s</span></span><input aria-label="Motion time" type="range" min={0} max={entry.clip!.duration} step="any" value={time} disabled={!ready} onChange={e=>{clock.current.time=Number(e.target.value);setTime(Number(e.target.value));}}/></label><small>Original articulated geometry · recorded body transforms · interpolation is illustrative</small></div>
  </div>;
}
