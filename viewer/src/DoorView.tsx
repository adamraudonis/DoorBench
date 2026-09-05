import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { BenchmarkJ, Manifest, ModelJ, ScenarioJ } from "./types";
import { FAMILY_LABELS } from "./types";
import { buildScene, type BuiltScene, type JointHandle } from "./scene";
import { buildEvaluationOverlay, type EvalOverlay } from "./evaluation";
import { GLOSSARY, REWARD_LABELS, type GlossaryEntry } from "./glossary";
import { activeLeaf, isLocked, openClosePhases, parsePoseQuery, sliderReaction, type Phase } from "./doorLogic";
import { ASSETS } from "./App";
import { BaselineBadges } from "./ResultBadges";
import { AppearancePanel } from "./Appearance";
import { fetchReference, buildReferencePlayer, type ReferenceClip, type ReferencePlayer } from "./referenceMotion";
import "./referenceMotion.css";

function fmt(x: any, digits = 3): string {
  if (x === null || x === undefined) return "–";
  if (typeof x === "boolean") return x ? "yes" : "no";
  if (typeof x === "number") return Math.abs(x) >= 1000 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(digits);
  return String(x);
}
const deg = (rad: number | null | undefined, d = 1) => rad == null ? "–" : `${(rad * 180 / Math.PI).toFixed(d)}°`;
const nice = (s: string | undefined | null) => (s ?? "–").replace(/_/g, " ");

let TIP_OPEN_KEY: string | null = null;   // deep link `tip=<row key>` opens that explanation on load

const LOOP_LABELS: Record<string, string> = { two_bar: "two-bar arm", telescoping: "telescoping strut", generic: "closed loop (numeric)" };

/** Accessible info icon: tooltip on hover, focus and click / tap.  `entry` overrides the glossary lookup by key. */
function Info({ k, entry, label }: { k?: string; entry?: GlossaryEntry; label: string }) {
  const [open, setOpen] = useState(!!k && TIP_OPEN_KEY === k);
  const ref = useRef<HTMLSpanElement>(null);
  const e = entry ?? (k ? GLOSSARY[k] : undefined);
  useEffect(() => {
    if (!open) return;
    const onDoc = (ev: MouseEvent | TouchEvent) => { if (ref.current && !ref.current.contains(ev.target as Node)) setOpen(false); };
    const onKey = (ev: KeyboardEvent) => { if (ev.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDoc); document.addEventListener("touchstart", onDoc); document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("touchstart", onDoc); document.removeEventListener("keydown", onKey); };
  }, [open]);
  const id = useMemo(() => "tip-" + Math.random().toString(36).slice(2, 8), []);
  if (!e) return null;
  return (
    <span className={"info-wrap" + (open ? " open" : "")} ref={ref}>
      <button type="button" className="info" aria-label={`About ${label}`} aria-describedby={id} aria-expanded={open} onClick={(ev) => { ev.preventDefault(); setOpen((v) => !v); }}>ⓘ</button>
      <span className="tip" role="tooltip" id={id}>
        <b>{label}</b>{e.unit ? <span className="u"> · {e.unit}</span> : null}
        <span className="what">{e.what}</span>
        {e.how ? <span className="how">{e.how}</span> : null}
      </span>
    </span>
  );
}

type Row = [string, any, string?, GlossaryEntry?];

function KV({ rows }: { rows: Row[] }) {
  return (
    <div className="kv">
      {rows.map(([k, v, key, entry]) => (
        <React.Fragment key={k}>
          <span className="k">{k}<Info k={key} entry={entry} label={k} /></span>
          <span className="v">{typeof v === "string" || React.isValidElement(v) ? v : fmt(v)}</span>
        </React.Fragment>
      ))}
    </div>
  );
}

/** Door page.  URL query (after the id): `eval=1` shows the evaluation overlay, `scenario=<name>` selects it, `t=<s>` positions the person,
 *  `tip=<row key>` opens one explanation (e.g. `tip=en_size`). */
/** Scenario <option>s grouped by suite: the core suite (no person; the default benchmark) first, the opt-in human suite after. */
function ScenarioOptions({ scenarios }: { scenarios: ScenarioJ[] }) {
  const idx = scenarios.map((s, i) => [s, i] as const);
  const core = idx.filter(([s]) => (s.suite ?? "core") === "core");
  const human = idx.filter(([s]) => s.suite === "human");
  return (
    <>
      <optgroup label="Core suite (default, no person)">{core.map(([s, i]) => <option key={s.name} value={i}>{nice(s.name)}</option>)}</optgroup>
      {human.length > 0 && <optgroup label="Human suite (advanced, opt-in)">{human.map(([s, i]) => <option key={s.name} value={i}>{nice(s.name)}</option>)}</optgroup>}
    </>
  );
}

export function DoorView({ manifest, id, query = "", embedded = false, initialDiagnostic = false }: { manifest: Manifest; id: string; query?: string; embedded?: boolean; initialDiagnostic?: boolean }) {
  const entry = manifest.doors.find((d) => d.id === id);
  const mountRef = useRef<HTMLDivElement>(null);
  const [model, setModel] = useState<ModelJ | null>(null);
  const [spec, setSpec] = useState<any>(null);
  const [qa, setQa] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [joints, setJoints] = useState<JointHandle[]>([]);
  const [, force] = useState(0);
  const [diagnostic, setDiagnostic] = useState(initialDiagnostic || new URLSearchParams(query).get("contrast") === "1");
  const diagnosticRef = useRef(diagnostic); diagnosticRef.current = diagnostic;
  const [reference, setReference] = useState<ReferenceClip | null>(null);
  const [referenceError, setReferenceError] = useState<string | null>(null);
  const [referenceVisible, setReferenceVisible] = useState(false);
  const [referenceTime, setReferenceTime] = useState(0);
  const [referencePlaying, setReferencePlaying] = useState(false);
  const [referencePhase, setReferencePhase] = useState("approach");
  const [referenceReach, setReferenceReach] = useState(0);
  const referencePlayer = useRef<ReferencePlayer | null>(null);
  const referenceState = useRef({time:0,playing:false,visible:false,last:0,speed:1,clip:null as ReferenceClip|null});
  const [showEnv, setShowEnv] = useState(true);
  const [showCol, setShowCol] = useState(false);
  const [showEval, setShowEval] = useState(false);
  const [scenIdx, setScenIdx] = useState(0);
  const [humanT, setHumanT] = useState(0);
  const [humanPlay, setHumanPlay] = useState(false);
  const [hint, setHint] = useState<string | null>(null);
  const hintTimer = useRef<number | null>(null);
  const built = useRef<BuiltScene | null>(null);
  const overlay = useRef<EvalOverlay | null>(null);
  const three = useRef<{ scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: OrbitControls; anim: number } | null>(null);
  const queue = useRef<Phase[]>([]);
  const humanRef = useRef({ t: 0, play: false, dur: 0, last: 0 });
  const scenarioRef = useRef<ScenarioJ | undefined>(undefined);
  const showEvalRef = useRef(false);
  showEvalRef.current = showEval;
  const builtModel = useRef<ModelJ | null>(null);   // the model the current scene was built from (pose is kept across rebuilds of the same model)

  useEffect(() => {
    setModel(null); setSpec(null); setQa(null); setErr(null); setScenIdx(0); setHumanT(0); setHumanPlay(false);
    let cancelled = false;
    Promise.all([
      fetch(`${ASSETS}/doors/${id}/model.json`).then((r) => r.json()),
      fetch(`${ASSETS}/doors/${id}/spec.json`).then((r) => r.json()),
      fetch(`${ASSETS}/doors/${id}/qa.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]).then(([m, s, q]) => { if (!cancelled) { setModel(m); setSpec(s); setQa(q); } }).catch((e) => { if (!cancelled) setErr(String(e)); });
    return () => { cancelled = true; };
  }, [id]);

  useEffect(() => {
    const abort = new AbortController();
    setReference(null); setReferenceError(null); setReferenceTime(0); setReferencePlaying(false); setReferenceVisible(false);
    referenceState.current = {time:0,playing:false,visible:false,last:0,speed:1,clip:null};
    fetchReference(id, abort.signal).then(c => {
      if (abort.signal.aborted) return;
      setReference(c); referenceState.current.clip = c;
    }).catch(e => { if (!abort.signal.aborted) setReferenceError(String(e.message || e)); });
    return () => abort.abort();
  }, [id]);
  useEffect(() => { built.current?.setDiagnostic(diagnostic); }, [diagnostic, joints]);
  useEffect(() => {
    const t = three.current;
    if (!t || !reference) return;
    const player = buildReferencePlayer(reference);
    referencePlayer.current = player; player.group.visible = referenceState.current.visible;
    t.scene.add(player.group);
    return () => { t.scene.remove(player.group); player.dispose(); if(referencePlayer.current===player) referencePlayer.current=null; };
  }, [reference]);
  const pauseReference = () => {
    referenceState.current.playing = false; referenceState.current.visible = false;
    setReferencePlaying(false); setReferenceVisible(false);
    if (referencePlayer.current) referencePlayer.current.group.visible = false;
  };
  const seekReference = (time:number) => {
    const state=referenceState.current; state.time=time; state.visible=true;
    setReferenceTime(time); setReferenceVisible(true);
    queue.current=[];
    if(referencePlayer.current && built.current) {
      referencePlayer.current.group.visible=true;
      const info=referencePlayer.current.setTime(time,built.current);
      setReferencePhase(info.phase);setReferenceReach(info.error);force(x=>x+1);
    }
  };

  const appliedReference = useRef<string | null>(null);
  useEffect(() => {
    if (!reference || reference.door_id!==id || !model || builtModel.current!==model || !built.current || new URLSearchParams(query).get("reference") !== "1" || appliedReference.current === `${id}|${query}`) return;
    appliedReference.current = `${id}|${query}`;
    const requested=Number(new URLSearchParams(query).get("rt") || 0);
    seekReference(Number.isFinite(requested) ? Math.max(0,Math.min(reference.duration,requested)) : 0); frameReference();
  }, [reference, joints, query, id]);

  const toast = (text: string, ms = 2600) => {
    setHint(text);
    if (hintTimer.current) window.clearTimeout(hintTimer.current);
    hintTimer.current = window.setTimeout(() => setHint(null), ms);
  };

  // three.js setup
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x11151c);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.02, 100);
    camera.up.set(0, 0, 1);
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    el.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x3a3226, 0.9);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2.5, -3.5, 4.5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -4; sun.shadow.camera.right = 4; sun.shadow.camera.top = 4; sun.shadow.camera.bottom = -4; sun.shadow.camera.far = 20;
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0xbfd4ff, 0.5);
    fill.position.set(-3, 3, 2);
    scene.add(fill);
    const ground = new THREE.Mesh(new THREE.CircleGeometry(8, 48), new THREE.MeshStandardMaterial({ color: 0x2a2e36, roughness: 0.95 }));
    ground.position.z = -0.001;
    ground.receiveShadow = true;
    scene.add(ground);
    const grid = new THREE.GridHelper(12, 24, 0x334155, 0x1f2937);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.0005;
    scene.add(grid);
    const resize = () => { const w = el.clientWidth, h = el.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    const t = { scene, camera, renderer, controls, anim: 0 };
    three.current = t;
    // dev server only: lets scripts / the browser tooling aim the camera and read the built scene (screenshots, checks)
    if ((import.meta as any).env?.DEV) (window as any).__doorbench = { three: t, get built() { return built.current; }, refresh: () => force((x) => x + 1) };
    const loop = (now: number) => {
      t.anim = requestAnimationFrame(loop);
      const b = built.current;
      const q = queue.current;
      if (b && q.length) {
        const ph = q[0];
        if (ph.t0 === undefined) ph.t0 = now;
        const s = Math.min(1, (now - ph.t0) / ph.dur);
        const e = s < 0.5 ? 2 * s * s : 1 - Math.pow(-2 * s + 2, 2) / 2;
        b.setJoint(ph.joint, ph.from + (ph.to - ph.from) * e);
        for (const f of ph.followers ?? []) b.setJoint(f.joint, f.from + (f.to - f.from) * e);
        if (s >= 1) q.shift();
        force((x) => x + 1);
      }
      const ref = referenceState.current;
      if (ref.visible && ref.clip && referencePlayer.current && b) {
        const dt = ref.last ? Math.min(.1, (now-ref.last)/1000) : 0;
        if(ref.playing) {
          ref.time = Math.min(ref.clip.duration, ref.time+dt*ref.speed);
          if(ref.time>=ref.clip.duration){ref.playing=false;setReferencePlaying(false);}
        }
        const state=referencePlayer.current.setTime(ref.time,b);
        setReferenceTime(ref.time);setReferencePhase(state.phase);setReferenceReach(state.error);
      }
      ref.last=now;
      // closed kinematic loops (closer arms, struts) follow the driver joints; no-op unless a joint moved this frame
      if (b && b.solveLoops().changed) force((x) => x + 1);
      const hr = humanRef.current;
      if (hr.play && overlay.current && hr.dur > 0) {
        const dt = hr.last ? (now - hr.last) / 1000 : 0;
        hr.t = (hr.t + dt) % (hr.dur + 1.0);
        overlay.current.setHumanTime(hr.t);
        setHumanT(hr.t);
      }
      hr.last = now;
      overlay.current?.update();
      controls.update();
      renderer.render(scene, camera);
    };
    t.anim = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(t.anim); ro.disconnect(); controls.dispose(); renderer.dispose(); el.removeChild(renderer.domElement); three.current = null; };
  }, []);

  // build door
  useEffect(() => {
    const t = three.current;
    if (!t || !model) return;
    let cancelled = false;
    // same model rebuilt (walls / collision toggled): keep the pose; new door: apply the `q=` pose deep link
    const keep = built.current && builtModel.current === model ? Array.from(built.current.joints.values()).filter((h) => !h.loopSolved).map((h) => [h.name, h.q] as const) : [];
    if (built.current) { t.scene.remove(built.current.root); built.current.dispose(); built.current = null; }
    queue.current = [];
    buildScene(model, { showEnv, showCollision: showCol }).then((b) => {
      if (cancelled) { b.dispose(); return; }
      built.current = b;
      builtModel.current = model;
      for (const [name, q] of keep) if (b.joints.get(name)?.loopSolved === false) b.setJoint(name, q);
      b.solveLoops();
      b.setDiagnostic(diagnosticRef.current);
      t.scene.add(b.root);
      const c = b.bounds.getCenter(new THREE.Vector3());
      const size = b.bounds.getSize(new THREE.Vector3()).length() || 3;
      const u = model.meta?.u ?? 1;
      // frame the opening (same extents the thumbnail cameras use), not the whole environment
      const ext: number = model.meta?.scene_extent ?? size * 0.5;
      const wy: number = model.meta?.wall_y ?? 0;
      const tgt = new THREE.Vector3(model.meta?.cam_target_x ?? c.x, wy, model.meta?.cam_target_z ?? c.z);
      t.camera.position.set(tgt.x + 0.9 * ext * u, wy - 1.7 * ext, tgt.z + 0.55 * ext);
      t.controls.target.copy(tgt);
      t.controls.update();
      setJoints(Array.from(b.joints.values()));
      if (showEvalRef.current) frameEvaluation();
    }).catch((e) => setErr(String(e)));
    return () => { cancelled = true; };
  }, [model, showEnv, showCol]);

  // `q=` pose deep link: applied once per (door, query) as soon as the scene exists, and again when the hash changes
  const appliedPose = useRef<string | null>(null);
  useEffect(() => {
    const b = built.current;
    const key = `${id}|${query}`;
    if (!b || appliedPose.current === key) return;
    appliedPose.current = key;
    const pose = parsePoseQuery(query);
    if (!pose.length) return;
    queue.current = [];
    for (const [name, q] of pose) if (b.joints.get(name)?.loopSolved === false) b.setJoint(name, q);
    b.solveLoops();
    force((x) => x + 1);
  }, [id, query, joints]);

  const bench: BenchmarkJ | undefined = spec?.benchmark;
  const scenarios: ScenarioJ[] = bench?.scenarios ?? [];
  const scenario: ScenarioJ | undefined = scenarios[Math.min(scenIdx, Math.max(0, scenarios.length - 1))];
  scenarioRef.current = scenario;
  TIP_OPEN_KEY = new URLSearchParams(query).get("tip");

  // deep links: #/door/<id>?eval=1&scenario=hold_open_for_human&t=4
  useEffect(() => {
    if (!bench) return;
    const p = new URLSearchParams(query);
    const want = p.get("scenario");
    if (want) { const i = scenarios.findIndex((s) => s.name === want); if (i >= 0) { setScenIdx(i); scenarioRef.current = scenarios[i]; } }
    const t = parseFloat(p.get("t") ?? "");
    if (!Number.isNaN(t)) { humanRef.current.t = t; setHumanT(t); }
    if (p.get("eval") === "1") { setShowEval(true); setTimeout(frameEvaluation, 0); }
  }, [bench]);

  // evaluation overlay
  useEffect(() => {
    const t = three.current;
    if (overlay.current) { t?.scene.remove(overlay.current.group); overlay.current.dispose(); overlay.current = null; }
    humanRef.current.dur = 0;
    if (!t || !showEval || !scenario || !model || !built.current) return;
    const ov = buildEvaluationOverlay(scenario, model, built.current);
    overlay.current = ov;
    t.scene.add(ov.group);
    humanRef.current.dur = ov.humanDuration;
    const t0 = Math.min(humanRef.current.t, ov.humanDuration);
    humanRef.current.t = t0;
    ov.setHumanTime(t0);
    setHumanT(t0);
    return () => { t.scene.remove(ov.group); ov.dispose(); if (overlay.current === ov) overlay.current = null; };
  }, [showEval, scenario, model, joints]);

  useEffect(() => { humanRef.current.play = humanPlay; }, [humanPlay]);

  function frameReference() {
    const t=three.current, b=built.current, clip=referenceState.current.clip;
    if(!t || !b || !clip) return;
    const box=b.bounds.clone();
    for(let i=0;i<clip.avatar.length;i+=5) for(let j=0;j<clip.avatar[i].length;j+=3) box.expandByPoint(new THREE.Vector3(clip.avatar[i][j],clip.avatar[i][j+1],clip.avatar[i][j+2]));
    const center=box.getCenter(new THREE.Vector3()), radius=box.getSize(new THREE.Vector3()).length()/2;
    const halfFov=Math.atan(Math.tan(THREE.MathUtils.degToRad(t.camera.fov/2))*Math.min(1,t.camera.aspect));
    const distance=Math.max(3.2,radius/Math.sin(halfFov)*1.08);
    t.camera.position.copy(center).addScaledVector(new THREE.Vector3(.65,-1,.55).normalize(),distance);
    if(entry?.family === "hatch_ceiling") t.camera.position.z = 1.45;
    t.controls.target.copy(center); t.controls.update();
  }

  function frameEvaluation() {
    const t = three.current;
    const scenario = scenarioRef.current;
    if (!t || !scenario) return;
    const pp = scenario.pass_plane.center, st = scenario.start.center, g = scenario.goal?.center ?? pp;
    const tgt = new THREE.Vector3((st[0] + g[0]) / 2, (st[1] + g[1]) / 2 - 0.4, 0.8);
    const span = Math.max(3.2, Math.hypot(st[0] - g[0], st[1] - g[1]) + 1.8);
    t.camera.position.set(tgt.x + span * 0.7, tgt.y - span * 0.95, span * 0.7);
    t.controls.target.copy(tgt);
    t.controls.update();
  }

  if (!entry) return <div className="err">Unknown door {id}</div>;
  if (err) return <div className="err">{err}</div>;
  const phys = spec?.physics ?? {};
  const primary = model?.meta?.primary_joint as string | undefined;
  const secondary = model?.meta?.secondary_joint as string | undefined;
  const operator = model?.meta?.operator_joint as string | undefined;
  const opJoint = model?.bodies.find((b) => b.joint?.name === operator)?.joint;
  const opType = opJoint?.type;
  const opLeaf = model ? activeLeaf(model) : undefined;
  const primaryH = opLeaf ? built.current?.joints.get(opLeaf) : undefined;
  const opH = operator ? built.current?.joints.get(operator) : undefined;

  const animate = (joint: string | undefined, to: number) => {
    const b = built.current;
    if (!b || !joint) return;
    const h = b.joints.get(joint);
    if (!h) return;
    pauseReference();
    queue.current = [{ joint, from: h.q, to, dur: 900 }];
  };

  /** Physically honest open / close: work the operator (retract the bolt through its coupling), move the leaf, release. */
  const openClose = () => {
    const b = built.current;
    if (!b || !model) return;
    pauseReference();
    const { phases, note } = openClosePhases(model, b.joints);
    queue.current = phases;
    if (note) toast(note);
  };

  const onSlider = (h: JointHandle, q: number) => {
    const b = built.current;
    if (!b || !model) return;
    pauseReference();
    queue.current = [];
    b.setJoint(h.name, q);
    const r = sliderReaction(model, b.joints, h.name, q);
    if (r.operatorTo != null && operator) b.setJoint(operator, r.operatorTo);
    if (r.mirror) b.setJoint(r.mirror.joint, r.mirror.q);
    if (r.note) toast(r.note);
    b.solveLoops();
    force((x) => x + 1);
  };
  const loopResults = built.current?.loopResults ?? [];

  const dl = entry.files ?? {};
  const fileLink = (label: string, rel: string | undefined) => rel ? <a key={label} href={`${ASSETS}/doors/${rel}`} download>{label}</a> : null;
  const rotary = opType !== "slide";
  const travelStr = phys.latch ? (rotary ? `${deg(phys.latch.operator_travel)} (${fmt(phys.latch.operator_travel)} rad)` : `${fmt(phys.latch.operator_travel * 1000, 1)} mm`) : "–";
  const springStr = phys.latch ? (rotary ? `${fmt(phys.latch.operator_spring_preload)} N·m + ${fmt(phys.latch.operator_spring_rate)} N·m/rad · q` : `${fmt(phys.latch.operator_spring_preload)} N + ${fmt(phys.latch.operator_spring_rate)} N/m · q`) : "–";
  const yieldStr = phys.latch ? `${fmt(phys.latch.operator_yield)} ${rotary ? "N·m" : "N"}` : "–";
  const backlashStr = (() => {
    if (!phys.lock) return "–";
    if (phys.lock.engaged && !phys.lock.robot_side_release && opJoint?.range) {
      const span = opJoint.range[1] - opJoint.range[0];
      return rotary ? `${deg(span)} (as built)` : `${fmt(span * 1000, 1)} mm (as built)`;
    }
    return rotary ? deg(phys.lock.handle_backlash_locked_rad ?? 0) : `${fmt((phys.lock.handle_backlash_locked_rad ?? 0) * 1000, 1)} mm`;
  })();
  const qaChecks: Record<string, boolean> = qa?.checks ?? {};
  const qaRows: Row[] = Object.entries(qaChecks).map(([k, v]) => [nice(k), <span className={v ? "ok" : "bad"}>{v ? "pass" : "FAIL"}</span>, GLOSSARY["qa_" + k] ? "qa_" + k : "qa_generic"]);
  if (qa && !("clearance" in qaChecks)) qaRows.splice(3, 0, ["clearance", "n/a (regenerate)", "qa_clearance"]);
  if (qa?.metrics?.clearance_n_failures != null) qaRows.push(["clearance failures", `${qa.metrics.clearance_n_failures}${qa.metrics.clearance_failures?.length ? ": " + qa.metrics.clearance_failures.slice(0, 3).map((f: any) => `${(f.geoms ?? []).join(" ↔ ")} ${f.depth != null ? (f.depth * 1000).toFixed(1) + " mm" : ""}${f.joint ? ` @ ${f.joint}=${f.q}` : ""}`).join("; ") : ""}`, "qa_clearance"]);
  const evEntry = (ev: string): GlossaryEntry | undefined => bench?.event_descriptions?.[ev] ? { what: bench.event_descriptions[ev], unit: "reward, once per episode" } : undefined;

  return (
    <div className={"doorview" + (embedded ? " embedded" : "")}>
      <div className="viewport">
        <div className="scene-mount" ref={mountRef} />
        <div className="hud">
          {primaryH && <button className="primary" onClick={openClose} title="Actuates the operator (retracts the latch), moves the leaf, releases the operator">Open / close door</button>}
          {opH && <button onClick={() => animate(operator, opH.range ? (opH.q > (opH.range[0] + opH.range[1]) / 2 ? opH.range[0] : opH.range[1]) : opH.q + 1)}>Actuate operator</button>}
          <button onClick={() => { pauseReference(); const b = built.current; queue.current = []; if (b) { for (const h of b.joints.values()) b.setJoint(h.name, h.modeledAt); b.solveLoops(); } force((x) => x + 1); }}>Reset</button>
          <button className={diagnostic ? "active" : ""} aria-pressed={diagnostic} title="Brown door, gold mechanisms, neutral surroundings; glass remains transparent" onClick={() => setDiagnostic(v=>!v)}>Mechanism contrast</button>
          <button onClick={() => setShowEnv((v) => !v)}>{showEnv ? "Hide" : "Show"} walls</button>
          <button onClick={() => setShowCol((v) => !v)}>{showCol ? "Hide" : "Show"} collision</button>
          <button className={showEval ? "active" : ""} aria-pressed={showEval} disabled={!scenario} title={scenario ? "Draw the benchmark scenario: start zone, approach, handle targets, pass plane, goal, human path" : "no benchmark block in spec.json"} onClick={() => { const v = !showEval; setShowEval(v); if (v) setTimeout(frameEvaluation, 0); }}>{showEval ? "Hide" : "Show"} evaluation</button>
          {showEval && scenarios.length > 1 && (
            <select aria-label="Scenario" value={scenIdx} onChange={(e) => setScenIdx(parseInt(e.target.value))}><ScenarioOptions scenarios={scenarios} /></select>
          )}
        </div>
        <div className="reference-player" data-review-shortcuts="off">
          <div className="reference-heading"><strong>Original illustrative reference</strong><span>Recorded MuJoCo door · kinematic figure</span></div>
          {reference ? <>
            <div className="reference-controls">
              <button className="primary" disabled={!model || builtModel.current!==model} onClick={() => {
                const state=referenceState.current;
                if(!state.visible || state.time>=reference.duration) {seekReference(0);frameReference();}
                state.playing=!state.playing;setReferencePlaying(state.playing);
              }}>{referencePlaying ? "Pause reference" : "Play reference"}</button>
              <input aria-label="Reference motion time" type="range" min={0} max={reference.duration} step={.025} value={referenceTime}
                onChange={e=>{referenceState.current.playing=false;setReferencePlaying(false);seekReference(Number(e.target.value));}} />
              <output>{referenceTime.toFixed(1)} / {reference.duration.toFixed(1)} s</output>
              <select aria-label="Reference playback speed" defaultValue="1" onChange={e=>referenceState.current.speed=Number(e.target.value)}><option value=".25">¼×</option><option value=".5">½×</option><option value="1">1×</option><option value="2">2×</option></select>
              {referenceVisible && <button onClick={pauseReference}>Hide figure</button>}
            </div>
            <div className="reference-status"><span className={reference.outcome.success ? "ok" : "bad"}>Door task: {nice(reference.outcome.outcome)}</span><span>{nice(reference.scenario)} · {nice(referencePhase)}</span>
              {referenceVisible && referenceReach>.08 && <span className="bad">Hand target out of reach: {(referenceReach*100).toFixed(0)} cm</span>}
              <a href={`#/door/${id}?reference=1&rt=${referenceTime.toFixed(2)}&contrast=${diagnostic?1:0}`}>Link to this moment</a>
              <a href={`./reference-motions/clips/${id}.json.gz`} download>Download clip</a>
              <a href="https://huggingface.co/datasets/adamraudonis/DoorBench" target="_blank" rel="noreferrer">Native trajectories ↗</a>
            </div>
          </> : <p>{referenceError || "Loading reference recording…"}</p>}
          <p className="reference-note">Generalized forces move the door; this original figure has known contact and clearance errors. <a href={`#/motions?door=${encodeURIComponent(id)}`}>Open Motion Lab for independently checked candidates and this door’s results →</a></p>
        </div>
        {showEval && scenario?.human && (
          <div className="timeline">
            <button onClick={() => setHumanPlay((v) => !v)} aria-label={humanPlay ? "Pause" : "Play"}>{humanPlay ? "❚❚" : "▶"}</button>
            <label>person t = {humanT.toFixed(1)} s
              <input type="range" min={0} max={humanRef.current.dur || scenario.human.path[scenario.human.path.length - 1][0]} step={0.1} value={humanT}
                onChange={(e) => { const v = parseFloat(e.target.value); humanRef.current.t = v; setHumanT(v); overlay.current?.setHumanTime(v); }} />
            </label>
          </div>
        )}
        {hint && <div className="toast" role="status">{hint}</div>}
        <div className="hint">drag to orbit · scroll to zoom · right-drag to pan</div>
        {!model && <div className="loading" style={{ position: "absolute", top: 50 }}>Loading model…</div>}
      </div>
      <div className="side">
        <h2>{entry.use_case || entry.id}</h2>
        <AppearancePanel id={id} />
        <div className="use">{entry.id} · <a href={`#/?family=${entry.family}`}>{FAMILY_LABELS[entry.family] ?? entry.family}</a> · {entry.context} · task: {nice(entry.task)} · difficulty {entry.difficulty}/5</div>
        <div style={{ marginTop: 6 }} className="chips">
          <span className={"chip " + (entry.signed_off ? "ok" : "bad")}>{entry.signed_off ? "Automated QA passed" : "QA: " + (entry.qa_failed?.join(", ") || "needs review")}</span>
          {scenarios.map((s, i) => <button key={s.name} className={"chip link" + (showEval && i === scenIdx ? " active" : "")} title={s.suite === "human" ? "human-interaction suite: advanced, opt-in (not part of the default core benchmark)" : "core suite: default benchmark, no person involved"} onClick={() => { setScenIdx(i); if (!showEval) { setShowEval(true); setTimeout(frameEvaluation, 0); } }}>{nice(s.name)}{s.suite === "human" ? <span className="suite-badge">human</span> : null}</button>)}
        </div>
        <div style={{ marginTop: 4 }} className="chips" title="baseline results on this door: successful episodes / episodes (core suite; human suite where the door lists one) - see the Results page">
          <BaselineBadges id={entry.id} compact={false} />
          {scenarios.some((s) => s.suite === "human") && <BaselineBadges id={entry.id} compact={false} suite="human" />}
        </div>
        <h3>Joints</h3>
        {joints.map((h) => (
          <div className="joint" key={h.name}>
            <div className="lbl"><span className="name" title={h.loopSolved ? "solved by the closed-loop linkage: follows the door, not user-driven" : h.label}>{h.name}{h.name === primary ? " ★" : ""}{h.loopSolved ? " (linkage)" : !h.interactive ? " (driven)" : ""}{isLocked(h) && h.role === "operator" ? " 🔒" : ""}</span><span className="val">{h.type === "hinge" ? `${(h.q * 180 / Math.PI).toFixed(1)}°` : `${(h.q * 1000).toFixed(1)} mm`}</span></div>
            <input type="range" min={h.range ? h.range[0] : -3.14} max={h.range ? h.range[1] : 3.14} step={0.001} value={h.q} aria-label={h.label || h.name} disabled={h.loopSolved}
              onChange={(e) => onSlider(h, parseFloat(e.target.value))} />
            <div style={{ fontSize: 11, color: "var(--muted)" }}>{h.label}</div>
          </div>
        ))}
        {loopResults.length > 0 && (
          <div className="loops" aria-label="Mechanism loops">
            {loopResults.map((r) => (
              <div className={"loop " + (r.ok ? "ok" : "bad")} key={r.name} title={`${r.equality}: ${r.joints.join(" + ")} · ${r.source === "schema" ? "from model.json linkages" : "derived from the connect equality"}`}>
                <span className="name">{r.name}</span>
                <span className="type">{LOOP_LABELS[r.type] ?? r.type} · {r.joints.join(" + ")}</span>
                <span className="val">{r.ok ? "closed" : `open ${(r.separation * 1000).toFixed(1)} mm${r.stretched ? " (reach)" : ""}`}</span>
              </div>
            ))}
          </div>
        )}
        <h3>Evaluation</h3>
        {!bench && <p style={{ fontSize: 12, color: "var(--muted)" }}>No benchmark block in spec.json (regenerate the dataset).</p>}
        {bench && scenario && (
          <div className="eval">
            <div className="kv" style={{ marginBottom: 6 }}>
              <span className="k">scenario<Info k="scenario" label="scenario" /></span>
              <span className="v">{scenarios.length > 1 ? <select value={scenIdx} onChange={(e) => setScenIdx(parseInt(e.target.value))}><ScenarioOptions scenarios={scenarios} /></select> : nice(scenario.name)}</span>
            </div>
            <p className="desc">{scenario.description}</p>
            <KV rows={[
              ["initial state", `${scenario.initial_state.door}${scenario.initial_state.lock_engaged ? ", locked" : ""}${scenario.initial_state.latched ? ", latched" : ""}`],
              ["time budget", `${fmt(scenario.time_budget_s)} s`, "time_budget"],
              ["expected transit", `${fmt(scenario.expected_transit_s)} s`, "expected_transit"],
              ["· approach", `${fmt(scenario.expected_transit_terms?.approach_s)} s`, "transit_approach"],
              ["· operate", `${fmt(scenario.expected_transit_terms?.operate_s)} s`, "transit_operate"],
              ["· open", `${fmt(scenario.expected_transit_terms?.open_s)} s`, "transit_open"],
              ["· pass", `${fmt(scenario.expected_transit_terms?.pass_s)} s`, "transit_pass"],
              ["· scenario extra", `${fmt(scenario.expected_transit_terms?.scenario_extra_s)} s`, "transit_extra"],
              ["start zone", `(${fmt(scenario.start.center[0], 2)}, ${fmt(scenario.start.center[1], 2)}) r ${fmt(scenario.start.radius, 2)} m · yaw ${deg(scenario.start.yaw, 0)} ± ${deg(scenario.start.randomize?.yaw_jitter_rad ?? 0, 0)}`, "start_zone"],
              ["approach point", `(${scenario.approach_point.map((c) => fmt(c, 2)).join(", ")})`, "approach"],
              ["handle targets", scenario.handle_targets.length ? scenario.handle_targets.join(", ") : "– (no operator: push through)", "handle_targets"],
              ["pass plane", `${fmt(scenario.pass_plane.width, 2)} × ${fmt(scenario.pass_plane.height, 2)} m at (${scenario.pass_plane.center.map((c) => fmt(c, 2)).join(", ")})`, "pass_plane"],
              ["goal zone", scenario.goal ? `(${scenario.goal.center.map((c) => fmt(c, 2)).join(", ")}) r ${fmt(scenario.goal.radius, 2)} m` : "– (none)", "goal_zone"],
              ["suite", scenario.suite === "human" ? "human interaction (advanced, opt-in; not part of the default core benchmark)" : "core (default benchmark; no person involved)", "suite"] as Row,
              ...(scenario.human ? [["person", `${scenario.human.direction === "same_as_robot" ? "follows the robot" : "comes through first"} · ${fmt(scenario.human.speed_m_s, 1)} m/s · starts at ${fmt(scenario.human.start_t_s, 1)} s · path ends ${fmt(scenario.human.path[scenario.human.path.length - 1][0], 1)} s${scenario.human.waits_at_closed_door ? " · waits at a closed door" : ""}`, "human"] as Row] : []),
            ]} />
            <h4>Reward table<Info k="rewards" label="reward table" /></h4>
            <KV rows={Object.entries(scenario.rewards).map(([ev, v]) => [REWARD_LABELS[ev] ?? nice(ev), <span className={v > 0 ? "ok" : "bad"}>{v > 0 ? "+" : ""}{Number.isInteger(v) ? v : v.toFixed(2)}</span>, undefined, evEntry(ev)] as Row)} />
            <h4>Success<Info k="success" label="success criteria" /></h4>
            <div className="chips">{scenario.success.map((c) => <span key={c} className={"chip " + (c.startsWith("!") ? "bad" : "ok")}>{c.startsWith("!") ? "no " + (REWARD_LABELS[c.slice(1)] ?? nice(c.slice(1))) : (REWARD_LABELS[c] ?? nice(c))}</span>)}</div>
          </div>
        )}
        <h3>Leaf</h3>
        <KV rows={[["mass (leaf + hardware)", `${fmt(phys.mass?.total_kg)} kg`, "mass_total"], ["slab", `${fmt(phys.mass?.slab_kg)} kg (${nice(spec?.leaf?.slab)})`, "mass_slab"], ["glass", `${fmt(phys.mass?.glass_kg)} kg`, "mass_glass"], ["hardware", `${fmt(phys.mass?.hardware_kg)} kg`, "mass_hardware"], ["size W×H×t", spec ? `${spec.leaf.width}×${spec.leaf.height}×${spec.leaf.thickness} m` : "–", "size"], ["panel style", nice(spec?.leaf?.panel_style), "panel_style"], ["finish", spec ? `${spec.leaf.finish.kind} ${spec.leaf.finish.color}` : "–", "finish"], ["inertia about hinge", phys.inertia_about_hinge_kg_m2 != null ? `${fmt(phys.inertia_about_hinge_kg_m2)} kg·m²` : "–", "inertia"], ["condition", spec?.condition, "condition"]]} />
        <h3>Hinge / motion</h3>
        <KV rows={[["hinge", nice(spec?.hinge?.model), "hinge"], ["count", spec?.hinge?.count != null ? String(spec.hinge.count) : "–", "hinge_count"], ["side / swing", spec ? `${spec.hinge.side} / ${spec.robot.is_push ? "push" : "pull"} (robot at −y)` : "–", "side_swing"], ["coulomb friction", phys.hinge ? `${fmt(phys.hinge.coulomb_torque_Nm)} N·m` : "–", "coulomb"], ["stiction (stuck)", phys.hinge ? `${fmt(phys.hinge.stick_torque_Nm)} N·m` : "–", "stiction"], ["bearing μ", phys.hinge?.bearing_mu, "bearing_mu"], ["damping", phys.hinge ? `${fmt(phys.hinge.total_damping_symmetric)} N·m·s/rad` : "–", "damping"], ["roller friction", phys.roller ? `${fmt(phys.roller.coulomb_force_N)} N (μ=${phys.roller.mu_rolling})` : "–", "roller_friction"], ["max open", spec?.kinematics?.max_open_deg != null ? `${spec.kinematics.max_open_deg}°` : spec?.kinematics?.travel_m != null ? `${spec.kinematics.travel_m} m` : "–", "max_open"], ["stop", nice(spec?.kinematics?.stop), "stop"]]} />
        <h3>Closer</h3>
        <KV rows={[["model", nice(phys.closer?.model), "closer_model"], ["EN 1154 size", phys.closer?.en_size, "en_size"], ["spring preload", phys.closer ? `${fmt(phys.closer.spring_preload_Nm)} N·m` : "–", "preload"], ["spring rate", phys.closer ? `${fmt(phys.closer.spring_stiffness_Nm_per_rad)} N·m/rad` : "–", "spring_rate"], ["damping close / open", phys.closer ? `${fmt(phys.closer.damping_closing)} / ${fmt(phys.closer.damping_opening)} N·m·s/rad` : "–", "closer_damping"], ["closing time (est.)", phys.closer?.closing_time_est_s != null ? `${fmt(phys.closer.closing_time_est_s)} s` : "–", "closing_time"]]} />
        <h3>Operator · latch · lock</h3>
        <KV rows={[["operator", `${nice(spec?.operator?.model)}${opType ? ` (${opType === "slide" ? "linear" : "rotary"})` : ""}`, "operator"], ["height", spec ? `${spec.operator.height} m` : "–", "op_height"], ["travel", travelStr, "op_travel"], ["return spring", springStr, "op_return_spring"], ["yield (damage)", yieldStr, "op_yield"], ["latch", nice(phys.latch?.model), "latch"], ["throw", phys.latch ? `${fmt((phys.latch.throw_m ?? 0) * 1000, 1)} mm` : "–", "throw"], ["bolt spring", phys.latch ? `${fmt(phys.latch.bolt_spring_preload_N)} N + ${fmt(phys.latch.bolt_spring_rate_N_per_m)} N/m` : "–", "bolt_spring"], ["lock", nice(phys.lock?.model), "lock"], ["engaged", phys.lock?.engaged, "lock_engaged"], ["robot-side release", phys.lock?.robot_side_release, "robot_side_release"], ["locked backlash", backlashStr, "backlash"], ["deadbolt throw", phys.lock ? `${fmt((phys.lock.deadbolt_throw_m ?? 0) * 1000, 1)} mm` : "–", "deadbolt_throw"], ["code", phys.lock?.code ?? "–", "code"]]} />
        <h3>Compliance (as simulated)</h3>
        <KV rows={[["opening force (start)", phys.compliance?.opening_force_start_N != null ? `${fmt(phys.compliance.opening_force_start_N)} N` : "–", "force_start"], ["opening force (90°)", phys.compliance?.opening_force_90deg_N != null ? `${fmt(phys.compliance.opening_force_90deg_N)} N` : "–", "force_90"], ["operator force", phys.compliance?.operator_force_N != null ? `${fmt(phys.compliance.operator_force_N)} N` : "–", "operator_force"], ["ADA 5 lbf interior", phys.compliance?.ada_interior_5lbf_ok, "ada_5lbf"], ["IBC fire/exterior", phys.compliance?.ibc_fire_exterior_ok, "ibc_fire"], ["hardware ≤ 5 lbf", phys.compliance?.hardware_operable_5lbf_ok, "hardware_5lbf"], ["ADA clear width", phys.compliance?.clear_width_ada_ok, "clear_width"]]} />
        <h3>Damage thresholds</h3>
        <KV rows={[["leaf dent", phys.damage ? `${fmt(phys.damage.leaf_dent_force_N)} N` : "–", "dent"], ["leaf puncture", phys.damage ? `${fmt(phys.damage.leaf_puncture_force_N)} N` : "–", "puncture"], ["glass break", phys.damage?.glass_break_force_N != null ? `${fmt(phys.damage.glass_break_force_N)} N` : "–", "glass_break"], ["operator yield", phys.damage ? `${fmt(phys.damage.operator_yield_torque_Nm)} ${rotary ? "N·m" : "N"}` : "–", "op_yield_dmg"], ["latch shear", phys.damage ? `${fmt(phys.damage.latch_shear_yield_N)} N` : "–", "latch_shear"], ["hinge tear-out", phys.damage ? `${fmt(phys.damage.hinge_tearout_force_N)} N` : "–", "hinge_tearout"], ["slam velocity", phys.damage ? `${fmt(phys.damage.slam_velocity_rad_s)} ${spec?.kinematics?.type?.startsWith("hinge") || spec?.kinematics?.type === "rotor" ? "rad/s" : "m/s"}` : "–", "slam_velocity"]]} />
        {qa && (<><h3>QA sign-off</h3><KV rows={qaRows} /></>)}
        <h3>Files</h3>
        <div className="dl">
          {fileLink("MJCF (full)", dl.mjcf?.full)}{fileLink("MJCF (simple)", dl.mjcf?.simple)}{fileLink("MJCF (minimal)", dl.mjcf?.minimal)}
          {fileLink("URDF", dl.urdf?.full)}{fileLink("USD", typeof dl.usd === "string" ? dl.usd : undefined)}
          <a href={`${ASSETS}/doors/${id}/spec.json`} download>spec.json</a><a href={`${ASSETS}/doors/${id}/model.json`} download>model.json</a>
        </div>
        <h3>Extras & tags</h3>
        <div className="chips">{[...(entry.extras ?? []), ...(entry.tags ?? [])].map((t, i) => <span className="chip" key={i}>{t.replace(/_/g, " ")}</span>)}</div>
        {spec?.physics?.mass?.source && <p style={{ fontSize: 11, color: "var(--muted)" }}>Mass source: {spec.physics.mass.source}</p>}
      </div>
    </div>
  );
}
