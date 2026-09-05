// Physics playground: the selected door's real door.xml running in MuJoCo (WebAssembly) with sliders for the
// physical constants, mouse / button interaction, QA-style experiments and live plots.  Route: #/playground/<id>.
// See docs/PLAYGROUND.md.  Everything MuJoCo-related is in ./mujoco/*; this file is the page.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Manifest, ModelJ } from "./types";
import { FAMILY_LABELS } from "./types";
import { ASSETS } from "./App";
import { computeTargets, defaults, formatValue, paramDefs, rebuildKey, rebuildTargets, rewriteMjcf, setParam, specOverride, type ParamDef, type ParamGroup, type ParamValues, type Targets } from "./mujoco/params";
import { errorText, fetchReader, loadMujocoModule } from "./mujoco/loader";
import { DoorSim, type SimJoint } from "./mujoco/sim";
import { MjRenderer } from "./mujoco/render";
import { drawPlot } from "./mujoco/plots";

const DEFAULT_DOOR = "db0012_swing_single";
const GROUP_LABELS: Record<ParamGroup, string> = { world: "World", leaf: "Leaf", hinge: "Hinge", closer: "Closer", roller: "Track / rollers", latch: "Latch", operator: "Operator", lock: "Lock" };
const GROUP_ORDER: ParamGroup[] = ["closer", "hinge", "roller", "leaf", "latch", "operator", "lock", "world"];
const RTFS = [0.1, 0.25, 0.5, 1, 2, 4];
const deg = (x: number) => (x * 180 / Math.PI);
const fmt = (x: number | null | undefined, d = 2) => (x == null || !Number.isFinite(x) ? "–" : x.toFixed(d));

type Phase = "idle" | "assets" | "wasm" | "compile" | "ready" | "error";

function Info({ d }: { d: ParamDef }) {
  return (
    <span className="info-wrap">
      <button className="info" type="button" aria-label={`About ${d.label}`}>ⓘ</button>
      <span className="tip" role="tooltip">
        <b>{d.label}</b> <span className="u">[{d.unit}]</span>
        <span className="what">{d.what}</span>
        <span className="how">MJCF: {d.mjcf}</span>
        <span className="how">USD / Isaac Lab: {d.usd}</span>
        {!d.live && <span className="how">Changing this recompiles the model (the state is kept).</span>}
      </span>
    </span>
  );
}

function ParamRow({ d, value, onChange }: { d: ParamDef; value: number; onChange: (x: number) => void }) {
  const changed = Math.abs(value - d.default) > 1e-9;
  const scale = d.display?.scale ?? 1;
  return (
    <div className={"joint pg-param" + (changed ? " changed" : "")}>
      <div className="lbl">
        <span className="name">{d.label}<Info d={d} /></span>
        <span className="val">{formatValue(d, value)}{changed && <button className="pg-reset" title={`back to the dataset value ${formatValue(d, d.default)}`} onClick={() => onChange(d.default)}>↺</button>}</span>
      </div>
      <div className="pg-slider">
        <input type="range" min={d.min} max={d.max} step={d.step} value={value} aria-label={d.label} onChange={(e) => onChange(parseFloat(e.target.value))} />
        <input type="number" className="pg-num" value={Number((value * scale).toFixed(d.display?.digits ?? 4))} step={d.step * scale} onChange={(e) => { const x = parseFloat(e.target.value); if (Number.isFinite(x)) onChange(x / scale); }} aria-label={`${d.label} value`} />
      </div>
    </div>
  );
}

function HoldButton({ label, title, onHold, onRelease, className }: { label: string; title?: string; onHold: () => void; onRelease: () => void; className?: string }) {
  return (
    <button className={className} title={title} onPointerDown={(e) => { e.preventDefault(); (e.target as HTMLElement).setPointerCapture?.(e.pointerId); onHold(); }} onPointerUp={onRelease} onPointerCancel={onRelease} onPointerLeave={onRelease}
      onKeyDown={(e) => { if (e.key === " " || e.key === "Enter") onHold(); }} onKeyUp={(e) => { if (e.key === " " || e.key === "Enter") onRelease(); }}>{label}</button>
  );
}

export function Playground({ manifest, id, query = "" }: { manifest: Manifest; id?: string; query?: string }) {
  const doorId = id && manifest.doors.some((d) => d.id === id) ? id : (manifest.doors.some((d) => d.id === DEFAULT_DOOR) ? DEFAULT_DOOR : manifest.doors[0]?.id);
  const entry = manifest.doors.find((d) => d.id === doorId);
  const mountRef = useRef<HTMLDivElement>(null);
  const plotRefs = [useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null)];
  const [phase, setPhase] = useState<Phase>("idle");
  const [err, setErr] = useState<string | null>(null);
  const [modelJ, setModelJ] = useState<ModelJ | null>(null);
  const [spec, setSpec] = useState<any>(null);
  const [qa, setQa] = useState<any>(null);
  const [defs, setDefs] = useState<ParamDef[]>([]);
  const [values, setValues] = useState<ParamValues>({});
  const [paused, setPaused] = useState(false);
  const [rtf, setRtf] = useState(1);
  const [showEnv, setShowEnv] = useState(true);
  const [showCol, setShowCol] = useState(false);
  const [effort, setEffort] = useState(60);
  const [, setTick] = useState(0);
  const [overrideOpen, setOverrideOpen] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const three = useRef<{ scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: OrbitControls; anim: number; dragLine: THREE.Line } | null>(null);
  const sim = useRef<DoorSim | null>(null);
  const view = useRef<MjRenderer | null>(null);
  const valuesRef = useRef<ParamValues>({});
  const defsRef = useRef<ParamDef[]>([]);
  const targetsRef = useRef<Targets | null>(null);
  const rebuildKeyRef = useRef<string>("");
  const rebuildTimer = useRef<number | null>(null);
  const pausedRef = useRef(false); pausedRef.current = paused;
  const rtfRef = useRef(1); rtfRef.current = rtf;
  const lastFrame = useRef(0);
  const carry = useRef(0);
  const lastTick = useRef(0);
  const effortRef = useRef(60); effortRef.current = effort;
  const gen = useRef(0);

  // --- door data + simulation --------------------------------------------------------------------------------
  useEffect(() => {
    if (!doorId) return;
    const my = ++gen.current;
    setPhase("assets"); setErr(null); setModelJ(null); setSpec(null); setQa(null); setDefs([]); setValues({}); setPaused(false);
    (async () => {
      try {
        const [m, s, q] = await Promise.all([
          fetch(`${ASSETS}/doors/${doorId}/model.json`).then((r) => { if (!r.ok) throw new Error(`model.json HTTP ${r.status}`); return r.json(); }),
          fetch(`${ASSETS}/doors/${doorId}/spec.json`).then((r) => { if (!r.ok) throw new Error(`spec.json HTTP ${r.status}`); return r.json(); }),
          fetch(`${ASSETS}/doors/${doorId}/qa.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
        ]);
        if (my !== gen.current) return;
        setModelJ(m); setSpec(s); setQa(q);
        const D = paramDefs(s, m);
        const V = defaults(D);
        defsRef.current = D; valuesRef.current = V; setDefs(D); setValues(V);
        setPhase("wasm");
        const mj = await loadMujocoModule();
        if (my !== gen.current) return;
        setPhase("compile");
        const ds = await DoorSim.create(mj, doorId, m, s, fetchReader(ASSETS));
        if (my !== gen.current) { ds.dispose(); return; }
        const t = computeTargets(s, m, V, D);
        targetsRef.current = t; rebuildKeyRef.current = rebuildKey(t);
        ds.applyLive(t);
        // hand effort default: enough to beat preload + friction with margin (what a person would use)
        const phys = s.physics ?? {};
        const prim = ds.primary;
        const base = prim?.type === "hinge" ? (phys.closer?.spring_preload_Nm ?? 0) + (phys.hinge?.coulomb_torque_Nm ?? 0) + 20 : (phys.roller?.coulomb_force_N ?? 0) + 40;
        setEffort(Math.round(Math.max(10, base)));
        if (sim.current) { sim.current.dispose(); }
        sim.current = ds;
        attachRenderer(ds, true);
        setPhase("ready");
        if ((import.meta as any).env?.DEV) (window as any).__playground = { sim: ds, three: three.current, snapshot, setParam: (k: string, x: number) => onParam(k, x), values: () => valuesRef.current };
      } catch (e) {
        if (my !== gen.current) return;
        setErr(errorText(e)); setPhase("error");
      }
    })();
    return () => { gen.current++; };
  }, [doorId]);

  useEffect(() => () => { if (sim.current) { sim.current.dispose(); sim.current = null; } if (view.current && three.current) { three.current.scene.remove(view.current.root); view.current.dispose(); view.current = null; } }, []);

  function attachRenderer(ds: DoorSim, frame: boolean) {
    const t = three.current;
    if (!t) return;
    if (view.current) { t.scene.remove(view.current.root); view.current.dispose(); }
    const r = new MjRenderer(ds.mj, ds.model, ds.modelJ);
    r.setVisibility(showEnv, showCol);
    r.update(ds.data);
    t.scene.add(r.root);
    view.current = r;
    if (frame) {
      const meta = ds.modelJ.meta ?? {};
      const b = r.computeBounds();
      const c = b.getCenter(new THREE.Vector3());
      const size = b.getSize(new THREE.Vector3()).length() || 3;
      const u = meta.u ?? 1;
      const ext: number = meta.scene_extent ?? size * 0.5;
      const wy: number = meta.wall_y ?? 0;
      const tgt = new THREE.Vector3(meta.cam_target_x ?? c.x, wy, meta.cam_target_z ?? c.z);
      t.camera.position.set(tgt.x + 0.9 * ext * u, wy - 1.7 * ext, tgt.z + 0.55 * ext);
      t.controls.target.copy(tgt);
      t.controls.update();
    }
  }

  // --- three.js -------------------------------------------------------------------------------------------------
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
    el.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x3a3226, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2.5, -3.5, 4.5); sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -4; sun.shadow.camera.right = 4; sun.shadow.camera.top = 4; sun.shadow.camera.bottom = -4; sun.shadow.camera.far = 20;
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0xbfd4ff, 0.5); fill.position.set(-3, 3, 2); scene.add(fill);
    const ground = new THREE.Mesh(new THREE.CircleGeometry(8, 48), new THREE.MeshStandardMaterial({ color: 0x2a2e36, roughness: 0.95 }));
    ground.position.z = -0.001; ground.receiveShadow = true; scene.add(ground);
    const grid = new THREE.GridHelper(12, 24, 0x334155, 0x1f2937); grid.rotation.x = Math.PI / 2; grid.position.z = 0.0005; scene.add(grid);
    const dragLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ color: 0xe0a458 }));
    dragLine.visible = false; scene.add(dragLine);
    const resize = () => { const w = el.clientWidth, h = el.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
    resize();
    const ro = new ResizeObserver(resize); ro.observe(el);
    const t = { scene, camera, renderer, controls, anim: 0, dragLine };
    three.current = t;

    // mouse pull: grab a moving part and drag it with a spring (the hand), like MuJoCo's simulate perturbation
    const ray = new THREE.Raycaster();
    const plane = new THREE.Plane();
    const ndc = new THREE.Vector2();
    let dragging = false;
    const toNdc = (ev: PointerEvent) => { const r = renderer.domElement.getBoundingClientRect(); ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1); };
    const onDown = (ev: PointerEvent) => {
      const ds = sim.current, v = view.current;
      if (!ds || !v || ev.button !== 0 || ev.shiftKey) return;
      toNdc(ev); ray.setFromCamera(ndc, camera);
      const hit = v.pick(ray);
      if (!hit) return;
      controls.enabled = false;
      dragging = true;
      const xpos = ds.data.xpos as ArrayLike<number>, xmat = ds.data.xmat as ArrayLike<number>;
      const b = hit.body, o = 9 * b;
      const d = [hit.point.x - xpos[3 * b], hit.point.y - xpos[3 * b + 1], hit.point.z - xpos[3 * b + 2]];
      const local: [number, number, number] = [xmat[o] * d[0] + xmat[o + 3] * d[1] + xmat[o + 6] * d[2], xmat[o + 1] * d[0] + xmat[o + 4] * d[1] + xmat[o + 7] * d[2], xmat[o + 2] * d[0] + xmat[o + 5] * d[1] + xmat[o + 8] * d[2]];
      const camDir = camera.getWorldDirection(new THREE.Vector3());
      plane.setFromNormalAndCoplanarPoint(camDir, hit.point);
      const W = ds.spec?.leaf?.width ?? 0.9;
      const maxF = ds.primary?.type === "hinge" ? effortRef.current / Math.max(0.3, 0.9 * W) : effortRef.current;
      ds.drag = { body: b, local, target: [hit.point.x, hit.point.y, hit.point.z], kp: Math.max(200, 4 * maxF), maxF, point: [hit.point.x, hit.point.y, hit.point.z], force: [0, 0, 0] };
      renderer.domElement.setPointerCapture(ev.pointerId);
      ev.preventDefault(); ev.stopPropagation();
    };
    const onMove = (ev: PointerEvent) => {
      const ds = sim.current;
      if (!dragging || !ds?.drag) return;
      toNdc(ev); ray.setFromCamera(ndc, camera);
      const p = new THREE.Vector3();
      if (ray.ray.intersectPlane(plane, p)) ds.drag.target = [p.x, p.y, p.z];
    };
    const onUp = (ev: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      if (sim.current) sim.current.drag = null;
      controls.enabled = true;
      try { renderer.domElement.releasePointerCapture(ev.pointerId); } catch { /* not captured */ }
    };
    renderer.domElement.addEventListener("pointerdown", onDown, { capture: true });
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("pointerup", onUp);
    renderer.domElement.addEventListener("pointercancel", onUp);

    const loop = (now: number) => {
      t.anim = requestAnimationFrame(loop);
      const ds = sim.current, v = view.current;
      const dtFrame = lastFrame.current ? Math.min(0.1, (now - lastFrame.current) / 1000) : 1 / 60;
      lastFrame.current = now;
      if (ds && v) {
        if (!pausedRef.current || ds.script) {
          const want = dtFrame * rtfRef.current / ds.timestep + carry.current;
          const n = Math.min(400, Math.floor(want));
          carry.current = Math.min(want - n, 400);
          if (n > 0) { try { ds.step(n); } catch (e) { console.error(e); setErr(`simulation error: ${errorText(e)}`); setPhase("error"); sim.current = null; } }
        }
        if (sim.current) {
          ds.contactForceOnLeaf();
          v.update(ds.data);
          if (ds.drag) { const g = dragLine.geometry.getAttribute("position") as THREE.BufferAttribute; g.setXYZ(0, ...ds.drag.point); g.setXYZ(1, ...ds.drag.target); g.needsUpdate = true; dragLine.visible = true; } else dragLine.visible = false;
          drawPlots(ds);
          if (now - lastTick.current > 120) { lastTick.current = now; setTick((x) => x + 1); }
        }
      }
      controls.update();
      renderer.render(scene, camera);
    };
    t.anim = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(t.anim); ro.disconnect(); controls.dispose(); renderer.dispose();
      renderer.domElement.removeEventListener("pointerdown", onDown, { capture: true } as any);
      el.removeChild(renderer.domElement); three.current = null;
    };
  }, []);

  useEffect(() => { view.current?.setVisibility(showEnv, showCol); }, [showEnv, showCol]);

  const drawPlots = (ds: DoorSim) => {
    const p = ds.primary;
    const isHinge = p?.type === "hinge";
    const rec = ds.recorder;
    const unit = isHinge ? "°" : "mm";
    const k = isHinge ? 180 / Math.PI : 1000;
    const phys = ds.spec?.physics ?? {};
    const c0 = plotRefs[0].current, c1 = plotRefs[1].current, c2 = plotRefs[2].current;
    if (c0) drawPlot(c0, rec, { title: `${p?.name ?? "primary joint"} position`, unit, window: 12, series: [{ label: "angle", color: "#e0a458", get: (i) => rec.q[i] * k }], yLines: p?.range ? [{ y: p.range[1] * k, label: "stop", color: "#f87171" }] : [] });
    if (c1) drawPlot(c1, rec, { title: `generalized force on ${p?.name ?? "joint"}`, unit: isHinge ? "N·m" : "N", window: 12, series: [{ label: "applied (hand + closer law)", color: "#5bc0eb", get: (i) => rec.a[i] }, { label: "passive (spring, damping, friction)", color: "#4ade80", get: (i) => rec.p[i] }, { label: "constraint (stops, latch, contacts)", color: "#f87171", get: (i) => rec.c[i] }] });
    if (c2) drawPlot(c2, rec, { title: "contact force on the leaf", unit: "N", window: 12, yMin: 0, series: [{ label: "Σ normal force", color: "#c084fc", get: (i) => rec.f[i] }], yLines: phys.damage?.leaf_dent_force_N ? [{ y: phys.damage.leaf_dent_force_N, label: "dent", color: "#f87171" }] : [] });
  };

  // --- parameters -----------------------------------------------------------------------------------------------
  const onParam = useCallback((key: string, x: number) => {
    const D = defsRef.current;
    const nv = setParam(valuesRef.current, D, key, x);
    valuesRef.current = nv; setValues(nv);
    const ds = sim.current;
    if (!ds) return;
    const t = computeTargets(ds.spec, ds.modelJ, nv, D);
    targetsRef.current = t;
    const rk = rebuildKey(t);
    if (rk !== rebuildKeyRef.current) {
      if (rebuildTimer.current) window.clearTimeout(rebuildTimer.current);
      rebuildTimer.current = window.setTimeout(async () => {
        const cur = sim.current, tt = targetsRef.current;
        if (!cur || !tt) return;
        try {
          setPhase("compile");
          await cur.rebuild(rewriteMjcf(cur.xml0, rebuildTargets(tt)));
          rebuildKeyRef.current = rebuildKey(tt);
          cur.applyLive(tt);
          attachRenderer(cur, false);
          setPhase("ready");
        } catch (e) { setErr(`recompile failed: ${errorText(e)}`); setPhase("error"); }
      }, 350);
    }
    ds.applyLive(t);
  }, []);

  const resetParams = () => { const D = defsRef.current; for (const d of D) if (Math.abs(valuesRef.current[d.key] - d.default) > 1e-12) onParam(d.key, d.default); };

  const override = useMemo(() => (spec && defs.length ? specOverride(doorId!, defs, values) : null), [spec, defs, values, doorId]);
  const overrideJson = override ? JSON.stringify(override, null, 2) : "";
  const nChanged = defs.filter((d) => Math.abs((values[d.key] ?? d.default) - d.default) > 1e-9).length;

  const copy = async (text: string, what: string) => {
    try { await navigator.clipboard.writeText(text); setCopied(what); window.setTimeout(() => setCopied(null), 1800); } catch { setOverrideOpen(true); }
  };
  const downloadXml = () => {
    const ds = sim.current, t = targetsRef.current;
    if (!ds || !t) return;
    const blob = new Blob([rewriteMjcf(ds.xml0, t)], { type: "text/xml" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${doorId}_tuned.xml`; a.click(); window.setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };

  // --- dev-only screenshot helper (POST to the vite save-snapshots plugin) ------------------------------------------
  async function snapshot(name: string, which: "door" | "plots" = "door") {
    const t = three.current, ds = sim.current;
    if (!t || !ds) return "no sim";
    const out = document.createElement("canvas");
    const ctx = out.getContext("2d")!;
    if (which === "door") {
      const src = t.renderer.domElement;
      out.width = src.width; out.height = src.height + 60 * (window.devicePixelRatio || 1);
      ctx.fillStyle = "#171a21"; ctx.fillRect(0, 0, out.width, out.height);
      ctx.drawImage(src, 0, 0);
      ctx.fillStyle = "#e6e8ee"; ctx.font = `${13 * (window.devicePixelRatio || 1)}px ui-sans-serif, sans-serif`;
      const changed = defsRef.current.filter((d) => Math.abs(valuesRef.current[d.key] - d.default) > 1e-9).map((d) => `${d.label}: ${formatValue(d, d.default)} → ${formatValue(d, valuesRef.current[d.key])}`);
      ctx.fillText(`${doorId} · MuJoCo ${ds.mj.mj_versionString()} in the browser · ${changed.length ? changed.join(" · ") : "dataset values"}`, 12, src.height + 24 * (window.devicePixelRatio || 1));
      ctx.fillStyle = "#9aa3b2";
      const m = ds.measures;
      ctx.fillText(`t = ${ds.time.toFixed(2)} s · ${ds.primary?.name} = ${ds.primary?.type === "hinge" ? deg(ds.q(ds.primary)).toFixed(1) + "°" : (ds.q(ds.primary) * 1000).toFixed(0) + " mm"} · peak speed ${m.peakSpeed.toFixed(2)} · closing time ${fmt(m.closingTime)} s · final ${m.finalAngle != null ? deg(m.finalAngle).toFixed(1) + "°" : "–"}`, 12, src.height + 48 * (window.devicePixelRatio || 1));
    } else {
      const cs = plotRefs.map((r) => r.current!).filter(Boolean);
      const w = Math.max(...cs.map((c) => c.width)), h = cs.reduce((s, c) => s + c.height, 0);
      out.width = w; out.height = h;
      ctx.fillStyle = "#171a21"; ctx.fillRect(0, 0, w, h);
      let y = 0; for (const c of cs) { ctx.drawImage(c, 0, y); y += c.height; }
    }
    const r = await fetch(`/__snapshot?name=${encodeURIComponent(name)}`, { method: "POST", body: out.toDataURL("image/png") });
    return r.text();
  }

  // --- derived --------------------------------------------------------------------------------------------------
  const ds = sim.current;
  const p = ds?.primary ?? null;
  const isHinge = p?.type === "hinge";
  const phys = spec?.physics ?? {};
  const qaM = qa?.metrics ?? {}, qaC = qa?.checks ?? {};
  const qaPush: number = qaM.qa_push ?? (isHinge ? 2 * ((phys.closer?.spring_preload_Nm ?? 0) + (phys.hinge?.coulomb_torque_Nm ?? 0)) + 60 : 2 * (phys.roller?.coulomb_force_N ?? 0) + 80);
  const m = ds?.measures;
  const posStr = (x: number | null | undefined) => (x == null ? "–" : isHinge ? `${deg(x).toFixed(1)}°` : `${(x * 1000).toFixed(1)} mm`);
  const velStr = (x: number | null | undefined) => (x == null ? "–" : isHinge ? `${x.toFixed(2)} rad/s` : `${x.toFixed(2)} m/s`);
  const otherJoints: SimJoint[] = ds ? ds.joints.filter((j) => j !== p && j.role !== "mechanism" && (j.role !== "primary" && j.role !== "secondary" || true)) : [];
  const effortFor = (j: SimJoint) => (j.role === "operator" ? (j.type === "hinge" ? (j.name.startsWith("dog_") ? 14 : j.name.includes("wheel") ? 10 : j.name.includes("exit_device") ? 8 : 4) : 120) : j.role === "primary" || j.role === "secondary" ? effort : (j.type === "hinge" ? 3 : 60));
  const grouped = useMemo(() => { const g = new Map<ParamGroup, ParamDef[]>(); for (const d of defs) { if (!g.has(d.group)) g.set(d.group, []); g.get(d.group)!.push(d); } return GROUP_ORDER.filter((k) => g.has(k)).map((k) => [k, g.get(k)!] as const); }, [defs]);
  const doorOptions = useMemo(() => { const q = search.trim().toLowerCase(); return manifest.doors.filter((d) => !d.error && (!q || d.id.includes(q) || d.family.includes(q) || (d.use_case ?? "").toLowerCase().includes(q) || d.operator.includes(q) || d.closer.includes(q))).slice(0, 400); }, [manifest, search]);
  const idx = manifest.doors.findIndex((d) => d.id === doorId);
  const go = (i: number) => { const d = manifest.doors[(i + manifest.doors.length) % manifest.doors.length]; if (d) window.location.hash = `#/playground/${d.id}`; };

  const holdPrimary = (sign: number) => { if (ds && p) ds.hold(p, sign * effort); };
  const releasePrimary = () => { if (ds && p) ds.release(p); };

  if (!doorId) return <div className="err">No doors in the manifest.</div>;

  return (
    <div className="pg">
      <div className="viewport pg-viewport" ref={mountRef}>
        <div className="hud">
          <button className={paused ? "" : "active"} onClick={() => setPaused((v) => !v)} title="space: run / pause">{paused ? "▶ Run" : "❚❚ Pause"}</button>
          <button onClick={() => { ds?.step(1); }} disabled={!ds} title="advance one physics step">Step</button>
          <button onClick={() => { ds?.step(50); }} disabled={!ds} title="advance 50 physics steps (0.1 s)">Step ×50</button>
          <button onClick={() => { ds?.reset(); }} disabled={!ds} title="mj_resetData: closed door, zero velocity, recorder cleared">Reset</button>
          <select aria-label="Real-time factor" value={rtf} onChange={(e) => setRtf(parseFloat(e.target.value))} title="simulated seconds per wall-clock second">{RTFS.map((r) => <option key={r} value={r}>{r}× real time</option>)}</select>
          <select aria-label="Timestep" value={ds?.timestep ?? 0.002} onChange={(e) => { if (ds) { ds.timestep = parseFloat(e.target.value); setTick((x) => x + 1); } }} title="integrator timestep (the dataset uses 2 ms, implicitfast)">{[0.001, 0.002, 0.004].map((dt) => <option key={dt} value={dt}>{dt * 1000} ms step</option>)}</select>
          <button onClick={() => setShowEnv((v) => !v)}>{showEnv ? "Hide" : "Show"} walls</button>
          <button onClick={() => setShowCol((v) => !v)}>{showCol ? "Hide" : "Show"} collision</button>
        </div>
        {phase !== "ready" && phase !== "error" && <div className="pg-status" role="status">{phase === "assets" ? "Loading model.json / spec.json…" : phase === "wasm" ? "Loading MuJoCo (WebAssembly, ≈ 10 MB, once per session)…" : phase === "compile" ? "Compiling door.xml + meshes in MuJoCo…" : "…"}</div>}
        {phase === "error" && <div className="pg-status bad" role="alert"><b>This door cannot run in the browser build.</b><br />{err}<br /><span style={{ color: "var(--muted)" }}>The dataset file itself is fine (it passed QA in MuJoCo {qa?.mujoco_version ?? "3.x"}); the WebAssembly build or the asset fetch failed here. Try another door or run it locally: <code>python -m mujoco.viewer --mjcf assets/doors/{doorId}/scene.xml</code></span></div>}
        {ds && ds.warnings.length > 0 && <div className="toast" role="status">MuJoCo warnings: {ds.warnings.join(", ")}</div>}
        {ds && ds.events.length > 0 && <div className="toast" role="status" style={{ top: 78 }}>{ds.events[ds.events.length - 1]}</div>}
        {ds?.drag && <div className="pg-force">hand: {Math.hypot(...ds.drag.force).toFixed(0)} N</div>}
        <div className="hint">drag the door / handle to pull it (spring hand, ≤ {isHinge ? `${effort} N·m at the hinge` : `${effort} N`}) · drag empty space to orbit · scroll to zoom · right-drag to pan</div>
      </div>
      <div className="pg-plots">
        <canvas ref={plotRefs[0]} aria-label="joint position vs time" />
        <canvas ref={plotRefs[1]} aria-label="joint generalized forces vs time" />
        <canvas ref={plotRefs[2]} aria-label="contact force vs time" />
      </div>
      <div className="side pg-side">
        <div className="pg-doorpick">
          <button onClick={() => go(idx - 1)} title="previous door">‹</button>
          <input type="search" placeholder="find a door (id, family, use case, closer…)" value={search} onChange={(e) => setSearch(e.target.value)} list="pg-doors" onKeyDown={(e) => { if (e.key === "Enter") { const hit = doorOptions[0]; if (hit) window.location.hash = `#/playground/${hit.id}`; } }} aria-label="Door" />
          <datalist id="pg-doors">{doorOptions.slice(0, 60).map((d) => <option key={d.id} value={d.id}>{d.use_case}</option>)}</datalist>
          <button onClick={() => go(idx + 1)} title="next door">›</button>
        </div>
        {search && doorOptions.length > 0 && doorOptions[0].id !== doorId && (
          <div className="pg-hits">{doorOptions.slice(0, 8).map((d) => <a key={d.id} href={`#/playground/${d.id}`} onClick={() => setSearch("")}>{d.id} <span>{d.use_case}</span></a>)}</div>
        )}
        {entry && <>
          <h2>{entry.use_case || entry.id}</h2>
          <div className="use">{entry.id} · {FAMILY_LABELS[entry.family] ?? entry.family} · {entry.closer !== "none" ? `closer ${entry.closer.replace(/_/g, " ")}` : "no closer"} · {entry.condition} · <a href={`#/door/${entry.id}`}>door page</a></div>
        </>}
        <div className="pg-note">Real MuJoCo {ds ? ds.mj.mj_versionString() : "3.12"} (WebAssembly) running <code>door.xml</code> with its meshes, latch tendons and closer-arm equalities. Sliders are <code>spec.json["physics"]</code> values mapped 1:1 onto the MJCF; the direction-dependent closer law is the one <code>DoorEnv</code> and the Isaac Lab <code>DoorMechanismAction</code> apply.</div>

        <h3>Physical constants {nChanged > 0 && <span className="pg-badge">{nChanged} changed</span>} {nChanged > 0 && <button className="pg-mini" onClick={resetParams}>reset all</button>}</h3>
        {grouped.length === 0 && <p className="pg-muted">{phase === "error" ? "–" : "Loading…"}</p>}
        {grouped.map(([g, list]) => (
          <div key={g} className="pg-group">
            <h4>{GROUP_LABELS[g]}{g === "closer" && phys.closer?.model ? ` · ${String(phys.closer.model).replace(/_/g, " ")}${phys.closer.en_size ? ` · EN 1154 size ${phys.closer.en_size}` : ""}` : ""}{g === "hinge" && spec?.hinge?.model ? ` · ${String(spec.hinge.model).replace(/_/g, " ")}` : ""}</h4>
            {list.map((d) => <ParamRow key={d.key} d={d} value={values[d.key] ?? d.default} onChange={(x) => onParam(d.key, x)} />)}
          </div>
        ))}

        <h3>Interact</h3>
        <div className="joint">
          <div className="lbl"><span className="name">hand effort</span><span className="val">{effort} {isHinge ? "N·m at the hinge" : "N"}</span></div>
          <input type="range" min={1} max={Math.max(400, Math.round(qaPush * 1.2))} step={1} value={effort} onChange={(e) => setEffort(parseInt(e.target.value))} aria-label="hand effort" />
          <div className="pg-muted">hold a button (or drag the leaf in the view) to apply it; QA used {fmt(qaPush, 0)} {isHinge ? "N·m" : "N"} for its hold / open checks</div>
        </div>
        {p && (
          <div className="pg-btnrow">
            <HoldButton className="primary" label={isHinge ? "Push open (+)" : "Slide open (+)"} title={`+${effort} on ${p.name} while held`} onHold={() => holdPrimary(1)} onRelease={releasePrimary} />
            <HoldButton label={isHinge ? "Pull closed (−)" : "Slide closed (−)"} title={`−${effort} on ${p.name} while held`} onHold={() => holdPrimary(-1)} onRelease={releasePrimary} />
          </div>
        )}
        {otherJoints.length > 0 && (
          <div className="pg-joints">
            {otherJoints.map((j) => (
              <div className="pg-jrow" key={j.name} title={j.label}>
                <span className="name">{j.name}{j.role === "operator" ? " ★" : ""}</span>
                <span className="val">{j.type === "hinge" ? `${deg(ds!.q(j)).toFixed(0)}°` : `${(ds!.q(j) * 1000).toFixed(1)} mm`}</span>
                <HoldButton label="−" title={`−${effortFor(j)} ${j.type === "hinge" ? "N·m" : "N"} while held`} onHold={() => ds!.hold(j, -effortFor(j))} onRelease={() => ds!.release(j)} />
                <HoldButton label="+" title={`+${effortFor(j)} ${j.type === "hinge" ? "N·m" : "N"} while held (${j.label})`} onHold={() => ds!.hold(j, effortFor(j))} onRelease={() => ds!.release(j)} />
              </div>
            ))}
          </div>
        )}

        <h3>Experiments (as in qa.py)</h3>
        <div className="pg-btnrow wrap">
          <button disabled={!ds || !p} onClick={() => { ds?.releaseTest(); setPaused(false); }} title="closer test: place the door at 60° (latch extended), let go, run 12 s">Release from 60°</button>
          <button disabled={!ds || !p} onClick={() => { ds?.flingTest(); setPaused(false); }} title="backcheck test: slam the door open at the slam velocity (4 rad/s or 1.5 m/s)">Fling open</button>
          <button disabled={!ds || !p} onClick={() => { ds?.pushTest(qaPush); setPaused(false); }} title={`QA hold test: ${fmt(qaPush, 0)} ${isHinge ? "N·m" : "N"} on the closed door for 1 s`}>Push at latch</button>
          <button disabled={!ds || !p} onClick={() => { ds?.actuateTest(qaPush); setPaused(false); }} title="QA actuate test: work the operator (and bolts), then push for 6.4 s">Actuate &amp; open</button>
        </div>
        {m?.testRunning && <div className="pg-muted">running {m.lastTest} test… t = {fmt(ds?.time, 2)} s</div>}

        <h3>Measured vs dataset QA</h3>
        <table className="pg-metrics">
          <thead><tr><th></th><th>this run</th><th>qa.json / spec</th></tr></thead>
          <tbody>
            <tr><td>{p?.name ?? "position"} now</td><td>{ds ? posStr(ds.q(p)) : "–"}</td><td>range {p?.range ? posStr(p.range[1]) : "–"}</td></tr>
            <tr><td>peak speed (since reset)</td><td>{velStr(m?.peakSpeed)}</td><td>slam threshold {phys.damage?.slam_velocity_rad_s != null ? `${phys.damage.slam_velocity_rad_s} ${isHinge ? "rad/s" : "m/s"}` : "–"}</td></tr>
            <tr className={m?.lastTest === "release" ? "hl" : ""}><td>closing time 60° → 2°</td><td>{m?.closingTime != null ? `${fmt(m.closingTime)} s` : m?.lastTest === "release" && !m.testRunning ? "did not close" : "–"}</td><td>{phys.closer?.closing_time_est_s != null ? `${fmt(phys.closer.closing_time_est_s)} s (spec estimate, 90° → 0)` : "–"}</td></tr>
            <tr className={m?.lastTest === "release" ? "hl" : ""}><td>final angle after 12 s</td><td>{posStr(m?.finalAngle)}{m?.relatched != null ? (m.relatched ? " · re-latched" : " · not latched") : ""}</td><td>{qaM.closer_final_angle != null ? `${posStr(qaM.closer_final_angle)} · closer_returns ${qaC.closer_returns ? "pass" : "FAIL"}` : "–"}</td></tr>
            <tr className={m?.lastTest === "fling" ? "hl" : ""}><td>fling: peak opening</td><td>{m?.flingPeak != null ? `${posStr(m.flingPeak)}${m.flingHitStop ? " · hit the stop" : " · backcheck held it"}` : "–"}</td><td>backcheck {phys.closer?.backcheck_angle_rad ? `from ${deg(phys.closer.backcheck_angle_rad).toFixed(0)}°, +${phys.closer.backcheck_damping} N·m·s/rad` : "none"}</td></tr>
            <tr className={m?.lastTest === "push" ? "hl" : ""}><td>push {fmt(qaPush, 0)} {isHinge ? "N·m" : "N"} for 1 s</td><td>{m?.pushDisplacement != null ? `${posStr(m.pushDisplacement)} · ${m.pushDisplacement < (isHinge ? 2 * Math.PI / 180 : 0.015) ? "holds" : "opens"}` : "–"}</td><td>{qaM.hold_displacement != null ? `${posStr(qaM.hold_displacement)} · ${"hold" in qaC ? `hold ${qaC.hold ? "pass" : "FAIL"}` : "free_opens" in qaC ? `free_opens ${qaC.free_opens ? "pass" : "FAIL"}` : ""}` : "–"}</td></tr>
            <tr className={m?.lastTest === "actuate" ? "hl" : ""}><td>actuate &amp; push: opened to</td><td>{posStr(m?.actuateOpened)}</td><td>{qaM.actuate_displacement != null ? `${posStr(qaM.actuate_displacement)} · actuate_opens ${qaC.actuate_opens ? "pass" : "FAIL"}` : qaM.locked_displacement != null ? `${posStr(qaM.locked_displacement)} (locked_holds ${qaC.locked_holds ? "pass" : "FAIL"})` : "–"}</td></tr>
            <tr><td>peak contact force on leaf</td><td>{m ? `${fmt(m.peakContact, 0)} N` : "–"}</td><td>dent {phys.damage?.leaf_dent_force_N != null ? `${fmt(phys.damage.leaf_dent_force_N, 0)} N` : "–"}</td></tr>
            {ds?.bolts.length ? <tr><td>latch bolt</td><td>{ds.boltsExtended() ? "extended" : "retracted"}</td><td>{qaM.bolt_after_release_m != null ? `${(qaM.bolt_after_release_m * 1000).toFixed(1)} mm after release` : "–"}</td></tr> : null}
            {ds?.law.maglock ? <tr><td>maglock</td><td>{ds.maglockBroken ? "forced open" : "holding"}</td><td>{fmt(ds.law.maglock.holdingForceN, 0)} N</td></tr> : null}
          </tbody>
        </table>

        <h3>Reproduce in MuJoCo / Isaac Lab</h3>
        <div className="pg-btnrow wrap">
          <button className="primary" onClick={() => copy(overrideJson, "override")} disabled={!override}>{copied === "override" ? "Copied" : "Copy as spec override"}</button>
          <button onClick={() => setOverrideOpen((v) => !v)} disabled={!override}>{overrideOpen ? "Hide" : "Show"} JSON</button>
          <button onClick={downloadXml} disabled={!ds}>Download tuned door.xml</button>
        </div>
        {overrideOpen && <textarea className="pg-json" readOnly value={overrideJson} rows={Math.min(16, overrideJson.split("\n").length + 1)} aria-label="spec override JSON" />}
        <div className="pg-note">
          The override is a partial <code>spec.json</code> (same key paths and units as the panel on the door page). <code>doorbench/export/playground.py</code> merges it into the spec and runs the real exporters, so MuJoCo and Isaac Lab consume exactly these numbers:
          <pre>{`# save the JSON above as override.json, then:
python -m doorbench.export.playground ${doorId} --override override.json --out out
python -m mujoco.viewer --mjcf out/doors/${doorId}/scene.xml     # MuJoCo
DOORBENCH_ASSETS=$PWD/out bash isaaclab/cloud/train.sh \\
    --task DoorBench-Open-Hand-v0 --doors ${doorId}             # Isaac Lab (door.usda / door_rl.usda)`}</pre>
          <span style={{ color: "var(--muted)" }}>Isaac Lab gets the springs as USD drives (stiffness / damping / target), Coulomb friction as <code>physxJointAxis:staticFrictionEffort</code>, the closer's closing-side damping, backcheck and hold-open through <code>DoorMechanismAction</code> (from <code>doorbench:closer</code>), masses as <code>physics:mass</code>. Full map: docs/PLAYGROUND.md.</span>
        </div>
      </div>
    </div>
  );
}
