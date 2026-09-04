import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Manifest, ModelJ } from "./types";
import { FAMILY_LABELS } from "./types";
import { buildScene, type BuiltScene, type JointHandle } from "./scene";
import { ASSETS } from "./App";

function fmt(x: any, digits = 3): string {
  if (x === null || x === undefined) return "–";
  if (typeof x === "boolean") return x ? "yes" : "no";
  if (typeof x === "number") return Math.abs(x) >= 1000 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(digits);
  return String(x);
}

function KV({ rows }: { rows: [string, any][] }) {
  return <div className="kv">{rows.map(([k, v]) => (<React.Fragment key={k}><span className="k">{k}</span><span className="v">{typeof v === "string" ? v : fmt(v)}</span></React.Fragment>))}</div>;
}

export function DoorView({ manifest, id }: { manifest: Manifest; id: string }) {
  const entry = manifest.doors.find((d) => d.id === id);
  const mountRef = useRef<HTMLDivElement>(null);
  const [model, setModel] = useState<ModelJ | null>(null);
  const [spec, setSpec] = useState<any>(null);
  const [qa, setQa] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [joints, setJoints] = useState<JointHandle[]>([]);
  const [, force] = useState(0);
  const [showEnv, setShowEnv] = useState(true);
  const [showCol, setShowCol] = useState(false);
  const built = useRef<BuiltScene | null>(null);
  const three = useRef<{ scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: OrbitControls; anim: number } | null>(null);
  const animating = useRef<{ t0: number; from: number; to: number; joint: string } | null>(null);

  useEffect(() => {
    setModel(null); setSpec(null); setQa(null); setErr(null);
    Promise.all([
      fetch(`${ASSETS}/doors/${id}/model.json`).then((r) => r.json()),
      fetch(`${ASSETS}/doors/${id}/spec.json`).then((r) => r.json()),
      fetch(`${ASSETS}/doors/${id}/qa.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]).then(([m, s, q]) => { setModel(m); setSpec(s); setQa(q); }).catch((e) => setErr(String(e)));
  }, [id]);

  // three.js setup
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x11151c);
    const camera = new THREE.PerspectiveCamera(50, 1, 0.02, 100);
    camera.up.set(0, 0, 1);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
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
    const loop = (now: number) => {
      t.anim = requestAnimationFrame(loop);
      const a = animating.current, b = built.current;
      if (a && b) {
        const s = Math.min(1, (now - a.t0) / 1400);
        const e = s < 0.5 ? 2 * s * s : 1 - Math.pow(-2 * s + 2, 2) / 2;
        b.setJoint(a.joint, a.from + (a.to - a.from) * e);
        if (s >= 1) animating.current = null;
        force((x) => x + 1);
      }
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
    if (built.current) { t.scene.remove(built.current.root); built.current.dispose(); built.current = null; }
    buildScene(model, { showEnv, showCollision: showCol }).then((b) => {
      if (cancelled) { b.dispose(); return; }
      built.current = b;
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
    }).catch((e) => setErr(String(e)));
    return () => { cancelled = true; };
  }, [model, showEnv, showCol]);

  if (!entry) return <div className="err">Unknown door {id}</div>;
  if (err) return <div className="err">{err}</div>;
  const phys = spec?.physics ?? {};
  const primary = model?.meta?.primary_joint as string | undefined;
  const operator = model?.meta?.operator_joint as string | undefined;
  const animate = (joint: string | undefined, to: number) => {
    const b = built.current;
    if (!b || !joint) return;
    const h = b.joints.get(joint);
    if (!h) return;
    animating.current = { t0: performance.now(), from: h.q, to, joint };
  };
  const primaryH = primary ? built.current?.joints.get(primary) : undefined;
  const opH = operator ? built.current?.joints.get(operator) : undefined;
  const dl = entry.files ?? {};
  const fileLink = (label: string, rel: string | undefined) => rel ? <a key={label} href={`${ASSETS}/doors/${rel}`} download>{label}</a> : null;

  return (
    <div className="doorview">
      <div className="viewport" ref={mountRef}>
        <div className="hud">
          {primaryH && <button className="primary" onClick={() => animate(primary, primaryH.range ? (primaryH.q > (primaryH.range[0] + primaryH.range[1]) / 2 ? primaryH.range[0] : primaryH.range[1]) : primaryH.q + 1.2)}>Open / close door</button>}
          {opH && <button onClick={() => animate(operator, opH.range ? (opH.q > (opH.range[0] + opH.range[1]) / 2 ? opH.range[0] : opH.range[1]) : opH.q + 1)}>Actuate operator</button>}
          <button onClick={() => { const b = built.current; if (b) for (const h of b.joints.values()) b.setJoint(h.name, h.modeledAt); force((x) => x + 1); }}>Reset</button>
          <button onClick={() => setShowEnv((v) => !v)}>{showEnv ? "Hide" : "Show"} walls</button>
          <button onClick={() => setShowCol((v) => !v)}>{showCol ? "Hide" : "Show"} collision</button>
        </div>
        <div className="hint">drag to orbit · scroll to zoom · right-drag to pan</div>
        {!model && <div className="loading" style={{ position: "absolute", top: 50 }}>Loading model…</div>}
      </div>
      <div className="side">
        <h2>{entry.use_case || entry.id}</h2>
        <div className="use">{entry.id} · <a href={`#/?family=${entry.family}`}>{FAMILY_LABELS[entry.family] ?? entry.family}</a> · {entry.context} · task: {entry.task.replace(/_/g, " ")} · difficulty {entry.difficulty}/5</div>
        <div style={{ marginTop: 6 }}>
          <span className={"chip " + (entry.signed_off ? "ok" : "bad")}>{entry.signed_off ? "QA signed off" : "QA: " + (entry.qa_failed?.join(", ") || "needs review")}</span>
        </div>
        <h3>Joints</h3>
        {joints.map((h) => (
          <div className="joint" key={h.name}>
            <div className="lbl"><span className="name" title={h.label}>{h.name}{h.name === primary ? " ★" : ""}{!h.interactive ? " (driven)" : ""}</span><span className="val">{h.type === "hinge" ? `${(h.q * 180 / Math.PI).toFixed(1)}°` : `${(h.q * 1000).toFixed(1)} mm`}</span></div>
            <input type="range" min={h.range ? h.range[0] : -3.14} max={h.range ? h.range[1] : 3.14} step={0.001} value={h.q}
              onChange={(e) => { built.current?.setJoint(h.name, parseFloat(e.target.value)); force((x) => x + 1); }} />
            <div style={{ fontSize: 11, color: "var(--muted)" }}>{h.label}</div>
          </div>
        ))}
        <h3>Leaf</h3>
        <KV rows={[["mass (leaf + hardware)", `${fmt(phys.mass?.total_kg)} kg`], ["slab", `${fmt(phys.mass?.slab_kg)} kg (${spec?.leaf?.slab?.replace(/_/g, " ")})`], ["glass", `${fmt(phys.mass?.glass_kg)} kg`], ["hardware", `${fmt(phys.mass?.hardware_kg)} kg`], ["size W×H×t", spec ? `${spec.leaf.width}×${spec.leaf.height}×${spec.leaf.thickness} m` : "–"], ["panel style", spec?.leaf?.panel_style?.replace(/_/g, " ")], ["finish", spec ? `${spec.leaf.finish.kind} ${spec.leaf.finish.color}` : "–"], ["inertia about hinge", phys.inertia_about_hinge_kg_m2 != null ? `${fmt(phys.inertia_about_hinge_kg_m2)} kg·m²` : "–"], ["condition", spec?.condition]]} />
        <h3>Hinge / motion</h3>
        <KV rows={[["hinge", spec?.hinge?.model?.replace(/_/g, " ")], ["count", spec?.hinge?.count], ["side / swing", spec ? `${spec.hinge.side} / ${spec.robot.is_push ? "push" : "pull"} (robot at −y)` : "–"], ["coulomb friction", phys.hinge ? `${fmt(phys.hinge.coulomb_torque_Nm)} N·m` : "–"], ["stiction (stuck)", phys.hinge ? `${fmt(phys.hinge.stick_torque_Nm)} N·m` : "–"], ["bearing μ", phys.hinge?.bearing_mu], ["damping", phys.hinge ? `${fmt(phys.hinge.total_damping_symmetric)} N·m·s/rad` : "–"], ["roller friction", phys.roller ? `${fmt(phys.roller.coulomb_force_N)} N (μ=${phys.roller.mu_rolling})` : "–"], ["max open", spec?.kinematics?.max_open_deg != null ? `${spec.kinematics.max_open_deg}°` : spec?.kinematics?.travel_m != null ? `${spec.kinematics.travel_m} m` : "–"], ["stop", spec?.kinematics?.stop]]} />
        <h3>Closer</h3>
        <KV rows={[["model", phys.closer?.model?.replace(/_/g, " ")], ["EN 1154 size", phys.closer?.en_size], ["spring preload", phys.closer ? `${fmt(phys.closer.spring_preload_Nm)} N·m` : "–"], ["spring rate", phys.closer ? `${fmt(phys.closer.spring_stiffness_Nm_per_rad)} N·m/rad` : "–"], ["damping close / open", phys.closer ? `${fmt(phys.closer.damping_closing)} / ${fmt(phys.closer.damping_opening)}` : "–"], ["closing time (est.)", phys.closer?.closing_time_est_s != null ? `${fmt(phys.closer.closing_time_est_s)} s` : "–"]]} />
        <h3>Operator · latch · lock</h3>
        <KV rows={[["operator", spec?.operator?.model?.replace(/_/g, " ")], ["height", spec ? `${spec.operator.height} m` : "–"], ["travel", phys.latch ? `${fmt(phys.latch.operator_travel)} ${spec?.operator?.model?.includes("panic") ? "m" : "rad"}` : "–"], ["return spring", phys.latch ? `${fmt(phys.latch.operator_spring_preload)} + ${fmt(phys.latch.operator_spring_rate)}·q` : "–"], ["yield (damage)", phys.latch ? `${fmt(phys.latch.operator_yield)}` : "–"], ["latch", phys.latch?.model?.replace(/_/g, " ")], ["throw", phys.latch ? `${fmt((phys.latch.throw_m ?? 0) * 1000, 1)} mm` : "–"], ["bolt spring", phys.latch ? `${fmt(phys.latch.bolt_spring_preload_N)} N + ${fmt(phys.latch.bolt_spring_rate_N_per_m)} N/m` : "–"], ["lock", phys.lock?.model?.replace(/_/g, " ")], ["engaged", phys.lock?.engaged], ["robot-side release", phys.lock?.robot_side_release], ["locked backlash", phys.lock ? `${fmt((phys.lock.handle_backlash_locked_rad ?? 0) * 57.3, 1)}°` : "–"], ["deadbolt throw", phys.lock ? `${fmt((phys.lock.deadbolt_throw_m ?? 0) * 1000, 1)} mm` : "–"], ["code", phys.lock?.code ?? "–"]]} />
        <h3>Compliance (as simulated)</h3>
        <KV rows={[["opening force (start)", phys.compliance?.opening_force_start_N != null ? `${fmt(phys.compliance.opening_force_start_N)} N` : "–"], ["opening force (90°)", phys.compliance?.opening_force_90deg_N != null ? `${fmt(phys.compliance.opening_force_90deg_N)} N` : "–"], ["operator force", phys.compliance?.operator_force_N != null ? `${fmt(phys.compliance.operator_force_N)} N` : "–"], ["ADA 5 lbf interior", phys.compliance?.ada_interior_5lbf_ok], ["IBC fire/exterior", phys.compliance?.ibc_fire_exterior_ok], ["hardware ≤ 5 lbf", phys.compliance?.hardware_operable_5lbf_ok], ["ADA clear width", phys.compliance?.clear_width_ada_ok]]} />
        <h3>Damage thresholds</h3>
        <KV rows={[["leaf dent", phys.damage ? `${fmt(phys.damage.leaf_dent_force_N)} N` : "–"], ["leaf puncture", phys.damage ? `${fmt(phys.damage.leaf_puncture_force_N)} N` : "–"], ["glass break", phys.damage?.glass_break_force_N != null ? `${fmt(phys.damage.glass_break_force_N)} N` : "–"], ["operator yield", phys.damage ? `${fmt(phys.damage.operator_yield_torque_Nm)} N·m` : "–"], ["latch shear", phys.damage ? `${fmt(phys.damage.latch_shear_yield_N)} N` : "–"], ["hinge tear-out", phys.damage ? `${fmt(phys.damage.hinge_tearout_force_N)} N` : "–"], ["slam velocity", phys.damage ? `${fmt(phys.damage.slam_velocity_rad_s)} rad/s` : "–"]]} />
        {qa && (<><h3>QA sign-off</h3><KV rows={Object.entries(qa.checks ?? {}).map(([k, v]) => [k.replace(/_/g, " "), v ? "pass" : "FAIL"])} /></>)}
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
