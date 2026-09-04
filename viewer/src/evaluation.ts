// "Show evaluation" overlay: start zone + sampled starts, approach point, handle targets with reward labels, the
// "open" / "clear" thresholds, the pass plane, the goal zone and (for human scenarios) the person's timed path.
import * as THREE from "three";
import type { ModelJ, ScenarioJ } from "./types";
import type { BuiltScene } from "./scene";
import { REWARD_LABELS } from "./glossary";

export interface EvalOverlay {
  group: THREE.Group;
  humanDuration: number;          // seconds (0 without a human)
  update: () => void;             // re-place markers attached to moving bodies (call once per frame)
  setHumanTime: (t: number) => void;
  dispose: () => void;
}

const C = { start: 0x5bc0eb, goal: 0x4ade80, plane: 0x5bc0eb, handle: 0xe0a458, human: 0xf28c4a, muted: 0x9aa3b2, open: 0xe0a458 };

// small deterministic PRNG so the sampled starts are stable between renders (the Python side uses random.Random with
// the same formula; the streams differ, the distribution is identical)
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleStarts(sc: ScenarioJ, n: number): { x: number; y: number; yaw: number }[] {
  const st = sc.start;
  const out = [];
  for (let i = 0; i < n; i++) {
    const rnd = mulberry32((st.randomize?.seed_base ?? 0) + i + 1);
    const u1 = rnd(), u2 = rnd(), u3 = rnd();
    const r = (st.randomize?.radius ?? st.radius) * Math.sqrt(u1);
    const phi = 2 * Math.PI * u2;
    out.push({ x: st.center[0] + r * Math.cos(phi), y: st.center[1] + r * Math.sin(phi), yaw: st.yaw + (2 * u3 - 1) * (st.randomize?.yaw_jitter_rad ?? 0.35) });
  }
  return out;
}

export function humanPose(path: [number, number, number][], t: number): [number, number] {
  if (!path.length) return [0, 0];
  if (t <= path[0][0]) return [path[0][1], path[0][2]];
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i], b = path[i + 1];
    if (t <= b[0]) { const s = (t - a[0]) / Math.max(1e-6, b[0] - a[0]); return [a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2])]; }
  }
  return [path[path.length - 1][1], path[path.length - 1][2]];
}

export function makeLabel(text: string, color = "#e6e8ee", height = 0.11, bg = "rgba(23,26,33,0.88)"): THREE.Sprite {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  const font = "600 30px ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif";
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 28, h = 46;
  canvas.width = w; canvas.height = h;
  ctx.font = font;
  ctx.fillStyle = bg;
  ctx.beginPath(); ctx.roundRect(1, 1, w - 2, h - 2, 10); ctx.fill();
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, 14, h / 2 + 1);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
  const sp = new THREE.Sprite(mat);
  sp.scale.set((w / h) * height, height, 1);
  sp.renderOrder = 1000;
  return sp;
}

function disc(cx: number, cy: number, cz: number, r: number, color: number, opacity = 0.18): THREE.Group {
  const g = new THREE.Group();
  const fill = new THREE.Mesh(new THREE.CircleGeometry(r, 48), new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide }));
  const ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.015, r, 64), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide }));
  fill.position.set(cx, cy, cz + 0.004); ring.position.set(cx, cy, cz + 0.005);
  g.add(fill, ring);
  return g;
}

function polyline(points: THREE.Vector3[], color: number, opacity = 1): THREE.Line {
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: false }));
}

function arrow(x: number, y: number, z: number, yaw: number, len: number, color: number): THREE.ArrowHelper {
  const dir = new THREE.Vector3(Math.cos(yaw), Math.sin(yaw), 0);
  const a = new THREE.ArrowHelper(dir, new THREE.Vector3(x, y, z), len, color, 0.09, 0.06);
  (a.line.material as THREE.Material).depthWrite = false;
  return a;
}

function siteWorld(built: BuiltScene, model: ModelJ, siteName: string): THREE.Vector3 | null {
  for (const b of model.bodies) {
    const s = b.sites.find((x) => x.name === siteName);
    if (!s) continue;
    const node = built.bodies.get(b.name);
    if (!node) return null;
    const container = (node as any).userData.container as THREE.Object3D;
    return container.localToWorld(new THREE.Vector3(s.pos[0], s.pos[1], s.pos[2]));
  }
  return null;
}

function bodyWorld(built: BuiltScene, bodyName: string): THREE.Vector3 | null {
  const node = built.bodies.get(bodyName);
  if (!node) return null;
  const container = (node as any).userData.container as THREE.Object3D;
  return container.getWorldPosition(new THREE.Vector3());
}

function rewardText(sc: ScenarioJ, ev: string, extra = ""): string | null {
  const v = sc.rewards[ev];
  if (v === undefined) return null;
  return `${v > 0 ? "+" : ""}${Number.isInteger(v) ? v : v.toFixed(2)} ${REWARD_LABELS[ev] ?? ev}${extra}`;
}

export function buildEvaluationOverlay(sc: ScenarioJ, model: ModelJ, built: BuiltScene): EvalOverlay {
  const group = new THREE.Group();
  group.name = "evaluation";
  const meta = model.meta ?? {};
  const disposables: THREE.Object3D[] = [];
  const add = (o: THREE.Object3D) => { group.add(o); disposables.push(o); return o; };

  // ---- start zone + sampled starts
  const st = sc.start;
  add(disc(st.center[0], st.center[1], st.center[2], st.radius, C.start, 0.16));
  add(arrow(st.center[0], st.center[1], st.center[2] + 0.02, st.yaw, 0.45, C.start));
  for (const s of sampleStarts(sc, 6)) add(arrow(s.x, s.y, st.center[2] + 0.015, s.yaw, 0.28, C.muted));
  const lStart = makeLabel("start zone (sampled starts)", "#5bc0eb"); lStart.position.set(st.center[0], st.center[1] - st.radius - 0.2, st.center[2] + 0.12); add(lStart);

  // ---- approach point
  const ap = sc.approach_point;
  const apm = new THREE.Mesh(new THREE.OctahedronGeometry(0.05), new THREE.MeshBasicMaterial({ color: C.start, depthWrite: false }));
  apm.position.set(ap[0], ap[1], ap[2] + 0.06); add(apm);
  const lAp = makeLabel("approach", "#5bc0eb", 0.085); lAp.position.set(ap[0] + 0.35, ap[1], ap[2] + 0.16); add(lAp);
  add(polyline([new THREE.Vector3(st.center[0], st.center[1], st.center[2] + 0.01), new THREE.Vector3(ap[0], ap[1], ap[2] + 0.01)], C.start, 0.5));

  // ---- pass plane
  const pp = sc.pass_plane;
  const horizontal = Math.abs(pp.normal[2]) > 0.5;
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(pp.width, pp.height), new THREE.MeshBasicMaterial({ color: C.plane, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(plane.geometry), new THREE.LineBasicMaterial({ color: C.plane, transparent: true, opacity: 0.8, depthWrite: false }));
  for (const o of [plane, edge]) { o.position.set(pp.center[0], pp.center[1], pp.center[2]); if (!horizontal) o.rotation.x = -Math.PI / 2; add(o); }
  const tTrav = rewardText(sc, "traversed");
  if (tTrav) { const l = makeLabel(tTrav, "#5bc0eb", 0.13); l.position.set(pp.center[0], pp.center[1], horizontal ? pp.center[2] + 0.25 : pp.center[2] + pp.height / 2 + 0.16); add(l); }
  const dir = new THREE.Vector3(pp.traverse_direction[0], pp.traverse_direction[1], pp.traverse_direction[2]);
  add(new THREE.ArrowHelper(dir, new THREE.Vector3(pp.center[0] - dir.x * 0.3, pp.center[1] - dir.y * 0.3, pp.center[2] - dir.z * 0.3), 0.6, C.plane, 0.12, 0.08));

  // ---- goal zone
  if (sc.goal) {
    const g = sc.goal;
    add(disc(g.center[0], g.center[1], g.center[2], g.radius, C.goal, 0.16));
    const tClose = rewardText(sc, "closed_behind");
    const l = makeLabel(tClose ? `goal · ${tClose}` : "goal", "#4ade80"); l.position.set(g.center[0], g.center[1] + g.radius + 0.1, g.center[2] + 0.25); add(l);
  }

  // ---- "open" threshold: arc of the leaf's free edge (hinged) or a travel mark (sliding)
  const thr = sc.thresholds;
  const W = (model.meta?.leaf_edge_x_local != null) ? Math.abs(model.meta.leaf_edge_x_local) : null;
  const hx = meta.hinge_x, u = meta.u, v = meta.v, wy = meta.wall_y ?? 0;
  const tOpen = rewardText(sc, "opened");
  if (thr.open_rad != null && hx != null && u != null && v != null && W) {
    const arcPts = (a0: number, a1: number) => { const pts = []; for (let i = 0; i <= 24; i++) { const th = a0 + (a1 - a0) * i / 24; pts.push(new THREE.Vector3(hx + u * W * Math.cos(th), wy + v * W * Math.sin(th), 0.006)); } return pts; };
    add(polyline(arcPts(0, thr.open_rad), C.open));
    const eo = arcPts(thr.open_rad, thr.open_rad)[0];
    add(polyline([new THREE.Vector3(hx, wy, 0.006), eo], C.open, 0.7));
    if (tOpen) { const l = makeLabel(`${tOpen} ≥ ${Math.round(thr.open_rad * 180 / Math.PI)}°`, "#e0a458", 0.1); l.position.set(eo.x, eo.y, 0.22); add(l); }
    if (thr.clear_rad != null && thr.clear_rad > thr.open_rad) {
      add(polyline(arcPts(thr.open_rad, thr.clear_rad), C.muted, 0.8));
      const ec = arcPts(thr.clear_rad, thr.clear_rad)[0];
      const l = makeLabel(`clear ≥ ${Math.round(thr.clear_rad * 180 / Math.PI)}°`, "#9aa3b2", 0.085); l.position.set(ec.x, ec.y, 0.14); add(l);
    }
  } else if (tOpen) {
    const t = thr.open_m != null ? `${tOpen} ≥ ${(thr.open_m * 100).toFixed(0)} cm` : thr.open_rad != null ? `${tOpen} ≥ ${Math.round(thr.open_rad * 180 / Math.PI)}°` : tOpen;
    const l = makeLabel(t, "#e0a458", 0.1); l.position.set(pp.center[0], pp.center[1], horizontal ? pp.center[2] + 0.12 : Math.max(0.3, pp.center[2] - pp.height / 2 + 0.3)); add(l);
  }

  // ---- handle targets (follow the hardware) + unlatch marker on the bolt
  const handleMarkers: { site: string; mesh: THREE.Mesh; label?: THREE.Sprite }[] = [];
  sc.handle_targets.forEach((site, i) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.024, 16, 12), new THREE.MeshBasicMaterial({ color: C.handle, depthTest: false }));
    m.renderOrder = 999;
    add(m);
    let label: THREE.Sprite | undefined;
    if (i === 0) { const t = rewardText(sc, "touch_handle"); if (t) { label = makeLabel(t, "#e0a458", 0.1); add(label); } }
    handleMarkers.push({ site, mesh: m, label });
  });
  let boltMarker: { body: string; mesh: THREE.Mesh; label: THREE.Sprite } | null = null;
  const tUnlatch = rewardText(sc, "unlatch");
  if (tUnlatch) {
    const bolt = model.bodies.find((b) => b.joint && b.joint.role === "latch" && b.joint.name.endsWith("latch_bolt_slide")) ?? model.bodies.find((b) => b.joint && b.joint.role === "latch");
    if (bolt) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 12), new THREE.MeshBasicMaterial({ color: C.handle, depthTest: false }));
      m.renderOrder = 999;
      const label = makeLabel(tUnlatch, "#e0a458", 0.09);
      add(m); add(label);
      boltMarker = { body: bolt.name, mesh: m, label };
    }
  }

  // ---- human path + figure
  let humanFigure: THREE.Group | null = null;
  let humanDuration = 0;
  const h = sc.human;
  if (h && h.path.length) {
    humanDuration = h.path[h.path.length - 1][0];
    add(polyline(h.path.map((p) => new THREE.Vector3(p[1], p[2], 0.02)), C.human));
    const t0 = h.path[0][0];
    for (let t = Math.ceil(t0); t <= humanDuration + 1e-6; t += 1) {
      const [x, y] = humanPose(h.path, t);
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 8), new THREE.MeshBasicMaterial({ color: C.human, depthWrite: false }));
      m.position.set(x, y, 0.03); add(m);
      if (t % 2 === 0 || t === Math.ceil(t0)) { const l = makeLabel(`${t.toFixed(0)} s`, "#f28c4a", 0.07); l.position.set(x, y, 0.12); add(l); }
    }
    const fig = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(h.radius_m, Math.max(0.1, h.height_m - 2 * h.radius_m), 6, 16), new THREE.MeshStandardMaterial({ color: C.human, transparent: true, opacity: 0.55, roughness: 0.7 }));
    body.rotation.x = Math.PI / 2; body.position.z = h.height_m / 2;
    fig.add(body);
    const head = makeLabel([rewardText(sc, "held_for_human") ?? rewardText(sc, "yielded_to_human") ?? "person", rewardText(sc, "collision_with_human")].filter(Boolean).join(" · "), "#f28c4a", 0.1);
    head.position.z = h.height_m + 0.18; fig.add(head);
    const [x0, y0] = humanPose(h.path, t0);
    fig.position.set(x0, y0, 0);
    add(fig);
    humanFigure = fig;
    const lp = makeLabel(h.direction === "same_as_robot" ? "person follows the robot" : "person comes through first", "#f28c4a", 0.085);
    const [xs, ys] = humanPose(h.path, t0); lp.position.set(xs, ys, 0.35); add(lp);   // at the path start, near the floor
  }

  function update() {
    built.root.updateMatrixWorld(true);
    for (const hm of handleMarkers) {
      const p = siteWorld(built, model, hm.site);
      if (!p) continue;
      hm.mesh.position.copy(p);
      if (hm.label) hm.label.position.set(p.x, p.y, p.z + 0.14);
    }
    if (boltMarker) {
      const p = bodyWorld(built, boltMarker.body);
      if (p) { boltMarker.mesh.position.copy(p); boltMarker.label.position.set(p.x, p.y, p.z - 0.14); }
    }
  }
  update();

  function setHumanTime(t: number) {
    if (!humanFigure || !h) return;
    const [x, y] = humanPose(h.path, t);
    humanFigure.position.set(x, y, 0);
  }

  function dispose() {
    group.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
      const mat = (m as any).material as THREE.Material | THREE.Material[] | undefined;
      if (mat) for (const mm of Array.isArray(mat) ? mat : [mat]) { const map = (mm as any).map as THREE.Texture | undefined; if (map) map.dispose(); mm.dispose(); }
    });
  }
  return { group, humanDuration, update, setHumanTime, dispose };
}
