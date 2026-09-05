// Minimal canvas line plots for the playground (no chart library: the whole page is redrawn at frame rate from the
// recorder's ring buffers).
import type { Recorder } from "./sim";

export interface Series { label: string; color: string; get: (i: number) => number }
export interface PlotSpec {
  title: string;
  unit: string;
  series: Series[];
  window: number;                  // seconds shown
  yLines?: { y: number; label: string; color?: string }[];
  yMin?: number;                   // optional fixed floor (e.g. 0 for forces)
}

const font = "11px ui-sans-serif, -apple-system, Segoe UI, Roboto, sans-serif";

export function drawPlot(canvas: HTMLCanvasElement, rec: Recorder, spec: PlotSpec) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (W === 0 || H === 0) return;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const padL = 44, padR = 8, padT = 18, padB = 16;
  const pw = W - padL - padR, ph = H - padT - padB;
  ctx.font = font;
  ctx.fillStyle = "#9aa3b2";
  ctx.fillText(spec.title, padL, 12);
  const idx = rec.indices();
  if (!idx.length) { ctx.fillText("no data yet", padL + 6, padT + ph / 2); return; }
  const tEnd = rec.t[idx[idx.length - 1]], tStart = Math.max(rec.t[idx[0]], tEnd - spec.window);
  const t0 = tEnd - spec.window;
  // visible range
  let lo = Infinity, hi = -Infinity;
  for (const i of idx) { if (rec.t[i] < t0) continue; for (const s of spec.series) { const y = s.get(i); if (Number.isFinite(y)) { if (y < lo) lo = y; if (y > hi) hi = y; } } }
  for (const l of spec.yLines ?? []) { if (l.y < lo) lo = l.y; if (l.y > hi) hi = l.y; }
  if (spec.yMin !== undefined) lo = Math.min(lo, spec.yMin);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = -1; hi = 1; }
  if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
  const m = 0.08 * (hi - lo); lo -= m; hi += m;
  const X = (t: number) => padL + ((t - t0) / spec.window) * pw;
  const Y = (y: number) => padT + (1 - (y - lo) / (hi - lo)) * ph;
  // axes + grid
  ctx.strokeStyle = "#2a2f3a"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.rect(padL, padT, pw, ph); ctx.stroke();
  const ticks = niceTicks(lo, hi, 4);
  ctx.fillStyle = "#9aa3b2"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const y of ticks) { const py = Y(y); ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(padL + pw, py); ctx.strokeStyle = "#20252f"; ctx.stroke(); ctx.fillText(fmtTick(y), padL - 4, py); }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let k = 0; k <= 4; k++) { const t = t0 + (k / 4) * spec.window; ctx.fillText(`${t.toFixed(1)} s`, X(t), padT + ph + 3); }
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.fillText(spec.unit, padL + 4, padT + 11);
  // reference lines
  for (const l of spec.yLines ?? []) {
    const py = Y(l.y);
    ctx.setLineDash([4, 4]); ctx.strokeStyle = l.color ?? "#e0a45888"; ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(padL + pw, py); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = l.color ?? "#e0a458"; ctx.textAlign = "right"; ctx.fillText(l.label, padL + pw - 4, py - 3); ctx.textAlign = "left";
  }
  // series
  ctx.save();
  ctx.beginPath(); ctx.rect(padL, padT, pw, ph); ctx.clip();
  spec.series.forEach((s, si) => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.4;
    ctx.beginPath();
    let first = true;
    for (const i of idx) {
      const t = rec.t[i];
      if (t < tStart) continue;
      const y = s.get(i);
      if (!Number.isFinite(y)) continue;
      const px = X(t), py = Y(y);
      if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    // legend
    ctx.fillStyle = s.color;
    ctx.fillText(s.label, padL + 40 + si * 110, 12);
  });
  ctx.restore();
}

function niceTicks(lo: number, hi: number, n: number): number[] {
  const span = hi - lo;
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let y = Math.ceil(lo / step) * step; y <= hi + 1e-9; y += step) out.push(Math.abs(y) < 1e-12 ? 0 : y);
  return out;
}
const fmtTick = (y: number) => Math.abs(y) >= 100 ? y.toFixed(0) : Math.abs(y) >= 10 ? y.toFixed(1) : y.toFixed(2);
