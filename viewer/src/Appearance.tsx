import React, { useEffect, useState } from "react";

export const APPEARANCE = "./appearance";
export type AppearanceRender = {
  door_id: string; variant: number; image: string | null; metadata: string; blend: string | null;
  recipe: { wall: string; floor: string; door_finish: string; lighting: string; seed: number };
  quality: "preview" | "photo";
};
type AppearanceIndex = { schema_version: number; renders: AppearanceRender[] };
let indexPromise: Promise<AppearanceIndex | null> | undefined;
export function useAppearance() {
  const [index, setIndex] = useState<AppearanceIndex | null>(null);
  useEffect(() => {
    let live = true;
    indexPromise ??= fetch(`${APPEARANCE}/index.json`).then((r) => r.ok ? r.json() : null)
      .then((x) => x?.schema_version === 1 && Array.isArray(x.renders) ? x : null).catch(() => null);
    indexPromise.then((x) => { if (live) setIndex(x); });
    return () => { live = false; };
  }, []);
  return index;
}

export function AppearanceThumb({ render, fallback, alt }: { render?: AppearanceRender; fallback: string; alt: string }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [render?.image]);
  const photo = render?.image && !failed;
  return <div style={{ position: "relative" }}>
    <img src={photo ? `${APPEARANCE}/${render.image}` : fallback} loading="lazy" alt={alt}
      style={photo ? { aspectRatio: "1 / 1", objectFit: "contain" } : undefined} onError={() => setFailed(true)} />
    {photo && <span style={{ position: "absolute", left: 10, bottom: 10, background: "#17201cce", color: "#e8f0e9", padding: "4px 8px", borderRadius: 4, fontSize: 11 }}>Blender · {render.quality === "photo" ? "photo" : "preview"}</span>}
  </div>;
}

const nice = (s: string) => s.replace(/^(wall|floor)_/, "").replace(/_/g, " ");
export function AppearancePanel({ id }: { id: string }) {
  const index = useAppearance();
  const renders = index?.renders.filter((r) => r.door_id === id && r.image) ?? [];
  const [selected, setSelected] = useState(0);
  useEffect(() => setSelected(0), [id]);
  if (!renders.length) return null;
  const r = renders[Math.min(selected, renders.length-1)];
  return <section style={{ marginTop: 18 }}>
    <h3>Blender appearance</h3>
    {renders.length > 1 && <select aria-label="Rendered appearance" value={selected} onChange={(e) => setSelected(Number(e.target.value))}>
      {renders.map((x,i) => <option key={`${x.variant}-${i}`} value={i}>{nice(x.recipe.wall)} · {nice(x.recipe.floor)} · {nice(x.recipe.door_finish)} · {nice(x.recipe.lighting)}</option>)}
    </select>}
    <a href={`${APPEARANCE}/${r.image}`} target="_blank" rel="noreferrer"><img src={`${APPEARANCE}/${r.image}`} alt={`${id} with ${nice(r.recipe.wall)} and ${nice(r.recipe.floor)}`} style={{ width: "100%", borderRadius: 8, marginTop: 8 }} /></a>
    <p className="sub">{nice(r.recipe.wall)} · {nice(r.recipe.floor)} · {nice(r.recipe.door_finish)} · {nice(r.recipe.lighting)}</p>
    <p className="sub">Saved pose. The interactive view above controls the simulation model.</p>
    <a href={`${APPEARANCE}/${r.metadata}`} target="_blank" rel="noreferrer">Appearance recipe</a>
    {r.blend && <> · <a href={`${APPEARANCE}/${r.blend}`} download>Blender scene</a></>}
  </section>;
}
