import React, { useMemo, useState } from "react";
import { FAMILY_LABELS, type Manifest, type ManifestDoor } from "./types";
import { ASSETS } from "./App";
import { BaselineBadges } from "./ResultBadges";
import { AppearanceThumb, useAppearance, type AppearanceRender } from "./Appearance";

function uniq(xs: string[]) { return Array.from(new Set(xs)).sort(); }

export function thumbUrl(d: ManifestDoor, which = "iso") {
  const t = d.thumbs.find((x) => x.includes(`thumb_${which}.jpg`)) ?? d.thumbs[0];
  return t ? `${ASSETS}/doors/${t}` : "";
}

export function DoorCard({ d, appearance }: { d: ManifestDoor; appearance?: AppearanceRender }) {
  return (
    <a className="card" href={`#/door/${d.id}`}>
      <AppearanceThumb render={appearance} fallback={thumbUrl(d)} alt={d.use_case} />
      <div className="body">
        <div className="title"><span>{d.use_case || d.id}</span><span style={{ color: "var(--muted)", fontWeight: 400 }}>{d.mass_kg.toFixed(0)} kg</span></div>
        <div className="sub">{d.id} · {d.leaf.width.toFixed(2)}×{d.leaf.height.toFixed(2)} m · {d.leaf.slab.replace(/_/g, " ")}</div>
        <div className="chips">
          <span className="chip fam">{FAMILY_LABELS[d.family] ?? d.family}</span>
          <span className="chip">{d.operator.replace(/_/g, " ")}</span>
          {d.lock !== "none" && <span className="chip">{d.lock.replace(/_/g, " ")}{d.lock_engaged ? " (locked)" : ""}</span>}
          {d.closer !== "none" && <span className="chip">closer</span>}
          <span className="chip">{d.condition}</span>
          <span className="chip">{d.task.replace(/_/g, " ")}</span>
          {d.benchmark?.has_human && <span className="chip">human scenario</span>}
          <span className="chip">L{d.difficulty}</span>
          <span className={"chip " + (d.signed_off ? "ok" : "bad")}>{d.signed_off ? "signed off" : "needs review"}</span>
          <BaselineBadges id={d.id} />
        </div>
      </div>
    </a>
  );
}

export function Catalogue({ manifest, query }: { manifest: Manifest; query: string }) {
  const appearance = useAppearance();
  const [imageMode, setImageMode] = useState("blender");
  const photoById = useMemo(() => {
    const result = new Map<string, AppearanceRender>();
    for (const r of appearance?.renders ?? []) {
      const current = result.get(r.door_id);
      if (r.image && (!current || (r.quality === "photo" && current.quality !== "photo"))) result.set(r.door_id, r);
    }
    return result;
  }, [appearance]);
  const params = new URLSearchParams(query);
  const [family, setFamily] = useState(params.get("family") ?? "");
  const [operator, setOperator] = useState("");
  const [lock, setLock] = useState("");
  const [closer, setCloser] = useState("");
  const [condition, setCondition] = useState("");
  const [task, setTask] = useState("");
  const [scenario, setScenario] = useState("");
  const [signed, setSigned] = useState("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("index");
  const doors = manifest.doors.filter((d) => !d.error);
  const opts = useMemo(() => ({
    operator: uniq(doors.map((d) => d.operator)), lock: uniq(doors.map((d) => d.lock)), closer: uniq(doors.map((d) => d.closer)),
    condition: uniq(doors.map((d) => d.condition)), task: uniq(doors.map((d) => d.task)),
    scenario: uniq(doors.flatMap((d) => d.benchmark?.scenarios ?? [])),
  }), [doors]);
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    let xs = doors.filter((d) =>
      (!family || d.family === family) && (!operator || d.operator === operator) && (!lock || d.lock === lock) && (!closer || (closer === "any" ? d.closer !== "none" : d.closer === closer)) &&
      (!condition || d.condition === condition) && (!task || d.task === task) && (!scenario || (d.benchmark?.scenarios ?? []).includes(scenario)) && (!signed || (signed === "yes") === d.signed_off) &&
      (!q || [d.id, d.use_case, d.family, d.context, d.leaf.slab, d.leaf.panel_style, d.operator, d.lock, d.hinge, ...d.tags, ...d.extras].join(" ").toLowerCase().includes(q)));
    const key: Record<string, (d: ManifestDoor) => number | string> = { index: (d) => d.index, mass: (d) => d.mass_kg, difficulty: (d) => d.difficulty, width: (d) => d.leaf.width, family: (d) => d.family };
    const f = key[sort] ?? key.index;
    xs = [...xs].sort((a, b) => (f(a) < f(b) ? -1 : f(a) > f(b) ? 1 : 0));
    return xs;
  }, [doors, family, operator, lock, closer, condition, task, scenario, signed, search, sort]);
  const sel = (v: string, set: (s: string) => void, list: string[], label: string) => (
    <select value={v} onChange={(e) => set(e.target.value)}><option value="">{label}</option>{list.map((x) => <option key={x} value={x}>{x.replace(/_/g, " ")}</option>)}</select>
  );
  return (
    <div>
      <div className="filters">
        <input type="search" placeholder="Search (id, use case, slab, tags…)" value={search} onChange={(e) => setSearch(e.target.value)} />
        <select value={family} onChange={(e) => setFamily(e.target.value)}><option value="">All door types</option>{manifest.families.map((f) => <option key={f} value={f}>{FAMILY_LABELS[f] ?? f}</option>)}</select>
        {sel(operator, setOperator, opts.operator, "Any operator")}
        {sel(lock, setLock, opts.lock, "Any lock")}
        <select value={closer} onChange={(e) => setCloser(e.target.value)}><option value="">Any closer</option><option value="any">Has closer</option>{opts.closer.map((x) => <option key={x} value={x}>{x.replace(/_/g, " ")}</option>)}</select>
        {sel(condition, setCondition, opts.condition, "Any condition")}
        {sel(task, setTask, opts.task, "Any task")}
        {opts.scenario.length > 0 && sel(scenario, setScenario, opts.scenario, "Any scenario")}
        <select value={signed} onChange={(e) => setSigned(e.target.value)}><option value="">QA: all</option><option value="yes">signed off</option><option value="no">needs review</option></select>
        <select value={sort} onChange={(e) => setSort(e.target.value)}><option value="index">Sort: id</option><option value="family">Sort: type</option><option value="mass">Sort: mass</option><option value="difficulty">Sort: difficulty</option><option value="width">Sort: width</option></select>
        {photoById.size > 0 && <select aria-label="Thumbnail rendering" value={imageMode} onChange={(e) => setImageMode(e.target.value)}><option value="blender">Blender renders ({photoById.size} available)</option><option value="simulation">Simulation thumbnails</option></select>}
        <span className="count">{filtered.length} / {doors.length}</span>
      </div>
      <div className="grid">{filtered.map((d) => <DoorCard key={d.id} d={d} appearance={imageMode === "blender" ? photoById.get(d.id) : undefined} />)}</div>
    </div>
  );
}
