import React, { useMemo, useState } from "react";
import { FAMILY_LABELS, type Manifest, type ManifestDoor } from "./types";
import { ASSETS } from "./App";

function uniq(xs: string[]) { return Array.from(new Set(xs)).sort(); }

export function thumbUrl(d: ManifestDoor, which = "iso") {
  const t = d.thumbs.find((x) => x.includes(`thumb_${which}.jpg`)) ?? d.thumbs[0];
  return t ? `${ASSETS}/doors/${t}` : "";
}

export function DoorCard({ d }: { d: ManifestDoor }) {
  return (
    <a className="card" href={`#/door/${d.id}`}>
      <img src={thumbUrl(d)} loading="lazy" alt={d.use_case} />
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
        </div>
      </div>
    </a>
  );
}

export function Catalogue({ manifest, query }: { manifest: Manifest; query: string }) {
  // Deep-linkable filters (`#/?family=…&context=…&tag=…&slab=…&operator=…&lock=…&closer=…&q=…`): the Hierarchy page
  // links every node to the catalogue with exactly the filter that reproduces the node's doors (taxonomy.variant_of).
  const params = new URLSearchParams(query);
  const [family, setFamily] = useState(params.get("family") ?? "");
  const [operator, setOperator] = useState(params.get("operator") ?? "");
  const [lock, setLock] = useState(params.get("lock") ?? "");
  const [closer, setCloser] = useState(params.get("closer") ?? "");
  const [condition, setCondition] = useState(params.get("condition") ?? "");
  const [task, setTask] = useState(params.get("task") ?? "");
  const [scenario, setScenario] = useState(params.get("scenario") ?? "");
  const [signed, setSigned] = useState("");
  const [search, setSearch] = useState(params.get("q") ?? "");
  const [sort, setSort] = useState(params.get("sort") ?? "index");
  const [context, setContext] = useState(params.get("context") ?? "");   // exact-match facets without a dropdown: shown as removable chips
  const [tag, setTag] = useState(params.get("tag") ?? "");
  const [slab, setSlab] = useState(params.get("slab") ?? "");
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
      (!context || d.context === context) && (!tag || d.tags.includes(tag)) && (!slab || d.leaf.slab === slab) &&
      (!q || [d.id, d.use_case, d.family, d.context, d.leaf.slab, d.leaf.panel_style, d.operator, d.lock, d.hinge, ...d.tags, ...d.extras].join(" ").toLowerCase().includes(q)));
    const key: Record<string, (d: ManifestDoor) => number | string> = { index: (d) => d.index, mass: (d) => d.mass_kg, difficulty: (d) => d.difficulty, width: (d) => d.leaf.width, family: (d) => d.family };
    const f = key[sort] ?? key.index;
    xs = [...xs].sort((a, b) => (f(a) < f(b) ? -1 : f(a) > f(b) ? 1 : 0));
    return xs;
  }, [doors, family, operator, lock, closer, condition, task, scenario, signed, search, sort, context, tag, slab]);
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
        {context && <span className="active-filter">context: {context.replace(/_/g, " ")}<button type="button" aria-label="Clear context filter" onClick={() => setContext("")}>✕</button></span>}
        {tag && <span className="active-filter">tag: {tag.replace(/_/g, " ")}<button type="button" aria-label="Clear tag filter" onClick={() => setTag("")}>✕</button></span>}
        {slab && <span className="active-filter">slab: {slab.replace(/_/g, " ")}<button type="button" aria-label="Clear slab filter" onClick={() => setSlab("")}>✕</button></span>}
        <select value={signed} onChange={(e) => setSigned(e.target.value)}><option value="">QA: all</option><option value="yes">signed off</option><option value="no">needs review</option></select>
        <select value={sort} onChange={(e) => setSort(e.target.value)}><option value="index">Sort: id</option><option value="family">Sort: type</option><option value="mass">Sort: mass</option><option value="difficulty">Sort: difficulty</option><option value="width">Sort: width</option></select>
        <span className="count">{filtered.length} / {doors.length}</span>
      </div>
      <div className="grid">{filtered.map((d) => <DoorCard key={d.id} d={d} />)}</div>
    </div>
  );
}
