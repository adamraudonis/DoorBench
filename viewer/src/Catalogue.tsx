import { isPetDoor } from "./collections";
import React, { useEffect, useMemo, useState } from "react";
import { FAMILY_LABELS, type Manifest, type ManifestDoor } from "./types";
import { ASSETS } from "./App";
import { AppearanceThumb, useAppearance, type AppearanceRender } from "./Appearance";
import { DATASET, formatMass, Icon } from "./SiteUI";

const nice = (s: string) => s.replace(/_/g, " ");
const uniq = (xs: string[]) => Array.from(new Set(xs)).sort();
const PAGE_SIZE = 24;

export function thumbUrl(d: ManifestDoor, which = "iso") {
  const t = d.thumbs.find((x) => x.includes(`thumb_${which}.jpg`)) ?? d.thumbs[0];
  return t ? `${ASSETS}/doors/${t}` : "";
}

export function DoorCard({ d, appearance }: { d: ManifestDoor; appearance?: AppearanceRender }) {
  return <a className="card door-card" href={`#/door/${d.id}`}>
    <div className="card-image"><AppearanceThumb render={appearance} fallback={thumbUrl(d)} alt={d.use_case || d.id} /><span className="card-open" aria-hidden="true"><Icon name="arrow" /></span>
      {!isPetDoor(d) && d.isaac_parity && d.isaac_parity !== "untested" && <span className={`card-parity ${d.isaac_parity === "ok" ? "passed" : "mismatch"}`} title={`MuJoCo / Isaac Sim parity gate: ${d.isaac_parity === "ok" ? "passed" : "mismatch"} (grade ${d.isaac_parity_grade ?? "?"}). See this door’s Isaac parity details.`}>
        Isaac {d.isaac_parity === "ok" ? "parity" : "mismatch"}{d.isaac_parity_grade ? ` ${d.isaac_parity_grade}` : ""}
      </span>}
    </div>
    <div className="body"><div className="card-eyebrow"><span>{FAMILY_LABELS[d.family] ?? nice(d.family)}</span>{!isPetDoor(d) && <span>L{d.difficulty}</span>}</div><h3>{d.use_case || nice(d.leaf.slab)}</h3><p className="card-material">{nice(d.leaf.slab)}</p>
      <div className="card-facts"><span>{d.leaf.width.toFixed(2)} × {d.leaf.height.toFixed(2)} <small>m</small></span><span>{formatMass(d.mass_kg)} <small>kg</small></span><span>{d.lock_engaged ? "Starts locked" : nice(d.operator)}</span></div>
      <div className="card-footer"><code>{d.id}</code><span className={`qa-indicator ${d.signed_off ? "passed" : "pending"}`} title={d.signed_off ? "Passed the automated generation QA gates. This is separate from a visual or physical review." : "One or more automated QA gates need attention."}><span />{d.signed_off ? "Automated QA" : "QA attention"}</span></div>
    </div>
  </a>;
}

export function Catalogue({ manifest, query, supplementaryCount = 0 }: { manifest: Manifest; query: string; supplementaryCount?: number }) {
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
  const [family, setFamily] = useState(new URLSearchParams(query).get("family") ?? "");
  useEffect(() => setFamily(new URLSearchParams(query).get("family") ?? ""), [query]);
  const [operator, setOperator] = useState("");
  const [lock, setLock] = useState("");
  const [closer, setCloser] = useState("");
  const [condition, setCondition] = useState("");
  const [task, setTask] = useState("");
  const [scenario, setScenario] = useState("");
  const [signed, setSigned] = useState("");
  const [parity, setParity] = useState(new URLSearchParams(query).get("parity") ?? "");
  useEffect(() => {
    const value = new URLSearchParams(query).get("parity") ?? "";
    setParity(value);
    if (value) setAdvanced(true);
  }, [query]);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("index");
  const [advanced, setAdvanced] = useState(() => !!new URLSearchParams(query).get("parity"));
  const [page, setPage] = useState(1);
  const doors = useMemo(() => manifest.doors.filter((d) => !d.error), [manifest]);
  const hasParity = !!manifest.isaac_parity || doors.some((d) => d.isaac_parity) || !!parity;
  const opts = useMemo(() => ({ operator: uniq(doors.map((d) => d.operator)), lock: uniq(doors.map((d) => d.lock)), closer: uniq(doors.map((d) => d.closer)), condition: uniq(doors.map((d) => d.condition)), task: uniq(doors.map((d) => d.task)), scenario: uniq(doors.flatMap((d) => d.benchmark?.scenarios ?? [])) }), [doors]);
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const xs = doors.filter((d) => (!family || d.family === family) && (!operator || d.operator === operator) && (!lock || d.lock === lock) && (!closer || (closer === "any" ? d.closer !== "none" : d.closer === closer)) && (!condition || d.condition === condition) && (!task || d.task === task) && (!scenario || (d.benchmark?.scenarios ?? []).includes(scenario)) && (!signed || (signed === "yes") === d.signed_off) && (!parity || (d.isaac_parity ?? "untested") === parity) && (!q || [d.id, d.use_case, d.family, d.context, d.leaf.slab, d.leaf.panel_style, d.operator, d.lock, d.hinge, ...d.tags, ...d.extras].join(" ").toLowerCase().includes(q)));
    const key: Record<string, (d: ManifestDoor) => number | string> = { index: (d) => d.index, mass: (d) => d.mass_kg, difficulty: (d) => d.difficulty, width: (d) => d.leaf.width, family: (d) => d.family };
    const f = key[sort] ?? key.index;
    return xs.sort((a, b) => (f(a) < f(b) ? -1 : f(a) > f(b) ? 1 : 0));
  }, [doors, family, operator, lock, closer, condition, task, scenario, signed, parity, search, sort]);
  useEffect(() => setPage(1), [family, operator, lock, closer, condition, task, scenario, signed, parity, search, sort]);
  const extraCount = [operator, lock, closer, condition, task, scenario, signed, parity].filter(Boolean).length;
  const hasFilters = !!(search || family || extraCount);
  const reset = () => { setFamily(""); setOperator(""); setLock(""); setCloser(""); setCondition(""); setTask(""); setScenario(""); setSigned(""); setParity(""); setSearch(""); if (query) window.location.hash = "#/"; };
  const sel = (label: string, value: string, set: (s: string) => void, list: string[]) => <label className="filter-field"><span>{label}</span><select value={value} onChange={(e) => set(e.target.value)}><option value="">All {label.toLowerCase()}</option>{list.map((x) => <option key={x} value={x}>{nice(x)}</option>)}</select></label>;
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount);
  const pageDoors = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);
  const goPage = (n: number) => { setPage(n); document.getElementById("collection")?.scrollIntoView({ behavior: "smooth", block: "start" }); };
  const heroDoors = ["db0079_sliding_single", "db0044_pivot"].map((id) => doors.find((d) => d.id === id)).filter((d): d is ManifestDoor => !!d);
  return <div className="catalogue page-shell">
    <section className="catalogue-hero">
      <div className="hero-copy"><p className="eyebrow"><span className="status-dot" /> An open benchmark for embodied AI</p><h1>Every door.<br />A new challenge.</h1><p className="hero-description">From a paper screen to a vault door. Explore {doors.length.toLocaleString()} articulated environments for training and evaluating robot interaction in simulation.</p><div className="hero-actions"><button className="button primary" onClick={() => document.getElementById("collection")?.scrollIntoView({ behavior: "smooth" })}>Explore the collection <Icon name="arrow" /></button><a className="text-link" href={DATASET} target="_blank" rel="noreferrer">Get the dataset <Icon name="external" size={14} /></a></div><div className="hero-stats"><div><strong>{doors.length.toLocaleString()}</strong><span>door environments</span></div><div><strong>{manifest.families.length}</strong><span>motion families</span></div><div><strong>3</strong><span>simulation formats</span></div></div></div>
      <div className="hero-gallery">{heroDoors.map((d, i) => <a className={`hero-door hero-door-${i}`} key={d.id} href={`#/door/${d.id}`}><AppearanceThumb render={photoById.get(d.id)} fallback={thumbUrl(d)} alt={d.use_case} /><div className="hero-image-caption"><div><span>{i === 0 ? "Sliding / solid timber" : "Pivot / architectural"}</span><strong>{i === 0 ? "Mechanics meet materials." : "A different way in."}</strong></div><Icon name="arrow" /></div></a>)}<span className="hero-caption">{heroDoors.every((d) => photoById.has(d.id)) ? "Actual dataset geometry · Blender appearance renders" : "Interactive articulated door models"}</span></div>
    </section>
    <div className="collection-heading" id="collection"><div><p className="eyebrow">The collection</p><h2>Find your next environment.</h2></div><a href="#/families" className="text-link">Explore all door types <Icon name="arrow" size={16} /></a></div>
    <section className="collection-controls" aria-label="Catalogue filters"><div className="filter-primary"><label className="search-field"><Icon name="search" /><input type="search" aria-label="Search doors" placeholder="Search doors, materials, hardware…" value={search} onChange={(e) => setSearch(e.target.value)} /></label><select aria-label="Door type" value={family} onChange={(e) => setFamily(e.target.value)}><option value="">All door types</option>{manifest.families.map((f) => <option key={f} value={f}>{FAMILY_LABELS[f] ?? f}</option>)}</select><button className={`filter-toggle ${advanced ? "active" : ""}`} onClick={() => setAdvanced(!advanced)} aria-expanded={advanced} aria-controls="advanced-filters"><Icon name="filter" />Filters{extraCount > 0 && <span className="filter-count">{extraCount}</span>}</button></div>
      {advanced && <div className="advanced-filters" id="advanced-filters">{sel("Operators", operator, setOperator, opts.operator)}{sel("Locks", lock, setLock, opts.lock)}<label className="filter-field"><span>Closers</span><select value={closer} onChange={(e) => setCloser(e.target.value)}><option value="">All closers</option><option value="any">Has a closer</option>{opts.closer.map((x) => <option key={x} value={x}>{nice(x)}</option>)}</select></label>{sel("Conditions", condition, setCondition, opts.condition)}{sel("Tasks", task, setTask, opts.task)}{sel("Scenarios", scenario, setScenario, opts.scenario)}<label className="filter-field"><span>Automated QA</span><select value={signed} onChange={(e) => setSigned(e.target.value)}><option value="">All statuses</option><option value="yes">Passed</option><option value="no">Needs attention</option></select></label>
        {hasParity && <label className="filter-field"><span>Isaac parity</span><select value={parity} onChange={(e) => setParity(e.target.value)} title="Recorded comparison of MuJoCo and Isaac Sim / PhysX gate outcomes. See the door’s Isaac parity section for grades and details."><option value="">All statuses</option><option value="ok">Parity passed</option><option value="fail">Mismatch</option><option value="untested">Untested</option></select></label>}
      </div>}
      <div className="collection-toolbar"><p aria-live="polite"><strong>{filtered.length.toLocaleString()}</strong> {hasFilters ? `of ${doors.length.toLocaleString()} doors` : "doors to explore"}{hasFilters && <button className="clear-filters" onClick={reset}>Clear filters <Icon name="close" size={12} /></button>}</p><div className="toolbar-right">{photoById.size > 0 && <div className="segmented" aria-label="Thumbnail style"><button aria-pressed={imageMode === "blender"} onClick={() => setImageMode("blender")}>Appearance</button><button aria-pressed={imageMode === "simulation"} onClick={() => setImageMode("simulation")}>Simulation</button></div>}<label className="sort-control"><span>Sort by</span><select aria-label="Sort doors" value={sort} onChange={(e) => setSort(e.target.value)}><option value="index">Door ID</option><option value="family">Door type</option><option value="mass">Mass</option><option value="difficulty">Difficulty</option><option value="width">Width</option></select></label></div></div>
    </section>
    {filtered.length > 0 ? <><div className="grid">{pageDoors.map((d) => <DoorCard key={d.id} d={d} appearance={imageMode === "blender" ? photoById.get(d.id) : undefined} />)}</div><div className="pagination"><span>Showing {(currentPage - 1) * PAGE_SIZE + 1}–{Math.min(currentPage * PAGE_SIZE, filtered.length)} of {filtered.length.toLocaleString()}</span><div><button disabled={currentPage === 1} onClick={() => goPage(currentPage - 1)}>Previous</button><label>Page <select aria-label="Catalogue page" value={currentPage} onChange={(e) => goPage(Number(e.target.value))}>{Array.from({ length: pageCount }, (_, i) => <option key={i} value={i + 1}>{i + 1}</option>)}</select> of {pageCount}</label><button disabled={currentPage === pageCount} onClick={() => goPage(currentPage + 1)}>Next <Icon name="arrow" size={15} /></button></div></div></> : <div className="empty-state"><Icon name="search" size={32} /><h3>No doors match these filters.</h3><p>Try another material, mechanism, or door type.</p><button onClick={reset}>Clear all filters</button></div>}
    {supplementaryCount > 0 && <p className="supplementary-link">Separate supplementary collection: <a href="#/pets">{supplementaryCount} pet doors</a> · downloadable assets, excluded from benchmarks.</p>}
    <aside className="catalogue-note"><Icon name="check" size={18} /><p><strong>Inspect the details.</strong> Each door includes physical parameters, articulated geometry, and downloadable simulation files. Automated QA is separate from visual and physical review.</p><a href="#/review">Open review workspace <Icon name="arrow" size={16} /></a></aside>
  </div>;
}
