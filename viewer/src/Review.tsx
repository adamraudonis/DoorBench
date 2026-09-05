import { isPetDoor } from "./collections";
import React, { useEffect, useMemo, useRef, useState } from "react";
import type { Manifest } from "./types";
import { FAMILY_LABELS } from "./types";
import { DoorView } from "./DoorView";
import { ASPECTS, EMPTY_FILTERS, ISSUE_LABELS, MAX_IMPORT_BYTES, STATUS_LABELS, canAccept, datasetFor, emptyReview,
  filterDoors, loadReviews, makeDocument, mergeReviews, parseDocument, reviewCounts, reviewShortcut, saveReviews, statusOf,
  storageKey, timestampAfter, type Aspect, type DoorReview, type Issue, type Rating, type ReviewDataset, type ReviewDocument,
  type ReviewFilters, type ReviewMap, type ReviewStatus } from "./reviewState";
import "./Review.css";

const ASPECT_COPY: Record<Aspect, { title: string; hint: string }> = {
  appearance: { title: "Appearance", hint: "Materials, proportions, detail and visual variety." },
  physical: { title: "Physical construction", hint: "Supports, attachments, clearances and believable assembly." },
  mechanism: { title: "Mechanism", hint: "Operator, latch, hinges, guides and closer through full travel." },
};
const RATING_LABELS: Record<Rating, string> = { unreviewed: "Not rated", pass: "Pass", issue: "Issue", uncertain: "Unsure" };

function initialReviews(dataset: ReviewDataset) {
  try { return { reviews: loadReviews(window.localStorage, dataset), error: "" }; }
  catch (error) { return { reviews: {} as ReviewMap, error: `Saved reviews could not be loaded: ${String(error)}. Existing saved data has been left intact.` }; }
}

export function Review({ manifest }: { manifest: Manifest }) {
  const dataset = useMemo(() => datasetFor(manifest), [manifest]);
  return <ReviewWorkspace key={dataset.id} manifest={manifest} dataset={dataset} />;
}

function ReviewWorkspace({ manifest, dataset }: { manifest: Manifest; dataset: ReviewDataset }) {
  const [initial] = useState(() => initialReviews(dataset));
  const [reviews, setReviews] = useState<ReviewMap>(initial.reviews);
  const [storageError, setStorageError] = useState(initial.error);
  const [filters, setFilters] = useState<ReviewFilters>(EMPTY_FILTERS);
  const [petReview, setPetReview] = useState(() => manifest.doors.some(d => isPetDoor(d) && d.id === new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("door")));
  const doors = useMemo(() => manifest.doors.filter(d => isPetDoor(d) === petReview).sort((a, b) => a.index - b.index || a.id.localeCompare(b.id)), [manifest, petReview]);
  const [selectedId, setSelectedId] = useState(() => {
    const fromUrl = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("door");
    if (doors.some((d) => d.id === fromUrl)) return fromUrl!;
    try {
      const saved = window.localStorage.getItem(`${storageKey(dataset)}:selection`);
      if (doors.some((d) => d.id === saved)) return saved!;
    } catch { /* The review panel displays storage failures separately. */ }
    return doors.find((d) => statusOf(initial.reviews[d.id]) === "unreviewed")?.id ?? doors[0]?.id ?? "";
  });
  const [notice, setNotice] = useState("");
  const [pendingImport, setPendingImport] = useState<ReviewDocument | null>(null);
  const [importBusy, setImportBusy] = useState(false);
  const [undo, setUndo] = useState<{ id: string; before: DoorReview } | null>(null);
  const noteRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const selectedQueueRef = useRef<HTMLButtonElement>(null);
  const queue = useMemo(() => filterDoors(doors, reviews, filters), [doors, reviews, filters]);
  const selected = doors.find((d) => d.id === selectedId);
  const current = reviews[selectedId] ?? emptyReview(selectedId);
  const status = statusOf(current);
  const counts = useMemo(() => reviewCounts(doors, reviews), [doors, reviews]);
  const queueIndex = queue.findIndex((d) => d.id === selectedId);
  const importPlan = pendingImport ? mergeReviews(reviews, pendingImport.reviews) : null;

  useEffect(() => {
    if (!selectedId) return;
    try { window.localStorage.setItem(`${storageKey(dataset)}:selection`, selectedId); } catch { /* Notes save reports failures. */ }
    window.history.replaceState(null, "", `#/review?door=${encodeURIComponent(selectedId)}`);
    // Reveal the item inside its queue only; scrollIntoView also shifts the outer
    // page on mount and can hide the review heading below the sticky site nav.
    const button = selectedQueueRef.current, list = button?.parentElement;
    if (button && list) {
      const item = button.getBoundingClientRect(), viewport = list.getBoundingClientRect();
      if (item.top < viewport.top) list.scrollTop += item.top - viewport.top;
      else if (item.bottom > viewport.bottom) list.scrollTop += item.bottom - viewport.bottom;
    }
  }, [selectedId, dataset]);

  useEffect(() => {
    const changed = (event: StorageEvent) => {
      if (event.key !== storageKey(dataset) || event.storageArea !== window.localStorage) return;
      try {
        const incoming = loadReviews(window.localStorage, dataset);
        setReviews((old) => mergeReviews(old, Object.values(incoming)).reviews);
        setNotice("Newer reviews from another tab were loaded.");
      } catch (error) { setStorageError(`Could not read reviews from another tab: ${String(error)}`); }
    };
    window.addEventListener("storage", changed);
    return () => window.removeEventListener("storage", changed);
  }, [dataset]);

  function persist(next: ReviewMap) {
    // No mount-time write: malformed stored data is never overwritten with an empty document.
    try {
      const saved = saveReviews(window.localStorage, dataset, next);
      setReviews(saved); setStorageError("");
    } catch (error) {
      setReviews(next);
      setStorageError(`Changes are only in this tab: ${String(error)}. Export JSON now to keep a backup.`);
    }
  }
  function update(patch: Partial<Omit<DoorReview, "door_id" | "updated_at">>) {
    if (!selected) return;
    setUndo({ id: selectedId, before: current });
    persist({ ...reviews, [selectedId]: { ...current, ...patch, updated_at: timestampAfter(current.updated_at) } });
  }
  function neighbor(direction: 1 | -1): string | null {
    if (!queue.length) return null;
    if (queueIndex >= 0) return queue[queueIndex + direction]?.id ?? null;
    const index = doors.findIndex((d) => d.id === selectedId);
    const possible = direction === 1 ? queue : [...queue].reverse();
    return possible.find((d) => direction === 1 ? doors.indexOf(d) > index : doors.indexOf(d) < index)?.id ?? null;
  }
  function move(direction: 1 | -1) {
    const next = neighbor(direction);
    if (next) { setSelectedId(next); setNotice(""); }
    else setNotice(direction === 1 ? "End of this queue. Change the filters to continue." : "Start of this queue.");
  }
  function accept() {
    if (!selected) return;
    if (!canAccept(current)) {
      setNotice("Resolve issue tags, flags and uncertain ratings before accepting. Existing findings have been kept."); return;
    }
    const next = neighbor(1);
    update({ ratings: { appearance: "pass", physical: "pass", mechanism: "pass" } });
    setNotice(`${selectedId} accepted by you.${next ? "" : " End of this queue."}`);
    if (next) setSelectedId(next);
  }
  function flag() { update({ flagged: true }); setNotice(`${selectedId} flagged. Add a tag or note describing the problem.`); noteRef.current?.focus(); }
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const action = reviewShortcut({ ...event, key: event.key, target: event.target as HTMLElement, ctrlKey: event.ctrlKey,
        metaKey: event.metaKey, altKey: event.altKey, shiftKey: event.shiftKey, repeat: event.repeat, defaultPrevented: event.defaultPrevented, isComposing: event.isComposing });
      if (!action || !selected) return;
      event.preventDefault();
      if (action === "next") move(1);
      if (action === "previous") move(-1);
      if (action === "accept") accept();
      if (action === "flag") flag();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  function exportReviews() {
    const blob = new Blob([JSON.stringify(makeDocument(dataset, reviews), null, 2) + "\n"], { type: "application/json" });
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `doorbench-review-${dataset.id}-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link); link.click(); link.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("Review JSON exported. Keep this file as your portable backup.");
  }
  async function importReviews(file?: File) {
    if (!file) return;
    setImportBusy(true); setPendingImport(null);
    try {
      if (file.size > MAX_IMPORT_BYTES) throw new Error("Review file exceeds the 64 MB limit.");
      const document = parseDocument(await file.text(), dataset);
      setPendingImport(document); setNotice("Import validated. Review the merge summary before applying it.");
    } catch (error) { setNotice(`Import rejected: ${String(error)} No reviews were changed.`); }
    finally { setImportBusy(false); if (fileRef.current) fileRef.current.value = ""; }
  }
  function undoLast() {
    if (!undo) return;
    const restored = { ...undo.before, updated_at: timestampAfter(reviews[undo.id]?.updated_at) };
    persist({ ...reviews, [undo.id]: restored }); setSelectedId(undo.id); setUndo(null); setNotice("Last edit undone.");
  }

  return <section className="review-page" aria-label="Human door review">
    <header className="review-heading">
      <div><div className="review-eyebrow">Your inspection workspace</div><h1>Review every door.</h1>
        <p>Inspect the assembly, operate its mechanisms, then record your judgement. Human ratings stay separate from automated QA.</p></div>
      <div className="review-transfer"><button onClick={exportReviews}>Export JSON</button>
        <button onClick={() => fileRef.current?.click()} disabled={importBusy}>{importBusy ? "Checking file…" : "Import JSON"}</button>
        <input ref={fileRef} className="review-hidden-input" type="file" accept=".json,application/json" aria-label="Import review JSON" onChange={(e) => void importReviews(e.target.files?.[0])} />
        <small>{storageError ? "Local save needs attention" : "Saved automatically in this browser"}</small></div>
    </header>
    <div className="review-progress">
      <div><strong>{counts.fully_rated}<span> / {counts.total}</span></strong><span> fully rated by you</span></div>
      <progress max={Math.max(counts.total, 1)} value={counts.fully_rated} aria-label="Doors with all three human ratings" />
      <div className="review-counts">{(Object.keys(STATUS_LABELS) as ReviewStatus[]).map((s) => <button key={s} className={`review-status review-status-${s}`} aria-pressed={filters.status === s}
        onClick={() => setFilters((f) => ({ ...f, status: f.status === s ? "" : s }))}>{counts[s]} {STATUS_LABELS[s].toLowerCase()}</button>)}</div>
    </div>
    {storageError && <div className="review-warning" role="alert">{storageError} Browser storage is local to this site address and browser profile.</div>}
    <div className="review-notice" role="status" aria-live="polite">{notice || "Start with a door, check all three aspects, and use Accept & next when they pass."}</div>
    {pendingImport && importPlan && <div className="review-import" role="region" aria-label="Import preview">
      <strong>Import: {importPlan.added} new · {importPlan.updated} newer · {importPlan.kept} kept locally</strong>
      <p>The newest timestamp wins per door; local records win ties. Other doors stay intact. Export a backup before merging if you need the previous version.</p>
      <button onClick={() => { persist(mergeReviews(reviews, pendingImport.reviews).reviews); setPendingImport(null); setUndo(null); setNotice("Validated reviews merged. Export JSON to keep a portable backup."); }}>Apply import</button>
      <button onClick={() => { setPendingImport(null); setNotice("Import cancelled. No reviews changed."); }}>Cancel</button>
    </div>}
    <div className="review-filters">
      <label>Search<input type="search" placeholder="ID, hardware, use, notes…" value={filters.search} onChange={(e) => setFilters((f) => ({ ...f, search: e.target.value }))} /></label>
      <label>Collection<select aria-label="Review collection" value={petReview ? "pets" : "standard"} onChange={e => { const pets = e.target.value === "pets"; setPetReview(pets); setFilters(EMPTY_FILTERS); setSelectedId(manifest.doors.find(d => isPetDoor(d) === pets)?.id ?? ""); }}><option value="standard">Standard doors</option><option value="pets">Supplementary pet doors · asset review</option></select></label>
      <label>Family<select value={filters.family} onChange={(e) => setFilters((f) => ({ ...f, family: e.target.value }))}><option value="">All families</option>{manifest.families.filter(f => (f === "pet_door") === petReview).map((f) => <option key={f} value={f}>{FAMILY_LABELS[f] ?? f}</option>)}</select></label>
      <label>Status<select value={filters.status} onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))}><option value="">All statuses</option>{Object.entries(STATUS_LABELS).map(([s, label]) => <option key={s} value={s}>{label}</option>)}</select></label>
      <label>Issue<select value={filters.issue} onChange={(e) => setFilters((f) => ({ ...f, issue: e.target.value }))}><option value="">All issues</option>{Object.entries(ISSUE_LABELS).map(([i, label]) => <option key={i} value={i}>{label}</option>)}</select></label>
      <button onClick={() => setFilters(EMPTY_FILTERS)}>Clear filters</button><span>{queue.length} matching doors</span>
    </div>
    <div className="review-workspace">
      <aside className="review-queue" aria-label="Review queue"><div className="review-queue-title">Door queue <span>{queue.length}</span></div>
        <div className="review-queue-list">{queue.map((d) => <button key={d.id} ref={d.id === selectedId ? selectedQueueRef : undefined} className={d.id === selectedId ? "selected" : ""}
          aria-current={d.id === selectedId ? "true" : undefined} onClick={() => { setSelectedId(d.id); setNotice(""); }}>
          <span><b>{d.id.split("_")[0]}</b><i className={`review-dot review-status-${statusOf(reviews[d.id])}`} title={STATUS_LABELS[statusOf(reviews[d.id])]} /></span>
          <span>{d.use_case || FAMILY_LABELS[d.family] || d.family}</span><small>{STATUS_LABELS[statusOf(reviews[d.id])]}</small></button>)}
          {!queue.length && <p className="review-empty">No doors match these filters. Clear them to see the full collection.</p>}</div>
      </aside>
      <main className="review-inspection">
        {selected ? <><div className="review-door-heading"><div><span>{FAMILY_LABELS[selected.family] ?? selected.family}</span><h2>{selected.use_case || selected.id}</h2><code>{selected.id}</code></div>
          <a href={`#/door/${selected.id}`} target="_blank" rel="noreferrer">Open full inspector ↗</a></div>
          <div className="review-navigation"><button onClick={() => move(-1)} disabled={!neighbor(-1)}>← Previous</button><span>{queueIndex >= 0 ? `${queueIndex + 1} of ${queue.length} in queue` : "Current door is outside the filters"}</span><button onClick={() => move(1)} disabled={!neighbor(1)}>Next →</button></div>
          <div className="review-preview"><DoorView key={selected.id} manifest={manifest} id={selected.id} embedded initialDiagnostic /></div>
        </> : <div className="review-empty">This dataset has no doors to review.</div>}
      </main>
      {selected && <aside className="review-form" aria-label={`Review ${selected.id}`}>
        <div className="review-form-title"><h2>Your assessment</h2><span className={`review-status review-status-${status}`}>{STATUS_LABELS[status]}</span></div>
        {ASPECTS.map((aspect) => <fieldset key={aspect}><legend>{ASPECT_COPY[aspect].title}</legend><p>{ASPECT_COPY[aspect].hint}</p><div className="review-rating-options">
          {(Object.keys(RATING_LABELS) as Rating[]).map((rating) => <label key={rating} className={current.ratings[aspect] === rating ? `selected rating-${rating}` : ""}>
            <input type="radio" name={`${selectedId}-${aspect}`} value={rating} checked={current.ratings[aspect] === rating} onChange={() => update({ ratings: { ...current.ratings, [aspect]: rating } })} />{RATING_LABELS[rating]}</label>)}
        </div></fieldset>)}
        <details className="review-checklist"><summary>What to inspect</summary><ol><li>Orbit both sides. Check the leaf, frame, hardware, materials and supports.</li><li>Watch the complete opening and closing sequence. Zoom into every joint, guide, latch and closer.</li><li>Hide walls and compare collision geometry where useful. Check endpoints and intermediate poses.</li><li>Compare the simulation reference where available. Mark unsure if the evidence does not establish correct behaviour.</li></ol><p>Viewer motion and your visual ratings do not certify force, contact, structural strength or simulator parity.</p></details>
        <fieldset className="review-issue-tags"><legend>Issue tags</legend>{(Object.entries(ISSUE_LABELS) as [Issue, string][]).map(([issue, label]) => <label key={issue}><input type="checkbox" checked={current.issues.includes(issue)}
          onChange={() => update({ issues: current.issues.includes(issue) ? current.issues.filter((i) => i !== issue) : [...current.issues, issue] })} />{label}</label>)}</fieldset>
        <label className="review-notes-label" htmlFor="review-notes">Notes <small>{current.notes.length}/10,000</small></label>
        <textarea ref={noteRef} id="review-notes" value={current.notes} maxLength={10000} rows={5} placeholder="Part name, pose, what is wrong, expected behaviour…" onChange={(e) => update({ notes: e.target.value })} />
        <div className="review-actions"><button className="review-accept" onClick={accept} disabled={!canAccept(current)}>Accept &amp; next <kbd>A</kbd></button>
          <button onClick={current.flagged ? () => update({ flagged: false }) : flag}>{current.flagged ? "Remove manual flag" : "Flag for follow-up"} <kbd>F</kbd></button></div>
        {!canAccept(current) && <p className="review-help">Resolve flags, issue tags and issue/unsure ratings before accepting. Removing a manual flag keeps the other findings.</p>}
        <p className="review-help">Accept marks all three aspects as passed by you. Flagging alone does not count as a fully rated door.</p>
        <div className="review-secondary"><button disabled={!undo} onClick={undoLast}>Undo last edit</button><button onClick={() => { update({ ...emptyReview(selectedId) }); setNotice("Assessment cleared. Undo last edit restores it."); }}>Clear assessment</button></div>
        <p className="review-shortcuts"><kbd>←</kbd> / <kbd>P</kbd> previous · <kbd>→</kbd> / <kbd>N</kbd> next<br /><kbd>A</kbd> accept &amp; next · <kbd>F</kbd> flag<br /><span>Shortcuts pause while a control or text field has focus.</span></p>
      </aside>}
    </div>
    <footer className="review-footer">Local human review · dataset v{dataset.version} · <code>{dataset.id}</code>. Progress is tied to this manifest, not a checksum of the model files. Export JSON before clearing browser data or changing site address.</footer>
  </section>;
}
