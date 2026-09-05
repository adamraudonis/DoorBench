import { describe, expect, test } from "bun:test";
import type { Manifest, ManifestDoor } from "./types";
import { EMPTY_FILTERS, canAccept, datasetFor, emptyReview, filterDoors, loadReviews, makeDocument, mergeReviews,
  parseDocument, reviewCounts, reviewShortcut, saveReviews, statusOf, storageKey, timestampAfter, type DoorReview,
  type ReviewMap, type ReviewStorage } from "./reviewState";

const doors = [
  { id: "db0001_swing", index: 1, family: "swing_single", context: "residential", use_case: "bedroom door", operator: "knob", closer: "none", hinge: "butt", latch: "tubular", lock: "none", tags: ["wood"], leaf: { width: .9 } },
  { id: "db0002_slide", index: 2, family: "sliding_single", context: "commercial", use_case: "warehouse slider", operator: "pull", closer: "none", hinge: "none", latch: "none", lock: "none", tags: ["metal"], leaf: { width: 1.2 } },
  { id: "db0003_swing", index: 3, family: "swing_single", context: "commercial", use_case: "exit door", operator: "panic", closer: "overhead", hinge: "butt", latch: "rim", lock: "none", tags: ["metal"], leaf: { width: 1.0 } },
] as ManifestDoor[];
const manifest = { name: "DoorBench", version: "0.1", generated: "2026-09-05", n_doors: 3, n_signed_off: 3, families: ["swing_single", "sliding_single"], doors } as Manifest;
const dataset = datasetFor(manifest);
function record(id = doors[0].id, patch: Partial<DoorReview> = {}): DoorReview {
  return { ...emptyReview(id), updated_at: "2026-09-05T12:00:00.000Z", ...patch };
}
const passing = { appearance: "pass", physical: "pass", mechanism: "pass" } as const;
function storage(): ReviewStorage & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => { data.set(k, v); } };
}

describe("human review identity and status", () => {
  test("manifest field order and time-only regeneration keep identity; geometry metadata changes isolate reviews", () => {
    expect(datasetFor({ ...manifest, generated: "later", doors: [...doors].reverse().map((d) => ({ ...d, time_s: 99 })) }).id).toBe(dataset.id);
    expect(datasetFor({ ...manifest, version: "0.2" }).id).not.toBe(dataset.id);
    expect(datasetFor({ ...manifest, doors: [{ ...doors[0], leaf: { ...doors[0].leaf, width: 2 } }, ...doors.slice(1)] }).id).not.toBe(dataset.id);
    expect(storageKey(dataset)).not.toBe(storageKey(datasetFor({ ...manifest, n_signed_off: 2 })));
  });
  test("automated QA never becomes a human acceptance; flags are not fully rated", () => {
    const reviews: ReviewMap = { [doors[0].id]: record(doors[0].id, { flagged: true }), [doors[1].id]: record(doors[1].id, { ratings: passing }) };
    expect(reviewCounts(doors, reviews)).toEqual({ total: 3, flagged: 1, accepted: 1, in_progress: 0, unreviewed: 1, fully_rated: 1 });
    expect(statusOf(undefined)).toBe("unreviewed");
    expect(statusOf(record(doors[0].id, { notes: "Need rear view" }))).toBe("in_progress");
    expect(statusOf(record(doors[0].id, { ratings: { ...passing, mechanism: "uncertain" } }))).toBe("in_progress");
  });
  test("issue evidence takes precedence over pass ratings; quick accept cannot erase it", () => {
    expect(statusOf(record(doors[0].id, { ratings: passing, issues: ["support"] }))).toBe("flagged");
    for (const patch of [{ flagged: true }, { issues: ["clearance"] }, { ratings: { ...passing, physical: "issue" } }, { ratings: { ...passing, mechanism: "uncertain" } }]) {
      expect(canAccept(record(doors[0].id, patch as Partial<DoorReview>))).toBe(false);
    }
    expect(canAccept(emptyReview(doors[0].id))).toBe(true);
  });
  test("search includes hardware and notes; all filters intersect", () => {
    const reviews = { [doors[2].id]: record(doors[2].id, { issues: ["clearance"], notes: "Arm hits frame" }) };
    expect(filterDoors(doors, reviews, { ...EMPTY_FILTERS, search: "panic frame", family: "swing_single", issue: "clearance", status: "flagged" }).map((d) => d.id)).toEqual([doors[2].id]);
    expect(filterDoors(doors, reviews, { ...EMPTY_FILTERS, status: "accepted" })).toEqual([]);
    expect(filterDoors(doors, reviews, { ...EMPTY_FILTERS, family: "sliding_single" }).map((d) => d.id)).toEqual([doors[1].id]);
  });
});

describe("versioned review import and local storage", () => {
  test("export/import round trip preserves all ratings, tags, notes and timestamps", () => {
    const r = record(doors[0].id, { ratings: { ...passing, mechanism: "issue" }, issues: ["mechanism", "reference"], notes: 'Hinge "jumps" at 45°\nCheck attachment.' });
    const doc = makeDocument(dataset, { [r.door_id]: r });
    expect(parseDocument(JSON.stringify(doc), dataset)).toEqual(doc);
  });
  test("wrong schema/dataset, unknown/duplicate IDs, bad types, invalid tags and extra fields reject the whole import", () => {
    const good = makeDocument(dataset, { [doors[0].id]: record() });
    const cases = [
      { ...good, schema_version: "2" }, { ...good, dataset: { ...dataset, id: "different" } },
      { ...good, dataset: { ...dataset, door_ids: dataset.door_ids.slice(1) } },
      { ...good, reviews: [record("unknown")] }, { ...good, reviews: [record(), record()] },
      { ...good, reviews: [record(doors[0].id, { flagged: "yes" as unknown as boolean })] },
      { ...good, reviews: [{ ...record(), ratings: { ...passing, mechanism: "excellent" } }] },
      { ...good, reviews: [{ ...record(), ratings: { appearance: "pass" } }] },
      { ...good, reviews: [{ ...record(), issues: ["__proto__"] }] },
      { ...good, reviews: [{ ...record(), issues: ["support", "support"] }] },
      { ...good, reviews: [{ ...record(), notes: "x".repeat(10001) }] },
      { ...good, reviews: [{ ...record(), updated_at: "not-a-date" }] },
      { ...good, reviews: [{ ...record(), signed_off: true }] },
      { ...good, exported_at: null },
    ];
    for (const bad of cases) expect(() => parseDocument(JSON.stringify(bad), dataset)).toThrow();
    expect(() => parseDocument("[", dataset)).toThrow("valid JSON");
  });
  test("newer whole record wins, ties stay local, unrelated reviews survive", () => {
    const old = record(), second = record(doors[1].id), third = record(doors[2].id);
    const next = { ...old, notes: "Updated", updated_at: "2026-09-05T13:00:00.000Z" };
    const merged = mergeReviews({ [old.door_id]: old, [second.door_id]: second }, [next, third]);
    expect([merged.added, merged.updated, merged.kept]).toEqual([1, 1, 0]);
    expect(Object.keys(merged.reviews)).toHaveLength(3);
    expect(mergeReviews(merged.reviews, [{ ...next, notes: "same timestamp" }, old]).reviews[old.door_id].notes).toBe("Updated");
  });
  test("edits remain newer even when clock moves backwards", () => {
    const future = "2099-01-01T00:00:00.000Z";
    expect(Date.parse(timestampAfter(future))).toBeGreaterThan(Date.parse(future));
  });
  test("save/load persists, merges another tab, isolates datasets and does not destroy corrupt storage", () => {
    const s = storage(), first = record(), second = record(doors[1].id);
    saveReviews(s, dataset, { [first.door_id]: first });
    const merged = saveReviews(s, dataset, { [second.door_id]: second });
    expect(Object.keys(merged)).toHaveLength(2);
    expect(loadReviews(s, dataset)).toEqual(merged);
    expect(loadReviews(s, { ...dataset, id: "another" })).toEqual({});
    s.setItem(storageKey(dataset), "damaged JSON");
    expect(() => saveReviews(s, dataset, {})).toThrow();
    expect(s.getItem(storageKey(dataset))).toBe("damaged JSON");
  });
  test("storage access/quota errors propagate instead of claiming a successful save", () => {
    expect(() => loadReviews({ getItem: () => { throw Error("denied"); }, setItem: () => {} }, dataset)).toThrow("denied");
    expect(() => saveReviews({ getItem: () => null, setItem: () => { throw Error("quota"); } }, dataset, {})).toThrow("quota");
  });
});

describe("review keyboard safety", () => {
  test("navigation and review commands work on non-interactive surfaces", () => {
    expect(reviewShortcut({ key: "a", target: { tagName: "DIV" } })).toBe("accept");
    expect(reviewShortcut({ key: "f" })).toBe("flag");
    expect(reviewShortcut({ key: "ArrowRight" })).toBe("next");
    expect(reviewShortcut({ key: "p" })).toBe("previous");
  });
  test("editing, focusable controls, timelines, modifiers, repeat, composition and consumed keys suppress shortcuts", () => {
    for (const tagName of ["INPUT", "SELECT", "TEXTAREA", "BUTTON", "A", "SUMMARY"]) expect(reviewShortcut({ key: "a", target: { tagName } })).toBeNull();
    expect(reviewShortcut({ key: "ArrowRight", target: { closest: () => ({ role: "slider" }) } })).toBeNull();
    expect(reviewShortcut({ key: "a", target: { isContentEditable: true } })).toBeNull();
    for (const prop of ["ctrlKey", "metaKey", "altKey", "shiftKey", "repeat", "defaultPrevented", "isComposing"]) expect(reviewShortcut({ key: "a", [prop]: true })).toBeNull();
  });
});
