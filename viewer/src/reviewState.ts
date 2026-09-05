import type { Manifest, ManifestDoor } from "./types";

export const REVIEW_SCHEMA = "doorbench-human-review/1" as const;
export const ASPECTS = ["appearance", "physical", "mechanism"] as const;
export const RATINGS = ["unreviewed", "pass", "issue", "uncertain"] as const;
export const ISSUE_LABELS = {
  appearance: "Appearance / materials", diversity: "Repetition / diversity", support: "Missing support / floating part",
  clearance: "Collision / clearance", alignment: "Attachment / alignment", mechanism: "Mechanism / motion",
  limits: "Joint limits / travel", reference: "Reference demo mismatch", other: "Other",
} as const;
export type Aspect = typeof ASPECTS[number];
export type Rating = typeof RATINGS[number];
export type Issue = keyof typeof ISSUE_LABELS;
export type ReviewStatus = "unreviewed" | "in_progress" | "accepted" | "flagged";
export const STATUS_LABELS: Record<ReviewStatus, string> = { unreviewed: "Unreviewed", in_progress: "In progress", accepted: "Accepted", flagged: "Flagged" };
export interface DoorReview {
  door_id: string;
  ratings: Record<Aspect, Rating>;
  flagged: boolean;
  issues: Issue[];
  notes: string;
  updated_at: string;
}
export interface ReviewDataset { id: string; name: string; version: string; door_ids: string[] }
export interface ReviewDocument {
  schema_version: typeof REVIEW_SCHEMA;
  dataset: ReviewDataset;
  exported_at: string;
  reviews: DoorReview[];
}
export type ReviewMap = Record<string, DoorReview>;
export interface ReviewFilters { search: string; family: string; status: string; issue: string }
export const EMPTY_FILTERS: ReviewFilters = { search: "", family: "", status: "", issue: "" };
export const MAX_IMPORT_BYTES = 64 * 1024 * 1024;

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).filter(([k]) => k !== "generated" && k !== "time_s")
    .sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`).join(",")}}`;
  return JSON.stringify(value) ?? "null";
}

/** Manifest fingerprint, not a geometry checksum: time-only regeneration keeps progress; changed manifest content isolates it. */
export function datasetFor(manifest: Manifest): ReviewDataset {
  const input = canonical({ ...manifest, doors: [...manifest.doors].sort((a, b) => a.id.localeCompare(b.id)) });
  let a = 0x811c9dc5, b = 0x9e3779b9;
  for (let i = 0; i < input.length; i++) {
    a = Math.imul(a ^ input.charCodeAt(i), 0x01000193);
    b = Math.imul(b ^ input.charCodeAt(i), 0x85ebca6b);
  }
  const digest = [a, b].map((n) => (n >>> 0).toString(16).padStart(8, "0")).join("");
  return { id: `manifest-${digest}`, name: manifest.name, version: manifest.version, door_ids: manifest.doors.map((d) => d.id).sort() };
}

export function storageKey(dataset: ReviewDataset): string { return `doorbench:human-review:v1:${dataset.id}`; }
export function emptyReview(id: string): DoorReview {
  return { door_id: id, ratings: { appearance: "unreviewed", physical: "unreviewed", mechanism: "unreviewed" }, flagged: false, issues: [], notes: "", updated_at: new Date(0).toISOString() };
}
export function statusOf(review?: DoorReview): ReviewStatus {
  if (!review) return "unreviewed";
  if (review.flagged || review.issues.length || ASPECTS.some((a) => review.ratings[a] === "issue")) return "flagged";
  if (ASPECTS.every((a) => review.ratings[a] === "pass")) return "accepted";
  if (review.notes.trim() || ASPECTS.some((a) => review.ratings[a] !== "unreviewed")) return "in_progress";
  return "unreviewed";
}
export function canAccept(review: DoorReview): boolean {
  return !review.flagged && !review.issues.length && !ASPECTS.some((a) => ["issue", "uncertain"].includes(review.ratings[a]));
}
export function timestampAfter(previous?: string): string {
  return new Date(Math.max(Date.now(), previous ? Date.parse(previous) + 1 : 0)).toISOString();
}
export function reviewCounts(doors: ManifestDoor[], reviews: ReviewMap) {
  const counts = { unreviewed: 0, in_progress: 0, accepted: 0, flagged: 0, fully_rated: 0, total: doors.length };
  for (const door of doors) {
    const review = reviews[door.id];
    counts[statusOf(review)]++;
    if (review && ASPECTS.every((a) => review.ratings[a] !== "unreviewed")) counts.fully_rated++;
  }
  return counts;
}
export function filterDoors(doors: ManifestDoor[], reviews: ReviewMap, filters: ReviewFilters): ManifestDoor[] {
  const words = filters.search.trim().toLowerCase().split(/\s+/).filter(Boolean);
  return doors.filter((d) => {
    const review = reviews[d.id];
    const text = [d.id, d.family, d.context, d.use_case, d.operator, d.closer, d.hinge, d.latch, d.lock, ...(d.tags ?? []), review?.notes ?? ""].join(" ").toLowerCase();
    return (!filters.family || d.family === filters.family) && (!filters.status || statusOf(review) === filters.status)
      && (!filters.issue || !!review?.issues.includes(filters.issue as Issue)) && words.every((w) => text.includes(w));
  });
}

export function makeDocument(dataset: ReviewDataset, reviews: ReviewMap): ReviewDocument {
  const known = new Set(dataset.door_ids);
  return { schema_version: REVIEW_SCHEMA, dataset, exported_at: new Date().toISOString(),
    reviews: Object.values(reviews).filter((r) => known.has(r.door_id)).sort((a, b) => a.door_id.localeCompare(b.door_id)) };
}

function object(value: unknown): value is Record<string, unknown> { return !!value && typeof value === "object" && !Array.isArray(value); }
function keys(value: Record<string, unknown>, allowed: string[], where: string) {
  if (Object.keys(value).some((k) => !allowed.includes(k))) throw new Error(`${where}: unrecognized field.`);
}
function validDate(value: unknown): value is string {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}T/.test(value) && Number.isFinite(Date.parse(value));
}

/** Strict, atomic validation. An incompatible dataset or one invalid record rejects the entire import. */
export function parseDocument(text: string, dataset: ReviewDataset): ReviewDocument {
  if (new TextEncoder().encode(text).byteLength > MAX_IMPORT_BYTES) throw new Error("Review file exceeds the 64 MB limit.");
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error("This file is not valid JSON."); }
  if (!object(value) || value.schema_version !== REVIEW_SCHEMA) throw new Error(`Expected review schema ${REVIEW_SCHEMA}.`);
  keys(value, ["schema_version", "dataset", "exported_at", "reviews"], "Review file");
  const ds = value.dataset;
  if (!object(ds) || ds.id !== dataset.id || ds.name !== dataset.name || ds.version !== dataset.version
    || !Array.isArray(ds.door_ids) || JSON.stringify(ds.door_ids) !== JSON.stringify(dataset.door_ids)) {
    throw new Error("This review belongs to a different dataset. Open the matching dataset before importing it.");
  }
  keys(ds, ["id", "name", "version", "door_ids"], "Dataset");
  if (!validDate(value.exported_at) || !Array.isArray(value.reviews) || value.reviews.length > dataset.door_ids.length) throw new Error("Invalid export date or review list.");
  const known = new Set(dataset.door_ids), seen = new Set<string>();
  const reviews: DoorReview[] = value.reviews.map((raw, index) => {
    const label = `Review ${index + 1}`;
    if (!object(raw) || typeof raw.door_id !== "string" || !known.has(raw.door_id) || seen.has(raw.door_id)) throw new Error(`${label}: unknown or duplicate door ID.`);
    seen.add(raw.door_id);
    keys(raw, ["door_id", "ratings", "flagged", "issues", "notes", "updated_at"], label);
    if (!object(raw.ratings)) throw new Error(`${label}: missing ratings.`);
    keys(raw.ratings, [...ASPECTS], `${label} ratings`);
    const ratings = raw.ratings;
    if (ASPECTS.some((a) => !RATINGS.includes(ratings[a] as Rating))) throw new Error(`${label}: invalid rating.`);
    if (typeof raw.flagged !== "boolean" || !Array.isArray(raw.issues) || raw.issues.some((i) => typeof i !== "string" || !Object.hasOwn(ISSUE_LABELS, i))
      || new Set(raw.issues).size !== raw.issues.length) throw new Error(`${label}: invalid flag or issue tags.`);
    if (typeof raw.notes !== "string" || raw.notes.length > 10000 || !validDate(raw.updated_at)) throw new Error(`${label}: invalid notes or timestamp.`);
    return { door_id: raw.door_id, ratings: Object.fromEntries(ASPECTS.map((a) => [a, ratings[a]])) as Record<Aspect, Rating>,
      flagged: raw.flagged, issues: [...raw.issues] as Issue[], notes: raw.notes, updated_at: raw.updated_at };
  });
  return { schema_version: REVIEW_SCHEMA, dataset, exported_at: value.exported_at, reviews };
}

/** Whole per-door record wins by edit timestamp; existing local record wins ties. Never silently delete other doors. */
export function mergeReviews(local: ReviewMap, incoming: DoorReview[]) {
  const reviews = { ...local };
  let added = 0, updated = 0, kept = 0;
  for (const review of incoming) {
    const old = reviews[review.door_id];
    if (!old) { reviews[review.door_id] = review; added++; }
    else if (Date.parse(review.updated_at) > Date.parse(old.updated_at)) { reviews[review.door_id] = review; updated++; }
    else kept++;
  }
  return { reviews, added, updated, kept };
}

export interface ReviewStorage { getItem(key: string): string | null; setItem(key: string, value: string): void }
export function loadReviews(storage: ReviewStorage, dataset: ReviewDataset): ReviewMap {
  const raw = storage.getItem(storageKey(dataset));
  return raw ? mergeReviews({}, parseDocument(raw, dataset).reviews).reviews : {};
}
export function saveReviews(storage: ReviewStorage, dataset: ReviewDataset, reviews: ReviewMap): ReviewMap {
  // Merge newer edits from another tab before writing; exceptions are handled visibly by the caller.
  const merged = mergeReviews(loadReviews(storage, dataset), Object.values(reviews)).reviews;
  storage.setItem(storageKey(dataset), JSON.stringify(makeDocument(dataset, merged)));
  return merged;
}

export type ShortcutTarget = { tagName?: string; isContentEditable?: boolean; closest?: (selector: string) => unknown };
export function reviewShortcut(event: { key: string; target?: ShortcutTarget | null; ctrlKey?: boolean; metaKey?: boolean; altKey?: boolean; shiftKey?: boolean; repeat?: boolean; defaultPrevented?: boolean; isComposing?: boolean }): "next" | "previous" | "accept" | "flag" | null {
  if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || event.repeat || event.defaultPrevented || event.isComposing) return null;
  const target = event.target;
  if (target?.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON|A|SUMMARY)$/.test(target?.tagName ?? "")
    || target?.closest?.("[contenteditable]:not([contenteditable='false']), [role='slider'], [role='textbox'], [data-review-shortcuts='off']")) return null;
  return ({ n: "next", j: "next", ArrowRight: "next", p: "previous", k: "previous", ArrowLeft: "previous", a: "accept", f: "flag" } as const)[event.key as "n"] ?? null;
}
