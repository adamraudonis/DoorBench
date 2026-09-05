import type { Manifest, ManifestDoor } from './types';

export const ELIGIBILITY_POLICY = 'doorbench.benchmark-eligibility.v1';
export const PET_COLLECTION_REASON = 'Standalone pet doors are downloadable supplementary assets and excluded from robot and human benchmark evaluation.';
// Derive from the actual family, including for older published manifests.
// A pet-flap extra on an ordinary door does not change its collection.
export function isPetDoor(door: { family: string }): boolean { return door.family === 'pet_door'; }
export function isPetDoorId(id: string): boolean { return /^db\d+_pet_door$/.test(id); }
export function standardManifest(manifest: Manifest): Manifest {
  const doors = manifest.doors.filter(d => !isPetDoor(d));
  return { ...manifest, doors, n_doors: doors.length, n_signed_off: doors.filter(d => d.signed_off).length,
    families: manifest.families.filter(f => f !== 'pet_door') };
}
export function referenceUnavailable(door: Pick<ManifestDoor, 'family' | 'reference_motion_available' | 'reference_motion_unavailable_reason'> | undefined): string | null {
  if (!door) return 'The source door is absent from the current collection.';
  if (isPetDoor(door)) return PET_COLLECTION_REASON;
  if (door.reference_motion_available === false) return door.reference_motion_unavailable_reason || 'Door geometry was revised; archived motion is unavailable for this version.';
  return null;
}
export function eligibleMotionEntries<T extends {door_id:string;family:string}>(entries:T[], manifest:Manifest): T[] {
  const eligible = new Set(manifest.doors.filter(d => !isPetDoor(d)).map(d => d.id));
  return entries.filter(d => !isPetDoor(d) && eligible.has(d.door_id));
}

/** Old combined indexes cannot be relabeled with a smaller denominator. The
 * publisher must recompute aggregates from original eligible episodes first. */
export function resultsRespectEligibility(index: { eligibility_policy?: string; n_doors_total: number; results: Array<{suites: Record<string, {by_family: Record<string, unknown>; doors: Record<string, unknown>} | undefined>}> }, manifest?: Manifest): boolean {
  if (index.eligibility_policy !== ELIGIBILITY_POLICY) return false;
  const eligible = manifest ? new Set(manifest.doors.filter(d => !isPetDoor(d)).map(d => d.id)) : null;
  if (eligible && index.n_doors_total !== eligible.size) return false;
  return index.results.every(r => Object.values(r.suites).every(s => !s ||
    !Object.hasOwn(s.by_family, 'pet_door') && Object.keys(s.doors).every(id => !isPetDoorId(id) && (!eligible || eligible.has(id)))));
}
