import { describe, expect, test } from 'bun:test';
import { ELIGIBILITY_POLICY, eligibleMotionEntries, isPetDoor, referenceUnavailable, resultsRespectEligibility, standardManifest } from './collections';
import { fetchReference } from './referenceMotion';
import { fetchPlannedClip, type MotionEntry } from './plannedReferenceMotion';
import type { Manifest, ManifestDoor } from './types';

const ordinary = {id:'db0002_swing_single',family:'swing_single',extras:['pet_flap'],signed_off:true} as ManifestDoor;
const pet = {id:'db0021_pet_door',family:'pet_door',signed_off:true,benchmark:{scenarios:['open_and_traverse']},benchmark_eligibility:{eligible:true}} as unknown as ManifestDoor;
const gate = {id:'db0060_baby_gate',family:'baby_gate',signed_off:false,reference_motion_available:false,reference_motion_unavailable_reason:'Door geometry was revised; archived motion uses the earlier overhead wall.'} as ManifestDoor;
const manifest = {doors:[ordinary,pet,gate],n_doors:3,n_signed_off:2,families:['swing_single','pet_door','baby_gate']} as Manifest;

describe('supplementary collection boundaries', () => {
  test('derives standard catalogue and counts from family despite stale eligibility and scenario metadata', () => {
    const result=standardManifest(manifest);
    expect(result.doors.map(d=>d.id)).toEqual([ordinary.id,gate.id]);
    expect(result.n_doors).toBe(2); expect(result.n_signed_off).toBe(1);
    expect(result.families).toEqual(['swing_single','baby_gate']);
    expect(isPetDoor(ordinary)).toBe(false);
    expect(manifest.doors).toHaveLength(3); expect(manifest.n_doors).toBe(3);
  });
  test('rejects pet and unknown archive rows even if an archived row misstates its family', () => {
    const rows=[{door_id:ordinary.id,family:ordinary.family},{door_id:pet.id,family:'swing_single'},{door_id:gate.id,family:gate.family},{door_id:'db9999_swing_single',family:'swing_single'}];
    expect(eligibleMotionEntries(rows,manifest).map(d=>d.door_id)).toEqual([ordinary.id,gate.id]);
    expect(referenceUnavailable(pet)).toContain('excluded');
    expect(referenceUnavailable(gate)).toContain('earlier overhead wall');
    expect(referenceUnavailable(ordinary)).toBeNull();
    expect(referenceUnavailable(undefined)).toContain('absent');
  });
  test('prevents both legacy and planned pet downloads before network access', async () => {
    const original=globalThis.fetch; let requests=0;
    globalThis.fetch=(async()=>{requests++;throw Error('Unexpected network');}) as typeof fetch;
    try {
      await expect(fetchReference(pet.id)).rejects.toThrow('supplementary');
      await expect(fetchPlannedClip({door_id:pet.id,family:'pet_door'} as MotionEntry,'https://example.org/index.json')).rejects.toThrow('supplementary');
      expect(requests).toBe(0);
    } finally { globalThis.fetch=original; }
  });
});

describe('historical result eligibility', () => {
  const valid = () => ({eligibility_policy:ELIGIBILITY_POLICY,n_doors_total:2,results:[{suites:{core:{by_family:{swing_single:{}},doors:{[ordinary.id]:[1,1]}}}}]});
  test('requires derived eligible aggregates, never relabels an old mixed denominator', () => {
    const index=valid();
    expect(resultsRespectEligibility(index,manifest)).toBe(true);
    expect(resultsRespectEligibility({...index,eligibility_policy:undefined},manifest)).toBe(false);
    expect(resultsRespectEligibility({...index,n_doors_total:3},manifest)).toBe(false);
  });
  test('rejects leaked pet families or per-door episodes despite new metadata', () => {
    const index=valid();
    (index.results[0].suites.core.doors as Record<string,number[]>)[pet.id]=[1,1];
    expect(resultsRespectEligibility(index,manifest)).toBe(false);
    expect(resultsRespectEligibility(index)).toBe(false);
    const family=valid(); (family.results[0].suites.core.by_family as Record<string,object>).pet_door={};
    expect(resultsRespectEligibility(family,manifest)).toBe(false);
  });
});
