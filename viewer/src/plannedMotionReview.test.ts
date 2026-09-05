import {describe,it,expect} from 'bun:test';
import {matchingReview,mergeReviews,parseReviewFile,REVIEW_SCHEMA,serializeReviews,type MotionVisualReview} from './plannedMotionReview';
const review:MotionVisualReview={door_id:'db0002_swing_single',clip_sha256:'a'.repeat(64),status:'needs_work',tags:['posture','timing'],note:'Shoulder at 4.2 s',updated_at:'2026-09-05T12:00:00.000Z'};
describe('reviewer-local motion style records',()=>{
  it('round trips only visual judgment and invalidates it for a changed clip hash',()=>{
    const parsed=parseReviewFile(serializeReviews([review]));expect(parsed.reviews).toEqual([review]);
    expect(matchingReview(parsed.reviews,review.door_id,'a'.repeat(64))?.status).toBe('needs_work');
    expect(matchingReview(parsed.reviews,review.door_id,'b'.repeat(64))).toBeUndefined();
    const next={...review,clip_sha256:'b'.repeat(64),status:'pass' as const};
    const merged=mergeReviews(parsed.reviews,[next]);expect(merged.length).toBe(2);expect(matchingReview(merged,review.door_id,'b'.repeat(64))?.status).toBe('pass');
    expect(parseReviewFile(serializeReviews(merged)).reviews).toEqual(merged);
  });
  it('rejects malformed imports before applying any notes',()=>{
    for(const patch of [{status:'accepted_kinematic'},{clip_sha256:'../bad'},{door_id:'../../private'},{tags:['unsafe']},{note:'x'.repeat(4001)},{updated_at:'2026-02-31T00:00:00.000Z'},{accepted:true}]) {
      expect(()=>parseReviewFile(JSON.stringify({schema:REVIEW_SCHEMA,reviews:[{...review,...patch}]}))).toThrow();
    }
    expect(()=>parseReviewFile(JSON.stringify({schema:REVIEW_SCHEMA,reviews:[review,review]}))).toThrow('Duplicate');
    expect(()=>parseReviewFile('x'.repeat(2_000_001))).toThrow('2 MB');
    expect(()=>parseReviewFile('{')).toThrow();
  });
});
