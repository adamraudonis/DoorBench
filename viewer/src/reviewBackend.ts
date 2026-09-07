import {makeDocument, type ReviewDataset, type ReviewMap} from './reviewState';
import type {ReviewScreenshot} from './reviewScreenshots';
export interface BackendState {reviews:ReviewMap;screenshots:ReviewScreenshot[];storage_path:string;updated_at:string}
export async function reviewRequest(datasetId:string,action:string,body?:unknown):Promise<BackendState> {
  const response=await fetch(`/__review/${action}?dataset=${encodeURIComponent(datasetId)}`,{
    method:body===undefined?'GET':'POST',headers:body===undefined?undefined:{'Content-Type':'application/json'},
    body:body===undefined?undefined:JSON.stringify(body),signal:AbortSignal.timeout(15000),
  });
  const result=await response.json();
  if(!response.ok)throw new Error(result.error||`Review server returned ${response.status}`);
  return result;
}
export function syncReviews(dataset:ReviewDataset,reviews:ReviewMap){return reviewRequest(dataset.id,'reviews',makeDocument(dataset,reviews));}
