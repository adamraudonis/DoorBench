import {reviewRequest} from './reviewBackend';
export interface ReviewScreenshot {
  id: string;
  dataset_id: string;
  door_id: string;
  filename: string;
  captured_at: string;
  image_data_url: string;
}
function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('doorbench-review-screenshots', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('screenshots', {keyPath: 'id'});
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function loadBrowserScreenshots(datasetId: string): Promise<ReviewScreenshot[]> {
  const db = await openDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const request = db.transaction('screenshots').objectStore('screenshots').getAll();
      request.onsuccess = () => resolve((request.result as ReviewScreenshot[]).filter(s => s.dataset_id === datasetId));
      request.onerror = () => reject(request.error);
    });
  } finally { db.close(); }
}
async function writeBrowserScreenshot(screenshot: ReviewScreenshot, remove = false) {
  const db = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction('screenshots', 'readwrite');
      const store = transaction.objectStore('screenshots');
      if (remove) store.delete(screenshot.id); else store.put(screenshot);
      transaction.oncomplete = () => resolve();
      transaction.onabort = () => reject(transaction.error ?? new Error('Screenshot storage failed'));
      transaction.onerror = () => reject(transaction.error);
    });
  } finally { db.close(); }
}
export function pngDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/** Migrate old browser attachments before switching to disk-backed storage. Originals remain as backup. */
export async function loadScreenshots(datasetId:string):Promise<ReviewScreenshot[]> {
  const key=`doorbench:screenshot-migration:${datasetId}`;
  let migrated=false;
  try{migrated=localStorage.getItem(key)==='done';}catch{}
  if(!migrated) {
    let browserShots:ReviewScreenshot[]=[];
    try{browserShots=await loadBrowserScreenshots(datasetId);}catch{/* Browser storage is optional. */}
    for(const shot of browserShots)await reviewRequest(datasetId,'screenshot',shot);
    try{localStorage.setItem(key,'done');}catch{/* Repeated migration is idempotent. */}
  }
  return (await reviewRequest(datasetId,'state')).screenshots;
}
export async function writeScreenshot(screenshot:ReviewScreenshot,remove=false):Promise<ReviewScreenshot[]> {
  // Keep a retry copy if the backend is briefly unavailable during a dev-server restart.
  if(!remove){try{await writeBrowserScreenshot(screenshot);localStorage.removeItem(`doorbench:screenshot-migration:${screenshot.dataset_id}`);}catch{/* Disk saves do not require browser storage. */}}
  const result=await reviewRequest(screenshot.dataset_id,remove?'remove-screenshot':'screenshot',screenshot);
  return result.screenshots;
}
