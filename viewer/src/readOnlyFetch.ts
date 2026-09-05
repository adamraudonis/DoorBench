/** Bounded retries for idempotent motion downloads; validation stays with callers. */
export class ReadRetryBudget {
  private retries=0;
  private delayMs=0;
  claim(delayMs:number) {
    if(!Number.isFinite(delayMs)||delayMs<0||delayMs>60_000||this.retries>=4||this.delayMs+delayMs>120_000)return false;
    this.retries++;this.delayMs+=delayMs;return true;
  }
}

function retryAfterMs(value:string|null) {
  if(value===null)return 0;
  const text=value.trim();
  if(/^\d+$/.test(text))return Number(text)*1000;
  // Accept the standard HTTP-date form, not Date.parse's permissive numeric
  // strings. A long valid server delay must exhaust this call, never retry early.
  if(/^(Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2} GMT$/.test(text)) {
    const date=Date.parse(text);if(Number.isFinite(date))return Math.max(0,date-Date.now());
  }
  return 0;
}

function waitForRetry(ms:number,signal?:AbortSignal) {
  signal?.throwIfAborted();
  return new Promise<void>((resolve,reject)=>{
    const abort=()=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);reject(signal?.reason??new DOMException('Download aborted','AbortError'));};
    const timer=setTimeout(()=>{signal?.removeEventListener('abort',abort);resolve();},ms);
    signal?.addEventListener('abort',abort,{once:true});
  });
}

/** GET only. Share one budget across a clip and all its source-resource requests.
 * Only HTTP 429/503 are retried. Network, body, checksum and schema failures are
 * not retried. Non-success response bodies are cancelled before return/backoff.
 */
export async function fetchReadOnly(url:string,options:{signal?:AbortSignal;cache?:RequestCache}={},budget=new ReadRetryBudget()) {
  for(let attempt=0;;attempt++) {
    options.signal?.throwIfAborted();
    const response=await fetch(url,{...options,method:'GET'});
    if(response.ok){options.signal?.throwIfAborted();return response;}
    // Starting cancellation releases the connection without waiting on a slow
    // server. A door change can therefore abort the following timer promptly.
    void response.body?.cancel().catch(()=>{});
    options.signal?.throwIfAborted();
    if(response.status!==429&&response.status!==503)return response;
    const delay=Math.max(2000*2**attempt,retryAfterMs(response.headers.get('Retry-After')));
    if(!budget.claim(delay))return response;
    await waitForRetry(delay,options.signal);
  }
}
