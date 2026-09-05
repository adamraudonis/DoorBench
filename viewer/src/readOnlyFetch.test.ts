import {afterEach,describe,expect,it} from 'bun:test';
import {fetchReadOnly,ReadRetryBudget} from './readOnlyFetch';

const originalFetch=globalThis.fetch,originalTimer=globalThis.setTimeout,originalClear=globalThis.clearTimeout,originalNow=Date.now;
afterEach(()=>{globalThis.fetch=originalFetch;globalThis.setTimeout=originalTimer;globalThis.clearTimeout=originalClear;Date.now=originalNow;});
function fastTimers(){const delays:number[]=[];globalThis.setTimeout=((callback:any,ms:number,...args:any[])=>{delays.push(ms);return originalTimer(callback,0,...args);}) as typeof setTimeout;return delays;}
function failed(status:number,events:string[]=[],header?:string){
  const body=new ReadableStream<Uint8Array>({start(c){c.enqueue(new Uint8Array([1]));},cancel(){events.push('cancel');}});
  return new Response(body,{status,headers:header===undefined?{}:{'Retry-After':header}});
}

describe('bounded read-only download retry',()=>{
  it('replays the same immutable GET after 429/503 and cancels each body before waiting',async()=>{
    const events:string[]=[],requests:any[]=[],delays=fastTimers();
    const timer=globalThis.setTimeout;globalThis.setTimeout=((...args:any[])=>{events.push('wait');return (timer as any)(...args);}) as typeof setTimeout;
    globalThis.fetch=(async(url,options)=>{requests.push([url,options]);return requests.length===1?failed(429,events):requests.length===2?failed(503,events):new Response('verified bytes');}) as typeof fetch;
    const url='https://huggingface.co/datasets/owner/data/resolve/'+'a'.repeat(40)+'/clips/content-hash.json.gz';
    const response=await fetchReadOnly(url,{cache:'no-cache'});
    expect(await response.text()).toBe('verified bytes');expect(delays).toEqual([2000,4000]);
    expect(events).toEqual(['cancel','wait','cancel','wait']);
    expect(requests).toEqual(Array.from({length:3},()=>[url,{cache:'no-cache',method:'GET'}]));
  });
  it('honors Retry-After seconds and standard HTTP dates',async()=>{
    const delays=fastTimers();let calls=0;const now=Date.UTC(2026,8,5,12,0,0);Date.now=()=>now;
    globalThis.fetch=(async()=>++calls===1?failed(429,[],'7'):calls===2?failed(503,[],new Date(now+30_000).toUTCString()):new Response('ok')) as typeof fetch;
    expect((await fetchReadOnly('https://example.test/file')).ok).toBeTrue();expect(delays).toEqual([7000,30000]);
  });
  it.each([400,401,403,404,409,500,502,504])('never retries permanent HTTP %s',async(status)=>{
    const events:string[]=[],delays=fastTimers();let calls=0;
    globalThis.fetch=(async()=>{calls++;return failed(status,events,'2');}) as typeof fetch;
    expect((await fetchReadOnly('https://example.test/file')).status).toBe(status);
    expect(calls).toBe(1);expect(events).toEqual(['cancel']);expect(delays).toEqual([]);
  });
  it('propagates network failures without retrying',async()=>{
    const delays=fastTimers();let calls=0;const error=new TypeError('network unavailable');
    globalThis.fetch=(async()=>{calls++;throw error;}) as typeof fetch;
    await expect(fetchReadOnly('https://example.test/file')).rejects.toBe(error);
    expect(calls).toBe(1);expect(delays).toEqual([]);
  });
  it('caps retries even when every HTTP response requests zero delay',async()=>{
    const delays=fastTimers();let calls=0;
    globalThis.fetch=(async()=>{calls++;return failed(429,[],'0');}) as typeof fetch;
    expect((await fetchReadOnly('https://example.test/file')).status).toBe(429);
    expect(calls).toBe(5);expect(delays).toEqual([2000,4000,8000,16000]);
  });
  it('shares the 120 s wait budget across a clip and its source requests',async()=>{
    const delays=fastTimers(),budget=new ReadRetryBudget();let calls=0;
    globalThis.fetch=(async()=>++calls===2?new Response('clip'):failed(429,[],'60')) as typeof fetch;
    expect((await fetchReadOnly('https://example.test/clip',{},budget)).ok).toBeTrue();
    expect((await fetchReadOnly('https://example.test/source',{},budget)).status).toBe(429);
    expect(calls).toBe(4);expect(delays).toEqual([60_000,60_000]);
  });
  it('shares the four-retry count across separate successful source requests',async()=>{
    const delays=fastTimers(),budget=new ReadRetryBudget();let calls=0;
    globalThis.fetch=(async()=>++calls%2===1?failed(503):new Response('source')) as typeof fetch;
    for(let i=0;i<4;i++)expect((await fetchReadOnly(`https://example.test/source${i}`,{},budget)).ok).toBeTrue();
    expect((await fetchReadOnly('https://example.test/source4',{},budget)).status).toBe(503);
    expect(calls).toBe(9);expect(delays).toEqual([2000,2000,2000,2000]);
  });
  it.each(['61','9999999999999999999999999'])('does not retry earlier than an excessive server delay %s',async(header)=>{
    const delays=fastTimers();let calls=0;
    globalThis.fetch=(async()=>{calls++;return failed(429,[],header);}) as typeof fetch;
    expect((await fetchReadOnly('https://example.test/file')).status).toBe(429);
    expect(calls).toBe(1);expect(delays).toEqual([]);
  });
  it.each(['NaN','Infinity','-2','not a date'])('uses exponential delay for malformed Retry-After %s',async(header)=>{
    const delays=fastTimers();let calls=0;
    globalThis.fetch=(async()=>++calls===1?failed(503,[],header):new Response('ok')) as typeof fetch;
    expect((await fetchReadOnly('https://example.test/file')).ok).toBeTrue();expect(delays).toEqual([2000]);
  });
  it('aborts during backoff promptly and never sends the next request',async()=>{
    const controller=new AbortController(),events:string[]=[];let calls=0,waiting!:()=>void,cleared=false;
    const started=new Promise<void>(resolve=>{waiting=resolve;});
    globalThis.setTimeout=((callback:any)=>{const id=originalTimer(callback,60_000);waiting();return id;}) as typeof setTimeout;
    globalThis.clearTimeout=((id:any)=>{cleared=true;originalClear(id);}) as typeof clearTimeout;
    globalThis.fetch=(async(_url,options)=>{calls++;expect(options?.signal).toBe(controller.signal);return failed(429,events);}) as typeof fetch;
    const promise=fetchReadOnly('https://example.test/old-door',{signal:controller.signal});
    await started;controller.abort();
    await expect(promise).rejects.toMatchObject({name:'AbortError'});
    expect(cleared).toBeTrue();expect(calls).toBe(1);expect(events).toEqual(['cancel']);
  });
  it('does not request an already-aborted door',async()=>{
    const controller=new AbortController();controller.abort();let calls=0;
    globalThis.fetch=(async()=>{calls++;return new Response('unexpected');}) as typeof fetch;
    await expect(fetchReadOnly('https://example.test/old-door',{signal:controller.signal})).rejects.toMatchObject({name:'AbortError'});
    expect(calls).toBe(0);
  });
});
