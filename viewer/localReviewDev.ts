import fs from 'node:fs';
import path from 'node:path';
import {randomUUID} from 'node:crypto';
import type {Plugin} from 'vite';
import {datasetFor, makeDocument, mergeReviews, parseDocument, type ReviewDataset, type ReviewMap} from './src/reviewState.ts';

type Screenshot = {id:string;dataset_id:string;door_id:string;filename:string;captured_at:string;file:string;deleted_at?:string};
type State = {schema:'doorbench.local-review.v1';dataset:ReviewDataset;reviews:ReviewMap;screenshots:Screenshot[];updated_at:string};
const safeId = (id:string) => /^[a-zA-Z0-9_-]{1,100}$/.test(id);
function atomic(file:string, contents:string|Buffer) {
  fs.mkdirSync(path.dirname(file),{recursive:true});
  const temp=`${file}.${randomUUID()}.tmp`;
  const fd=fs.openSync(temp,'wx');
  try {fs.writeFileSync(fd,contents);fs.fsyncSync(fd);} finally {fs.closeSync(fd);}
  fs.renameSync(temp,file);
}
/** Synchronous read/merge/atomic-write transactions cannot interleave within the dev server. */
export class LocalReviewStore {
  constructor(readonly root:string, readonly dataset:ReviewDataset) {
    if(!safeId(dataset.id)||dataset.door_ids.some(id=>!safeId(id)))throw Error('Invalid dataset identifiers');
  }
  get directory(){return path.join(this.root,this.dataset.id);}
  read():State {
    const file=path.join(this.directory,'feedback.json');
    if(!fs.existsSync(file))return {schema:'doorbench.local-review.v1',dataset:this.dataset,reviews:{},screenshots:[],updated_at:new Date(0).toISOString()};
    const state=JSON.parse(fs.readFileSync(file,'utf8')) as State;
    if(state.schema!=='doorbench.local-review.v1'||state.dataset.id!==this.dataset.id)throw Error('Invalid stored review data');
    parseDocument(JSON.stringify(makeDocument(this.dataset,state.reviews)),this.dataset);
    return state;
  }
  save(state:State) {
    state.updated_at=new Date().toISOString();
    // Keep the preceding revision as an additional recovery copy.
    const file=path.join(this.directory,'feedback.json');
    if(fs.existsSync(file))atomic(path.join(this.directory,'feedback.previous.json'),fs.readFileSync(file));
    atomic(file,JSON.stringify(state,null,2)+'\n');
  }
  merge(document:unknown) {
    const incoming=parseDocument(JSON.stringify(document),this.dataset);
    const state=this.read();
    state.reviews=mergeReviews(state.reviews,incoming.reviews).reviews;
    if(JSON.stringify(state.reviews)!==JSON.stringify(this.read().reviews))this.save(state);
    return state;
  }
  screenshot(input:any,remove=false) {
    if(!input||!safeId(input.id)||!this.dataset.door_ids.includes(input.door_id)||input.dataset_id!==this.dataset.id)throw Error('Invalid screenshot identifiers');
    const state=this.read();
    const existing=state.screenshots.find(s=>s.id===input.id);
    if(existing&&existing.door_id!==input.door_id)throw Error('Screenshot belongs to another door');
    if(remove){if(existing)existing.deleted_at=new Date().toISOString();this.save(state);return;}
    // Idempotent browser migration; removed attachments must not be reattached on a later visit.
    if(existing)return;
    if(typeof input.captured_at!=='string'||!Number.isFinite(Date.parse(input.captured_at)))throw Error('Invalid capture timestamp');
    if(typeof input.image_data_url!=='string'||!/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(input.image_data_url))throw Error('Expected a PNG screenshot');
    const png=Buffer.from(input.image_data_url.split(',')[1],'base64');
    if(png.length>20*1024*1024||png.length<24||!png.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))throw Error('Invalid PNG screenshot');
    const file=`screenshots/${input.door_id}/${input.id}.png`;
    atomic(path.join(this.directory,file),png);
    state.screenshots.push({id:input.id,dataset_id:this.dataset.id,door_id:input.door_id,filename:path.basename(input.filename||`${input.door_id}.png`),captured_at:input.captured_at,file});
    this.save(state);
  }
  response() {
    const state=this.read();
    return {reviews:state.reviews,screenshots:state.screenshots.filter(s=>!s.deleted_at).map(s=>({...s,image_data_url:`/__review/image?dataset=${this.dataset.id}&id=${s.id}`})),storage_path:this.directory,updated_at:state.updated_at};
  }
}
async function withWriteLock<T>(root:string,operation:()=>T):Promise<T> {
  fs.mkdirSync(root,{recursive:true});
  const lock=path.join(root,'.write-lock'),started=Date.now();
  while(true) {
    try{fs.mkdirSync(lock);break;}catch(e){
      if((e as NodeJS.ErrnoException).code!=='EEXIST')throw e;
      try{if(Date.now()-fs.statSync(lock).mtimeMs>30000){fs.rmdirSync(lock);continue;}}catch{}
      if(Date.now()-started>5000)throw Error('Review storage busy; retry the save');
      await new Promise(resolve=>setTimeout(resolve,20));
    }
  }
  try{return operation();}finally{fs.rmdirSync(lock);}
}
export function localReviewDev(root:string,assetRoot:string):Plugin {
  let cached:{mtime:number;dataset:ReviewDataset}|undefined;
  function getStore(){
    const file=path.join(assetRoot,'manifest.json'),mtime=fs.statSync(file).mtimeMs;
    if(!cached||cached.mtime!==mtime)cached={mtime,dataset:datasetFor(JSON.parse(fs.readFileSync(file,'utf8')))};
    return new LocalReviewStore(root,cached.dataset);
  }
  return {name:'local-review-backend',apply:'serve',configureServer(server){
    server.middlewares.use(async(req,res,next)=>{
      if(!req.url?.startsWith('/__review/'))return next();
      res.setHeader('Cache-Control','no-store');res.setHeader('X-Content-Type-Options','nosniff');
      const json=(code:number,data:unknown)=>{res.statusCode=code;res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));};
      // Review writes are available only on loopback and from this server's own origin.
      const address=req.socket.remoteAddress;
      if(!['127.0.0.1','::1','::ffff:127.0.0.1'].includes(address??''))return json(403,{error:'Local access only'});
      if(req.headers.origin&&req.headers.origin!==`http://${req.headers.host}`)return json(403,{error:'Same-origin access only'});
      try {
        const url=new URL(req.url,'http://localhost'),store=getStore();
        if(url.searchParams.get('dataset')!==store.dataset.id)return json(409,{error:'Dataset changed. Reload the review page.'});
        if(req.method==='GET'&&url.pathname==='/__review/state')return json(200,store.response());
        if(req.method==='GET'&&url.pathname==='/__review/image'){
          const shot=store.read().screenshots.find(s=>s.id===url.searchParams.get('id')&&!s.deleted_at);
          if(!shot)return json(404,{error:'Screenshot not found'});
          res.setHeader('Content-Type','image/png');res.end(fs.readFileSync(path.join(store.directory,shot.file)));return;
        }
        if(req.method!=='POST')return json(405,{error:'Method not allowed'});
        if(!req.headers['content-type']?.startsWith('application/json'))return json(415,{error:'Expected JSON'});
        let size=0;const chunks:Buffer[]=[];
        for await(const chunk of req){size+=chunk.length;if(size>30*1024*1024)return json(413,{error:'Request too large'});chunks.push(Buffer.from(chunk));}
        const body=JSON.parse(Buffer.concat(chunks).toString('utf8'));
        if(!['/__review/reviews','/__review/screenshot','/__review/remove-screenshot'].includes(url.pathname))return json(404,{error:'Unknown review endpoint'});
        await withWriteLock(root,()=>{
          if(url.pathname==='/__review/reviews')store.merge(body);
          else store.screenshot(body,url.pathname==='/__review/remove-screenshot');
        });
        return json(200,store.response());
      }catch(e){return json(500,{error:`Local review save/load failed: ${e instanceof Error?e.message:String(e)}`});}
    });
  }};
}
