import {test,expect} from 'bun:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {LocalReviewStore} from './localReviewDev';
import {emptyReview,makeDocument} from './src/reviewState';
const dataset={id:'manifest-test',name:'Test',version:'1',door_ids:['db0001_rollup','db0002_swing_single']};
function fixture(run:(store:LocalReviewStore)=>void){const root=fs.mkdtempSync(path.join(os.tmpdir(),'doorbench-review-test-'));try{run(new LocalReviewStore(root,dataset));}finally{fs.rmSync(root,{recursive:true,force:true});}}
test('persists comments, merges doors, ignores delayed writes, and keeps a previous revision',()=>fixture(store=>{
 const first={...emptyReview(dataset.door_ids[0]),notes:'Latch clips through frame',updated_at:'2026-09-07T18:00:00.000Z'};
 const second={...emptyReview(dataset.door_ids[1]),notes:'Handle floats',updated_at:'2026-09-07T18:00:01.000Z'};
 store.merge(makeDocument(dataset,{[first.door_id]:first}));store.merge(makeDocument(dataset,{[second.door_id]:second}));
 store.merge(makeDocument(dataset,{[first.door_id]:{...first,notes:'stale',updated_at:'2026-09-06T18:00:00.000Z'}}));
 const reopened=new LocalReviewStore(store.root,dataset);
 expect(reopened.read().reviews[first.door_id].notes).toBe(first.notes);
 expect(reopened.read().reviews[second.door_id].notes).toBe(second.notes);
 expect(fs.existsSync(path.join(store.directory,'feedback.previous.json'))).toBe(true);
}));
test('PNG migration is idempotent and deleted attachments stay removed',()=>fixture(store=>{
 const png=Buffer.alloc(24);Buffer.from([137,80,78,71,13,10,26,10]).copy(png);
 const shot={id:'test-shot',dataset_id:dataset.id,door_id:dataset.door_ids[0],filename:'view.png',captured_at:'2026-09-07T18:00:00.000Z',image_data_url:`data:image/png;base64,${png.toString('base64')}`};
 store.screenshot(shot);store.screenshot(shot);expect(store.response().screenshots).toHaveLength(1);
 const file=path.join(store.directory,store.read().screenshots[0].file);
 expect(fs.readFileSync(file).equals(png)).toBe(true);
 store.screenshot(shot,true);store.screenshot(shot);expect(store.response().screenshots).toHaveLength(0);expect(fs.existsSync(file)).toBe(true);
}));
test('invalid data and traversal cannot modify saved feedback',()=>fixture(store=>{
 const before=store.read();
 expect(()=>store.merge({...makeDocument(dataset,{}),reviews:[emptyReview('../escape')]})).toThrow();
 expect(()=>store.screenshot({id:'../escape',dataset_id:dataset.id,door_id:dataset.door_ids[0]})).toThrow();
 expect(()=>store.screenshot({id:'valid-id',dataset_id:dataset.id,door_id:dataset.door_ids[0],captured_at:'2026-09-07',image_data_url:'data:image/png;base64,YmFk'})).toThrow();expect(store.read()).toEqual(before);
}));

test('local HTTP API saves concurrent updates, rejects foreign origins, and rejects a mismatched dataset',async()=>{
 const {createServer}=await import('node:http');
 const {localReviewDev}=await import('./localReviewDev');
 const {datasetFor}=await import('./src/reviewState');
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'doorbench-review-api-'));
 fs.mkdirSync(path.join(root,'assets'));
 const manifest={name:'Test',version:'1',doors:dataset.door_ids.map((id,index)=>({id,index})),families:[]} as any;
 fs.writeFileSync(path.join(root,'assets','manifest.json'),JSON.stringify(manifest));
 const ds=datasetFor(manifest);
 let handler:any;
 const plugin=localReviewDev(path.join(root,'reviews'),path.join(root,'assets'));
 (plugin.configureServer as Function)({middlewares:{use:(fn:any)=>handler=fn}});
 const server=createServer((req,res)=>handler(req,res,()=>{res.statusCode=404;res.end();}));
 await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
 const port=(server.address() as any).port,url=`http://127.0.0.1:${port}/__review/`;
 try{
  const post=(body:unknown,origin?:string)=>fetch(`${url}reviews?dataset=${ds.id}`,{method:'POST',headers:{'Content-Type':'application/json',...(origin?{Origin:origin}:{})},body:JSON.stringify(body)});
  const docs=ds.door_ids.map(id=>makeDocument(ds,{[id]:{...emptyReview(id),notes:`Issue for ${id}`,updated_at:'2026-09-07T18:00:00.000Z'}}));
  const responses=await Promise.all(docs.map(doc=>post(doc)));
  expect(responses.map(r=>r.status)).toEqual([200,200]);
  const state=await (await fetch(`${url}state?dataset=${ds.id}`)).json() as any;
  expect(Object.keys(state.reviews)).toHaveLength(2);
  expect((await post(docs[0],'https://other.example')).status).toBe(403);
  expect((await fetch(`${url}state?dataset=wrong`)).status).toBe(409);
 }finally{await new Promise<void>(resolve=>server.close(()=>resolve()));fs.rmSync(root,{recursive:true,force:true});}
});
