import {afterEach, describe, expect, it} from 'bun:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import {humanReferenceDev} from './humanReferenceDev';
const roots:string[]=[], servers:http.Server[]=[];
afterEach(async()=>{await Promise.all(servers.splice(0).map(s=>new Promise<void>(resolve=>s.close(()=>resolve()))));for(const root of roots.splice(0))fs.rmSync(root,{recursive:true,force:true});});
async function fixture(){
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'doorbench-human-dev-'));roots.push(root);
 const dir=path.join(root,'out/human-reference/ceti-d02-o03');fs.mkdirSync(dir,{recursive:true});fs.mkdirSync(path.join(root,'docs'));
 fs.writeFileSync(path.join(dir,'motion.json'),'{}');fs.writeFileSync(path.join(dir,'contact-fit.json'),'report');fs.writeFileSync(path.join(dir,'animation.glb'),'binary');fs.writeFileSync(path.join(dir,'normal-speed.mp4'),'video');fs.writeFileSync(path.join(dir,'private.txt'),'private');fs.writeFileSync(path.join(root,'docs/HUMAN_REFERENCE.md'),'method');
 let handler:any;const plugin=humanReferenceDev(root);(plugin.configureServer as Function)({middlewares:{use(h:any){handler=h;}}});
 const server=http.createServer((req,res)=>handler(req,res,()=>{res.statusCode=404;res.end('fallback');}));servers.push(server);await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
 return {root,dir,plugin,url:`http://127.0.0.1:${(server.address() as any).port}/__human-reference/`};
}
describe('development-only human asset allowlist',()=>{
 it('serves only named files with no cache, including current local methodology',async()=>{const f=await fixture();expect(f.plugin.apply).toBe('serve');for(const [file,body] of [['motion.json','{}'],['contact-fit.json','report'],['animation.glb','binary'],['normal-speed.mp4','video'],['methodology.md','method']]){const r=await fetch(f.url+file);expect(r.status).toBe(200);expect(r.headers.get('cache-control')).toBe('no-store');expect(await r.text()).toBe(body);}expect((await fetch(f.url+'animation.glb',{method:'HEAD'})).headers.get('content-length')).toBe('6');});
 it('rejects writes, unknown files and escaped paths',async()=>{const f=await fixture();expect((await fetch(f.url+'motion.json',{method:'POST',body:'bad'})).status).toBe(405);for(const name of ['private.txt','%2e%2e%2fprivate.txt','%','motion.json/extra','constructor'])expect((await fetch(f.url+name)).status).toBe(404);expect(fs.readFileSync(path.join(f.dir,'motion.json'),'utf8')).toBe('{}');});
 it('rejects a symlink outside the configured capture root',async()=>{const f=await fixture();fs.unlinkSync(path.join(f.dir,'animation.glb'));fs.symlinkSync(path.join(f.root,'docs/HUMAN_REFERENCE.md'),path.join(f.dir,'animation.glb'));expect((await fetch(f.url+'animation.glb')).status).toBe(404);});
});
