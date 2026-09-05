// Integration check against a running viewer with generated assets.
// PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node viewer/qa/pet-collection.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.VIEWER_URL || 'http://127.0.0.1:5190/';
const out = process.env.QA_OUT || 'out/pet-collection-browser';
await fs.mkdir(out,{recursive:true});
const manifestBytes = Buffer.from(await (await fetch(new URL('assets/manifest.json',base))).arrayBuffer());
const manifest = JSON.parse(manifestBytes.toString());
const sha256=bytes=>createHash('sha256').update(bytes).digest('hex');
const pets=manifest.doors.filter(d=>d.family==='pet_door');
const standard=manifest.doors.filter(d=>d.family!=='pet_door');
const pet=pets[0]; assert(pet);
const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const context=await browser.newContext({viewport:{width:1440,height:1050}});
const page=await context.newPage();
const errors=[],requests=[],checks=[];
page.on('pageerror',error=>errors.push(String(error)));
page.on('request',r=>requests.push(r.url()));
async function check(name,fn){await fn();checks.push(name);console.log(`PASS ${name}`);}
async function go(hash,reload=false){await page.goto(`${base}${hash}`);if(reload)await page.reload();await page.locator('.loading').first().waitFor({state:'hidden',timeout:45000}).catch(()=>{});}
async function noOverflow(){assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));}
try {
  await go('#/');
  await check('standard catalogue excludes standalone pets',async()=>{
    await page.waitForSelector('.door-card');
    assert((await page.locator('.catalogue-hero').innerText()).includes(standard.length.toLocaleString()));
    assert.equal(await page.locator('select option[value="pet_door"]').count(),0);
    assert.equal(await page.locator('.door-card[href$="_pet_door"]').count(),0);
    assert((await page.locator('a[href="#/pets"]').first().innerText()).includes(String(pets.length)));
  });
  await page.screenshot({path:path.join(out,'catalogue-desktop.png')});
  await go('#/pets');
  await check('supplementary collection keeps every downloadable pet',async()=>{
    await page.waitForSelector('.pet-collection .door-card');
    assert.equal(await page.locator('.pet-collection .door-card').count(),pets.length);
    assert.equal(await page.locator('.card-parity').count(),0);
    assert((await page.locator('.pet-scope').innerText()).includes('outside the benchmark'));
    await noOverflow();
  });
  await page.screenshot({path:path.join(out,'pets-desktop.png')});
  await go('#/?family=pet_door');
  await check('legacy pet family deep link opens supplementary collection',async()=>{assert.equal(await page.locator('.pet-collection').count(),1);});
  const start=requests.length;
  await go(`#/door/${pet.id}?eval=1&reference=1&scenario=open_and_traverse`);
  await page.waitForSelector('.joint');
  await check('pet detail rejects stale evaluation/reference deep links before fetch',async()=>{
    assert((await page.locator('.side').innerText()).includes('Supplementary pet-door asset'));
    assert.equal(await page.getByRole('button',{name:/evaluation/i}).count(),0);
    assert.equal(await page.locator('.reference-player').count(),0);
    assert.equal(await page.locator('.side h3').filter({hasText:/^Evaluation$|^Isaac parity$/}).count(),0);
    assert.equal(await page.locator('.side .chip.res').count(),0);
    assert.equal(await page.locator('.doorview a[href*="#/motions"]').count(),0);
    assert.equal(requests.slice(start).filter(u=>u.includes('/reference-motions/')||u.includes('/planned-references/')).length,0);
    const href=await page.getByRole('link',{name:'MJCF (full)',exact:true}).getAttribute('href');
    assert.equal((await fetch(new URL(href,base))).status,200);
    assert.equal(await page.getByRole('link',{name:'model.json',exact:true}).count(),1);
  });
  await page.screenshot({path:path.join(out,'pet-detail-desktop.png')});
  await go('#/families');
  await check('families distinguish standard taxonomy from supplements',async()=>{
    assert.equal(await page.locator('.famcard[href*="pet_door"]').count(),0);
    assert.equal(await page.locator('.famcard').count(),new Set(standard.map(d=>d.family)).size);
    assert((await page.locator('main').innerText()).includes('Supplementary pet-door collection'));
  });
  await go('#/results');
  await check('results fail closed or show eligible historical subset',async()=>{
    const text=await page.locator('main').innerText();
    assert(text.includes('Results require the standard-door subset.')||text.includes('Historical run · eligible-door subset'));
    assert.equal(await page.locator('.results a[href$="_pet_door"]').count(),0);
    assert(!text.includes('/ 1000'));
  });
  await page.screenshot({path:path.join(out,'results-desktop.png')});
  const motionStart=requests.length;
  await go(`#/motions?door=${pet.id}`);
  await page.getByRole('heading',{name:'Supplementary pet-door asset'}).waitFor({timeout:60000});
  await check('archived MotionLab excludes pets and blocks explicit selected pet playback',async()=>{
    assert((await page.locator('.motion-counts').innerText()).includes(String(standard.length)));
    assert.equal(await page.locator('.motion-queue button').filter({hasText:'Pet door'}).count(),0);
    assert.equal(await page.locator('.motion-player').count(),0);
    assert.equal(requests.slice(motionStart).filter(u=>/\/clips\//.test(u)).length,0);
  });
  await page.screenshot({path:path.join(out,'pet-motion-block-desktop.png')});
  const revised=manifest.doors.find(d=>d.reference_motion_available===false&&d.family!=='pet_door');
  if(revised){
    const before=requests.length;
    await go(`#/door/${revised.id}?reference=1`);
    await page.getByText('Archived motion unavailable',{exact:true}).waitFor();
    await check('revised geometry blocks legacy motion before network',async()=>{
      assert.equal(await page.locator('.reference-player').count(),0);
      assert.equal(requests.slice(before).filter(u=>u.includes('/reference-motions/')).length,0);
    });
    await go(`#/motions?door=${revised.id}`);
    await page.getByRole('heading',{name:'Archived motion unavailable'}).waitFor({timeout:60000});
    await check('revised archive is separate from current accepted count and playback',async()=>{
      assert((await page.locator('.motion-selected').innerText()).includes('Geometry revised'));
      assert.equal(await page.locator('.motion-player').count(),0);
      assert.equal(requests.slice(before).filter(u=>/\/clips\//.test(u)).length,0);
    });
    await page.screenshot({path:path.join(out,'revised-gate-motion.png')});
  }
  await page.evaluate(id=>{window.location.hash=`#/motions?door=${id}`;},pet.id);
  await page.getByRole('heading',{name:'Supplementary pet-door asset'}).waitFor();
  await check('same-page motion deep-link changes preserve pet exclusion',async()=>{assert.equal(await page.locator('.motion-player').count(),0);});
  await go(`#/review?door=${pet.id}`);
  await check('asset review segregates pet queue while preserving access',async()=>{
    assert.equal(await page.getByLabel('Review collection').inputValue(),'pets');
    assert((await page.locator('.review-filters').innerText()).includes(`${pets.length} matching doors`));
    await page.getByLabel('Review collection').selectOption('standard');
    assert((await page.locator('.review-filters').innerText()).includes(`${standard.length} matching doors`));
    assert.equal(await page.locator('.review-queue button').filter({hasText:'pet_door'}).count(),0);
  });
  await page.route('**/assets/manifest.json',async route=>{
    const stale=structuredClone(manifest);
    for(const d of stale.doors){delete d.benchmark_eligibility;delete d.reference_motion_available;delete d.reference_motion_unavailable_reason;}
    const stalePet=stale.doors.find(d=>d.id===pet.id);stalePet.benchmark={scenarios:['open_and_traverse'],human:['hold_open_for_human']};
    await route.fulfill({json:stale});
  });
  const staleStart=requests.length;
  await go(`#/door/${pet.id}?eval=1&reference=1&scenario=hold_open_for_human`,true);
  await page.waitForSelector('.joint');
  await check('legacy manifest without eligibility metadata cannot revive pet evaluation',async()=>{
    assert.equal(await page.getByRole('button',{name:/evaluation/i}).count(),0);
    assert.equal(await page.locator('.reference-player,.side .chip.res').count(),0);
    assert.equal(requests.slice(staleStart).filter(u=>u.includes('/reference-motions/')).length,0);
  });
  await page.unroute('**/assets/manifest.json');
  const resultBytes=Buffer.from(await (await fetch(new URL('results/index.json',base))).arrayBuffer());
  await page.route('**/results/index.json',async route=>{const stale=JSON.parse(resultBytes);delete stale.eligibility_policy;stale.n_doors_total=manifest.doors.length;await route.fulfill({json:stale});});
  await go('#/results',true);
  await check('legacy combined results are hidden instead of relabeled',async()=>{
    await page.getByRole('heading',{name:'Results require the standard-door subset.'}).waitFor();
    assert.equal(await page.locator('.result-summary,.rtable').count(),0);
  });
  await page.unroute('**/results/index.json');
  await page.setViewportSize({width:390,height:844});
  await go('#/pets',true);
  await check('mobile supplementary layout fits viewport',noOverflow);
  await page.screenshot({path:path.join(out,'pets-mobile.png')});
  await go(`#/door/${pet.id}?eval=1&reference=1`);
  await page.waitForSelector('.joint');
  await check('mobile pet detail keeps downloads and no horizontal overflow',async()=>{await noOverflow();assert.equal(await page.getByRole('link',{name:'MJCF (full)',exact:true}).count(),1);});
  await page.screenshot({path:path.join(out,'pet-detail-mobile.png')});
  assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(out,'receipt.json'),JSON.stringify({base,at:new Date().toISOString(),standard_count:standard.length,pet_count:pets.length,pet_id:pet.id,revised_id:revised?.id??null,checks,page_errors:errors,requests,manifest_sha256:sha256(manifestBytes),results_index_sha256:sha256(resultBytes)},null,2));
  console.log(JSON.stringify({checks:checks.length,out,pet:pet.id,revised:revised?.id??null}));
} finally {await browser.close();}
