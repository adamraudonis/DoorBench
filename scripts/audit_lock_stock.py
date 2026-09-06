#!/usr/bin/env python3
"""Build the current catalogue privately and inspect all internal bolt clearances."""
from __future__ import annotations
import argparse,hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
from doorbench.build import build_model,_json_default
from doorbench.export.mjcf import build_mjcf
from doorbench.lock_stock_qa import run_lock_stock_qa
from doorbench.spec import generate_all

def sha(content):return hashlib.sha256(content).hexdigest()

def audit(tiers):
    sources={str(p):sha(p.read_bytes()) for p in Path('doorbench').rglob('*.py')};rows=[];start=time.time()
    for spec in generate_all():
        ir=build_model(spec)
        if not ir.meta.get('lock_stock'):continue
        assets={g.mesh_name+'.obj':g.mesh.export(file_type='obj',include_normals=False,include_texture=False).encode()
                for b in ir.bodies for g in b.geoms if g.type=='mesh'}
        row={'door_id':spec['id'],'records':len(ir.meta['lock_stock']),'tiers':{},
             'thin_cartridges':sum(r['thin_stock_edge_cartridge'] for r in ir.meta['lock_stock']),
             'spec_sha256':sha(json.dumps(spec,sort_keys=True).encode()),
             'model_sha256':sha(json.dumps(ir.to_dict('full'),default=_json_default,sort_keys=True).encode())}
        for tier in tiers:
            xml=ET.tostring(build_mjcf(ir,tier=tier,mesh_dir_rel=''),encoding='utf-8')
            model=mujoco.MjModel.from_xml_string(xml,assets)
            report=run_lock_stock_qa(model,ir.meta,tier=tier)
            report['xml_sha256']=sha(xml);row['tiers'][tier]=report
        rows.append(row)
        if not all(r['ok'] for r in row['tiers'].values()):print(spec['id'],'FAILED',flush=True)
    return {'scope':'Direct native bolt-to-parent geometry distances; full scalar travel, authored materials and housings. Not a full mechanical or strength certificate.',
            'source_sha256':sources,'sources_unchanged':all(sha(Path(p).read_bytes())==h for p,h in sources.items()),
            'mujoco_version':mujoco.__version__,'doors':len(rows),'instances':sum(r['records'] for r in rows),
            'passed':sum(all(t['ok'] for t in r['tiers'].values()) for r in rows),'rows':rows,'wall_s':time.time()-start}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tiers',nargs='+',choices=('full','simple','minimal'),default=['full','simple','minimal']);args=parser.parse_args()
    report=audit(args.tiers);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(f"{report['passed']}/{report['doors']} doors; {report['instances']} bolt instances; {report['wall_s']:.2f}s")
    return 0 if report['passed']==report['doors'] else 1

if __name__=='__main__':raise SystemExit(main())
