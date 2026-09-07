#!/usr/bin/env python3
"""Cluster-independent launcher. Provisioning and credentials stay outside it."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def command(config,root,output,overrides):
    if config['schema_version']!='doorbench.experiment.v1':raise ValueError('Unknown config version')
    if (config['stage'],config['robot_adapter'],config['simulator']) != ('privileged_body_reach','h1-shadow-v1','mujoco-native'):
        raise ValueError('This launcher has only verified the H1/Shadow native body-reaching adapter')
    args=[sys.executable,str(root/'scripts/dexterous/train_reach.py'),'--output',str(output)]
    for key in ('upstream','robot','door'):
        path=Path(config[key]);args+=['--'+key,str(path if path.is_absolute() else root/path)]
    values=dict(config['training']);values.update({k:v for k,v in overrides.items() if v is not None})
    if set(values)-{'steps','envs','distance','device','seed','checkpoint','standing_weight'}:raise ValueError('Unknown training option')
    for key,value in values.items():args+=['--'+key.replace('_','-'),str(value)]
    return args


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/dexterous/h1-shadow-reach-v1.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--envs',type=int);p.add_argument('--steps',type=int);p.add_argument('--device');p.add_argument('--checkpoint')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    cfg=json.loads(a.config.read_text())
    args=command(cfg,root,a.output.resolve(),{k:getattr(a,k) for k in ('envs','steps','device','checkpoint')})
    if a.dry_run:print(json.dumps(args,indent=2));return
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'experiment.json').write_text(json.dumps(cfg,indent=2)+'\n')
    raise SystemExit(subprocess.call(args,cwd=root))

if __name__=='__main__':main()
