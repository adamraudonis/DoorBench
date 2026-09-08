#!/usr/bin/env python3
"""Fetch the pinned official H1 actor, deployment configuration and license."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
from doorbench.dexterous.locomotion import UPSTREAM_URL,UPSTREAM_REVISION,POLICY_RELATIVE_PATH,POLICY_SHA256

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    files=(POLICY_RELATIVE_PATH,'deploy/deploy_mujoco/configs/h1.yaml','deploy/deploy_mujoco/deploy_mujoco.py','LICENSE')
    args.output.mkdir(parents=True,exist_ok=True);manifest={'upstream':UPSTREAM_URL,'revision':UPSTREAM_REVISION,'files':{}}
    for relative in files:
        path=args.output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/{UPSTREAM_REVISION}/{relative}'
        content=urllib.request.urlopen(url,timeout=60).read();digest=hashlib.sha256(content).hexdigest()
        if relative==POLICY_RELATIVE_PATH and digest!=POLICY_SHA256:raise ValueError('Pinned H1 policy hash mismatch')
        path.write_bytes(content);manifest['files'][relative]={'url':url,'sha256':digest,'bytes':len(content)}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(args.output/POLICY_RELATIVE_PATH)
if __name__=='__main__':main()
