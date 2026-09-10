#!/usr/bin/env python3
"""Bind an extracted terminal grasp to its completed Isaac source evidence."""
import argparse,json
from pathlib import Path
from doorbench.dexterous.qualified_isaac_grasp import load_qualified_isaac_grasp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','extracted','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new admission receipt')
    _,receipt=load_qualified_isaac_grasp(a.run,a.extracted)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'state_sha256':receipt['state_sha256']}))


if __name__=='__main__':main()
