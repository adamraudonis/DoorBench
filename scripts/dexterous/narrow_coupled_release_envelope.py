#!/usr/bin/env python3
"""Declare a prospective progress/aperture domain; new independent audit required."""
import argparse
import json
from pathlib import Path
import shutil
from doorbench.dexterous.coupled_release_geometry import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envelope',type=Path,required=True);p.add_argument('--upper-nodes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--operator-min',type=float);p.add_argument('--operator-max',type=float)
    a=p.parse_args();source=json.loads(a.envelope.read_text());nodes=json.loads(a.upper_nodes.read_text())
    if source.get('schema')!='doorbench.coupled-release-envelope.v1':raise ValueError('Source envelope required')
    a.output.mkdir(parents=True,exist_ok=False);frozen=a.output/'domain-source.py';shutil.copy2(__file__,frozen)
    source['admitted_leaf_upper_nodes']=nodes
    if a.operator_min is not None or a.operator_max is not None:
        if a.operator_min is None or a.operator_max is None or not -.01<=a.operator_min<0<a.operator_max<=.10:raise ValueError('Explicit bounded release-operator interval required; bridge entry gate is unchanged')
        source['operator_envelope_rad']=[a.operator_min,a.operator_max]
    source['input_sha256'].update({str(a.envelope.resolve()):sha(a.envelope),str(a.upper_nodes.resolve()):sha(a.upper_nodes),str(frozen.resolve()):sha(frozen)})
    source['physical_admission']=False
    (a.output/'envelope.json').write_text(json.dumps(source,indent=2)+'\n')


if __name__=='__main__':main()
