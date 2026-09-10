#!/usr/bin/env python3
"""Independently reduce the recorded standing-transfer palm-force stream."""
import argparse,gzip,hashlib,json
from pathlib import Path
from doorbench.dexterous.json_record_stream import iter_json_object_array
from doorbench.dexterous.standing_transfer_evaluation import standing_transfer_checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new audit receipt')
    report_path=a.trial/'operation-report.json';stream_path=a.trial/'standing-transfer-steps.json.gz'
    r=json.loads(report_path.read_text());transfer=r['standing_transfer']
    with gzip.open(stream_path,'rt') as stream:
        checks=standing_transfer_checks(iter_json_object_array(stream),seconds=r['duration_s'],dt=r['physics_dt_s'],started_s=transfer['started_s'])
    matching=all(r['checks'].get(k)==v for k,v in checks.items())
    result=dict(passed=bool(r['passed'] and matching and all(checks.values())),checks=checks,producer_matches=matching,
        scope='Independent reduction of recorded actual panel force vectors; original physics and raw handle audits remain separately required',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [report_path,stream_path]})
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
