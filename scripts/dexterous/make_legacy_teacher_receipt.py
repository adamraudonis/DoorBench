"""Create a separate acquisition-only receipt without changing a legacy archive."""
import argparse
import json
from pathlib import Path
from doorbench.dexterous.legacy_teacher_provenance import audit_legacy_acquisition_prefix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing provenance receipts')
    result=audit_legacy_acquisition_prefix(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('scope','sensor_samples','sample_end_time_s','qualified_hold_physics_samples')}))


if __name__=='__main__':main()
