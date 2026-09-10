import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def command(tmp_path):
    args=[sys.executable,str(ROOT/'scripts/dexterous/probe_acquisition_operation.py')]
    for name in ('robot','door','reference','motors','output'):
        args+=['--'+name,str(tmp_path/name)]
    return args+['--portable-wrapper','--hold-attained-grasp','--attained-hold-stage','operator',
                 '--operation-operator-follow-after-leaf-rad','.05','--validate-arguments-only']

def test_hold_follow_cli_preflight_needs_no_assets_and_starts_no_physics(tmp_path):
    result=subprocess.run(command(tmp_path),cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)=={'arguments_valid':True,'physics_started':False}
    assert not (tmp_path/'output').exists()

def test_preflight_still_rejects_unsupported_transfer_composition(tmp_path):
    args=command(tmp_path)+['--standing-transfer-path',str(tmp_path/'transfer')]
    result=subprocess.run(args,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},capture_output=True,text=True)
    assert result.returncode==2
    assert 'Operator follow requires standalone operation' in result.stderr
    assert not (tmp_path/'output').exists()
