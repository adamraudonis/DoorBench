"""The direct bridge remains explicit and cannot silently replace a return."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('extra,valid',[
    ([],True),
    (['--standing-measured-rest'],False),
    (['--standing-withdrawal-path','release'],False),
    (['--standing-withdrawal-path','release','--standing-transfer-path','transfer'],False),
    (['--standing-measured-rest','--standing-withdrawal-path','release','--standing-transfer-path','transfer','--record-transitions'],True),
    (['--standing-measured-rest','--standing-withdrawal-path','release','--standing-transfer-path','transfer','--record-transitions','--standing-return-path','return'],False),
    (['--standing-measured-rest','--standing-withdrawal-path','release','--standing-transfer-path','transfer','--record-transitions','--portable-wrapper','--standing-transfer-attained-arm','--standing-transfer-no-fixed-pads','--operation-operator-lead-limit-rad','.1'],True),
])
def test_native_preflight_keeps_historical_default_and_requires_explicit_bridge(tmp_path,extra,valid):
    command=[sys.executable,str(ROOT/'scripts/dexterous/probe_acquisition_operation.py')]
    for name in ('robot','door','reference','motors','output'):
        command+=['--'+name,str(tmp_path/name)]
    result=subprocess.run(command+['--validate-arguments-only']+extra,
        cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),capture_output=True,text=True)
    assert (result.returncode==0)==valid,result.stderr
    if valid:assert json.loads(result.stdout)==dict(arguments_valid=True,physics_started=False)
    assert not (tmp_path/'output').exists()
