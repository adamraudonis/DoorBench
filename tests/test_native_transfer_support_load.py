"""Explicit support targets may change the controller reference, not its caps."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous import standing_transfer

ROOT=Path(__file__).resolve().parents[1]


def route_fixture(tmp_path):
    proof=tmp_path/'proof.json';proof.write_text('{"passed":true}')
    config=dict(schema='doorbench.standing-transfer.v1',geometric_screen_passed=True,
        robot_xml_sha256='robot',joint_names=['joint'],left_joint_names=['joint'],
        root_path=[[0.,0.,1.,1.,0.,0.,0.]]*101,joint_path=[[0.]]*101,
        dense_audit_path=str(proof),dense_audit_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
        left_targets=[dict(position=[0,0,0],normal=[0,1,0],nominal=[0])]*101)
    path=tmp_path/'route.json';path.write_text(json.dumps(config))
    operation=SimpleNamespace(grasp_offset=np.zeros(3),acquisition=SimpleNamespace(names=['joint']))
    return operation,dict(source_xml_sha256='robot'),path


@pytest.mark.parametrize('explicit,expected',[(None,4.),(6.,6.)])
def test_target_plumbed_without_changing_existing_force_and_offset_caps(tmp_path,monkeypatch,explicit,expected):
    # Geometry admission is separately tested; observe the exact instantiated
    # contact controller configuration without building or stepping a plant.
    monkeypatch.setattr(standing_transfer,'validate_route_geometry',lambda config:None)
    monkeypatch.setattr(standing_transfer,'LeftPalmContact',lambda *args,**kwargs:SimpleNamespace(**kwargs))
    options={} if explicit is None else dict(support_load_target=explicit)
    teacher=standing_transfer.StandingTransferTeacher(*route_fixture(tmp_path),**options)
    assert teacher.left.support_load_target==expected
    assert teacher.left.contact_force==8.
    assert teacher.left.maximum_normal_offset==.008
    assert teacher.left.reach_seconds==8.


@pytest.mark.parametrize('target',[2.,0.,8.001,float('nan'),float('inf')])
def test_target_rejected_before_asset_or_controller_access(target):
    with pytest.raises(ValueError,match='support target'):
        standing_transfer.StandingTransferTeacher(None,None,None,support_load_target=target)


def test_opt_in_cli_requires_route_and_keeps_default_asset_free(tmp_path):
    command=[sys.executable,str(ROOT/'scripts/dexterous/probe_acquisition_operation.py')]
    for name in ('robot','door','reference','motors','output'):
        command+=['--'+name,str(tmp_path/name)]
    command+=['--validate-arguments-only']
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    for extra,success in [([],True),(['--standing-transfer-support-load','6'],False),
        (['--standing-transfer-path','missing','--standing-transfer-support-load','6'],True),
        (['--standing-transfer-path','missing','--standing-transfer-support-load','8.1'],False),
        (['--standing-transfer-path','missing','--standing-transfer-support-load','nan'],False)]:
        result=subprocess.run(command+extra,cwd=ROOT,env=env,capture_output=True,text=True)
        assert (result.returncode==0)==success,result.stderr
        if success:assert json.loads(result.stdout)==dict(arguments_valid=True,physics_started=False)
    assert not (tmp_path/'output').exists()
