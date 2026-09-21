"""Prospective contract selection and independent raw anatomical classification."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import mujoco
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('contact_profile_auditor',ROOT/'scripts/dexterous/audit_sensor_acquisition_contacts.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def test_profile_cli_requires_declaration_and_full_record(tmp_path):
    command=[sys.executable,str(ROOT/'scripts/dexterous/probe_acquisition_operation.py')]
    for name in ('robot','door','reference','motors','output'):
        command+=['--'+name,str(tmp_path/name)]
    command+=['--grasp-profile','volar-phalange-v1','--validate-arguments-only']
    assert subprocess.run(command,capture_output=True).returncode==2
    command+=['--grasp-profile-definition',str(tmp_path/'declaration.json')]
    assert subprocess.run(command,capture_output=True).returncode==2
    result=subprocess.run(command+['--record-transitions'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['physics_started'] is False
    assert not (tmp_path/'output').exists()

@pytest.mark.parametrize('body,selected', [('rh_ffmiddle',True),('rh_thmiddle',False),('rh_ffknuckle',False)])
def test_raw_auditor_keeps_anatomical_exclusions(body,selected):
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="lever"><geom name="lever_geom" type="cylinder" size=".01 .1"/></body>
      <body name="robot/{body}"><geom name="finger" type="sphere" size=".01"/></body>
    </worldbody></mujoco>''')
    lever=model.geom('lever_geom').id;finger=model.geom('finger').id
    lb=int(model.geom_bodyid[lever]);fb=int(model.geom_bodyid[finger])
    raw=dict(body_ids=[lb,fb],body_positions_world_m=[[0,0,0],[0,.02,0]],
        body_rotations_world=[np.eye(3).tolist(),np.eye(3).tolist()],contacts=[dict(
        geom=[finger,lever],position_world_m=[0,.01,.01],
        frame_world=[[0,-1,0],[1,0,0],[0,0,1]],wrench_contact_frame=[2,0,0,0,0,0])])
    _,strict=audit.raw_pad_evidence(model,raw,lever)
    _,volar=audit.raw_pad_evidence(model,raw,lever,profile='volar-phalange-v1')
    assert strict[0]['pad_qualified'] is False
    assert volar[0]['pad_qualified'] is selected

def test_raw_auditor_rejects_unknown_profile_even_without_contacts():
    with pytest.raises(ValueError,match='Unknown Shadow grasp profile'):
        audit.raw_pad_evidence(None,{'contacts':[]},0,profile='loose')
