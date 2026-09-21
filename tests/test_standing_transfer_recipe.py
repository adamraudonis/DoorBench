import copy
import json
from pathlib import Path
import pytest
from scripts.isaac.standing_transfer_recipe import build_command

ROOT=Path(__file__).resolve().parents[1]
PROFILE=json.loads((ROOT/'configs/isaac/standing-transfer-h1-shadow-v1.json').read_text())
@pytest.fixture
def inputs(tmp_path):
    return dict(source=str(tmp_path/'snapshot'),ready=str(tmp_path/'cache/ready.json'),reference=str(tmp_path/'reference.json'),
                route=str(tmp_path/'plan/transfer.json'),output=str(tmp_path/'results/run'),work=str(tmp_path/'workspace'),
                deadline=100.,now=0.,python='python3')


def test_relocatable_paths_and_explicit_experiment(inputs):
    command=build_command(PROFILE,**inputs)
    assert command[:2]==['python3',str(Path(inputs['source'])/'scripts/isaac/run_standing_operation.py')]
    assert command[command.index('--camera-profile')+1]==str(Path(inputs['source'])/'configs/dexterous/h1-manipulation-cameras.json')
    assert command[command.index('--standing-transfer-route')+1]==inputs['route']
    assert command.count('--deadline-unix')==1
    assert '--actual-material-pads' in command


@pytest.mark.parametrize('change',[['--output','/other'],['--operation-leaf-target-rad','.12'],['--unknown']])
def test_recipe_cannot_override_lifecycle_or_duplicate_options(change,inputs):
    p=copy.deepcopy(PROFILE);p['arguments']+=change
    with pytest.raises(ValueError):build_command(p,**inputs)


def test_expired_deadline_and_escaping_camera_rejected(inputs):
    with pytest.raises(ValueError):build_command(PROFILE,**{**inputs,'now':101.})
    p=copy.deepcopy(PROFILE);p['camera_profile']='../camera.json'
    with pytest.raises(ValueError):build_command(p,**inputs)


def test_hybrid_recipe_changes_only_explicit_force_control(inputs):
    profile=json.loads((ROOT/'configs/isaac/standing-transfer-h1-shadow-hybrid-v1.json').read_text())
    original=build_command(PROFILE,**inputs)
    command=build_command(profile,**inputs)
    assert command.count('--standing-transfer-hybrid-support')==1
    command.remove('--standing-transfer-hybrid-support')
    assert command==original
    profile['arguments'].append('--standing-transfer-hybrid-support')
    with pytest.raises(ValueError):build_command(profile,**inputs)
