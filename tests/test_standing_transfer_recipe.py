import copy
import json
from pathlib import Path
import pytest
from scripts.isaac.standing_transfer_recipe import build_command

ROOT=Path(__file__).resolve().parents[1]
PROFILE=json.loads((ROOT/'configs/isaac/standing-transfer-h1-shadow-v1.json').read_text())
INPUTS=dict(source='/snapshot',ready='/cache/ready.json',reference='/reference.json',
            route='/plan/transfer.json',output='/results/run',work='/workspace',deadline=100.,now=0.,python='python3')


def test_relocatable_paths_and_explicit_experiment():
    command=build_command(PROFILE,**INPUTS)
    assert command[:2]==['python3','/snapshot/scripts/isaac/run_standing_operation.py']
    assert command[command.index('--camera-profile')+1]=='/snapshot/configs/dexterous/h1-manipulation-cameras.json'
    assert command[command.index('--standing-transfer-route')+1]=='/plan/transfer.json'
    assert command.count('--deadline-unix')==1
    assert '--actual-material-pads' in command


@pytest.mark.parametrize('change',[['--output','/other'],['--operation-leaf-target-rad','.12'],['--unknown']])
def test_recipe_cannot_override_lifecycle_or_duplicate_options(change):
    p=copy.deepcopy(PROFILE);p['arguments']+=change
    with pytest.raises(ValueError):build_command(p,**INPUTS)


def test_expired_deadline_and_escaping_camera_rejected():
    with pytest.raises(ValueError):build_command(PROFILE,**{**INPUTS,'now':101.})
    p=copy.deepcopy(PROFILE);p['camera_profile']='../camera.json'
    with pytest.raises(ValueError):build_command(p,**INPUTS)
