import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from doorbench.dexterous.bimanual_transfer import load_screen_targets


CONFIG = Path(__file__).parents[1] / 'configs/dexterous/bimanual-left-contact-v2.json'


def inputs(tmp_path):
    robot = tmp_path / 'robot.xml'
    robot.write_text('fixture model identity')
    config = json.loads(CONFIG.read_text())
    config['robot_sha256'] = hashlib.sha256(robot.read_bytes()).hexdigest()
    path = tmp_path / 'targets.json'
    path.write_text(json.dumps(config))
    return robot, path, config


def test_portable_targets_retain_leaf_coordinates_and_do_not_require_scene(tmp_path):
    robot, path, config = inputs(tmp_path)
    names, targets = load_screen_targets(path, robot, tmp_path / 'absent-door')
    assert names == config['joint_names']
    assert len(targets) == 61
    assert np.array_equal(targets[-1]['position'], config['targets'][-1]['position'])
    assert np.isclose(np.linalg.norm(targets[-1]['normal']), 1.)


def test_robot_identity_cannot_be_silently_changed(tmp_path):
    robot, path, _ = inputs(tmp_path)
    robot.write_text('different model')
    with pytest.raises(ValueError, match='different robot'):
        load_screen_targets(path, robot, None)


@pytest.mark.parametrize('field,value', [('position',[0,0]),('nominal',[0]*7),('normal',[0,0,0]),('position',[0,float('nan'),0])])
def test_corrupted_pose_targets_are_rejected(tmp_path, field, value):
    robot, path, config = inputs(tmp_path)
    config['targets'][20][field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_screen_targets(path, robot, None)
