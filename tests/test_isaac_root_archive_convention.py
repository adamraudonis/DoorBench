"""Evaluate recording metadata across actual controller modes without Isaac imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('continuous,locomotion,stance,actor_origin', [
    (False, None, None, False),
    (True, None, None, True),
    (False, 'calibration.json', None, True),
    (False, None, 'landed-foot-v1', True),
])
def test_archive_root_convention_matches_controller_mode(continuous, locomotion, stance, actor_origin):
    source = Path(__file__).resolve().parents[1] / 'scripts/dexterous/isaac_opening.py'
    tree = ast.parse(source.read_text())
    expressions = [node.value for node in ast.walk(tree)
                   if isinstance(node, ast.keyword) and node.arg == 'root_state_convention']
    assert len(expressions) == 1
    value = eval(compile(ast.Expression(expressions[0]), str(source), 'eval'),
                 {'continuous': continuous, 'a': SimpleNamespace(
                     sensor_locomotion_calibration=locomotion, acquisition_stance_profile=stance)})
    assert ('world actor-origin linear/angular velocity' in value) == actor_origin
    assert ('COM' in value) != actor_origin
