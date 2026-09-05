import json
from pathlib import Path
import pytest

from doorbench.appearance.pipeline import digest, find_blender, select_doors


def test_family_selection_and_unknown_ids():
    manifest = {'doors':[{'id':'a','family':'swing'},{'id':'b','family':'slide'},{'id':'c','family':'swing'}]}
    assert [d['id'] for d in select_doors(manifest,'families')] == ['a','b']
    assert [d['id'] for d in select_doors(manifest,'c,a')] == ['a','c']
    with pytest.raises(ValueError,match='Unknown door'):
        select_doors(manifest,'missing')


def test_job_digest_detects_state_and_recipe_changes():
    a = {'state':{'qpos':[0]},'recipe':{'floor':'oak','wall':'plaster'}}
    b = {'recipe':{'wall':'plaster','floor':'oak'},'state':{'qpos':[0]}}
    assert digest(a) == digest(b)
    b['state']['qpos'] = [.2]
    assert digest(a) != digest(b)
    with pytest.raises(ValueError):
        digest({'state':float('nan')})


def test_explicit_blender_executable(tmp_path):
    executable = tmp_path/'Blender with spaces'
    executable.write_text('binary placeholder')
    assert find_blender(executable) == str(executable)
