import os
from pathlib import Path

import pytest
from doorbench.dexterous.standing_withdrawal import bound_input_digest


def test_relative_and_absolute_names_bind_the_identical_file(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);target=tmp_path/'recorded route'/'report.json'
    relative=os.path.relpath(target,tmp_path)
    assert bound_input_digest({relative:'abc'},target)=='abc'
    assert bound_input_digest({str(target):'abc'},relative)=='abc'


def test_another_path_with_the_same_filename_does_not_match(tmp_path):
    assert bound_input_digest({str(tmp_path/'source-a'/'report.json'):'abc'},tmp_path/'source-b'/'report.json') is None


def test_conflicting_aliases_are_rejected(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='Conflicting'):
        bound_input_digest({'report.json':'a',str(tmp_path/'report.json'):'b'},tmp_path/'report.json')
