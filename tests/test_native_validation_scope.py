"""A partial recording must be explicitly scoped; missing coverage never passes silently."""
import json
import pytest
from doorbench.reference.native_validation import validate_native
from doorbench.reference.record import NATIVE_SCHEMA, digest


def fixture(tmp_path):
    assets=tmp_path/'assets';root=tmp_path/'motion';assets.mkdir();root.mkdir()
    (assets/'manifest.json').write_text(json.dumps({'doors':[{'id':'a','family':'swing_single'},{'id':'b','family':'swing_single'},{'id':'pet','family':'pet_door'}]}))
    (root/'index.json').write_text(json.dumps({'schema':NATIVE_SCHEMA,'clips':[],'manifest_sha256':digest(assets/'manifest.json')}))
    return root,assets


def test_missing_recordings_fail_default_coverage(tmp_path):
    root,assets=fixture(tmp_path)
    with pytest.raises(AssertionError,match='coverage'):validate_native(root,assets)


@pytest.mark.parametrize('ids',[[],['a','a'],['unknown'],['pet']])
def test_subset_must_be_explicit_nonempty_unique_eligible(tmp_path,ids):
    root,assets=fixture(tmp_path)
    with pytest.raises(AssertionError,match='subset'):validate_native(root,assets,door_ids=ids)


def test_subset_still_requires_every_requested_recording(tmp_path):
    root,assets=fixture(tmp_path)
    with pytest.raises(AssertionError,match='coverage'):validate_native(root,assets,door_ids=['a'])
