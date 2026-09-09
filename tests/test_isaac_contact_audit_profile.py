"""An old failed run cannot acquire a new grasp contract during audit."""
import json
import pytest
from scripts.dexterous.audit_isaac_acquisition_contacts import audit


def test_rejects_retrospective_profile_upgrade_before_loading_results(tmp_path):
    (tmp_path / 'configuration.json').write_text(json.dumps({'args': {'grasp_profile': 'distal-pad-v1'}}))
    with pytest.raises(ValueError, match='no retrospective upgrade'):
        audit(tmp_path, profile='volar-phalange-v1')


def test_unknown_contract_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='Unknown prospective'):
        audit(tmp_path, profile='any-contact-is-fine')
