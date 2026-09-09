import json

import numpy as np
import pytest

from doorbench.dexterous.native_correction_demonstrations import NativeCorrectionDemonstration, REQUIRED_CHECKS


def test_missing_or_substituted_audit_check_rejected(tmp_path):
    checks=dict.fromkeys(REQUIRED_CHECKS,True)
    checks.pop('exact_counterfactual_labels');checks['unrelated_check']=True
    (tmp_path/'independent-correction-audit.json').write_text(json.dumps(dict(
        schema='doorbench.native-approach-correction-audit.v1',passed=True,checks=checks,physics_steps=0)))
    with pytest.raises(ValueError,match='Complete independent'):
        NativeCorrectionDemonstration(tmp_path)


def test_counterfactual_target_does_not_replace_actual_previous_command(monkeypatch):
    import doorbench.dexterous.native_correction_demonstrations as module
    demo=object.__new__(NativeCorrectionDemonstration)
    demo.times=np.array([0.,.002,.004]);demo.dimensions=None
    demo.numeric={'previous_action':np.array([[0.,0.],[.1,.2],[.3,.4]])}
    demo.targets=np.array([[.8,.9],[-.8,-.9]])
    demo.packet=lambda i:{'previous_action':demo.numeric['previous_action'][i].copy()}
    monkeypatch.setattr(module,'prepare_actor_packet',lambda packet,*_:packet)
    inputs,targets=demo.sequence(0,2)
    np.testing.assert_array_equal(inputs['previous_action'],[[0.,0.],[.1,.2]])
    np.testing.assert_array_equal(targets,demo.targets)
    targets[:]=0
    assert demo.targets.any()
    with pytest.raises(ValueError,match='boundary'):demo.sequence(1,2)
