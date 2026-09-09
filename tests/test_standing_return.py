import copy
from types import SimpleNamespace
import numpy as np
import pytest
from doorbench.dexterous.standing_return import validate_return_targets


def data():
    model=SimpleNamespace(nq=8,joint=lambda name:SimpleNamespace(qposadr=np.array([0 if name=='robot/free_base' else 7])))
    q=[0.,0.,1.,1.,0.,0.,0.,.2]
    return model,dict(rows=[dict(qpos=q.copy(),root_qpos_address=0,joints={'torso':.2})],source_state={'qpos':q.copy()})


def test_consumed_coordinates_match_independent_scene():
    model,plan=data();validate_return_targets(plan,model)
    changed=copy.deepcopy(plan);changed['rows'][0]['joints']['torso']=.3
    with pytest.raises(ValueError,match='Consumed'):validate_return_targets(changed,model)


def test_rejects_changed_attained_start_and_normalized_root():
    model,plan=data();plan['source_state']['qpos'][-1]=.1
    with pytest.raises(ValueError,match='attained'):validate_return_targets(plan,model)
    model,plan=data();plan['rows'][0]['qpos'][3]=2.
    with pytest.raises(ValueError,match='normalized'):validate_return_targets(plan,model)
