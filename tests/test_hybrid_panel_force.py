import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('_hybrid_panel_test',Path(__file__).resolve().parents[1]/'doorbench/dexterous/bimanual_transfer.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
replace_normal_acceleration=module.replace_normal_acceleration


def test_unequal_inertia_does_not_turn_tangential_servo_into_normal_acceleration():
    mass=np.diag([1.,4.]);normal=np.array([1.,1.]);servo=np.array([2.,-2.])
    assert normal@servo==0.  # Euclidean torque projection would miss the problem.
    assert normal@np.linalg.solve(mass,servo)==1.5
    force=replace_normal_acceleration(servo,mass,normal,0.)
    assert normal@np.linalg.solve(mass,force)==pytest.approx(0.,abs=1e-12)


def test_actual_opposing_contact_force_cancels_only_the_requested_normal_acceleration():
    mass=np.array([[2.,.3],[.3,.5]]);normal=np.array([.4,-.2]);servo=np.array([1.,3.])
    command=replace_normal_acceleration(servo,mass,normal,5.)
    actual_contact_reaction=-normal*5.
    acceleration=np.linalg.solve(mass,command+actual_contact_reaction)
    assert normal@acceleration==pytest.approx(0.,abs=1e-12)
    assert np.linalg.norm(acceleration)>0.  # Tangential/posture action remains.


@pytest.mark.parametrize('mass,normal',[(np.diag([1.,-1.]),[1.,1.]),(np.eye(2),[0.,0.])])
def test_invalid_inertia_or_uncontrollable_normal_is_rejected(mass,normal):
    with pytest.raises(ValueError):replace_normal_acceleration([1.,2.],mass,normal,5.)


def test_extraction_preserves_the_measured_trial_arithmetic_bit_for_bit():
    rng=np.random.default_rng(3)
    for _ in range(20):
        a=rng.normal(size=(7,7));mass=a@a.T+np.eye(7)
        j=rng.normal(size=7);servo=rng.normal(size=7);gravity=rng.normal(size=7)
        inv=np.linalg.solve(mass,j);norm=float(j@inv);requested=5.
        original=gravity+servo-j*float(inv@servo)/norm+j*requested
        extracted=replace_normal_acceleration(servo,mass,j,requested,gravity=gravity)
        np.testing.assert_array_equal(extracted,original)
