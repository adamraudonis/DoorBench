import numpy as np
import pytest
from doorbench.dexterous.left_approach_clearance import clearance_envelope


def test_clearance_preserves_endpoints_and_never_inverts_side():
    values=np.array([clearance_envelope(u,.01) for u in np.linspace(0,1,1001)])
    assert abs(values[0])<1e-15 and abs(values[-1])<1e-15
    assert np.all(values>=0) and values.max()<=.01


@pytest.mark.parametrize('u,d',[(-.1,.01),(1.1,.01),(.5,0),(.5,.026),(.5,float('nan'))])
def test_clearance_rejects_undeclared_geometry(u,d):
    with pytest.raises(ValueError):clearance_envelope(u,d)
