"""Execute the producer's acceptance boundary with actual observer history."""
import ast
from types import SimpleNamespace

import numpy as np
import pytest

from test_isaac_transfer_prefix_wiring import MAIN, execute
from test_isaac_live_transfer_handoff import fixture


ACCEPT=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If)
    and 'live_handoff_observer is not None' in ast.unparse(n.test)
    and any(isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute)
        and v.func.attr=='observe_completed_interval' for v in ast.walk(n)))
LOOP=next(n for n in ast.walk(MAIN) if isinstance(n,ast.For) and ACCEPT in n.body)
GUARD=LOOP.body[LOOP.body.index(ACCEPT)-1]


@pytest.mark.parametrize('failed',[False,True])
def test_only_valid_completed_delivery_populates_actual_handoff(failed):
    observer,calls,command,info=fixture()
    returned,_=observer.force(0.)
    scope=dict(live_handoff_observer=observer,standing_controller=observer,delivery_failure='bad delivery' if failed else None,
        step=0,dt=.002,forces=returned,pad_steps=[{'valid_pad_grasp':True}],
        surface={'palm_normal_load_N':3.},dnames=['leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide'],
        door=SimpleNamespace(data=SimpleNamespace(joint_pos=np.array([[.091,0.,0.]]))))
    if failed:
        with pytest.raises(RuntimeError,match='bad delivery'):execute([GUARD,ACCEPT],scope)
        assert observer.accepted_intervals==0 and observer.pending is not None
    else:
        execute([GUARD,ACCEPT],scope)
        assert observer.accepted_intervals==1 and observer.pending is None
        np.testing.assert_array_equal(observer.motor_capture.capture(.002),command)
    assert len(calls)==1


def test_missing_opt_in_performs_no_observation():
    execute([GUARD,ACCEPT],dict(delivery_failure=None,live_handoff_observer=None))
