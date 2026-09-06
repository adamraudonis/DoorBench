"""Independent physical transmission gate and missing pocket-wheel control."""
import json
from pathlib import Path
import mujoco
import numpy as np
import pytest
from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.rollup_hoist_qa import run_rollup_hoist_qa


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp('hoist-gate')
    source=next(s for s in generate_all()if s['id']=='db0419_rollup')
    result=export_door(source,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
    path=Path(result['files']['mjcf']['full'])
    return path,json.loads(path.with_name('model.json').read_text())['meta']


def test_material_wheel_transmission_passes_and_missing_wheel_cannot(fixture):
    path,meta=fixture
    for connected in (True,False):
        model=mujoco.MjModel.from_xml_path(str(path))
        if not connected:
            for gid in range(model.ngeom):
                if (model.geom(gid).name or '').startswith('hoist_hand_wheel'):
                    model.geom_contype[gid]=0;model.geom_conaffinity[gid]=0
        before=model.qpos0.copy();mass=model.body_mass.copy();friction=model.geom_friction.copy()
        result=run_rollup_hoist_qa(model,meta,phase_duration_s=2.)
        if connected:
            assert result['ok'],result
            assert result['pocket_contact_pairs'] and result['opening_end_bottom_z_m']>.07
            assert result['opening_end_bottom_z_m']-result['final_bottom_z_m']>.02
            assert run_rollup_hoist_qa(model,meta,phase_duration_s=2.)['cache_hit']
        else:
            assert not result['ok']
            assert 'missing_material_to_pocket_wheel_contact' in result['failures']
            assert 'native_opening_transmission_not_observed' in result['failures']
        assert result['peak_force_N']<=120
        assert not any(result['warnings'])
        assert np.array_equal(model.qpos0,before) and np.array_equal(model.body_mass,mass) and np.array_equal(model.geom_friction,friction)


def test_transmission_gate_rejects_too_short_or_invalid_trials(fixture):
    path,meta=fixture;model=mujoco.MjModel.from_xml_path(str(path))
    for duration in (.1,False,float('nan')):
        with pytest.raises(ValueError):run_rollup_hoist_qa(model,meta,phase_duration_s=duration)
