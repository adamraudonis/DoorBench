"""An actual frame rebate covers the top/sill gap of a closed vault."""
import json
import mujoco
import numpy as np
import pytest
from doorbench.build import export_door
from doorbench.spec import generate_all


@pytest.fixture(scope='module')
def frames(tmp_path_factory):
    root=tmp_path_factory.mktemp('vault-frames');rows=[]
    for spec in generate_all():
        if spec['family'] not in ('vault','blast'):continue
        summary=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('json','mjcf'))
        ir=json.loads((root/'doors'/spec['id']/'model.json').read_text())
        rows.append((spec,ir,summary['files']['mjcf']))
    assert len(rows)==14
    return rows


@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_closed_frame_gaps_have_connected_overlapping_stock(frames,tier):
    for spec,ir,files in frames:
        m=mujoco.MjModel.from_xml_path(files[tier]);d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
        for row in ir['meta']['vault_frame_rebates']['rows']:
            g=m.geom(row['geom']).id;a=m.geom(row['anchor_geom']).id
            assert m.geom_contype[g] and m.geom_conaffinity[g]
            # These plates share welded/bolted backing stock. Test their real
            # primitives even though same-static-body contacts are filtered.
            distance=mujoco.mj_geomDistance(m,d,g,a,.1,None)
            assert distance<=1e-7,(spec['id'],row['geom'],distance)
            z=.05+spec['leaf']['height']+.010 if 'head' in row['geom'] else .045
            hit=np.array([-1],np.int32)
            side=float(ir['meta']['v']);origin=np.array([0.,side*(spec['leaf']['thickness']/2+.2),z])
            distance=mujoco.mj_ray(m,d,origin,np.array([0.,-side,0.]),None,True,-1,hit)
            assert distance>0 and hit[0]==g,(spec['id'],row['geom'],m.geom(hit[0]).name if hit[0]>=0 else None)
        # Geometric aperture fit is not a structural or weather-tightness rating.
        assert 'security' in ir['meta']['vault_frame_rebates']['scope']
