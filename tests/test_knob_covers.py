import json
import pytest
from doorbench.spec import generate_all
from doorbench.build import export_door
from doorbench.knob_cover_qa import run_knob_cover_qa


@pytest.fixture(scope='module')
def covers(tmp_path_factory):
    root=tmp_path_factory.mktemp('knob-covers');result=[]
    for spec in generate_all():
        if spec['operator']['model']!='knob_childproof':continue
        exported=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('json','mjcf'))
        meta=json.loads((root/'doors'/spec['id']/'model.json').read_text())['meta']
        result.append((spec,meta,exported['files']['mjcf']))
    assert len(result)==8
    return result


@pytest.mark.parametrize('tier',['full','simple','minimal'])
def test_every_cover_has_native_free_shell_and_accessible_inner_grip(covers,tier):
    for spec,meta,paths in covers:
        report=run_knob_cover_qa(paths[tier],meta)
        assert report['ok'],(spec['id'],tier,report)


def test_blocked_apertures_and_rigidly_coupled_shell_fail(covers,tmp_path):
    import mujoco as mj
    spec,meta,paths=covers[0];face=meta['knob_covers'][0]['faces'][0]
    for kind in ('blocked_aperture','coupled_cover'):
        source=mj.MjSpec.from_file(paths['full'])
        if kind=='blocked_aperture':
            source.body(face['cover_body']).add_geom(name='incorrect_solid_cover',type=mj.mjtGeom.mjGEOM_SPHERE,
                pos=face['knob_center_local'],size=[.041,0,0],contype=1,conaffinity=1)
        else:
            source.add_equality(name='incorrect_rigid_cover',type=mj.mjtEq.mjEQ_JOINT,
                name1=face['cover_joint'],name2=meta['knob_covers'][0]['operator_joint'],data=[0,1,0,0,0,0,0,0,0,0,0])
        # Compile first so the serialized absolute mesh paths are resolved.
        source.compile()
        import xml.etree.ElementTree as ET
        from pathlib import Path
        xml=ET.fromstring(source.to_xml());compiler=xml.find('compiler')
        for field in ('meshdir','texturedir'):
            compiler.set(field,str((Path(paths['full']).parent/compiler.get(field,'.')).resolve()))
        path=tmp_path/(kind+'.xml');path.write_text(ET.tostring(xml,encoding='unicode'))
        altered=json.loads(json.dumps(meta))
        if kind=='blocked_aperture':altered['knob_covers'][0]['faces'][0]['shell_geoms'].append('incorrect_solid_cover')
        report=run_knob_cover_qa(path,altered)
        assert not report['ok'],kind


def test_baseline_uses_opposed_inner_finger_contacts(covers):
    import numpy as np
    from pathlib import Path
    from doorbench.benchmark.runner import Job,run_episode
    from doorbench.reference.record import Recorder
    for spec,meta,paths in covers:
        rec=Recorder(10);path=Path(paths['full']).parent
        row={'id':spec['id'],'family':spec['family']}
        result=run_episode(Job(row,str(path),'open_and_traverse',0,'full','scripted_hand',randomize=False),observer=rec)
        assert not result.get('error'),result
        assert result['success'],result
        near=next(f for f in meta['knob_covers'][0]['faces'] if f['face']==-1)
        commanded=[f['site_forces'] for f in rec.frames if f['site_forces']]
        assert commanded and all(set(f)==set(near['grip_sites']) for f in commanded)
        assert all(np.linalg.norm(force)<=20.+1e-8 for frame in commanded for force in frame.values())
        for frame in commanded:
            np.testing.assert_allclose(np.sum(list(frame.values()),axis=0),0.,atol=1e-8)
