import hashlib,importlib.util,json
from pathlib import Path
import pytest


def test_same_named_robot_and_door_with_different_xml_cannot_render(tmp_path):
    spec=importlib.util.spec_from_file_location('render',Path(__file__).parents[1]/'scripts/dexterous/render_acquisition_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    robot=tmp_path/'robot.xml';robot.write_text('original robot')
    door=tmp_path/'door.xml';door.write_text('original door')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(inputs=dict(robot=dict(sha256=digest(robot)),door={'door.xml':digest(door)}))))
    assert len(module.verify_recorded_xml(tmp_path,robot,tmp_path))==2
    robot.write_text('same version name, different physics')
    with pytest.raises(ValueError,match='exact recorded'):module.verify_recorded_xml(tmp_path,robot,tmp_path)
    robot.write_text('original robot');door.write_text('different inertial frame')
    with pytest.raises(ValueError,match='exact recorded'):module.verify_recorded_xml(tmp_path,robot,tmp_path)


def test_render_times_stream_records_and_reject_truncation(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('render',Path(__file__).parents[1]/'scripts/dexterous/render_acquisition_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'trace.json'
    rows=[{'sim_time_s':i*.02,'diagnostics':{'unused':'x'*70000}} for i in range(4)]
    path.write_text(json.dumps(rows))
    def no_whole_file(*args,**kwargs):
        raise AssertionError('Renderer must not read the whole trace at once')
    monkeypatch.setattr(Path,'read_text',no_whole_file)
    assert module.recorded_trace_times(path).tolist()==[r['sim_time_s'] for r in rows]
    path.write_text('[{"sim_time_s":0.0},')
    with pytest.raises(ValueError):module.recorded_trace_times(path)
