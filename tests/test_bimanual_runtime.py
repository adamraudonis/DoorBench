import json
import pytest
from doorbench.dexterous.bimanual_runtime import validate_runtime_rescreen,SCHEMA,CHECKS


def fixture(tmp_path):
    kwargs=dict(config={'runtime_rescreen_trajectory_sha256':'trace','compiled_robot_identity':{'sha256':'source'}},config_sha256='targets',compiled_identity={'sha256':'destination','mujoco_version':'3.12.0'},design_identity={'sha256':'design'},door_xml_sha256='door')
    report=dict(schema=SCHEMA,passed=True,checks={k:True for k in CHECKS},target_config_sha256='targets',trajectory_sha256='trace',source_design_sha256='design',source_compiled_robot_sha256='source',destination_compiled_robot_sha256='destination',destination_mujoco_version='3.12.0',door_xml_sha256='door',recorded_samples=1700,physics_steps=0)
    return tmp_path/'receipt.json',kwargs,report


def test_runtime_bound_receipt_accepted(tmp_path):
    p,k,r=fixture(tmp_path);p.write_text(json.dumps(r));assert validate_runtime_rescreen(p,**k)==r


@pytest.mark.parametrize('key',['target_config_sha256','trajectory_sha256','source_design_sha256','source_compiled_robot_sha256','destination_compiled_robot_sha256','destination_mujoco_version','door_xml_sha256'])
def test_changed_inputs_rejected(tmp_path,key):
    p,k,r=fixture(tmp_path);r[key]='changed';p.write_text(json.dumps(r))
    with pytest.raises(ValueError):validate_runtime_rescreen(p,**k)


@pytest.mark.parametrize('broken',['missing_gate','failed','wrong_scope'])
def test_incomplete_or_failed_evidence_rejected(tmp_path,broken):
    p,k,r=fixture(tmp_path)
    if broken=='missing_gate':r['checks'].pop(CHECKS[0])
    elif broken=='failed':r['passed']=False
    else:r['physics_steps']=1
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):validate_runtime_rescreen(p,**k)
