import json
import pytest
from doorbench.dexterous.release_motion_admission import digest,validate_motion_screen


def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir();trajectory=source/'trajectory.npz';trajectory.write_bytes(b'bound physical state')
    screen=tmp_path/'screen.json';screen.write_text('{}')
    audit=tmp_path/'audit.json'
    d=dict(schema='doorbench.release-leaf-motion-screen.v1',passed=True,physics_steps=0,sampled_configurations=3003,failed_configurations=0,leaf_envelope_rad=.012,minimum_clearance_m=.0001,input_sha256={str(p):digest(p) for p in [screen,trajectory]})
    audit.write_text(json.dumps(d));cfg=dict(leaf_motion_audit_path=str(audit),leaf_motion_audit_sha256=digest(audit))
    return source,screen,audit,d,cfg


def test_binds_source_and_route(tmp_path):
    source,screen,audit,d,cfg=fixture(tmp_path)
    validate_motion_screen(cfg,screen,source)
    screen.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='another route'):validate_motion_screen(cfg,screen,source)


def test_rejects_failed_screen_even_with_matching_digest(tmp_path):
    source,screen,audit,d,cfg=fixture(tmp_path)
    d.update(passed=False,failed_configurations=1,minimum_clearance_m=-.001)
    audit.write_text(json.dumps(d));cfg['leaf_motion_audit_sha256']=digest(audit)
    with pytest.raises(ValueError,match='Passing explicit'):validate_motion_screen(cfg,screen,source)


def test_detects_modified_audit(tmp_path):
    source,screen,audit,d,cfg=fixture(tmp_path)
    audit.write_text('{}')
    with pytest.raises(ValueError,match='bytes changed'):validate_motion_screen(cfg,screen,source)
