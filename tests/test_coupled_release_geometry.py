import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

import doorbench.dexterous.coupled_release_geometry as geometry
from doorbench.dexterous.coupled_release_geometry import CoupledReleaseGeometry,sha
from doorbench.dexterous.coupled_release_reference import CoupledReleaseReference,validate_admission


def query_map():
    g=CoupledReleaseGeometry.__new__(CoupledReleaseGeometry)
    g.elapsed=np.linspace(0.,16.,5);g.angles=np.linspace(.08,.4,5)
    g.plan=dict(duration_s=16.,admitted_leaf_upper_nodes=[[0.,.12],[7.2,.12],[10.,.4],[16.,.4]])
    values=g.elapsed[:,None]**2+3.*g.angles[None,:]
    g.splines=[RectBivariateSpline(g.elapsed,g.angles,values)]
    return g


def test_analytic_tensor_interpolation_uses_actual_leaf_angle():
    g=query_map()
    assert g.coordinates(4.,.10)[0]==pytest.approx(16.3)
    assert g.coordinates(4.,.11)[0]==pytest.approx(16.33)
    assert g.coordinates(12.,.4)[0]==pytest.approx(145.2)


@pytest.mark.parametrize('time,angle',[(4.,.121),(12.,.401),(-.1,.10),(16.1,.10),(0.,.079),(float('nan'),.1)])
def test_no_progress_or_actual_angle_extrapolation(time,angle):
    with pytest.raises(ValueError,match='outside'):query_map().coordinates(time,angle)


def test_same_explicit_domain_is_interpolated_at_boundary():
    g=query_map();upper=.12+(.4-.12)*.5
    assert g.upper_angle(8.6)==pytest.approx(upper)
    assert g.coordinates(8.6,upper)[0]==pytest.approx(8.6**2+3*upper)
    with pytest.raises(ValueError):g.coordinates(8.6,upper+.00001)


@pytest.mark.parametrize('nodes',[[[1.,.12],[16.,.4]],[[0.,.12],[5.,.4],[4.,.4],[16.,.4]],[[0.,.12],[16.,.5]],[[0.,float('nan')],[16.,.4]]])
def test_malformed_or_expanded_domain_is_rejected(nodes):
    g=query_map();g.plan['admitted_leaf_upper_nodes']=nodes
    with pytest.raises(ValueError):g.coordinates(4.,.1)


def receipt(tmp_path,**changes):
    envelope=tmp_path/'envelope.json';envelope.write_text('{}')
    p=Path(geometry.__file__).resolve()
    report=dict(schema='doorbench.coupled-release-envelope-audit.v1',passed=True,coarse_diagnostic=False,samples=51339,exact_initial_state=True,physics_steps=0,
        limits=dict(left_position_error_m=.001,left_rotation_error_rad=.01,right_position_error_m=.001,right_rotation_error_rad=.01,foot_position_error_m=.001,foot_rotation_error_rad=.01,torso_tilt_deg=4.,root_translation_m=.03,com_xy_displacement_m=.015,palm_panel_gap_change_m=.001),
        input_sha256={str(envelope.resolve()):sha(envelope),str(p):sha(p)})
    report.update(changes);audit=tmp_path/'audit.json';audit.write_text(json.dumps(report))
    return dict(coupled_envelope_path=str(envelope),coupled_envelope_sha256=sha(envelope),coupled_audit_path=str(audit),coupled_audit_sha256=sha(audit))


def test_binds_passing_dense_audit_and_consumed_evaluator(tmp_path):
    config=receipt(tmp_path)
    assert validate_admission(config)[1]['passed']
    Path(config['coupled_envelope_path']).write_text('{"changed":true}')
    with pytest.raises(ValueError,match='bytes changed'):validate_admission(config)


@pytest.mark.parametrize('changes',[dict(passed=False),dict(coarse_diagnostic=True),dict(samples=9999),dict(exact_initial_state=False),dict(physics_steps=1),dict(limits={}),dict(input_sha256={})])
def test_failed_coarse_partial_or_unbound_receipts_cannot_admit(tmp_path,changes):
    with pytest.raises(ValueError):validate_admission(receipt(tmp_path,**changes))


def test_failed_reference_is_terminal_even_if_caller_catches_exception():
    reference=CoupledReleaseReference.__new__(CoupledReleaseReference)
    reference.failure_snapshot=None;reference.accepted=7
    calls=[]
    def fail(*args):
        calls.append(args)
        raise ValueError('outside admitted geometry')
    reference._update=fail
    with pytest.raises(ValueError,match='outside'):
        reference.update(50.014,.014,{'leaf':.5},None,None)
    assert reference.failure_snapshot['accepted_samples']==7
    assert reference.failure_snapshot['angles']=={'leaf':.5}
    with pytest.raises(ValueError,match='terminal'):
        reference.update(50.016,.016,{'leaf':.1},None,None)
    assert len(calls)==1
