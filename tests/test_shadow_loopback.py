"""Native unilateral mechanics; no hand-object contacts or added actuator strength."""
import mujoco
import numpy as np
import pytest

from doorbench.dexterous.shadow_loopback import add_loopbacks,assert_only_passive_loopbacks_changed


def pair_spec():
    return mujoco.MjSpec.from_string('''<mujoco><compiler angle="radian"/>
      <option gravity="0 0 0" timestep=".002"/>
      <worldbody><body name="distal"><joint name="rh_FFJ1" type="hinge" axis="1 0 0" range="0 1.5708"/>
        <geom size=".02" mass=".1" contype="0" conaffinity="0"/></body>
      <body name="middle" pos=".1 0 0"><joint name="rh_FFJ2" type="hinge" axis="1 0 0" range="0 1.5708"/>
        <geom size=".02" mass=".1" contype="0" conaffinity="0"/></body></worldbody>
      <tendon><fixed name="rh_FFJ0"><joint joint="rh_FFJ1" coef="1"/><joint joint="rh_FFJ2" coef="1"/></fixed></tendon>
      </mujoco>''')


def pair_model():
    spec=pair_spec();before=spec.compile();receipt=add_loopbacks(spec,before,hands=['rh'],digits=['FF']);after=spec.compile()
    assert_only_passive_loopbacks_changed(before,after,receipt)
    return after,receipt


def test_slack_inequality_stays_free_without_motor_or_spring_force():
    m,_=pair_model();d=mujoco.MjData(m);d.qpos[:]=[.2,.7];mujoco.mj_forward(m,d)
    assert m.nu==0 and m.neq==0
    assert not np.any(d.qfrc_constraint) and not np.any(d.qfrc_passive)
    for _ in range(200):mujoco.mj_step(m,d)
    np.testing.assert_allclose(d.qpos,[.2,.7],atol=1e-12)


def test_violating_difference_is_corrected_passively_not_as_an_equality():
    m,_=pair_model();d=mujoco.MjData(m);d.qpos[:]=[.7,.2];mujoco.mj_forward(m,d)
    assert d.qfrc_constraint[0]<0<d.qfrc_constraint[1]
    np.testing.assert_allclose(d.qfrc_constraint.sum(),0,atol=1e-12)
    for _ in range(300):mujoco.mj_step(m,d)
    assert d.qpos[0]-d.qpos[1]<.001
    assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
    # A different slack configuration remains valid; it is not welded at equality.
    d.qpos[:]=[.1,1.];d.qvel[:]=0.;mujoco.mj_forward(m,d)
    assert not np.any(d.qfrc_constraint)


def test_profile_adds_only_unactuated_difference_limit_and_rejects_duplicate():
    spec=pair_spec();before=spec.compile();receipt=add_loopbacks(spec,before,hands=['rh'],digits=['FF']);after=spec.compile()
    assert_only_passive_loopbacks_changed(before,after,receipt)
    tendon=after.tendon('rh_FF_loopback').id
    assert after.tendon_range[tendon,1]==0
    assert after.tendon_range[tendon,0]<-1.5708
    assert after.tendon_stiffness[tendon]==0 and after.tendon_damping[tendon]==0
    with pytest.raises(ValueError,match='already exists'):add_loopbacks(spec,after,hands=['rh'],digits=['FF'])


def test_contract_rejects_accidental_strength_or_inertia_edit():
    spec=pair_spec();before=spec.compile();receipt=add_loopbacks(spec,before,hands=['rh'],digits=['FF']);after=spec.compile()
    after.body_mass[1]*=1.1
    with pytest.raises(ValueError,match='body_mass'):assert_only_passive_loopbacks_changed(before,after,receipt)


def test_contract_rejects_motor_transmission_coefficient_change():
    spec=pair_spec();before=spec.compile();receipt=add_loopbacks(spec,before,hands=['rh'],digits=['FF']);after=spec.compile()
    after.wrap_prm[0]=2.
    with pytest.raises(ValueError,match='wrap_prm'):assert_only_passive_loopbacks_changed(before,after,receipt)


def test_time_constant_cannot_be_shorter_than_two_physics_ticks():
    spec=pair_spec();spec.option.timestep=.003
    with pytest.raises(ValueError,match='two physics timesteps'):
        add_loopbacks(spec,spec.compile(),hands=['rh'],digits=['FF'])


def test_chain_compliance_under_original_motor_scale_exposes_entry_transient():
    # This mechanism-only differential excitation uses the original 1 Nm motor
    # scale. It does not assert that a J0 sum motor produces opposing torques.
    from scripts.dexterous.probe_shadow_loopback import fixture_spec, measure
    model=fixture_spec(.004).compile()
    assert model.nu==0 and model.neq==0
    slack,_=measure(model,[.2,.6],0.)
    np.testing.assert_allclose(slack[:,1:3],np.broadcast_to([.2,.6],slack[:,1:3].shape),atol=1e-12)
    assert not np.any(slack[:,4:])
    _,active=measure(model,[.6,.6],1.)
    assert 0 < active['max_difference_rad'] < .02
    np.testing.assert_allclose(active['steady_constraint_Nm'],[-1.,1.],atol=1e-5)
    _,entry=measure(model,[.2,.6],1.)
    assert entry['max_difference_rad'] > .02  # Retain known stress failure.
    assert 0 < entry['final_difference_rad'] < .01
    _,soft=measure(fixture_spec(.02).compile(),[.6,.6],1.)
    assert soft['final_difference_rad'] > .1  # Do not silently restore soft v1 prototype.
