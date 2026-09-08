import copy
import numpy as np
import pytest
from doorbench.dexterous.isaac_joint_passive import passive_profile, configure_backend, PassivePropertyInvariant


def motors():
    return {'passive': {'a': dict(damping=.05, friction=.01, armature=.0002,
                                stiffness=0., springref=0.),
                        'b': dict(damping=2., friction=0., armature=.1,
                                  stiffness=0., springref=0.)}}


def test_original_coefficients_are_assigned_once_in_named_order():
    original = motors(); snapshot = copy.deepcopy(original)
    legacy = passive_profile(original, ['b', 'a'], 'legacy-tanh-v1')
    backend = passive_profile(original, ['b', 'a'], 'backend-dry-v2')
    np.testing.assert_array_equal(legacy['explicit_damping'], [2., .05])
    np.testing.assert_array_equal(legacy['explicit_friction'], [0., .01])
    assert not legacy['backend_friction_properties'].any()
    assert not backend['explicit_damping'].any() and not backend['explicit_friction'].any()
    np.testing.assert_array_equal(backend['backend_friction_properties'], [[0., 0., 2.], [.01, .01, .05]])
    np.testing.assert_array_equal(backend['native_armature'], [.1, .0002])
    assert original == snapshot


@pytest.mark.parametrize('names,profile', [(['a'], 'backend-dry-v2'),
    (['a', 'a'], 'backend-dry-v2'), (['a', 'b'], 'unknown')])
def test_bad_coverage_or_profile_fails(names, profile):
    with pytest.raises(ValueError): passive_profile(motors(), names, profile)


@pytest.mark.parametrize('field,value', [('stiffness', 1.), ('friction', -.01), ('damping', float('nan'))])
def test_unsupported_terms_fail(field, value):
    source = motors(); source['passive']['a'][field] = value
    with pytest.raises(ValueError): passive_profile(source, ['a', 'b'], 'backend-dry-v2')


def test_backend_receipt_rejects_stale_or_changed_readback():
    import torch
    class View:
        def set_dof_friction_properties(self, target, indices):
            assert indices.tolist() == [0]
            self.value = target.clone()
        def get_dof_friction_properties(self): return self.value
        def get_dof_armatures(self): return torch.tensor([[.0002, .1]], dtype=torch.float32)
    view = View(); declaration = passive_profile(motors(), ['a', 'b'], 'backend-dry-v2')
    receipt = configure_backend(view, declaration)
    assert receipt['profile'] == 'backend-dry-v2'
    class Broken(View):
        def get_dof_friction_properties(self): return self.value + torch.tensor(.001)
    with pytest.raises(ValueError): configure_backend(Broken(), declaration)
    class WrongArmature(View):
        def get_dof_armatures(self): return super().get_dof_armatures() + .0001
    with pytest.raises(ValueError, match='armature'): configure_backend(WrongArmature(), declaration)


def configured_view():
    import torch
    declaration = passive_profile(motors(), ['a', 'b'], 'backend-dry-v2')
    class View:
        def __init__(self):
            self.properties=torch.tensor(declaration['backend_friction_properties'][None], dtype=torch.float32)
            self.armature=torch.tensor(declaration['native_armature'][None], dtype=torch.float32)
            self.queries=0
        def get_dof_friction_properties(self): self.queries+=1; return self.properties
        def get_dof_armatures(self): self.queries+=1; return self.armature
    return declaration, View()


def test_guard_is_read_only_tracks_every_step_and_receipt_is_a_copy():
    import json
    declaration, view=configured_view();guard=PassivePropertyInvariant(declaration)
    assert view.queries==0 and not guard.receipt()['passed']
    declaration['backend_friction_properties'][:]=99  # frozen target owns its values
    for i in range(5):guard.check(view,time_s=(i+1)*.002)
    receipt=guard.receipt();assert receipt['passed'] and receipt['checked_intervals']==5 and view.queries==10
    assert receipt['maximum_absolute_property_difference']==[0.,0.,0.]
    receipt['expected_float32_properties'][0][0]=99
    assert guard.receipt()['expected_float32_properties'][0][0]!=99
    json.dumps(guard.receipt(),allow_nan=False)


@pytest.mark.parametrize('mutation', ['friction','viscous','armature','nan','missing','query_error'])
def test_guard_retains_exact_failed_step_and_never_repairs_or_retries(mutation):
    import json,torch
    declaration,view=configured_view();guard=PassivePropertyInvariant(declaration)
    guard.check(view,time_s=.002)
    if mutation=='friction':view.properties[0,0,0]+=1e-6
    elif mutation=='viscous':view.properties[0,0,2]=0
    elif mutation=='armature':view.armature[0,0]=0
    elif mutation=='nan':view.properties[0,0,0]=float('nan')
    elif mutation=='missing':view.properties=view.properties[:,:,:2]
    else:
        def fail():raise RuntimeError('Backend getter failed')
        view.get_dof_friction_properties=fail
    with pytest.raises((ValueError,RuntimeError)):guard.check(view,time_s=.004)
    receipt=guard.receipt();assert not receipt['passed'] and receipt['first_failure']['interval_index']==2
    assert receipt['first_failure']['observed_interval_end_s']==.004 and receipt['valid_intervals']==1
    previous=receipt.copy();queries=view.queries
    with pytest.raises(ValueError,match='already failed'):guard.check(view,time_s=.006)
    assert view.queries==queries and guard.receipt()==previous
    json.dumps(receipt,allow_nan=False)


@pytest.mark.parametrize('time_s', [0., .004, True, float('nan')])
def test_guard_rejects_missing_clock_without_reading_plant(time_s):
    declaration,view=configured_view();guard=PassivePropertyInvariant(declaration)
    with pytest.raises(ValueError):guard.check(view,time_s=time_s)
    assert view.queries==0 and guard.receipt()['first_failure']['interval_index']==1


def test_guard_cannot_be_enabled_for_legacy_or_duplicate_explicit_terms():
    with pytest.raises(ValueError):PassivePropertyInvariant(passive_profile(motors(), ['a','b'], 'legacy-tanh-v1'))
    declaration,_=configured_view();declaration['explicit_damping'][0]=.05
    with pytest.raises(ValueError,match='duplicated'):PassivePropertyInvariant(declaration)


def test_si_terminal_speed_and_legacy_oscillation_prediction():
    # Independent analytic checks, using the measured fixture's original inertia.
    inertia=.0000027+.017*.0125**2+.0002
    assert (.02-.01)/.05==pytest.approx(.2)
    amplitude=.01*.002/(2*inertia-.05*.002)
    assert amplitude==pytest.approx(.0643681860240576)


@pytest.mark.parametrize('profile', ['legacy-tanh-v1','backend-dry-v2'])
def test_original_61_motor_recovery_uses_only_submitted_explicit_terms(profile):
    from test_isaac_traversal_runner import HELPERS,transmission
    matrix,inverse=transmission();names=['joint'+str(i) for i in range(69)]
    original={'passive':{n:dict(damping=.01*(i+1),friction=.01,armature=.0002,
                                stiffness=0.,springref=0.) for i,n in enumerate(names)}}
    declaration=passive_profile(original,names,profile)
    velocity=np.linspace(-.3,.3,69);requested=np.linspace(-.5,.5,61)
    damping=declaration['explicit_damping'];friction=declaration['explicit_friction']
    submitted=(matrix.T@requested-damping*velocity-friction*np.tanh(velocity/.001)).astype(np.float32)
    actual,residual=HELPERS['actual_motor_delivery'](submitted,velocity,matrix,inverse,damping,friction)
    np.testing.assert_allclose(actual,requested,rtol=0,atol=1e-6);assert residual<1e-5
    if profile=='backend-dry-v2':
        assert np.array_equal(submitted,(matrix.T@requested).astype(np.float32))
        # Backend solver reactions are not submitted motor commands. Incorrectly
        # restoring those passive terms contaminates the8 differential coordinates.
        legacy=passive_profile(original,names,'legacy-tanh-v1')
        with pytest.raises(ValueError,match='transmission'):
            HELPERS['actual_motor_delivery'](submitted,velocity,matrix,inverse,
                legacy['explicit_damping'],legacy['explicit_friction'])


def test_runner_default_keeps_legacy_profile_and_opt_in_is_explicit(monkeypatch):
    from test_isaac_traversal_runner import run_parser
    assert run_parser(monkeypatch,[]).joint_passive_profile=='legacy-tanh-v1'
    assert run_parser(monkeypatch,['--joint-passive-profile','backend-dry-v2']).joint_passive_profile=='backend-dry-v2'
