import copy
import numpy as np
import pytest
from doorbench.dexterous.isaac_joint_passive import passive_profile, configure_backend


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
    view = View(); declaration = passive_profile(motors(), ['a', 'b'], 'backend-dry-v2')
    receipt = configure_backend(view, declaration)
    assert receipt['profile'] == 'backend-dry-v2'
    class Broken(View):
        def get_dof_friction_properties(self): return self.value + torch.tensor(.001)
    with pytest.raises(ValueError): configure_backend(Broken(), declaration)
