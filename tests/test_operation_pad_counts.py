from doorbench.dexterous.operation_pad_counts import operation_pad_counts


def test_one_shot_iteration_preserves_boundary_and_independent_failures():
    rows=[dict(sim_time_s=t,valid_pad_grasp=valid,digit_forces_N={'rf':force},
               contacts=[{'pad_qualified':patch}],maximum_thumb_finger_dot=dot)
          for t,valid,force,patch,dot in [(0,False,0,False,0),(1,False,.1,True,-1),(2,False,1,False,0),(3,True,.2,True,-1)]]
    counts=operation_pad_counts(iter(rows),1)
    assert counts==dict(operation_invalid_grasp_samples=2,operation_digit_unload_samples=1,
                       operation_opposition_failure_samples=1,operation_invalid_pad_patch_samples=1)
    assert not any(operation_pad_counts(iter(rows),None).values())


def test_original_positive_time_filter_is_preserved_for_nonfinite_time():
    row=dict(sim_time_s=float('nan'),valid_pad_grasp=False,
             digit_forces_N={'rf':0.},contacts=[{'pad_qualified':False}])
    assert not any(operation_pad_counts(iter([row]),1.).values())
    row['sim_time_s']=1.
    assert not any(operation_pad_counts(iter([row]),float('nan')).values())
