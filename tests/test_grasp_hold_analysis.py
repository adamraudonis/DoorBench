import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('hold',Path(__file__).parents[1]/'scripts/dexterous/analyze_grasp_hold.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_opposition_failure_is_not_reported_as_a_digit_unload():
    rows=[]
    for i in range(4):
        rows.append(dict(sim_time_s=i*.002,valid_pad_grasp=i>=2,
            qualified_pad_forces_N={d:(.1 if i==0 and d=='mf' else 2.) for d in module.DIGITS},
            minimum_pairwise_finger_alignment=.2 if i==1 else .8,
            maximum_thumb_finger_dot=-.2 if i==1 else -.8,contacts=[]))
    report=module.summarize(rows)
    assert report['failed_grasp_samples']==2
    assert report['low_load_samples']==1
    assert report['opposed_geometry_failure_with_all_digits_loaded']==1
    assert report['per_digit_low_load']['mf']==[dict(time_s=0.,force_N=.1)]
    assert report['valid_windows']==[dict(start_s=.004,end_s=.006,span_s=.002)]
