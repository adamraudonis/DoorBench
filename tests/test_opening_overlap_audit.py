import numpy as np

from scripts.dexterous.audit_opening_overlap import spans


def test_half_second_hold_needs251_endpoints_not250():
    times=np.arange(251)*.002
    held=spans(np.ones(251,bool),times)[0]
    assert held['samples']==251 and held['span_s']==.5
    interrupted=np.ones(251,bool);interrupted[250]=False
    assert spans(interrupted,times)[0]['span_s']==.498


def test_one_bad_interval_breaks_hold_without_filling_the_gap():
    times=30.+np.arange(251)*.002;mask=np.ones(251,bool);mask[210]=False
    runs=spans(mask,times)
    assert len(runs)==2 and sum(row['samples'] for row in runs)==250
    assert max(row['span_s'] for row in runs)<.5
    assert all(not(row['start_s']<=times[210]<=row['end_s']) for row in runs)


def test_no_positive_samples_cannot_create_overlap():
    assert spans([False,False],[0,.002])==[]
