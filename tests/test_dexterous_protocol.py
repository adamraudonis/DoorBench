import pytest
from doorbench.dexterous.protocol import score

def test_missing_doors_and_failed_scenarios_stay_in_denominator():
    required={'one':['open','unlock'],'two':['open']}
    rows=[dict(door_id='one',scenario=s,seed=k,success=True,physically_valid=True)
          for s in ['open','unlock'] for k in range(9)]
    result=score(required,rows)
    assert result['target_doors']==2 and result['coverage']==.5
    assert result['expected_trials']==30 and result['recorded_trials']==18

def test_success_without_physical_validation_does_not_count():
    rows=[dict(door_id='one',scenario='open',seed=k,success=True) for k in range(10)]
    assert score({'one':['open']},rows)['reliably_solved']==0

def test_duplicate_or_out_of_scope_trials_rejected():
    row=dict(door_id='one',scenario='open',seed=0,success=True,physically_valid=True)
    with pytest.raises(ValueError):score({'one':['open']},[row,row])
    with pytest.raises(ValueError):score({'two':['open']},[row])
