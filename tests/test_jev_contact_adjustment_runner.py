"""The live-study shell is bounded and strips provider text without real calls."""
import copy
import json
import urllib.error

import pytest

from scripts.dexterous.build_jev_contact_adjustment_study import questions
from scripts.dexterous.run_jev_contact_adjustment_study import MODEL, run_cases, typed_answers


def cases():
    request=dict(model=MODEL,state=dict(history=[dict(relative_time_s=-.2),dict(relative_time_s=0.)]),questions=questions())
    return [dict(case_id='case-'+str(i),request=copy.deepcopy(request)) for i in range(10)]


def response(request):
    answers={}
    for key,q in request['questions'].items():
        if q['type']=='noul':answers[key]=dict(type='noul',noul=.5,untrusted_text='DO_NOT_LOG_PROVIDER_TEXT')
        else:
            selected=next(iter(q['criteria']));answers[key]=dict(type='choice',choice=selected,confidence=1.,
                probabilities={k:float(k==selected) for k in q['criteria']})
    return dict(model=MODEL,answers=answers,provider_text='DO_NOT_LOG_PROVIDER_TEXT')


def test_ten_sequential_attempts_no_retry_and_no_provider_exception_text(tmp_path):
    calls=[];pack=cases()
    def client(request):
        calls.append(request)
        assert not {'expected','label','source_run','case_id'}&request.keys()
        if len(calls)==1:raise urllib.error.HTTPError('secret_url',503,'DO_NOT_LOG_SECRET_EXCEPTION',{},None)
        return response(request)
    file=tmp_path/'receipts.jsonl';receipts=run_cases(pack,client,file)
    assert len(calls)==10
    assert receipts[0]['error_type']=='provider_http_503'
    assert sum(r['status']=='typed_reply' for r in receipts)==9
    text=file.read_text()
    assert 'DO_NOT_LOG' not in text and 'secret_url' not in text
    assert len(text.splitlines())==10


@pytest.mark.parametrize('size',[0,9,11])
def test_wrong_case_count_makes_no_calls(tmp_path,size):
    called=[]
    with pytest.raises(ValueError):run_cases((cases()+cases())[:size],lambda p:called.append(p),tmp_path/'receipts.jsonl')
    assert called==[]


def test_future_leak_in_late_case_is_rejected_before_first_call(tmp_path):
    pack=cases();pack[-1]['request']['state']['future']=[1,2,3];called=[]
    with pytest.raises(ValueError):run_cases(pack,lambda p:called.append(p),tmp_path/'receipts.jsonl')
    assert called==[]


@pytest.mark.parametrize('change',['nan','boolean','wrong_model','out_of_range','incomplete_choices'])
def test_invalid_typed_reply_is_rejected(change):
    request=cases()[0]['request'];reply=response(request)
    if change=='nan':reply['answers']['loaded_ff']['noul']=float('nan')
    if change=='boolean':reply['answers']['loaded_ff']['noul']=True
    if change=='wrong_model':reply['model']='another-model'
    if change=='out_of_range':reply['answers']['loaded_ff']['noul']=2.
    if change=='incomplete_choices':reply['answers']['correction_candidate']['probabilities'].pop('hold_and_observe')
    with pytest.raises(ValueError):typed_answers(reply,request)
