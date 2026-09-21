#!/usr/bin/env python3
"""One sequential, no-retry live request per frozen archived study case.

Only the request field is sent. Evaluator labels are opened after all requests.
No simulator, motor or fresh permission lease is created by this experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import statistics
import time
import urllib.error

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from doorbench.dexterous.jev_advisor import JevClient, MODEL
from scripts.dexterous.build_jev_contact_adjustment_study import assert_payload_boundary, sha, write


def finite_probability(value):
    return type(value) in (int,float) and math.isfinite(value) and 0<=value<=1


def typed_answers(response,request):
    """Keep only validated typed scalars; discard provider free text entirely."""
    if response.get('model')!=MODEL:raise ValueError('Pinned model required')
    result={}
    for key,question in request['questions'].items():
        answer=response['answers'][key]
        if answer['type']!=question['type']:raise ValueError('Typed answer required')
        if question['type']=='noul':
            probability=answer['noul']
            if not finite_probability(probability):raise ValueError('Finite probability required')
            result[key]=dict(type='noul',noul=float(probability))
        elif question['type']=='choice':
            choice=answer['choice'];probabilities=answer['probabilities'];confidence=answer['confidence']
            if (choice not in question['criteria'] or set(probabilities)!=set(question['criteria'])
                    or not finite_probability(confidence) or not all(finite_probability(v) for v in probabilities.values())
                    or not math.isclose(sum(probabilities.values()),1.,abs_tol=.02)
                    or probabilities[choice]<max(probabilities.values())):
                raise ValueError('Complete typed choice probabilities required')
            result[key]=dict(type='choice',choice=choice,confidence=float(confidence),
                probabilities={k:float(v) for k,v in probabilities.items()})
        else:raise ValueError('Study question type unsupported')
    return result


def run_cases(cases,transport,receipt_path):
    """At most ten calls; failures are recorded once and never retried."""
    if len(cases)!=10 or len({r['case_id'] for r in cases})!=10:
        raise ValueError('Exactly ten distinct study cases required')
    for item in cases:
        if set(item)!={'case_id','request'} or item['request']['model']!=MODEL:raise ValueError('Frozen request wrapper required')
        assert_payload_boundary(item['request'])
    receipts=[]
    with Path(receipt_path).open('x',encoding='utf-8') as stream:
        for attempt,item in enumerate(cases,1):
            request=item['request'];started=time.monotonic()
            receipt=dict(attempt=attempt,case_id=item['case_id'],
                request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                status='error',model=None,answers=None,error_type=None)
            try:
                response=transport(request)
                try:answers=typed_answers(response,request)
                except (KeyError,TypeError,ValueError,AttributeError):receipt['error_type']='invalid_typed_response'
                else:receipt.update(status='typed_reply',model=MODEL,answers=answers)
            except Exception as exc:
                receipt['error_type']='provider_http_'+str(exc.code) if isinstance(exc,urllib.error.HTTPError) else 'provider_error_'+type(exc).__name__
            receipt['latency_ms']=(time.monotonic()-started)*1000.
            stream.write(json.dumps(receipt,allow_nan=False)+'\n');stream.flush();receipts.append(receipt)
            print(json.dumps(dict(case_id=receipt['case_id'],status=receipt['status'],error_type=receipt['error_type'],latency_ms=receipt['latency_ms'])),flush=True)
    return receipts


def binary_metrics(pairs):
    return dict(count=len(pairs),accuracy_at_0_5=None if not pairs else sum((p>=.5)==y for p,y in pairs)/len(pairs),
        brier=None if not pairs else sum((p-float(y))**2 for p,y in pairs)/len(pairs))


def score(receipts,labels):
    expected={r['case_id']:r for r in labels};good=[r for r in receipts if r['status']=='typed_reply']
    contact=[];geometry=[];forecast=[];persistence=[];recent=[];per_case=[]
    for r in good:
        label=expected[r['case_id']];a=r['answers']
        if r['request_sha256']!=label['request_sha256']:raise ValueError('Scoring input differs from preregistered case')
        contact.extend((a['loaded_'+d]['noul'],truth) for d,truth in label['current_loaded'].items())
        geometry.append((a['all_thumb_pairs_opposed']['noul'],label['current_all_thumb_pairs_opposed']))
        forecast.append((a['next_100ms_hold']['noul'],label['next_100ms_all_intervals_opposed_grasp']))
        persistence.append((float(label['comparator_persistence_prediction']),label['next_100ms_all_intervals_opposed_grasp']))
        recent.append((float(label['comparator_recent_all_valid_prediction']),label['next_100ms_all_intervals_opposed_grasp']))
        choice=a['correction_candidate']['choice']
        per_case.append(dict(case_id=r['case_id'],forecast_probability=a['next_100ms_hold']['noul'],forecast_observed=label['next_100ms_all_intervals_opposed_grasp'],
            current_geometry_probability=a['all_thumb_pairs_opposed']['noul'],current_geometry_observed=label['current_all_thumb_pairs_opposed'],
            inspection_priority=a['inspection_priority']['choice'],correction_candidate=choice,
            correction_within_authored_joint_range=choice in label['admissible_joint_range_candidates'],
            correction_physical_success=None))
    duplicates=[]
    for i,a in enumerate(good):
        for b in good[i+1:]:
            if a['request_sha256']==b['request_sha256']:
                deltas={k:abs(v['noul']-b['answers'][k]['noul']) for k,v in a['answers'].items() if v['type']=='noul'}
                duplicates.append(dict(cases=[a['case_id'],b['case_id']],noul_probability_absolute_differences=deltas,
                    correction_choice_agrees=a['answers']['correction_candidate']['choice']==b['answers']['correction_candidate']['choice']))
    unique=[];seen=set()
    for r in good:
        if r['request_sha256'] not in seen:
            seen.add(r['request_sha256']);unique.append((r['answers']['next_100ms_hold']['noul'],expected[r['case_id']]['next_100ms_all_intervals_opposed_grasp']))
    return dict(schema='doorbench.jev-contact-adjustment-evaluation.v1',api_attempts=len(receipts),automatic_retries=0,
        typed_replies=len(good),errors=dict(Counter(r['error_type'] for r in receipts if r['status']=='error')),
        current_digit_contact_consistency=binary_metrics(contact),current_pair_geometry_consistency=binary_metrics(geometry),
        held_out_100ms_forecast=binary_metrics(forecast),held_out_forecast_unique_payloads=binary_metrics(unique),
        comparator_persistence_same_successful_cases=binary_metrics(persistence),comparator_recent_all_valid_same_successful_cases=binary_metrics(recent),
        median_latency_ms=float(statistics.median(r['latency_ms'] for r in receipts)) if receipts else None,
        maximum_latency_ms=max((r['latency_ms'] for r in receipts),default=None),
        correction_choices=dict(Counter(r['correction_candidate'] for r in per_case)),
        correction_joint_range_violations=sum(not r['correction_within_authored_joint_range'] for r in per_case),
        physical_success_score=None,physics_steps=0,motor_commands=0,
        limitation='Ten curated windows from two dependent trajectories include one byte-identical input pair. Current arithmetic consistency is not new perception. Forecasts are held out but this is not a generalization benchmark. Angle candidates were never physically executed and receive no physical-success score.',
        duplicate_input_consistency=duplicates,cases=per_case)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    manifest=json.loads((a.study/'manifest.json').read_text())
    if sha(a.study/'model-inputs.jsonl')!=manifest['model_inputs_sha256']:raise ValueError('Frozen model input changed')
    if any(sha(p)!=digest for p,digest in manifest['input_sha256'].items()):raise ValueError('Frozen study source changed')
    cases=[json.loads(line) for line in (a.study/'model-inputs.jsonl').read_text().splitlines()]
    a.output.mkdir(parents=True,exist_ok=False)
    client=JevClient(timeout_s=10.)
    os.environ.pop('TYPESAFE_API_KEY',None)
    try:receipts=run_cases(cases,client,a.output/'receipts.jsonl')
    finally:client._key=None;os.environ.pop('TYPESAFE_API_KEY',None)
    # Expected answers and actual future evidence stay out of the request path.
    label_path=a.study/'evaluator-only/labels.json'
    if sha(label_path)!=manifest['evaluator_labels_sha256']:raise ValueError('Frozen evaluator labels changed')
    summary=score(receipts,json.loads(label_path.read_text()))
    summary['input_sha256']={str(p):sha(p) for p in [Path(__file__),a.study/'manifest.json',a.study/'model-inputs.jsonl',label_path,a.output/'receipts.jsonl']}
    write(a.output/'results.json',summary)
    print(json.dumps(dict(attempts=summary['api_attempts'],typed_replies=summary['typed_replies'],errors=summary['errors'],output=str(a.output))),flush=True)


if __name__=='__main__':main()
