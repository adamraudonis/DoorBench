import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('_transition_archive_test',Path(__file__).resolve().parents[1]/'doorbench/dexterous/native_transition_archive.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


def rows():
    rng=np.random.default_rng(55);out=[]
    for i in range(3):
        row={key:rng.normal(size=4).tolist() for key in module.STATE_FIELDS}
        row.update(interval_start_s=i*.002,interval_end_s=(i+1)*.002,geometry_time_s=i*.002)
        row['contacts']=[]
        for _ in range(i):
            row['contacts'].append(dict(geom=[2,3],body=[0,1],position_world_m=rng.normal(size=3).tolist(),
                frame_world=rng.normal(size=(3,3)).tolist(),wrench_contact_frame=rng.normal(size=6).tolist(),distance_m=-.001))
        row.update(body_ids=[0,1],body_positions_world_m=rng.normal(size=(2,3)).tolist(),body_rotations_world=rng.normal(size=(2,3,3)).tolist())
        out.append(row)
    return out


def test_all_contact_and_state_values_roundtrip_bit_exactly(tmp_path):
    src=rows();writer=module.NativeTransitionArchive(tmp_path/'evidence',chunk_size=2)
    for row in src:writer.write(row)
    writer.close();writer.close()
    assert list(module.NativeTransitionArchive.read(tmp_path/'evidence'))==src
    receipt=json.loads((tmp_path/'evidence/manifest.json').read_text())
    assert receipt['rows']==3 and len(receipt['chunks'])==2 and receipt['complete']
    with pytest.raises(ValueError,match='closed'):writer.write(src[0])


def test_partial_chunks_remain_explicitly_incomplete(tmp_path):
    writer=module.NativeTransitionArchive(tmp_path/'evidence',chunk_size=2)
    src=rows()
    for row in src:writer.write(row)
    writer.close(complete=False)
    with pytest.raises(ValueError,match='Incomplete'):list(module.NativeTransitionArchive.read(tmp_path/'evidence'))
    assert list(module.NativeTransitionArchive.read(tmp_path/'evidence',allow_incomplete=True))==src


def test_corrupted_numeric_chunk_rejected(tmp_path):
    writer=module.NativeTransitionArchive(tmp_path/'evidence',chunk_size=1)
    writer.write(rows()[0]);writer.close()
    path=next((tmp_path/'evidence').glob('*.npz'))
    with path.open('ab') as stream:stream.write(b'corruption')
    with pytest.raises(ValueError,match='hash mismatch'):list(module.NativeTransitionArchive.read(tmp_path/'evidence'))


def warning_rows():
    from doorbench.dexterous.native_warning_audit import warning_interval
    import mujoco
    counters=np.zeros(int(mujoco.mjtWarning.mjNWARNING),dtype=np.int64)
    src=rows()
    for i,row in enumerate(src):
        after=counters.copy()
        if i==1:after[int(mujoco.mjtWarning.mjWARN_CONTACTFULL)]+=1
        row['mujoco_warning_interval']=warning_interval(counters,after)
        counters=after
    return src


def test_capacity_warning_is_preserved_losslessly_in_numeric_archive(tmp_path):
    src=warning_rows();writer=module.NativeTransitionArchive(tmp_path/'evidence',chunk_size=2)
    for row in src:writer.write(row)
    writer.close()
    restored=list(module.NativeTransitionArchive.read(tmp_path/'evidence'))
    assert restored==src
    assert [r['mujoco_warning_interval']['passed'] for r in restored]==[True,False,True]
    receipt=json.loads((tmp_path/'evidence/manifest.json').read_text())
    assert all(c['warning_counters_recorded'] for c in receipt['chunks'])


def test_missing_or_falsified_warning_evidence_cannot_be_packed():
    src=warning_rows();src[1]['mujoco_warning_interval']['passed']=True
    with pytest.raises(ValueError,match='does not match'):module.packed(src)
    del src[1]['mujoco_warning_interval']
    with pytest.raises(ValueError,match='missing'):module.packed(src)
    src=warning_rows();src[0]['mujoco_warning_interval']['before'][0]=.5
    with pytest.raises(ValueError,match='counters'):module.packed(src)


def test_legacy_evidence_remains_without_invented_warning_qualification():
    restored=list(module.unpacked(module.packed(rows())))
    assert all('mujoco_warning_interval' not in r for r in restored)
