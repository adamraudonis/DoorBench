"""Independent negative controls for occupied contact buffers in smoke evidence."""
import copy

import pytest

from scripts.dexterous.audit_traversal_smoke import unpack_contacts


def fixture():
    layout=dict(capacity=8,sensor_paths=['/World/H1/left_ankle_link'],filter_paths=[['/World/floor','/World/Door/Articulation/leaf']])
    row=dict(normal=dict(pairs=[[0,0,3,2]],slots=[3,4],force_N=[[100],[120]],
        point_world=[[0,0,0],[0,1,0]],normal_world=[[0,0,1],[0,0,1]],distance_m=[[-.001],[.001]]),
        friction=dict(pairs=[[0,0,0,1]],slots=[0],force_N=[[2,1,0]],point_world=[[0,.5,0]]))
    return row,layout


def test_normal_and_friction_slot_numbers_are_independent():
    row,layout=fixture();result=unpack_contacts(row,layout)
    assert len(result['normal'][0])==2 and len(result['friction'][0])==1
    assert result['normal'][1]['force_N'].sum()==220
    assert result['friction'][1]['force_N'].tolist()==[[2,1,0]]


@pytest.mark.parametrize('mutation',[
    lambda r,l:r['normal']['slots'].__setitem__(0,4),
    lambda r,l:r['friction']['slots'].__setitem__(0,False),
    lambda r,l:r['normal']['pairs'].__setitem__(0,[0,0,7,2]),
    lambda r,l:r['normal']['pairs'].__setitem__(0,[0,0,3,-2]),
    lambda r,l:r['normal']['pairs'].__setitem__(0,[1,0,3,2]),
    lambda r,l:r['normal']['pairs'].__setitem__(0,[0,2,3,2]),
    lambda r,l:r['normal']['pairs'].__setitem__(0,[0,0,3.0,2]),
    lambda r,l:r['normal']['force_N'].__setitem__(0,[-1]),
    lambda r,l:r['normal']['normal_world'].__setitem__(0,[0,0,2]),
    lambda r,l:r['friction']['force_N'].__setitem__(0,[float('nan'),0,0]),
    lambda r,l:r['normal']['distance_m'].__setitem__(0,[float('inf')]),
    lambda r,l:l.update(capacity=2),
])
def test_corrupt_contact_evidence_is_rejected(mutation):
    row,layout=fixture();mutation(row,layout)
    with pytest.raises(ValueError):unpack_contacts(row,layout)


def test_duplicate_pair_cannot_double_count_normal_force():
    row,layout=fixture()
    row['normal']['pairs']=[[0,0,3,1],[0,0,4,1]]
    with pytest.raises(ValueError):unpack_contacts(row,layout)
