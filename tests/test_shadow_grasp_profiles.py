"""Declared contact surface expansion preserves opposition and physical gates."""
import copy

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doorbench.dexterous.grasp_verification import shadow_surface_qualified, shadow_lever_pad_grasp, audited_native_step
from doorbench.dexterous.isaac_pad_audit import shadow_physx_pad_grasp
from test_isaac_pad_audit import fixture


def mixed_fixture(segment='middle',point=(0.,-.008,.020)):
    patches,poses=fixture();old=patches[0]['body'];new=old.replace('distal',segment)
    poses[new]=poses.pop(old);patches[0]['body']=new
    rotation=Rotation.from_quat(poses[new][3:])
    poses[new][:3]=patches[0]['position']-rotation.apply(point)
    return patches,poses


def evaluate(patches,poses,profile='volar-phalange-v1'):
    return shadow_physx_pad_grasp(patches,poses,[0,0,0],[1,0,0],half_length=.053,radius=.007,profile=profile)


@pytest.mark.parametrize('segment', ['middle','proximal'])
def test_expansion_is_opt_in_and_distal_reference_remains_failed(segment):
    patches,poses=mixed_fixture(segment)
    assert not evaluate(patches,poses,'distal-pad-v1')['valid_pad_grasp']
    report=evaluate(patches,poses)
    assert report['valid_pad_grasp']
    assert report['grasp_profile']=='volar-phalange-v1'
    assert not report['distal_pad_grasp']['valid_pad_grasp']
    assert not report['contacts'][0]['distal_pad_qualified']


@pytest.mark.parametrize('defect', ['dorsal','wrong_side','cap','off_segment','proximal_joint','knuckle','lateral','rogue_patch','thumb_middle'])
def test_expanded_profile_rejects_wrong_surface_or_opposition(defect):
    patches,poses=mixed_fixture()
    path=patches[0]['body'];rotation=Rotation.from_quat(poses[path][3:])
    if defect=='dorsal':poses[path][:3]=patches[0]['position']-rotation.apply([0,.008,.02])
    elif defect=='wrong_side':
        patches[0]['position'][1]=.007;patches[0]['normal']=np.array([0.,1.,0.])
        poses[path]=np.r_[patches[0]['position']-[0.,-.008,.02],[0.,0.,0.,1.]]
    elif defect=='cap':
        shift=np.array([-.0229,0,0]);patches[0]['position']+=shift;poses[path][:3]+=shift
    elif defect=='off_segment':poses[path][:3]=patches[0]['position']-rotation.apply([0,-.008,.026])
    elif defect=='proximal_joint':
        new=path.replace('middle','proximal');poses[new]=poses.pop(path);patches[0]['body']=new
        poses[new][:3]=patches[0]['position']-rotation.apply([0,-.008,0.])
    elif defect=='knuckle':
        new=path.replace('middle','knuckle');poses[new]=poses.pop(path);patches[0]['body']=new
    elif defect=='lateral':patches[0]['normal']=np.array([1.,0.,0.])
    elif defect=='rogue_patch':
        bad=copy.deepcopy(patches[0]);bad['normal']*=-1;patches.append(bad)
    elif defect=='thumb_middle':
        old=patches[-1]['body'];new=old.replace('distal','middle');poses[new]=poses.pop(old);patches[-1]['body']=new
    assert not evaluate(patches,poses)['valid_pad_grasp']


def test_unknown_profile_rejected_even_with_no_contact():
    with pytest.raises(ValueError,match='profile'):
        evaluate([],{},'automatic')
    # Invalid selection must not advance the physical plant.
    with pytest.raises(ValueError,match='profile'):
        audited_native_step(None,'lever',handle_joint='handle',profile='automatic')


def test_native_middle_patch_has_same_profile_semantics():
    model=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 -9.81 0"/>
      <worldbody><geom name="lever" type="capsule" size=".007 .053"/>
      <body name="robot/rh_ffmiddle" pos="0 .022 -.02"><joint type="slide" axis="0 1 0"/>
      <geom type="sphere" pos="0 -.012 .02" size=".004" mass=".1"/>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    original=shadow_lever_pad_grasp(model,data,'lever')
    expanded=shadow_lever_pad_grasp(model,data,'lever',profile='volar-phalange-v1')
    assert not original['contacts'][0]['pad_qualified']
    assert expanded['contacts'][0]['pad_qualified']
    assert not expanded['contacts'][0]['distal_pad_qualified']
    assert not expanded['valid_pad_grasp']  # Four missing digits cannot be waived.


def test_distal_contract_predicate_is_unchanged_on_boundary_grid():
    for segment in ('distal','middle','proximal','knuckle'):
        for y in (-.002,-.001,.002):
            for z in (.001,.002,.025,.040,.041):
                for ny in (-1.,-.5,1.):
                    old=segment=='distal' and y<-.001 and .002<=z<=.040 and -ny>.5
                    assert shadow_surface_qualified('ff',segment,[0,y,z],[0,ny,0])==old
