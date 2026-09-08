import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doorbench.dexterous.isaac_pad_audit import PhysXShadowPadAudit, shadow_physx_pad_grasp


def fixture():
    patches, poses = [], {}
    for digit, axial in zip(("ff", "mf", "rf", "lf", "th"), (-.03, -.01, .01, .03, 0.)):
        thumb = digit == "th"
        point = np.array([axial, .007 if thumb else -.007, 0.])
        rotation = Rotation.identity() if thumb else Rotation.from_euler("z", 180, degrees=True)
        path = "/World/H1/pelvis/rh_" + digit + "distal"
        poses[path] = np.r_[point - rotation.apply([0., -.008, .020]), rotation.as_quat()]
        patches.append(dict(body=path, position=point, normal=np.array([0., 1. if thumb else -1., 0.]), normal_force_N=1.))
    return patches, poses


def evaluate(patches, poses, center=(0., 0., 0.), axis=(1., 0., 0.)):
    return shadow_physx_pad_grasp(patches, poses, center, axis, half_length=.053, radius=.007)


def test_all_five_volar_pads_and_global_rigid_transform_invariance():
    patches, poses = fixture()
    expected = evaluate(patches, poses)
    assert expected["valid_pad_grasp"]
    rotation = Rotation.from_euler("xyz", [.7, -.4, 1.1])
    translation = np.array([10., -2., 7.])
    changed = copy.deepcopy(patches)
    transformed = {}
    for contact in changed:
        contact["position"] = rotation.apply(contact["position"]) + translation
        contact["normal"] = rotation.apply(contact["normal"])
    for path, pose in poses.items():
        transformed[path] = np.r_[rotation.apply(pose[:3]) + translation,
            (rotation * Rotation.from_quat(pose[3:])).as_quat()]
    result = evaluate(changed, transformed, translation, rotation.apply([1., 0., 0.]))
    assert result["valid_pad_grasp"]
    for before, after in zip(expected["contacts"], result["contacts"]):
        np.testing.assert_allclose(before["body_position_m"], after["body_position_m"], atol=1e-12)
        np.testing.assert_allclose(before["hand_outward_normal_body"], after["hand_outward_normal_body"], atol=1e-12)


@pytest.mark.parametrize("defect", ["dorsal", "middle_link", "endcap", "wrong_normal", "extra_bad_patch"])
def test_centroid_success_cannot_hide_anatomical_failure(defect):
    patches, poses = fixture()
    path = patches[0]["body"]
    if defect == "dorsal":
        rotation = Rotation.from_quat(poses[path][3:])
        poses[path][:3] = patches[0]["position"] - rotation.apply([0., .008, .02])
    elif defect == "middle_link":
        new_path = path.replace("distal", "middle")
        poses[new_path] = poses.pop(path)
        patches[0]["body"] = new_path
    elif defect == "endcap":
        displacement = np.array([.0229, 0., 0.])
        patches[3]["position"] += displacement
        poses[patches[3]["body"]][:3] += displacement
    elif defect == "wrong_normal":
        patches[0]["normal"] *= -1
    else:
        bad = copy.deepcopy(patches[0])
        bad["normal"] *= -1
        patches.append(bad)
    assert not evaluate(patches, poses)["valid_pad_grasp"]


def test_missing_transform_fails_closed():
    patches, poses = fixture()
    del poses[patches[0]["body"]]
    with pytest.raises(KeyError):
        evaluate(patches, poses)


class Bodies:
    def __init__(self, poses):
        self.prim_paths = list(poses)
        self.transforms = np.array(list(poses.values()))

    def get_transforms(self):
        return self.transforms


class Contacts:
    filter_count = 1

    def __init__(self, patches):
        self.sensor_paths = [patch["body"] for patch in patches]
        self.force = np.r_[np.ones(5), np.zeros(3)][:, None]
        self.points = np.vstack([[patch["position"] for patch in patches], np.zeros((3, 3))])
        self.normals = np.vstack([[patch["normal"] for patch in patches], np.zeros((3, 3))])
        self.count = np.ones((5, 1), dtype=int)
        self.start = np.arange(5)[:, None]
        self.matrix = self.normals[:5, None].copy()

    def get_contact_data(self, dt):
        return self.force, self.points, self.normals, np.zeros((8, 1)), self.count, self.start

    def get_contact_force_matrix(self, dt):
        # Mimic a subsequent backend getter reusing count/start buffers.
        self.count[:] = 0
        self.start[:] = 0
        return self.matrix


def test_native_buffers_are_copied_before_subsequent_getters():
    patches, poses = fixture()
    evaluator = PhysXShadowPadAudit(Bodies(poses), Contacts(patches))
    result = evaluator.read(physics_dt=.002, time_s=1., center=[0., 0., 0.],
        axis=[1., 0., 0.], half_length=.053, radius=.007)
    assert result["valid_pad_grasp"]
    assert result["active_contact_count"] == 5
    assert result["normal_pair_force_consistency_error_N"] == 0.


def test_reversed_force_normal_sign_and_mismatched_row_order_are_rejected():
    patches, poses = fixture()
    contacts = Contacts(patches)
    contacts.normals *= -1
    evaluator = PhysXShadowPadAudit(Bodies(poses), contacts)
    with pytest.raises(ValueError, match="directions/loads"):
        evaluator.read(physics_dt=.002, time_s=1., center=[0., 0., 0.],
            axis=[1., 0., 0.], half_length=.053, radius=.007)
    bodies = Bodies(poses)
    bodies.prim_paths.reverse()
    with pytest.raises(ValueError, match="rows"):
        PhysXShadowPadAudit(bodies, Contacts(patches))


def test_opt_in_raw_evidence_reuses_single_read_and_preserves_default_result():
    import json
    patches,poses=fixture();contacts=Contacts(patches);bodies=Bodies(poses)
    calls={'contact':0,'matrix':0,'body':0}
    def counted(obj,name,key):
        old=getattr(obj,name)
        def call(*args):calls[key]+=1;return old(*args)
        setattr(obj,name,call)
    counted(contacts,'get_contact_data','contact');counted(contacts,'get_contact_force_matrix','matrix');counted(bodies,'get_transforms','body')
    actual=PhysXShadowPadAudit(bodies,contacts).read(physics_dt=.002,time_s=1.,center=[0,0,0],axis=[1,0,0],half_length=.053,radius=.007,include_evidence=True)
    raw=actual.pop('raw_evidence');default=PhysXShadowPadAudit(Bodies(poses),Contacts(patches)).read(physics_dt=.002,time_s=1.,center=[0,0,0],axis=[1,0,0],half_length=.053,radius=.007)
    assert actual==default and calls=={'contact':1,'matrix':1,'body':1}
    assert raw['clock']=='physx-interval-end' and raw['geometry_time_s']==1. and raw['interval_start_s']==.998
    assert len(raw['contacts'])==5 and raw['active_contact_count']==5
    restored=json.loads(json.dumps(raw,allow_nan=False))
    result=shadow_physx_pad_grasp(restored['contacts'],restored['body_transforms_xyzw'],**restored['lever'])
    assert result['valid_pad_grasp'] and result['qualified_pad_forces_N']==default['qualified_pad_forces_N']
    bodies.transforms.fill(99);contacts.points.fill(99);contacts.force.fill(99)
    assert raw==restored
