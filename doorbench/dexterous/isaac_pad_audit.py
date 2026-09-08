"""Privileged Shadow pad anatomy evaluation for native PhysX contact buffers.

This module is deliberately separate from sensor_contract/isaac_sensors actor
packets. Body identities, world poses and exact lever geometry are evaluator
inputs. They must never be concatenated into actor observations or used to
route a deployed sensor policy.
"""
import re

import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_verification import grasp_profile, profile_pad_opposition, shadow_surface_qualified


def _copy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.array(value, copy=True)


def shadow_physx_pad_grasp(contacts, body_transforms_xyzw, center, axis, *,
                         half_length, radius, side="rh", axial_margin=.001,
                         surface_tolerance=.004, profile="distal-pad-v1"):
    """Check every loaded handle-body patch in the actual distal link frame.

    ``normal`` is PhysX's normal force direction ON the sensor hand body. Its
    negation is the outward hand surface normal. ``body_transforms_xyzw`` maps
    the exact contact row paths to copied RigidBodyView transforms (xyz,xyzw).

    The contact view filters the complete handle rigid body, not an individual
    child collider. Extra hub/far-side patches are retained as unqualified;
    they cannot disappear behind a centroid or a geometry-selection filter.
    This is consequently stricter than native lever-collider-only evidence.
    """
    center = np.asarray(center, dtype=float)
    grasp_profile(profile)
    axis = np.asarray(axis, dtype=float)
    if side not in ("rh", "lh") or center.shape != (3,) or axis.shape != (3,):
        raise ValueError("Expected a Shadow hand and finite straight-lever geometry")
    values = np.r_[center, axis, half_length, radius, axial_margin, surface_tolerance]
    if not np.isfinite(values).all() or np.linalg.norm(axis) < 1e-8 or min(half_length, radius, surface_tolerance) <= 0 or axial_margin < 0:
        raise ValueError("Invalid straight-lever geometry")
    axis = axis / np.linalg.norm(axis)
    patches = []
    ignored_non_digit_force = 0.
    for contact in contacts:
        path = contact["body"]
        match = re.fullmatch(side + r"_(ff|mf|rf|lf|th)(.+)", path.rsplit("/", 1)[-1])
        force = float(contact["normal_force_N"])
        if not np.isfinite(force) or force < 0:
            raise ValueError("Invalid measured normal load")
        if force == 0:
            continue
        if match is None:
            ignored_non_digit_force += force
            continue
        pose = np.asarray(body_transforms_xyzw[path], dtype=float)
        point = np.asarray(contact["position"], dtype=float)
        normal = np.asarray(contact["normal"], dtype=float)
        if pose.shape != (7,) or point.shape != (3,) or normal.shape != (3,) or not np.isfinite(np.r_[pose, point, normal]).all():
            raise ValueError("Missing or malformed synchronized patch/body transform")
        if abs(np.linalg.norm(pose[3:]) - 1) > 1e-4 or abs(np.linalg.norm(normal) - 1) > 1e-4:
            raise ValueError("Expected a unit body quaternion and measured normal")
        rotation = Rotation.from_quat(pose[3:]).as_matrix()
        local_point = rotation.T @ (point - pose[:3])
        outward = rotation.T @ -normal
        relative = point - center
        axial = float(relative @ axis)
        radial = relative - axial * axis
        length = float(np.linalg.norm(radial))
        alignment = float((-normal) @ (-radial / length)) if length > 1e-8 else 0.
        clearance = float(half_length - abs(axial))
        on_side = bool(clearance >= axial_margin and abs(length - radius) <= surface_tolerance)
        qualified = bool(shadow_surface_qualified(match.group(1),match.group(2),local_point,outward,
            profile=profile) and on_side and alignment > .8)
        patches.append(dict(digit=match.group(1), body=path, position=point.tolist(),
            normal_force_N=force, body_position_m=local_point.tolist(),
            hand_outward_normal_body=outward.tolist(), pad_qualified=qualified,
            axial_clearance_m=clearance, inward_radial_normal_alignment=alignment,
            on_lever_cylindrical_side=on_side))
        if profile=="volar-phalange-v1":
            patches[-1]['distal_pad_qualified']=bool(shadow_surface_qualified(
                match.group(1),match.group(2),local_point,outward) and on_side and alignment>.8)
    result = profile_pad_opposition(patches, center, axis, profile=profile)
    return dict(**result, contacts=patches, non_digit_handle_force_N=ignored_non_digit_force,
        hand=side, grasp_profile=profile,
        anatomy_contract="shadow-distal-volar-minus-y-v1" if profile=="distal-pad-v1" else "shadow-volar-phalange-minus-y-v1",
        contract_scope="Privileged PhysX handle-body patch audit; non-lever digit contacts count as misplaced",
        minimum_axial_clearance_m=axial_margin)


class PhysXShadowPadAudit:
    """Read already-created hand views without changing any simulator state.

    Views must use the same body row ordering. ``handle_filter_index`` must
    identify the complete handle body; the caller owns that evaluator-only
    declaration. Call once after every robot.update()/physics step when using
    an every-step success gate. A 50 Hz trace cannot certify a 500 Hz hold.
    """
    def __init__(self, hand_bodies, hand_contacts, *, handle_filter_index=0, side="rh", profile="distal-pad-v1"):
        self.bodies = hand_bodies
        self.contacts = hand_contacts
        self.paths = tuple(hand_bodies.prim_paths)
        if tuple(hand_contacts.sensor_paths) != self.paths:
            raise ValueError("Contact and transform rows do not match")
        if not 0 <= handle_filter_index < hand_contacts.filter_count:
            raise ValueError("Invalid handle-body filter index")
        self.filter_index = handle_filter_index
        self.side = side
        self.profile = grasp_profile(profile)

    def read(self, *, physics_dt, time_s, center, axis, half_length, radius, include_evidence=False):
        if type(include_evidence) is not bool:
            raise ValueError('Raw evidence capture must be explicitly boolean')
        if not np.isfinite([physics_dt, time_s]).all() or physics_dt <= 0 or time_s < 0:
            raise ValueError("Invalid evidence clock")
        # PhysX getters can reuse count/start arrays: copy before another getter.
        force, point, normal, separation, count, start = [
            _copy(value) for value in self.contacts.get_contact_data(physics_dt)]
        transforms = _copy(self.bodies.get_transforms())
        matrix = _copy(self.contacts.get_contact_force_matrix(physics_dt))
        capacity = force.size
        force = force.reshape(capacity)
        point = point.reshape(capacity, 3)
        normal = normal.reshape(capacity, 3)
        count = count.reshape(len(self.paths), self.contacts.filter_count)
        start = start.reshape(count.shape)
        if transforms.shape != (len(self.paths), 7) or matrix.shape != (*count.shape, 3):
            raise ValueError("Malformed PhysX evidence shapes")
        if np.any(count < 0) or np.any(start < 0) or np.any(start + count > capacity) or count.sum() >= capacity:
            raise ValueError("Contact evidence buffer is invalid or exhausted")
        used = set()
        for offset, length in zip(start.flat, count.flat):
            indices = set(range(int(offset), int(offset + length)))
            if used & indices:
                raise ValueError("Overlapping contact evidence ranges")
            used |= indices
        patches = []
        force_error = 0.
        for row, path in enumerate(self.paths):
            first = int(start[row, self.filter_index])
            end = first + int(count[row, self.filter_index])
            vectors = force[first:end, None] * normal[first:end]
            difference = vectors.sum(axis=0) - matrix[row, self.filter_index]
            if not np.isfinite(difference).all():
                raise ValueError("Nonfinite PhysX pair-force evidence")
            force_error = max(force_error, float(np.linalg.norm(difference)))
            for index in range(first, end):
                patches.append(dict(body=path, position=point[index], normal=normal[index],
                    normal_force_N=float(force[index])))
        if not np.isfinite(force_error) or force_error > 1e-3:
            raise ValueError("Patch normal directions/loads disagree with independent PhysX pair forces")
        result = shadow_physx_pad_grasp(patches, dict(zip(self.paths, transforms)),
            center, axis, half_length=half_length, radius=radius, side=self.side, profile=self.profile)
        output = dict(**result, sim_time_s=float(time_s), physics_dt_s=float(physics_dt),
            contact_capacity=capacity, active_contact_count=int(count.sum()),
            normal_pair_force_consistency_error_N=force_error)

        if include_evidence:
            # Reuse the synchronized copies already reduced above. No second
            # contact getter, transform fetch, inference or simulation step.
            output['raw_evidence'] = dict(schema='doorbench.shadow-raw-pad-evidence.v1',
                interval_start_s=max(0.,float(time_s)-float(physics_dt)), interval_end_s=float(time_s),
                geometry_time_s=float(time_s), clock='physx-interval-end', scope='complete-handle-body',
                contacts=[dict(body=p['body'],position=p['position'].tolist(),normal=p['normal'].tolist(),normal_force_N=p['normal_force_N']) for p in patches],
                body_transforms_xyzw={path:pose.tolist() for path,pose in zip(self.paths,transforms)},
                handle_pair_forces_world_N={path:matrix[i,self.filter_index].tolist() for i,path in enumerate(self.paths)},
                lever=dict(center=np.asarray(center,dtype=float).tolist(),axis=np.asarray(axis,dtype=float).tolist(),half_length=float(half_length),radius=float(radius)),
                contact_capacity=int(capacity),active_contact_count=int(count.sum()),
                normal_pair_force_consistency_error_N=force_error)
        return output
