#!/usr/bin/env python3
"""One-door, original stick human/contact-hand research prototype. All SI units.

Only actor joints have actuators. No door actuation, mocap hand, hand weld or
qpos writes during stepping. The lever/bolt equality is a mechanical linkage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from doorbench.reference.rig import rig_xml


def s(v):
    return " ".join(str(float(x)) for x in v)


def sub(p, tag, **kw):
    return ET.SubElement(p, tag, {k: str(v) for k, v in kw.items()})


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * t * (10 + t * (-15 + 6 * t))


def mix(a, b, t):
    return np.asarray(a) + (np.asarray(b) - a) * smooth(t)


def make_model(anchors=False, no_touch=False, latch_blocked=False):
    root = ET.Element("mujoco", model="DoorBench — one physical hand")
    sub(root, "compiler", angle="radian", autolimits="true")
    sub(
        root,
        "option",
        timestep=".001",
        integrator="implicitfast",
        solver="Newton",
        iterations="60",
        cone="elliptic",
        impratio="5",
    )
    vis = sub(root, "visual")
    sub(vis, "global", offwidth="1600", offheight="1000")
    sub(vis, "quality", shadowsize="4096")
    sub(
        vis, "headlight", ambient=".35 .35 .35", diffuse=".6 .6 .6", specular=".2 .2 .2"
    )
    asset = sub(root, "asset")
    sub(
        asset,
        "texture",
        name="floor",
        type="2d",
        builtin="checker",
        rgb1=".16 .20 .23",
        rgb2=".19 .23 .26",
        width="512",
        height="512",
    )
    sub(
        asset,
        "material",
        name="floor",
        texture="floor",
        texrepeat="4 4",
        reflectance=".12",
    )
    default = sub(root, "default")
    sub(
        default,
        "geom",
        solref=".004 1",
        solimp=".95 .99 .001",
        friction="1 .005 .0001",
        condim="4",
    )
    sub(default, "joint", damping="1", armature=".005")
    world = sub(root, "worldbody")
    sub(
        world,
        "light",
        pos="1 -3 4",
        dir="-.2 .5 -1",
        directional="true",
        diffuse=".9 .85 .75",
        castshadow="true",
    )
    sub(world, "geom", name="floor", type="plane", size="5 5 .1", material="floor")
    brown = ".36 .17 .07 1"
    gold = ".85 .57 .18 1"

    def box(parent, name, pos, size, rgba, **kw):
        return sub(
            parent,
            "geom",
            name=name,
            type="box",
            pos=s(pos),
            size=s(size),
            rgba=rgba,
            **kw,
        )

    box(world, "hinge_jamb", [-0.045, 0, 1.05], [0.04, 0.075, 1.05], ".32 .38 .4 1")
    # Strike jamb has a real pocket; latch cannot rotate out through solid walls.
    for label, z, h in [("lower", 0.49, 0.49), ("upper", 1.57, 0.53)]:
        box(world, "strike_" + label, [0.854, 0, z], [0.04, 0.075, h], ".32 .38 .4 1")
    box(world, "strike_near", [0.854, -0.034, 1.01], [0.04, 0.015, 0.03], gold)
    box(world, "strike_far", [0.854, 0.034, 1.01], [0.04, 0.015, 0.03], gold)
    box(world, "header", [0.415, 0, 2.13], [0.5, 0.075, 0.045], ".32 .38 .4 1")
    door = sub(world, "body", name="door", pos="0 -.028 0")
    sub(
        door,
        "joint",
        name="door_hinge",
        axis="0 0 1",
        range="-1.7 .005",
        damping="1.4",
        frictionloss=".25",
    )
    box(
        door,
        "door_leaf",
        [0.405, 0.028, 1.045],
        [0.405, 0.021, 1.037],
        brown,
        mass="19",
    )
    for z in [0.26, 1.05, 1.85]:
        sub(
            door,
            "geom",
            name=f"hinge_{z}",
            type="cylinder",
            pos=s([0, 0, z]),
            size=".016 .06",
            rgba=gold,
            mass=".05",
            contype="0",
            conaffinity="0",
        )
    sub(
        door,
        "geom",
        name="rose",
        type="cylinder",
        pos=".745 .001 1.01",
        quat=".70710678 .70710678 0 0",
        size=".027 .008",
        rgba=gold,
        mass=".06",
    )
    lever = sub(door, "body", name="lever", pos=".745 -.002 1.01")
    sub(
        lever,
        "joint",
        name="lever_hinge",
        axis="0 -1 0",
        range="0 .65",
        stiffness="1.8",
        damping=".10",
        frictionloss=".045",
        armature=".0005",
    )
    sub(
        lever,
        "geom",
        name="lever_stem",
        type="capsule",
        fromto="0 0 0 0 -.066 0",
        size=".011",
        rgba=gold,
        mass=".04",
    )
    sub(
        lever,
        "geom",
        name="lever_grip",
        type="capsule",
        fromto="0 -.066 0 -.12 -.066 0",
        size=".012",
        rgba=gold,
        mass=".08",
    )
    sub(lever, "site", name="grip", pos="-.078 -.066 0", size=".004", rgba="0 0 0 0")
    bolt = sub(door, "body", name="bolt", pos=".794 .028 1.01")
    sub(
        bolt,
        "joint",
        name="latch_slide",
        type="slide",
        axis="-1 0 0",
        range="0 .030",
        stiffness="35",
        damping=".3",
    )
    box(bolt, "latch_bolt", [0.012, 0, 0], [0.02, 0.014, 0.012], gold, mass=".045")
    actor = ET.fromstring(rig_xml(root_pos=(0.9, -0.60, 0.94), root_yaw=0.0)).find(
        "worldbody"
    )[0]
    world.append(actor)
    for geom in actor.iter("geom"):
        name = geom.get("name")
        geom.set("rgba", ".24 .68 .74 1")
        # Self-collision enabled; MuJoCo excludes only parent-child pairs.
        # Actor/scene and finger/mechanism contacts remain fully native.
        geom.set("contype", "2")
        geom.set("conaffinity", "3")
        if "torso" in name:
            geom.set("size", ".06")
        elif "pelvis" in name:
            geom.set("size", ".07")
        elif "thigh" in name:
            geom.set("size", ".039")
        elif "shin" in name:
            geom.set("size", ".030")
        elif "upper_arm" in name:
            geom.set("size", ".028")
        elif "forearm" in name:
            geom.set("size", ".025")
    for body in actor.iter("body"):
        if body.get("name", "").startswith("actor_wrist_"):
            side = body.get("name")[-1]
            body.set("quat", ".70710678 -.70710678 0 0")
            for g in list(body.findall("geom")):
                body.remove(g)
            box(
                body,
                "hand_" + side + "_palm",
                [0, 0.033, 0],
                [0.037, 0.033, 0.011],
                ".82 .88 .85 1",
                mass=".20",
                contype="2",
                conaffinity="3" if not no_touch else "0",
            )
            for k, x in enumerate([-0.029, -0.010, 0.010, 0.029]):
                b = body
                for j, length in enumerate([0.035, 0.024, 0.019]):
                    b = sub(
                        b,
                        "body",
                        name=f"hand_{side}_finger{k}_{j}",
                        pos=s(
                            [x, 0.065, 0] if j == 0 else [0, [0.035, 0.024][j - 1], 0]
                        ),
                    )
                    sub(
                        b,
                        "joint",
                        name=f"finger_{side}_{k}_{j}",
                        axis="-1 0 0",
                        range="0 1.75",
                        damping=".025",
                        armature=".00002",
                    )
                    sub(
                        b,
                        "geom",
                        name=f"hand_{side}_finger{k}_{j}",
                        type="capsule",
                        fromto=s([0, 0, 0, 0, length, 0]),
                        size=".0075",
                        rgba=".84 .90 .87 1",
                        mass=".014",
                        contype="2",
                        conaffinity="3" if not no_touch else "0",
                    )
                    sub(
                        b,
                        "site",
                        name=f"touch_{side}_{k}_{j}",
                        type="capsule",
                        fromto=s([0, 0, 0, 0, length, 0]),
                        size=".0085",
                        rgba="0 0 0 0",
                    )
            # Thumb opposable across the palm, with two flexion joints.
            b = sub(
                body,
                "body",
                name="hand_" + side + "_thumb",
                pos=s([0.045 if side == "l" else -0.045, 0.045, -0.022]),
                quat=s([0.92387953, 0, 0, 0.38268343 if side == "l" else -0.38268343]),
            )
            for j, length in enumerate([0.032, 0.026]):
                if j:
                    b = sub(b, "body", name=f"hand_{side}_thumb{j}", pos="0 .032 0")
                sub(
                    b,
                    "joint",
                    name=f"thumb_{side}_{j}",
                    axis="-1 0 0",
                    range="-.4 1.5",
                    damping=".025",
                    armature=".00002",
                )
                sub(
                    b,
                    "geom",
                    name=f"hand_{side}_thumb{j}",
                    type="capsule",
                    fromto=s([0, 0, 0, 0, length, 0]),
                    size=".009",
                    rgba=".84 .90 .87 1",
                    mass=".02",
                    contype="2",
                    conaffinity="3" if not no_touch else "0",
                )
                sub(
                    b,
                    "site",
                    name=f"touch_{side}_thumb{j}",
                    type="capsule",
                    fromto=s([0, 0, 0, 0, length, 0]),
                    size=".010",
                    rgba="0 0 0 0",
                )
            sub(
                body,
                "site",
                name="touch_" + side + "_palm",
                type="box",
                pos="0 .033 0",
                size=".038 .034 .012",
                rgba="0 0 0 0",
            )
            sub(
                body,
                "site",
                name="palm_" + side,
                pos="0 0 0",
                size=".004",
                rgba="0 0 0 0",
            )
    for b in actor.iter("body"):
        n = b.get("name", "")
        if any(x in n for x in ["shoulder_", "elbow_", "hip_", "knee_", "ankle_"]):
            sub(
                b,
                "geom",
                type="sphere",
                size=".028",
                rgba=".34 .78 .8 1",
                mass="0",
                contype="0",
                conaffinity="0",
            )
    chest = actor.find(".//body[@name='actor_chest']")
    sub(
        chest,
        "geom",
        type="capsule",
        fromto="-.18 0 .06 .18 0 .06",
        size=".022",
        rgba=".24 .68 .74 1",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    sub(
        actor,
        "geom",
        type="capsule",
        fromto="-.105 0 -.06 .105 0 -.06",
        size=".025",
        rgba=".24 .68 .74 1",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    head = actor.find(".//body[@name='actor_head']")
    for x in [-0.031, 0.031]:
        sub(
            head,
            "geom",
            type="sphere",
            pos=s([x, 0.091, 0.026]),
            size=".011",
            rgba=".03 .12 .15 1",
            mass="0",
            contype="0",
            conaffinity="0",
        )
    # Forearm pronation is separate from wrist flexion/deviation.
    for side in ["l", "r"]:
        b = actor.find(f".//body[@name='actor_elbow_{side}']")
        sub(
            b,
            "joint",
            name="actor_pronation_" + side,
            axis="0 0 1",
            range="-2.8 2.8",
            damping=".2",
        )
    eq = sub(root, "equality")
    sub(
        eq,
        "joint",
        name="spindle_to_latch",
        joint1="latch_slide",
        joint2="lever_hinge",
        polycoef="0 .043 0 0 0",
        solref=".003 1",
    )
    act = sub(root, "actuator")
    for j in actor.iter("joint"):
        name = j.get("name")
        finger = name.startswith(("finger_", "thumb_"))
        lower = any(x in name for x in ["hip_", "knee_", "ankle_"])
        spine = "spine_" in name
        kp = 1.5 if finger else 2400 if lower else 1000 if spine else 320
        kd = 0.025 if finger else 100 if lower else 45 if spine else 18
        sub(
            act,
            "position",
            name="motor_" + name,
            joint=name,
            kp=kp,
            kv=kd,
            forcelimited="true",
            forcerange="-.18 .18" if finger else "-160 160",
        )
    sensor = sub(root, "sensor")
    for site in actor.iter("site"):
        if site.get("name", "").startswith("touch_"):
            sub(sensor, "touch", name=site.get("name"), site=site.get("name"))
    if latch_blocked:
        sub(
            eq,
            "joint",
            name="blocked_latch",
            joint1="latch_slide",
            polycoef="0 0 0 0 0",
            solref=".002 1",
        )
    if anchors:
        for side in ["l", "r"]:
            sub(
                eq,
                "weld",
                name="support_" + side,
                body1="actor_ankle_" + side,
                solref=".005 1",
            )
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    return model, xml


def joint_indices(m, names):
    return np.array([m.joint(n).qposadr[0] for n in names]), np.array(
        [m.joint(n).dofadr[0] for n in names]
    )


def initial(m):
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    a = math.acos((0.94 - 0.06 - 0.055) / 0.86)
    for side in ["l", "r"]:
        for name, v in [
            ("hip_" + side + "_pitch", a),
            ("knee_" + side, -2 * a),
            ("ankle_" + side + "_pitch", a),
            ("elbow_" + side, 0.45),
            ("shoulder_" + side + "_roll", 0.1 if side == "l" else -0.1),
        ]:
            d.qpos[m.joint("actor_" + name).qposadr[0]] = v
    mujoco.mj_forward(m, d)
    return d


def arm_ik(m, d, side, pos, rotation, iterations=15):
    names = (
        [f"actor_shoulder_{side}_{v}" for v in ["pitch", "roll", "yaw"]]
        + [f"actor_elbow_{side}", f"actor_pronation_{side}"]
        + [f"actor_wrist_{side}_{v}" for v in ["pitch", "roll", "yaw"]]
    )
    q, v = joint_indices(m, names)
    sid = m.site("palm_" + side).id
    jp = np.zeros((3, m.nv))
    jr = jp.copy()
    for _ in range(iterations):
        mujoco.mj_forward(m, d)
        er = Rotation.from_matrix(
            rotation @ d.site_xmat[sid].reshape(3, 3).T
        ).as_rotvec()
        e = np.r_[pos - d.site_xpos[sid], er * 0.18]
        if np.linalg.norm(e) < 0.0002:
            break
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.r_[jp[:, v], jr[:, v] * 0.18]
        delta = J.T @ np.linalg.solve(J @ J.T + np.eye(6) * 0.00008, e)
        d.qpos[q] += np.clip(delta, -0.15, 0.15)
        for name, qi in zip(names, q):
            d.qpos[qi] = np.clip(d.qpos[qi], *m.joint(name).range)
    return float(np.linalg.norm(e))


def run(out, no_touch=False, duration=6.6666667, anchors=False, latch_blocked=False):
    started = datetime.now(timezone.utc).isoformat()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m, xml = make_model(anchors, no_touch, latch_blocked)
    (out / "scene.xml").write_text(xml)
    d = initial(m)
    ik = initial(m)
    if anchors:
        for side in ["l", "r"]:
            i = m.equality("support_" + side).id
            b = m.body("actor_ankle_" + side).id
            m.eq_data[i, 3:6] = -d.xmat[b].reshape(3, 3).T @ d.xpos[b]
            m.eq_data[i, 6:10] = d.xquat[b] * np.array([1, -1, -1, -1])
        mujoco.mj_saveLastXML(str(out / "scene.xml"), m)
    qids = np.array([m.jnt_qposadr[i] for i in m.actuator_trnid[:, 0]])
    vids = np.array([m.jnt_dofadr[i] for i in m.actuator_trnid[:, 0]])
    saved = ET.parse(out / "scene.xml").getroot()
    key = sub(saved, "keyframe")
    sub(
        key,
        "key",
        name="initial",
        qpos=s(d.qpos),
        ctrl=s(d.qpos[qids] + d.qfrc_bias[vids] / m.actuator_gainprm[:, 0]),
    )
    (out / "scene.xml").write_text(ET.tostring(saved, encoding="unicode"))
    desired = d.qpos.copy()
    # Left-hand rest and working frame. Lever is +X across palm width.
    wrist0 = d.site_xpos[m.site("palm_l").id].copy()
    R0 = d.site_xmat[m.site("palm_l").id].reshape(3, 3).copy()
    grip = np.array([0.667, -0.096, 1.01])
    ready = grip + [0, -0.069, 0.031]
    rows = []
    poses = []
    velocities = []
    controls = []
    torques = []
    sensors = []
    forces = []
    maxpen = 0
    peakforce = 0
    maxselfpen = 0
    nonhand_impulse = 0
    peak_total = 0
    touch_impulse = 0
    feet = [m.body("actor_ankle_" + x).id for x in ["l", "r"]]
    feet0 = d.xpos[feet].copy()
    max_foot_drift = 0.0
    min_floor_force = 1e9
    body_names = [m.body(i).name or "" for i in range(m.nbody)]
    phase = "settle"
    for k in range(round(duration / m.opt.timestep)):
        clock = k * m.opt.timestep
        t = clock * 1.35
        # A smooth, feedforward task-space proposal. Native mechanics determine
        # achieved handle/door motion; no simulated mechanism state is assigned.
        if t < 0.65:
            phase = "settle"
            p = wrist0
            R = R0
            close = 0
        elif t < 2.0:
            phase = "reach"
            u = smooth((t - 0.65) / 1.35)
            p = (
                wrist0 * (1 - u)
                + ready * u
                + np.array([0, 0, 0.13]) * math.sin(math.pi * u)
            )
            R = Rotation.from_rotvec(
                Rotation.from_matrix(R0).as_rotvec() * (1 - u)
            ).as_matrix()
            close = 0
        elif t < 2.65:
            phase = "grasp"
            p = ready
            R = np.eye(3)
            close = smooth((t - 2.0) / 0.65)
        elif t < 3.35:
            phase = "press lever"
            v = 0.50 * smooth((t - 2.65) / 0.7)
            R = Rotation.from_rotvec([0, -v, 0]).as_matrix()
            p = np.array([0.745, -0.03, 1.01]) + R @ (
                np.array([-0.078, -0.066, 0]) + [0, -0.069, 0.031]
            )
            close = 1
        elif t < 6.7:
            phase = "pull"
            v = 0.50
            theta = -0.85 * smooth((t - 3.35) / 3.35)
            D = Rotation.from_rotvec([0, 0, theta]).as_matrix()
            L = Rotation.from_rotvec([0, -v, 0]).as_matrix()
            R = D @ L
            p = np.array([0, -0.028, 0]) + D @ (
                np.array([0.745, -0.002, 1.01])
                + L @ (np.array([-0.078, -0.066, 0]) + [0, -0.069, 0.031])
            )
            close = 1
        else:
            phase = "release"
            v = 0.50
            theta = -0.85
            D = Rotation.from_rotvec([0, 0, theta]).as_matrix()
            L = Rotation.from_rotvec([0, -v, 0]).as_matrix()
            R = D @ L
            p = (
                np.array([0, -0.028, 0])
                + D
                @ (
                    np.array([0.745, -0.002, 1.01])
                    + L @ (np.array([-0.078, -0.066, 0]) + [0, -0.069, 0.031])
                )
                + np.array([0.1, -0.1, 0.06]) * smooth((t - 7.1) / 1.4)
            )
            close = 1 - smooth((t - 6.7) / 0.55)
        if k % 5 == 0:
            ik.qpos[:] = desired
            look = smooth((t - 0.65) / 1.35)
            ik.qpos[m.joint("actor_neck_yaw").qposadr[0]] = look * (
                0.35 + 0.5 * smooth((t - 3.35) / 3.35)
            )
            ik.qpos[m.joint("actor_neck_pitch").qposadr[0]] = -0.20 * look
            # Solve against the intended upright stance, independent of physics.
            err = arm_ik(m, ik, "l", p, R, 8)
            desired = ik.qpos.copy()
            for side in ["l", "r"]:
                c = close if side == "l" else 0.18
                for n in range(4):
                    for j, angle in enumerate([0.75, 1.35, 1.1]):
                        desired[m.joint(f"finger_{side}_{n}_{j}").qposadr[0]] = (
                            c * angle
                        )
                for j, angle in enumerate([0.35, 0.55]):
                    desired[m.joint(f"thumb_{side}_{j}").qposadr[0]] = c * angle
        # Inverse-dynamics gravity terms apply to motor coordinates only.
        d.ctrl[:] = desired[qids] + d.qfrc_bias[vids] / m.actuator_gainprm[:, 0]
        mujoco.mj_step(m, d)
        # Refresh contacts, sensors and world poses at the recorded post-step state.
        mujoco.mj_forward(m, d)
        if abs(d.time - (k + 1) * m.opt.timestep) > 1e-7:
            raise RuntimeError("Simulator reset or clock mismatch")
        touch = 0.0
        contacts = []
        ground_force = 0.0
        max_foot_drift = max(
            max_foot_drift,
            float(np.linalg.norm(d.xpos[feet, :2] - feet0[:, :2], axis=1).max()),
        )
        for ci in range(d.ncon):
            c = d.contact[ci]
            a = m.geom(c.geom1).name or ""
            b = m.geom(c.geom2).name or ""
            if (a.startswith("hand_") and b.startswith(("lever", "latch", "door"))) or (
                b.startswith("hand_") and a.startswith(("lever", "latch", "door"))
            ):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f)
                force = float(np.linalg.norm(f[:3]))
                touch += force
                maxpen = max(maxpen, -float(c.dist))
                contacts.append([*c.pos, force, a, b])
                peakforce = max(peakforce, force)
            is_actor = [
                body_names[m.geom_bodyid[g]].startswith(("actor_", "hand_"))
                for g in [c.geom1, c.geom2]
            ]
            if all(is_actor):
                maxselfpen = max(maxselfpen, -float(c.dist))
            if (
                any(is_actor)
                and not (a.startswith("hand_") or b.startswith("hand_"))
                and (
                    a.startswith(("door_", "lever_", "latch_"))
                    or b.startswith(("door_", "lever_", "latch_"))
                )
            ):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f)
                nonhand_impulse += float(np.linalg.norm(f[:3])) * m.opt.timestep
            if ("floor" in [a, b]) and any(is_actor):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f)
                ground_force += float(f[0])
        touch_impulse += touch * m.opt.timestep
        peak_total = max(peak_total, touch)
        if t > 0.3:
            min_floor_force = min(min_floor_force, ground_force)
        if k % 20 == 0:
            rows.append(
                {
                    "t": round(d.time, 4),
                    "phase": phase,
                    "door_deg": float(
                        -d.qpos[m.joint("door_hinge").qposadr[0]] * 180 / np.pi
                    ),
                    "lever_deg": float(
                        d.qpos[m.joint("lever_hinge").qposadr[0]] * 180 / np.pi
                    ),
                    "latch_mm": float(d.qpos[m.joint("latch_slide").qposadr[0]] * 1000),
                    "touch_n": touch,
                    "pelvis_z": float(d.xpos[m.body("actor_pelvis").id, 2]),
                    "ik_error": err if t > 0.65 else 0,
                    "ground_n": ground_force,
                    "sensor_n": float(d.sensordata.sum()),
                    "wrist_error_m": float(
                        np.linalg.norm(d.site_xpos[m.site("palm_l").id] - p)
                    ),
                }
            )
            poses.append(d.qpos.copy())
            velocities.append(d.qvel.copy())
            controls.append(d.ctrl.copy())
            torques.append(d.actuator_force.copy())
            sensors.append(d.sensordata.copy())
            forces.append(contacts)
        if not np.isfinite(d.qpos).all():
            raise RuntimeError("nonfinite state")
    np.savez_compressed(
        out / "trajectory.npz",
        qpos=np.array(poses),
        qvel=np.array(velocities),
        ctrl=np.array(controls),
        actuator_force=np.array(torques),
        touch_sensor=np.array(sensors),
        time=np.array([r["t"] for r in rows]),
    )
    report = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rig_sha256": hashlib.sha256(
            (ROOT / "doorbench/reference/rig.py").read_bytes()
        ).hexdigest(),
        "scene_sha256": hashlib.sha256((out / "scene.xml").read_bytes()).hexdigest(),
        "schema": "doorbench.physical-human-prototype.v1",
        "simulator": mujoco.__version__,
        "timestep_s": m.opt.timestep,
        "foot_welds": 2 if anchors else 0,
        "free_root": True,
        "latch_blocked": latch_blocked,
        "no_touch": no_touch,
        "door_actuators": 0,
        "hand_welds": 0,
        "max_contact_penetration_m": maxpen,
        "peak_hand_contact_n": peakforce,
        "max_door_deg": max(r["door_deg"] for r in rows),
        "min_pelvis_z": min(r["pelvis_z"] for r in rows),
        "max_self_penetration_m": maxselfpen,
        "nonhand_door_impulse_ns": nonhand_impulse,
        "peak_total_hand_contact_n": peak_total,
        "hand_contact_impulse_ns": touch_impulse,
        "max_foot_drift_m": max_foot_drift,
        "min_floor_force_n": min_floor_force,
        "warnings": d.warning.number.tolist(),
        "peak_motor_torque_nm": float(np.max(np.abs(torques))),
        "rows": rows,
        "contacts": forces,
    }
    (out / "report.json").write_text(json.dumps(report))
    print(
        json.dumps({k: v for k, v in report.items() if k not in ["rows", "contacts"]})
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--anchors", action="store_true")
    p.add_argument("--no-touch", action="store_true")
    p.add_argument("--latch-blocked", action="store_true")
    p.add_argument("--duration", type=float, default=6.6666667)
    a = p.parse_args()
    run(a.out, a.no_touch, a.duration, a.anchors, a.latch_blocked)
