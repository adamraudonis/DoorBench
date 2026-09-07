"""MyoHand kinematics with COCO-WholeBody's 21 hand landmarks.

The vendored parameter extract retains upstream positions, axes and limits.
Mirror polar vectors with S and hinge axes with det(S)*S. All joints are in
radians, lengths in metres. No Sapiens inference or motion data are used.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DATA = Path(__file__).with_name("anatomy") / "myohand.json"
ANATOMY = json.loads(DATA.read_text())
DIGITS = ["thumb", "index", "middle", "ring", "little"]
COLORS = [
    [0.96, 0.56, 0.20, 1],
    [0.78, 0.41, 0.70, 1],
    [0.27, 0.65, 0.86, 1],
    [0.78, 0.32, 0.33, 1],
    [0.40, 0.70, 0.38, 1],
]
CHAINS = [
    ["firstmc_r", "proximal_thumb_r", "distal_thumb_r"],
    ["2proxph_r", "midph2_r", "distph2_r"],
    ["3proxph_r", "midph3_r", "distph3_r"],
    ["4proxph_r", "midph4_r", "distph4_r"],
    ["5proxph_r", "midph5_r", "distph5_r"],
]
TIPS = [
    [0.010, -0.018, -0.007],
    [0.002, -0.019, 0],
    [0.002, -0.020, -0.002],
    [-0.003, -0.020, -0.002],
    [-0.004, -0.018, -0.002],
]


def nums(v):
    return " ".join(f"{float(x):.10g}" for x in v)


def vec(s):
    return np.fromstring(s, sep=" ")


def add(p, t, **kw):
    return ET.SubElement(p, t, {k: str(v) for k, v in kw.items()})


def canonical_name(name, side):
    return "hand_" + side + "_" + name.removesuffix("_r")


def joint_name(name, side):
    return "hand_" + side + "_" + name.removesuffix("_r")


def attach_hand(wrist, side, no_touch=False):
    """Append a source-derived hand; return contact exclusions and manifest."""
    if side not in ["l", "r"]:
        raise ValueError("Hand side must be l or r")
    S = np.diag([1 if side == "l" else -1, -1, 1])
    axial = np.linalg.det(S) * S
    created = {}

    def emit(src, parent, is_root=False):
        source_name = src["body"]["name"]
        name = canonical_name(source_name, side)
        b = add(
            parent,
            "body",
            name=name,
            pos=nums(
                [0, 0, 0] if is_root else S @ vec(src["body"].get("pos", "0 0 0"))
            ),
        )
        created[source_name] = b
        # The wrist and pronation are supplied by the actor, with the anatomical
        # hand beginning at the lunate. The 20 digit DoFs stay upstream-exact.
        if not is_root:
            for j in src["joints"]:
                add(
                    b,
                    "joint",
                    name=joint_name(j["name"], side),
                    axis=nums(axial @ vec(j["axis"])),
                    pos=nums(S @ vec(j.get("pos", "0 0 0"))),
                    range=j["range"],
                    damping=".025",
                    armature=".0001",
                    limited="true",
                    solreflimit=".002 1",
                    solimplimit=".999 .9999 .0001",
                )
        digit = next(
            (i for i, chain in enumerate(CHAINS) if source_name in chain), None
        )
        rgba = nums(COLORS[digit]) if digit is not None else ".82 .83 .78 1"
        for i, g in enumerate(src["collision"]):
            attrs = {k: v for k, v in g.items() if k in ["type", "size"]}
            attrs.setdefault("type", "capsule")
            if "fromto" in g:
                attrs["fromto"] = nums((vec(g["fromto"]).reshape(2, 3) @ S.T).ravel())
            else:
                attrs["pos"] = nums(S @ vec(g.get("pos", "0 0 0")))
                R = Rotation.from_euler("XYZ", vec(g.get("euler", "0 0 0"))).as_matrix()
                R = S @ R @ S
                q = Rotation.from_matrix(R).as_quat()
                attrs["quat"] = nums(q[[3, 0, 1, 2]])
            add(
                b,
                "geom",
                name=name + "_contact" + str(i),
                rgba=rgba,
                contype="4",
                conaffinity="1" if not no_touch else "0",
                mass=("0" if i else ".015") if digit is not None else ".035",
                group="4",
                **attrs,
            )
            add(
                b,
                "site",
                name="touch_" + side + "_" + source_name + "_" + str(i),
                rgba="0 0 0 0",
                **attrs,
            )
        # Anatomical bones are thinner than their contact envelopes; draw an
        # explicit skeleton so the CMC and metacarpal cannot hide in a palm box.
        children = src["children"]
        points = [S @ vec(c["body"].get("pos", "0 0 0")) for c in children]
        if digit is not None and source_name == CHAINS[digit][-1]:
            points = [S @ np.array(TIPS[digit])]
        for i, p in enumerate(points):
            if np.linalg.norm(p) > 0.002:
                add(
                    b,
                    "geom",
                    name=name + "_bone" + str(i),
                    type="capsule",
                    fromto=nums([0, 0, 0, *p]),
                    size=".0028",
                    rgba=rgba,
                    mass="0",
                    contype="8",
                    conaffinity="10",
                    group="0",
                )
        if not src["collision"]:
            add(
                b,
                "inertial",
                mass=".005",
                diaginertia=".000002 .000002 .000002",
                pos="0 0 0",
            )
        for child in children:
            emit(child, b)

    emit(ANATOMY["tree"], wrist, True)
    # COCO-WholeBody order: wrist, thumb CMC/MCP/IP/tip, then MCP/PIP/DIP/tip.
    landmarks = [("wrist", created["lunate_r"], np.zeros(3))]
    for i, chain in enumerate(CHAINS):
        for j, n in enumerate(chain):
            landmarks.append(
                (
                    DIGITS[i]
                    + (
                        "_cmc"
                        if i == 0 and j == 0
                        else "_mcp"
                        if (i == 0 and j == 1) or (i > 0 and j == 0)
                        else "_ip"
                        if i == 0
                        else "_pip"
                        if j == 1
                        else "_dip"
                    ),
                    created[n],
                    np.zeros(3),
                )
            )
        landmarks.append(
            (DIGITS[i] + "_tip", created[chain[-1]], S @ np.array(TIPS[i]))
        )
    for i, (label, b, pos) in enumerate(landmarks):
        add(
            b,
            "site",
            name=f"hand_keypoint_{side}_{i:02}",
            pos=nums(pos),
            size=".0035",
            rgba="0 0 0 0",
        )
        add(
            b,
            "geom",
            name=f"hand_{side}_landmark_{i:02}",
            type="sphere",
            pos=nums(pos),
            size=".0038",
            rgba=".95 .95 .86 1",
            mass="0",
            contype="0",
            conaffinity="0",
            group="0",
        )
    add(wrist, "site", name="palm_" + side, pos="0 0 0", size=".004", rgba="0 0 0 0")
    # Tissue envelopes intentionally overlap in the source hand. They interact
    # with the environment (mask 1), not with neighbouring tissue envelopes.
    # The thinner bone capsules use mask 8 for hand/hand and hand/body separation.
    # This avoids making connected palm tissue shove the fingers apart at rest.
    return [], {
        "source": ANATOMY["source"],
        "commit": ANATOMY["commit"],
        "side": side,
        "landmarks": [x[0] for x in landmarks],
        "coco_wholebody_offset": 91 if side == "l" else 112,
    }


def target_pose(side, close):
    pose = {}
    for digit in range(2, 6):
        for key, v in [
            ("mcp" + str(digit) + "_flexion", 0.85),
            ("mcp" + str(digit) + "_abduction", 0),
            ("pm" + str(digit) + "_flexion", 1.1),
            ("md" + str(digit) + "_flexion", 0.75),
        ]:
            pose[joint_name(key, side)] = v * close
    for n, v in [
        ("cmc_flexion", -0.09762254),
        ("cmc_abduction", -0.21790465),
        ("mp_flexion", -0.4949852),
        ("ip_flexion", -0.35630553),
    ]:
        pose[joint_name(n, side)] = v * close
    return pose
