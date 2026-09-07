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
    [0.12, 0.80, 0.92, 1],
    [0.92, 0.91, 0.82, 1],
    [0.92, 0.91, 0.82, 1],
    [0.92, 0.91, 0.82, 1],
    [0.92, 0.91, 0.82, 1],
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
                priority="1",
                solref=".010 1",
                solimp=".90 .95 .001",
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


# Authored overhand lever grasp. Native collision envelopes, not rendered bone
# rods, determine the required separation around the 24 mm diameter grip.
LEVER_GRASP = {
    "cmc_flexion": -0.6055724,
    "cmc_abduction": 0.2817054,
    "mp_flexion": -0.6332882,
    "ip_flexion": -0.497506,
    "mcp2_flexion": 1.132385,
    "mcp2_abduction": -0.0385688,
    "pm2_flexion": 0.4288332,
    "md2_flexion": 0.4823154,
    "mcp3_flexion": 1.064923,
    "mcp3_abduction": -0.0904397,
    "pm3_flexion": 0.450094,
    "md3_flexion": 0.5217353,
    "mcp4_flexion": 0.8397097,
    "mcp4_abduction": -0.0496644,
    "pm4_flexion": 0.6462594,
    "md4_flexion": 0.6550941,
    "mcp5_flexion": 0.3921547,
    "mcp5_abduction": 0.1443957,
    "pm5_flexion": 0.8837379,
    "md5_flexion": 0.75,
}


def target_pose(side, close):
    return {
        joint_name(name, side): value * close for name, value in LEVER_GRASP.items()
    }


PREGRASP = {
    "cmc_flexion": -0.5403678,
    "cmc_abduction": 0.7649984,
    "mp_flexion": 0.1689725,
    "ip_flexion": -0.5000769,
    "mcp2_flexion": 0.0197571,
    "mcp2_abduction": -0.0341836,
    "pm2_flexion": 0.6978674,
    "md2_flexion": 0.6499485,
    "mcp3_flexion": 0.015,
    "mcp3_abduction": 0.0942573,
    "pm3_flexion": 0.5593461,
    "md3_flexion": 0.9848486,
    "mcp4_flexion": 0.0165593,
    "mcp4_abduction": 0.0872705,
    "pm4_flexion": 0.4281981,
    "md4_flexion": 0.6907838,
    "mcp5_flexion": 0.0155151,
    "mcp5_abduction": 0.0971099,
    "pm5_flexion": 0.1402493,
    "md5_flexion": 0.3249749,
}


def grasp_pose(side, amount):
    return {
        joint_name(n, side): PREGRASP[n] + amount * (v - PREGRASP[n])
        for n, v in LEVER_GRASP.items()
    }
