"""Simulator-agnostic articulated model IR.

Conventions (MJCF-like):
  * Z up, SI units.
  * Body pos/quat are relative to the parent body frame.
  * Geom pos/quat and joint pos/axis are in the body frame.
  * Quaternions are (w, x, y, z).
  * Box size = half-extents; cylinder/capsule size = (radius, half_length) along local z;
    sphere size = (radius,).
  * Joint positive direction is chosen so that q > 0 always means "opening" / "actuating".

Fidelity tiers: each body carries a set of tiers it belongs to.
  full    -> every mechanism body (bolts, thumbturns, closer arms, keypad buttons ...)
  simple  -> leaf + primary operator + latch bolt (primitive collision only)
  minimal -> leaf only (hinge with friction/damping; latch expressed as joint range)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np

TIERS = ("full", "simple", "minimal")
ALL_TIERS = frozenset(TIERS)
FULL_ONLY = frozenset({"full"})
FULL_SIMPLE = frozenset({"full", "simple"})


# ---------------------------------------------------------------------------
# quaternion helpers
# ---------------------------------------------------------------------------
def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0, 0, 0])
    axis = axis / n
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(m):
    m = np.asarray(m, dtype=float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rpy(q):
    """Quaternion (w,x,y,z) -> URDF roll/pitch/yaw (extrinsic XYZ)."""
    w, x, y, z = q
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def quat_rotate(q, v):
    return quat_to_mat(q) @ np.asarray(v, dtype=float)


def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


QUAT_ID = np.array([1.0, 0.0, 0.0, 0.0])
# handy fixed rotations
QX90 = quat_from_axis_angle([1, 0, 0], math.pi / 2)    # local z -> -y ... (rotates z axis to -y)
QY90 = quat_from_axis_angle([0, 1, 0], math.pi / 2)    # local z -> +x
QXm90 = quat_from_axis_angle([1, 0, 0], -math.pi / 2)  # local z -> +y


def quat_z_to(direction):
    """Quaternion rotating local +z onto `direction`."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    z = np.array([0, 0, 1.0])
    c = float(np.dot(z, d))
    if c > 1 - 1e-9:
        return QUAT_ID.copy()
    if c < -1 + 1e-9:
        return quat_from_axis_angle([1, 0, 0], math.pi)
    axis = np.cross(z, d)
    return quat_from_axis_angle(axis, math.acos(c))


# ---------------------------------------------------------------------------
# IR dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Material:
    name: str
    rgba: tuple = (0.7, 0.7, 0.7, 1.0)
    roughness: float = 0.6
    metallic: float = 0.0
    texture: Optional[str] = None
    transparent: bool = False
    emissive: tuple = (0, 0, 0)

    def to_dict(self):
        return {"name": self.name, "rgba": list(self.rgba), "roughness": self.roughness, "metallic": self.metallic,
                "texture": self.texture, "transparent": self.transparent, "emissive": list(self.emissive)}


@dataclass
class Geom:
    name: str
    type: str                        # box | cylinder | capsule | sphere | mesh
    size: tuple
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    material: str = "default"
    collision: bool = True
    visual: bool = True
    density: float = 1000.0          # kg/m^3 used for mass/inertia (effective)
    mass_override: Optional[float] = None
    friction: tuple = (0.6, 0.005, 0.0001)
    mesh: Any = None                 # trimesh.Trimesh (for type == mesh)
    mesh_name: Optional[str] = None  # library key (content hash) when shared
    mesh_shared: bool = False
    solref: Optional[tuple] = None   # soft contact override (MJCF)
    solimp: Optional[tuple] = None
    margin: float = 0.0
    tiers: frozenset = ALL_TIERS
    semantic: str = "structure"      # leaf | frame | operator | latch | lock | hinge | closer | glass | seal | wall | floor | track | decor | sensor | mechanism
    part_label: str = ""             # human-readable
    contact_priority: int = 0         # native material contact mixing priority; zero preserves defaults

    def validate_contact_priority(self):
        if (isinstance(self.contact_priority, bool) or not isinstance(self.contact_priority, int)
                or not 0 <= self.contact_priority <= 2147483647):
            raise ValueError('Geom contact_priority must be an integer in 0..2147483647')

    def volume(self) -> float:
        if self.type == "box":
            hx, hy, hz = self.size
            return 8 * hx * hy * hz
        if self.type == "cylinder":
            r, hl = self.size[:2]
            return math.pi * r * r * 2 * hl
        if self.type == "capsule":
            r, hl = self.size[:2]
            return math.pi * r * r * 2 * hl + 4 / 3 * math.pi * r ** 3
        if self.type == "sphere":
            return 4 / 3 * math.pi * self.size[0] ** 3
        if self.type == "mesh" and self.mesh is not None:
            try:
                v = float(abs(self.mesh.volume))
                if v > 1e-12:
                    return v
            except Exception:
                pass
            ext = self.mesh.extents
            return float(ext[0] * ext[1] * ext[2]) * 0.4
        return 0.0

    def mass(self) -> float:
        if self.mass_override is not None:
            return self.mass_override
        return self.density * self.volume()

    def local_inertia(self) -> np.ndarray:
        """Inertia tensor about the geom's own center, in geom frame (3x3)."""
        m = self.mass()
        if self.type == "box":
            hx, hy, hz = self.size
            a, b, c = 2 * hx, 2 * hy, 2 * hz
            return np.diag([m / 12 * (b * b + c * c), m / 12 * (a * a + c * c), m / 12 * (a * a + b * b)])
        if self.type == "cylinder":
            r, hl = self.size[:2]
            h = 2 * hl
            ixx = m / 12 * (3 * r * r + h * h)
            return np.diag([ixx, ixx, 0.5 * m * r * r])
        if self.type == "capsule":
            r, hl = self.size[:2]
            h = 2 * hl
            m_cyl = m * (math.pi * r * r * h) / max(self.volume(), 1e-12)
            m_sph = m - m_cyl
            ixx = m_cyl / 12 * (3 * r * r + h * h) + m_sph * (0.4 * r * r + 0.5 * h * h / 2 + 3 / 8 * r * h)
            izz = 0.5 * m_cyl * r * r + 0.4 * m_sph * r * r
            return np.diag([ixx, ixx, izz])
        if self.type == "sphere":
            r = self.size[0]
            return np.eye(3) * (0.4 * m * r * r)
        if self.type == "mesh" and self.mesh is not None:
            try:
                mesh = self.mesh
                if mesh.is_watertight:
                    it = np.asarray(mesh.moment_inertia, dtype=float)
                    vol = abs(mesh.volume)
                    if vol > 1e-12:
                        return it * (m / vol)  # trimesh uses density 1 -> scale
            except Exception:
                pass
            ext = np.asarray(self.mesh.extents) / 2
            a, b, c = 2 * ext
            return np.diag([m / 12 * (b * b + c * c), m / 12 * (a * a + c * c), m / 12 * (a * a + b * b)])
        return np.zeros((3, 3))

    def center(self) -> np.ndarray:
        """Center of mass in body frame."""
        c = np.asarray(self.pos, dtype=float)
        if self.type == "mesh" and self.mesh is not None:
            try:
                cm = self.mesh.center_mass if self.mesh.is_watertight else self.mesh.centroid
            except Exception:
                cm = self.mesh.bounding_box.centroid
            c = c + quat_rotate(self.quat, cm)
        return c

    def to_dict(self):
        self.validate_contact_priority()
        d = {
            "name": self.name, "type": self.type, "size": [float(s) for s in self.size],
            "pos": [float(p) for p in self.pos], "quat": [float(q) for q in self.quat],
            "material": self.material, "collision": self.collision, "visual": self.visual,
            "density": self.density, "friction": list(self.friction), "tiers": sorted(self.tiers),
            "semantic": self.semantic, "part_label": self.part_label,
        }
        if self.type == "mesh":
            d["mesh_name"] = self.mesh_name
            d["mesh_shared"] = self.mesh_shared
        if self.mass_override is not None:
            d["mass_override"] = self.mass_override
        if self.solref:
            d["solref"] = list(self.solref)
        if self.solimp:
            d["solimp"] = list(self.solimp)
        if self.contact_priority:
            d['contact_priority'] = self.contact_priority
        return d


@dataclass
class Joint:
    name: str
    type: str                     # hinge | slide | free (world-root, 7 qpos / 6 qvel)
    axis: tuple = (0.0, 0.0, 1.0)
    pos: tuple = (0.0, 0.0, 0.0)
    range: Optional[tuple] = (0.0, 1.5708)
    damping: float = 0.0
    frictionloss: float = 0.0
    stiffness: float = 0.0
    springref: float = 0.0
    armature: float = 0.0
    limit_solref: Optional[tuple] = None   # soft limit (e.g. gasket compression, rubber stop)
    role: str = "primary"         # primary | operator | latch | lock | mechanism | secondary | decor
    label: str = ""
    robot_interactive: bool = True
    initial: float = 0.0          # initial q (spec state at episode start)
    modeled_at: float = 0.0       # joint value at which the body geometry was authored (MJCF `ref`)
    notes: str = ""
    # asymmetric damping for door closers: not representable natively; exported as metadata
    damping_closing: Optional[float] = None
    damping_opening: Optional[float] = None
    backcheck_angle: Optional[float] = None
    backcheck_damping: Optional[float] = None
    ratchet_one_way: bool = False

    def to_dict(self):
        return {
            "name": self.name, "type": self.type, "axis": [float(a) for a in self.axis],
            "pos": [float(p) for p in self.pos], "range": None if self.range is None else [float(r) for r in self.range],
            "damping": self.damping, "frictionloss": self.frictionloss, "stiffness": self.stiffness,
            "springref": self.springref, "armature": self.armature, "role": self.role, "label": self.label,
            "robot_interactive": self.robot_interactive, "initial": self.initial, "notes": self.notes,
            "damping_closing": self.damping_closing, "damping_opening": self.damping_opening,
            "backcheck_angle": self.backcheck_angle, "backcheck_damping": self.backcheck_damping,
            "limit_solref": None if self.limit_solref is None else list(self.limit_solref),
            "ratchet_one_way": self.ratchet_one_way, "modeled_at": self.modeled_at,
            **({"qpos_width": 7, "qvel_width": 6,
                "initial_pose_source": "body world position and WXYZ quaternion"}
               if self.type == "free" else {}),
        }


@dataclass
class Site:
    name: str
    pos: tuple
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    size: float = 0.01
    role: str = "marker"   # grip | push | pass_plane | approach | sensor | closer_anchor
    tiers: frozenset = ALL_TIERS

    def to_dict(self):
        return {"name": self.name, "pos": [float(p) for p in self.pos], "quat": [float(q) for q in self.quat],
                "size": self.size, "role": self.role, "tiers": sorted(self.tiers)}


@dataclass
class Body:
    name: str
    parent: Optional[str]          # None for world children
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    joint: Optional[Joint] = None
    geoms: list = field(default_factory=list)
    sites: list = field(default_factory=list)
    tiers: frozenset = ALL_TIERS
    semantic: str = "structure"
    label: str = ""
    extra_mass: float = 0.0        # non-geometric mass (internal mechanism), added at COM of geoms
    mass_override: Optional[float] = None
    static: bool = False           # world-fixed (frame/wall)

    # ---- inertial -----
    def inertial(self, tier: str = "full"):
        geoms = [g for g in self.geoms if tier in g.tiers]
        masses = np.array([g.mass() for g in geoms]) if geoms else np.zeros(0)
        total = float(masses.sum()) + self.extra_mass
        if self.mass_override is not None and total > 0:
            scale = self.mass_override / total
            masses = masses * scale
            extra = self.extra_mass * scale
            total = self.mass_override
        else:
            extra = self.extra_mass
        if total <= 1e-9:
            return 0.0, np.zeros(3), np.zeros((3, 3))
        centers = np.array([g.center() for g in geoms]) if geoms else np.zeros((0, 3))
        com = (centers * masses[:, None]).sum(axis=0) / max(masses.sum(), 1e-12) if len(geoms) else np.zeros(3)
        # extra mass placed at com (does not move com)
        I = np.zeros((3, 3))
        for g, m, c in zip(geoms, masses, centers):
            R = quat_to_mat(g.quat)
            Ig = g.local_inertia()
            if g.mass_override is not None or self.mass_override is not None:
                Ig = Ig * (m / max(g.mass(), 1e-12))
            Iw = R @ Ig @ R.T
            d = c - com
            Iw = Iw + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
            I += Iw
        if extra > 0 and len(geoms):
            # spread extra mass as a small sphere-ish inertia at COM to avoid singular tensors
            r = 0.05
            I += np.eye(3) * (0.4 * extra * r * r)
        # guard against degenerate (thin) bodies: enforce minimum inertia ratio
        ev = np.linalg.eigvalsh(I)
        if ev.min() < 1e-9 * max(ev.max(), 1e-9):
            I += np.eye(3) * (1e-6 * total)
        return total, com, I

    def to_dict(self, tier: str = "full"):
        m, com, I = self.inertial(tier)
        return {
            "name": self.name, "parent": self.parent, "pos": [float(p) for p in self.pos], "quat": [float(q) for q in self.quat],
            "joint": None if self.joint is None else self.joint.to_dict(),
            "geoms": [g.to_dict() for g in self.geoms if tier in g.tiers],
            "sites": [s.to_dict() for s in self.sites if tier in s.tiers],
            "tiers": sorted(self.tiers), "semantic": self.semantic, "label": self.label, "static": self.static,
            "mass": m, "com": [float(c) for c in com], "inertia": [[float(x) for x in row] for row in I],
        }


@dataclass
class Equality:
    kind: str                       # joint | connect | weld
    name: str
    a: str                          # joint1 / body1
    b: Optional[str] = None         # joint2 / body2
    polycoeff: tuple = (0, 1, 0, 0, 0)   # for joint: qa = c0 + c1*qb + c2*qb^2 ...
    anchor: tuple = (0, 0, 0)       # connect: anchor in body a frame
    tiers: frozenset = FULL_ONLY
    label: str = ""
    active: bool = True
    solref: Optional[tuple] = None  # explicit constraint material/time response; None preserves exporter default
    solimp: Optional[tuple] = None

    def to_dict(self):
        result = {"kind": self.kind, "name": self.name, "a": self.a, "b": self.b, "polycoeff": list(self.polycoeff),
                "anchor": list(self.anchor), "tiers": sorted(self.tiers), "label": self.label, "active": self.active}
        if self.solref is not None:
            result['solref'] = list(self.solref)
        if self.solimp is not None:
            result['solimp'] = list(self.solimp)
        return result


@dataclass
class Tendon:
    name: str
    sites: list                     # spatial tendon through body sites (body, site)
    range: tuple
    stiffness: float = 0.0
    damping: float = 0.0
    tiers: frozenset = FULL_ONLY
    label: str = ""

    def to_dict(self):
        return {"name": self.name, "sites": list(self.sites), "range": list(self.range), "stiffness": self.stiffness,
                "damping": self.damping, "tiers": sorted(self.tiers), "label": self.label}


@dataclass
class SpatialSpring:
    """Two-site linear extension spring; distinct from joint-coordinate tendons."""
    name: str
    sites: tuple
    stiffness: float
    springlength: float
    damping: float = 0.0
    width: float = 0.004
    tiers: frozenset = ALL_TIERS
    label: str = ""

    def to_dict(self):
        return {"name": self.name, "sites": list(self.sites), "stiffness": self.stiffness,
                "springlength": self.springlength, "damping": self.damping, "width": self.width,
                "tiers": sorted(self.tiers), "label": self.label}


@dataclass
class SpatialCable:
    """Inextensible, tension-only routed cable, with optional pulley wraps.

    Path entries are {'site': name} or {'geom': name, 'sidesite': name}.
    A side site selects the wrapping branch; it is not a cable endpoint.
    """
    name: str
    path: tuple
    max_length: float
    width: float = .002
    tiers: frozenset = ALL_TIERS
    label: str = ""

    def to_dict(self):
        return {"name":self.name,"path":[dict(p) for p in self.path],
                "max_length":self.max_length,"width":self.width,
                "tiers":sorted(self.tiers),"label":self.label}

    def validate_path(self, sites, geoms):
        assert math.isfinite(self.max_length) and self.max_length>0, "cable length must be positive"
        assert math.isfinite(self.width) and self.width>0, "cable width must be positive"
        assert len(self.path)>=2 and 'site' in self.path[0] and 'site' in self.path[-1], "cable needs site endpoints"
        previous_geom=False
        for point in self.path:
            if 'site' in point:
                assert set(point)=={'site'} and point['site'] in sites, "missing/invalid cable site"
                previous_geom=False
            else:
                assert set(point)<= {'geom','sidesite'} and 'geom' in point, "invalid cable path entry"
                assert not previous_geom, "cable wrapping geoms require an intervening site"
                assert point['geom'] in geoms and geoms[point['geom']].type in ('sphere','cylinder'), "invalid cable wrapping geom"
                assert 'sidesite' not in point or point['sidesite'] in sites, "missing cable side site"
                previous_geom=True


@dataclass
class Model:
    name: str
    bodies: list = field(default_factory=list)
    materials: dict = field(default_factory=dict)
    equalities: list = field(default_factory=list)
    tendons: list = field(default_factory=list)
    contact_excludes: list = field(default_factory=list)  # (body1, body2)
    meta: dict = field(default_factory=dict)
    spatial_springs: list = field(default_factory=list)
    spatial_cables: list = field(default_factory=list)

    # ---- helpers ----
    def body(self, name) -> Body:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(name)

    def add_body(self, b: Body) -> Body:
        assert all(x.name != b.name for x in self.bodies), f"duplicate body {b.name}"
        self.bodies.append(b)
        return b

    def add_material(self, m: Material):
        self.materials[m.name] = m
        return m

    def children(self, parent):
        return [b for b in self.bodies if b.parent == parent]

    def joints(self, tier="full"):
        return [b.joint for b in self.bodies if b.joint is not None and tier in b.tiers]

    def bodies_in_tier(self, tier):
        # a body is in a tier if it and all its ancestors are
        out = []
        for b in self.bodies:
            ok = tier in b.tiers
            p = b.parent
            while ok and p is not None:
                pb = self.body(p)
                ok = tier in pb.tiers
                p = pb.parent
            if ok:
                out.append(b)
        return out

    def world_transform(self, body_name):
        """Return (pos, quat) of a body frame in world at q=0."""
        b = self.body(body_name)
        chain = []
        while b is not None:
            chain.append(b)
            b = self.body(b.parent) if b.parent else None
        pos = np.zeros(3)
        quat = QUAT_ID.copy()
        for b in reversed(chain):
            pos = pos + quat_rotate(quat, b.pos)
            quat = quat_mul(quat, np.asarray(b.quat))
        return pos, quat

    def total_mass(self, tier="full", moving_only=True):
        return sum(b.inertial(tier)[0] for b in self.bodies_in_tier(tier) if (not moving_only or not b.static))

    def to_dict(self, tier="full"):
        bodies = self.bodies_in_tier(tier)
        names = {b.name for b in bodies}
        return {
            "name": self.name,
            "tier": tier,
            "bodies": [b.to_dict(tier) for b in bodies],
            "materials": {k: v.to_dict() for k, v in self.materials.items()},
            "equalities": [e.to_dict() for e in self.equalities if tier in e.tiers and e.a in self._joint_names(names) | names],
            "tendons": [t.to_dict() for t in self.tendons if tier in t.tiers],
            **({"spatial_springs": [s.to_dict() for s in self.spatial_springs if tier in s.tiers]}
               if self.spatial_springs else {}),
            **({"spatial_cables": [c.to_dict() for c in self.spatial_cables if tier in c.tiers]}
               if self.spatial_cables else {}),
            "contact_excludes": [list(x) for x in self.contact_excludes if x[0] in names and x[1] in names],
            "meta": self.meta,
        }

    def _joint_names(self, body_names):
        return {b.joint.name for b in self.bodies if b.joint is not None and b.name in body_names}

    def bake_initial(self):
        """Transform geoms/sites of bodies whose joint has initial != modeled_at so that the authored geometry
        corresponds to q = initial (then modeled_at = initial).  Child bodies are unaffected because their frames
        are expressed relative to this body's frame, which is what the joint moves."""
        self.validate_free_joints()
        for b in self.bodies:
            j = b.joint
            if j is not None and j.type == "free":
                continue  # The full rest pose is b.pos/b.quat, not a scalar.
            if j is None or abs(j.initial - j.modeled_at) < 1e-12:
                continue
            dq = j.initial - j.modeled_at
            axis = np.asarray(j.axis, float)
            axis = axis / np.linalg.norm(axis)
            if j.type == "slide":
                dp = axis * dq
                for g in b.geoms:
                    g.pos = tuple(np.asarray(g.pos) + dp)
                for st in b.sites:
                    st.pos = tuple(np.asarray(st.pos) + dp)
            else:
                q = quat_from_axis_angle(axis, dq)
                R = quat_to_mat(q)
                jp = np.asarray(j.pos, float)
                for g in b.geoms:
                    g.pos = tuple(jp + R @ (np.asarray(g.pos) - jp))
                    g.quat = tuple(quat_mul(q, np.asarray(g.quat)))
                for st in b.sites:
                    st.pos = tuple(jp + R @ (np.asarray(st.pos) - jp))
                    st.quat = tuple(quat_mul(q, np.asarray(st.quat)))
                # child bodies attached to this body: their frames rotate with it (they are defined in this body's frame at q=modeled_at)
                for c in self.bodies:
                    if c.parent == b.name:
                        c.pos = tuple(jp + R @ (np.asarray(c.pos) - jp))
                        c.quat = tuple(quat_mul(q, np.asarray(c.quat)))
            j.modeled_at = j.initial
        return self

    def uniquify(self):
        """Ensure geom and site names are globally unique (append _2, _3 ...)."""
        seen = {}
        for b in self.bodies:
            for g in list(b.geoms) + list(b.sites):
                n = g.name
                if n in seen:
                    seen[n] += 1
                    g.name = f"{n}_{seen[n]}"
                else:
                    seen[n] = 1
        return self

    def validate_free_joints(self):
        """Free roots have real inertia and no scalar control/offset semantics."""
        free=set()
        for b in self.bodies:
            j=b.joint
            if j is None or j.type != 'free':continue
            free.add(j.name)
            assert b.parent is None and not b.static, 'free joint requires a dynamic world-root body'
            assert j.range is None and not j.robot_interactive and j.role == 'mechanism', 'free joint cannot be a scalar interactive DOF'
            assert not any(j.pos) and j.initial == j.modeled_at == 0, 'free initial pose belongs to body pos/quat, not scalar offsets'
            assert not any((j.damping,j.frictionloss,j.stiffness,j.springref,j.armature)), 'free root cannot inherit scalar joint forces'
            assert j.limit_solref is None and j.damping_closing is None and j.damping_opening is None and j.backcheck_angle is None and not j.ratchet_one_way, 'free root cannot use scalar mechanism rules'
            mass,_,inertia=b.inertial()
            assert mass>1e-9 and np.isfinite(inertia).all() and np.linalg.eigvalsh(inertia).min()>0, 'free root requires its actual positive body inertia'
            assert len(b.pos)==3 and all(math.isfinite(v)for v in b.pos), 'free root needs a finite position'
            assert len(b.quat)==4 and all(math.isfinite(v)for v in b.quat) and abs(float(np.linalg.norm(b.quat))-1)<1e-6, 'free root needs a unit WXYZ quaternion'
        assert not any(e.kind=='joint' and (e.a in free or e.b in free)for e in self.equalities), 'free joints cannot use scalar equalities'
        assert not any(name in free for t in self.tendons for name,_ in t.sites), 'free joints cannot use scalar fixed tendons'
        assert not any(a.get('joint')in free for a in self.meta.get('actuators',[])), 'free joints cannot use scalar actuators'
        return True

    def validate(self):
        self.validate_free_joints()
        names = set()
        for b in self.bodies:
            assert b.name not in names, f"dup body {b.name}"
            for geom in b.geoms:
                geom.validate_contact_priority()
            names.add(b.name)
            if b.parent is not None:
                assert b.parent in names, f"parent {b.parent} of {b.name} must be defined before child"
        jn = set()
        for j in self.joints():
            assert j.name not in jn, f"dup joint {j.name}"
            jn.add(j.name)
        for equality in self.equalities:
            if equality.solref is not None:
                assert len(equality.solref)==2 and all(math.isfinite(v) for v in equality.solref), 'invalid equality solref'
                assert equality.solref[1]>0, 'equality damping ratio must be positive'
            if equality.solimp is not None:
                assert len(equality.solimp) in (3,4,5) and all(math.isfinite(v) for v in equality.solimp), 'invalid equality solimp'
                assert 0<equality.solimp[0]<=equality.solimp[1]<1 and equality.solimp[2]>0, 'invalid equality impedance'
        spring_names = set()
        for spring in self.spatial_springs:
            assert spring.name not in spring_names, f"duplicate spatial spring {spring.name}"
            spring_names.add(spring.name)
            assert len(spring.sites) == 2 and len(set(spring.sites)) == 2, "spatial spring needs two distinct sites"
            assert all(math.isfinite(v) for v in (spring.stiffness, spring.springlength, spring.damping, spring.width))
            assert spring.stiffness > 0 and spring.springlength >= 0 and spring.damping >= 0 and spring.width > 0
            for tier in spring.tiers:
                available = {s.name for b in self.bodies_in_tier(tier) for s in b.sites if tier in s.tiers}
                assert set(spring.sites) <= available, f"missing spatial spring site in {tier}: {spring.name}"
        tendon_names={t.name for t in self.tendons}|spring_names
        for cable in self.spatial_cables:
            assert cable.name not in tendon_names, f"duplicate spatial cable {cable.name}"
            tendon_names.add(cable.name)
            for tier in cable.tiers:
                bodies=self.bodies_in_tier(tier)
                sites={s.name for b in bodies for s in b.sites if tier in s.tiers}
                geoms={g.name:g for b in bodies for g in b.geoms if tier in g.tiers}
                cable.validate_path(sites,geoms)
        return True
