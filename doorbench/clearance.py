"""Deterministic kinematic clearance gate.

The force-driven QA (qa.py) only ever sees *collision* geometry and only visits the configurations that the test
forces happen to reach.  This gate is geometric and exhaustive instead: every joint of a door is swept through its
full range with ALL geometry made collidable (visual-only parts included, because that is what a viewer shows) and
with MuJoCo's parent-child contact filter disabled, and any interpenetration deeper than a small tolerance is a
failure.  Sweeps:

* ``initial``            - the shipped configuration
* ``open:<leaf joint>``  - each leaf joint through its range with every releasable mechanism in its released state
                           (bolts retracted, hooks lifted, handles turned) - the door must open cleanly
* ``latched:<leaf joint>`` - same sweep with mechanisms at rest; pairs that are *supposed* to block (latch/lock
                           against strike/frame) are ignored, everything else must still clear
* ``mech:<joint>``       - each mechanism joint through its range with the leaf closed (coupled joints follow)
* ``coupling:<joint>``   - every joint equality: the driven joint must be able to follow its driver over the driver's
                           whole range without leaving its own limits (a driven hinge parked on a limit that the
                           coupling pushes against locks the mechanism - the accordion folds of 2026-09)

Hinge knuckles/leaves are allowed a larger overlap (they are mortised into leaf and jamb by design).

The second gate in this module is the RUNNING CLEARANCE gate (``Clearance.run_running``, published as
``checks["running_clearance"]``): the same sweeps, but it measures the *gap* instead of the overlap.  A part
authored EXACTLY touching a static member (0.000 m) passes the penetration gate above - MuJoCo with margin 0
never generates a force for it - while PhysX creates and resolves contacts inside its contact offset (the USD
export sets ``physxCollision:contactOffset`` 5 mm), so the same door jams, drifts or explodes in Isaac Sim; and a
real door does not touch its frame anyway, it runs 3-5 mm clear at the jambs and head, 6-20 mm above the floor and
10-20 mm clear on a revolving/turnstile rotor.  Scope: MOVING geom vs STATIC (world-welded) geom, and only geoms
that are simulated colliders in both engines - visual-only trim cannot jam anything, and its overlap is the
penetration gate's business.  See ``required_gap`` for the per-pair minimum and the seal / bearing / latch
allow-list.
"""
from __future__ import annotations

import fnmatch
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

TOL = 0.002           # m; general interpenetration tolerance
TOL_HINGE = 0.012     # m; hinge hardware overlaps the members it is mortised into
LEAF_ROLES = ("primary", "secondary")
MECH_ROLES = ("operator", "latch", "lock", "mechanism")
BLOCKING = ("latch", "lock")          # semantics that are expected to block a latched leaf against the frame
# pairs that interpenetrate BY DESIGN (a part sliding inside its own housing); models may add more via meta["clearance_allow"]
DEFAULT_ALLOW = [("*pushbutton_geom", "*pushbutton_housing"), ("*_pin_geom", "*_pin_housing"), ("*_pin_geom", "*_pin_bracket"), ("*ring_geom", "*ring_recess"),
                 ("*_bolt_geom", "*_bolt_housing"), ("*_hasp", "*_staple"),
                 # spindles / thumbturn and cylinder stubs pass through the lock body inside the leaf; the rim bolt lives in its case
                 ("*deadbolt_box", "*thumbturn_mesh"), ("*deadbolt_box", "*cylinder_face"), ("*_spindle", "*_bolt_capsule"), ("*_knob_*", "*_bolt_capsule"),
                 ("*_device_case", "*_bolt_capsule"),
                 # a surface hasp plate lies flat across the door/frame joint (frame face modelled as a solid member)
                 ("*_hasp", "jamb_*"), ("*_hasp", "post_*"), ("*_hasp", "stop_*"), ("*_hasp", "seal_*"),
                 ("*thumbturn_mesh", "*_escutcheon_*"), ("*cylinder_face", "*_escutcheon_*"),
                 # a spindle runs through the hole in the rose / escutcheon plate it turns in, and a wall button
                 # presses into the recess of the plate it is mounted on
                 ("*_spindle", "*_escutcheon_*"), ("*_spindle", "*_rose_*"), ("*_spindle", "*handleset*"),
                 ("*_spindle", "*_device_case"), ("*_spindle", "*_backplate_*"), ("rex_button_geom", "rex_plate"),
                 ("call_button_geom", "*_plate"),
                 # a push/pull paddle is a rocker on a fixed pivot pin: the arm root envelops the pin it turns on,
                 # exactly like the hub does (the pin is 4 mm and the neck 8 mm across it)
                 ("*_paddle_neck_*", "*_paddle_pin_*")]
FRAME_LIKE = ("frame", "latch", "lock", "wall", "track")
HARDWARE = ("operator", "latch", "lock", "mechanism", "closer", "track", "hinge")


# ---------------------------------------------------------------------------
# running clearance: what a moving part must keep between itself and static structure
# ---------------------------------------------------------------------------
RUN_MARGIN = 0.025     # m; contact margin the gap scan runs with (must exceed every required minimum below)
RUN_MIN = 0.003        # m; structural running clearance: leaf/panel edge to jamb, head, casing, track (real doors 3-5 mm)
RUN_MIN_FLOOR = 0.006  # m; undercut under a moving leaf / panel (real doors 6-20 mm; 6 mm is a tight carpet-less undercut)
RUN_MIN_ROTOR = 0.010  # m; revolving canopy / turnstile rotor running clearance (real 10-20 mm), via meta["running_clearance_min"]
RUN_EPS = 1e-5         # m; float slack on the comparison, not on the requirement.  mj_geomDistance itself is exact to
#                        well under a micron on these shapes (measured), but a part authored at exactly the minimum
#                        reaches the query as 0.0029999999999999947 after the kinematic chain, so a strict "<" would
#                        make the design value a coin flip.  0.01 mm is four orders above that float noise and four
#                        orders below the 3 mm it guards: nothing physical hides inside it.
# semantics whose members touch or compress BY DESIGN and therefore need no running gap:
#   seal      weatherstrip / brush seal / gasket / sweep - it is *meant* to be in contact
#   hinge     knuckles, pins, pivots, bearings - a bearing surface carries the leaf
#   latch/lock  a bolt seats in its strike / keeper; a hook drops on its keep
#   closer    the closer arm and its foot plate are bolted to the frame
#   mechanism spindles, shafts, drums running in their static housings
#   sensor    wall readers / REX plates the leaf hardware sweeps past on its mount
RUN_TOUCH_SEM = ("seal", "hinge", "latch", "lock", "closer", "mechanism", "sensor")
# structural members: everything here has to run clear of everything else here
RUN_STRUCT_SEM = ("leaf", "glass", "frame", "wall", "floor", "decor", "track", "operator")
# ... except the names below, which are contact faces even though they are modelled as frame/decor:
#   *stop*     the leaf closes ONTO the stop / bumper / floor stop - that contact is the door being shut
#   *bumper*   wall and rail bumpers exist to be hit
#   *threshold*/*sill*/*saddle*  the sill carries the door's sweep seal
#   *_seal*/*gasket*/*brush*/*sweep*/*astragal*  soft parts, whatever semantic they were given
RUN_TOUCH_NAME = ("*stop*", "*bumper*", "*threshold*", "*sill*", "*saddle*", "*seal*", "*gasket*", "*weatherstrip*",
                  "*brush*", "*sweep*", "*astragal*", "*_pad*", "*catch*", "*keeper*", "*strike*", "*boss*", "*bearing*",
                  "*pivot*", "*roller*", "*glide*", "*guide*", "*_shoe*", "*caster*", "*wheel*")


def gate_model(xml_path: str, margin: float = 0.0):
    import mujoco
    spec = mujoco.MjSpec.from_file(xml_path)
    for g in spec.geoms:
        g.contype = 1
        g.conaffinity = 1
        g.margin = margin
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    return spec.compile()


COUPLING_TOL = 1e-3   # rad / m; a driven joint may leave its range by no more than this over the driver's travel


def coupling_range_failures(m, tol: float = COUPLING_TOL, n_samples: int = 49) -> List[dict]:
    """Joint equalities whose driven joint cannot follow its driver over the driver's whole range.

    MuJoCo's joint equality is q_a = qpos0_a + poly(q_b - qpos0_b).  If the image of the driver's range leaves the
    driven joint's own limited range, the joint limit and the equality fight: the pair is locked (or the driver is
    capped short of its range) - a mechanism that looks fine in every kinematic pose and never moves under a push.
    Unlimited drivers are skipped (their image is unbounded by construction)."""
    import mujoco
    out = []
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    for e in range(m.neq):
        if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT) or not m.eq_active0[e]:
            continue
        a, b = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if b < 0 or not m.jnt_limited[a] or not m.jnt_limited[b]:
            continue
        lo_b, hi_b = (float(x) for x in m.jnt_range[b])
        lo_a, hi_a = (float(x) for x in m.jnt_range[a])
        qa0, qb0 = float(m.qpos0[m.jnt_qposadr[a]]), float(m.qpos0[m.jnt_qposadr[b]])
        c = [float(x) for x in m.eq_data[e][:5]]
        xs = np.linspace(lo_b, hi_b, n_samples)
        ys = qa0 + sum(c[k] * (xs - qb0) ** k for k in range(5))
        over = np.maximum(lo_a - ys, ys - hi_a)
        i = int(np.argmax(over))
        if over[i] > tol:
            out.append({"equality": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e), "driven": jname(a), "driver": jname(b),
                        "driver_q": float(xs[i]), "driven_q": float(ys[i]), "driven_range": [lo_a, hi_a], "overshoot": float(over[i])})
    return out


def _joint_info(model_json: dict) -> Dict[str, dict]:
    out = {}
    for b in model_json["bodies"]:
        j = b.get("joint")
        if j:
            out[j["name"]] = dict(j, body=b["name"])
    return out


def _semantics(model_json: dict) -> Dict[str, str]:
    out = {}
    for b in model_json["bodies"]:
        for g in b["geoms"]:
            out[g["name"]] = g.get("semantic", "")
    return out


def _collision(model_json: dict) -> Dict[str, bool]:
    """Which geoms are actually simulated colliders (MJCF contype/conaffinity 1, USD CollisionAPI)."""
    out = {}
    for b in model_json["bodies"]:
        for g in b["geoms"]:
            out[g["name"]] = bool(g.get("collision", True))
    return out


class Clearance:
    def __init__(self, door_dir: str, tier: str = "full"):
        import mujoco
        self.mj = mujoco
        self.dir = door_dir
        xml = os.path.join(door_dir, {"full": "door.xml", "simple": "door_simple.xml", "minimal": "door_minimal.xml"}[tier])
        self.m = gate_model(xml)
        self.d = mujoco.MjData(self.m)
        with open(os.path.join(door_dir, "model.json")) as f:
            mj_ = json.load(f)
        self.meta = mj_["meta"]
        self.contact_preview = None
        self.track_preview = None
        self.turnstile_preview = None
        self.turnstile_drop_held = {}
        self.elevator_released={};self.elevator_locked={}
        self.vault_native_model=mujoco.MjModel.from_xml_path(xml) if self.meta.get('vault_boltwork') else None
        self.rotary_release=None
        if self.meta.get('rotary_locksets'):
            from .rotary_lockset_qa import rotary_release_snapshot
            self.rotary_release=rotary_release_snapshot(mujoco.MjModel.from_xml_path(xml),self.meta)
            if not self.rotary_release['ok']:raise ValueError(f"Rotary catch native preparation failed: {self.rotary_release}")
        self.security_controlled=set();self.security_samples=[]
        self.paired_samples=[]
        if self.meta.get('paired_leaf_holds'):
            from .paired_hold_qa import run_paired_hold_qa
            proof=run_paired_hold_qa(mujoco.MjModel.from_xml_path(xml),self.meta)
            if not proof['ok']:raise ValueError(f"Inactive bolt native prerequisite failed: {proof['failures']}")
            self.paired_samples=proof['inspection_samples']
            if not self.paired_samples:raise ValueError('Inactive bolt proof has no native inspection states')
        if self.meta.get('security_guards'):
            from .security_mechanics_qa import run_security_service_qa
            with open(os.path.join(door_dir,'spec.json')) as f:security_spec=json.load(f)
            closer=security_spec['physics'].get('closer',{})
            proof=run_security_service_qa(mujoco.MjModel.from_xml_path(xml),self.meta,
                opening_preload=closer.get('spring_preload_Nm',0.),
                opening_stiffness=closer.get('spring_stiffness_Nm_per_rad',0.))
            if not proof['ok']:raise ValueError(f"Security native mechanism prerequisite failed: {proof['failures']}")
            self.security_controlled={self.meta['primary_joint'],*(n for r in self.meta['security_guards'] for n in r['guard_joints'])}
            self.security_samples=[sample for row in proof['measurements'] for sample in row['inspection_samples']]
            if not self.security_samples:raise ValueError('Security mechanism proof has no native inspection states')
        if self.meta.get('elevator_interlocks'):
            from .elevator_qa import run_elevator_qa
            proof=run_elevator_qa(mujoco.MjModel.from_xml_path(xml),self.meta)
            if not proof['ok']:raise ValueError(f"Elevator native mechanism prerequisite failed: {proof['failures']}")
            self.elevator_released=proof['released_configuration']
            self.elevator_locked=proof['locked_leaf_positions']
        if self.meta.get('turnstile_locks'):
            from .turnstile_contact_preview import TurnstileContactPreview
            self.turnstile_preview=TurnstileContactPreview(mujoco.MjModel.from_xml_path(xml),self.meta)
        if self.meta.get('turnstile_drop_arm'):
            from .turnstile_drop_qa import run_turnstile_drop_qa
            proof=run_turnstile_drop_qa(mujoco.MjModel.from_xml_path(xml),self.meta)
            if not proof['ok']:
                raise ValueError(f"Turnstile indexed drop/reset prerequisite failed: {proof['failures']}")
            self.turnstile_drop_held=proof['held_configuration']
        self.multipoint_released = {}
        if self.meta.get('multipoint_locks'):
            from .multipoint_qa import run_multipoint_qa
            proof=run_multipoint_qa(mujoco.MjModel.from_xml_path(xml),self.meta)
            if not proof['ok']:
                raise ValueError(f"Multipoint native inspection configuration failed: {proof['failures']}")
            for row in proof['results']:
                self.multipoint_released.update(row['released_joints'])
        if self.meta.get('closer_track_holds'):
            from .closer_track_qa import TrackContactPreview
            self.track_preview = TrackContactPreview(mujoco.MjModel.from_xml_path(xml),self.meta)
        if any(r.get('kind') == 'contact_suffolk' for r in self.meta.get('gate_hardware', [])):
            from .gate_hardware_qa import SuffolkContactPreview
            # Inspection geometry enables visual-only colliders. Passive
            # response must use the original native collision model; settling
            # decorative screws/bearings as solid obstructions would invent jams.
            self.contact_preview = SuffolkContactPreview(mujoco.MjModel.from_xml_path(xml), self.meta)
        self.allow = list(DEFAULT_ALLOW) + [tuple(a[:2]) for a in self.meta.get("clearance_allow", [])]
        # running-clearance exceptions: a pair allowed to interpenetrate is certainly allowed to touch, plus the
        # model's own documented [g1, g2, reason] entries
        self.run_allow = list(self.allow) + [tuple(a[:2]) for a in self.meta.get("running_clearance_allow", [])]
        self.run_min = max(RUN_MIN, float(self.meta.get("running_clearance_min", 0.0)))
        self.locked_shut = False
        try:
            with open(os.path.join(door_dir, "spec.json")) as f:
                sp_ = json.load(f)
            self.locked_shut = bool(sp_["lock"].get("engaged")) and not sp_["lock"].get("robot_side_release", True)
        except Exception:
            pass
        self.joints = _joint_info(mj_)
        self.material_flexures = set(self.meta.get('material_flexure_joints', []))
        strip_controls = (self.meta.get('strip_curtain') or {}).get('controls', [])
        strip_geoms = {body+'_pvc' for row in strip_controls for body in row['segment_bodies']}
        strip_supports = {name for row in strip_controls for name in [row['fixed_tab_geom'], *row['clamp_geoms']]}
        # Flexible material may meet its own bonded tab or touch neighboring
        # tabs/jaws. This changes only the rest-gap requirement, never contacts
        # or penetration checks; floor, jamb, header and rail are not included.
        self.strip_bearing_pairs = {(moving, fixed) for moving in strip_geoms for fixed in strip_supports}
        self.sem = _semantics(mj_)
        self.collide = _collision(mj_)
        m = self.m
        self.jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
        self.gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
        self.bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        # a geom is STATIC when its body is welded to the world (no joint anywhere up the chain)
        self.static = np.asarray(m.body_weldid)[np.asarray(m.geom_bodyid)] == 0
        # the running-clearance gate only looks at simulated colliders: those are the geoms both MuJoCo and PhysX
        # resolve contacts for, and a PhysX contact offset can only bite on a collider.  Visual-only trim (casings,
        # visual weatherstrips) is the penetration gate's business, not this one.
        col = np.array([bool(self.collide.get(n, True)) for n in self.gname])
        self.static_ids = np.flatnonzero(self.static & col)
        self.moving_ids = np.flatnonzero(~self.static & col)
        # bounding radii for the gap prefilter; a plane has rbound 0 but unbounded extent
        self.rbound = np.array(m.geom_rbound, dtype=float)
        self.rbound[np.asarray(m.geom_type) == int(mujoco.mjtGeom.mjGEOM_PLANE)] = 1e6
        # fixed tendons (one-sided couplings): list of (range_lo, [(qadr, coef)])
        self.tendons = []
        for t in range(m.ntendon):
            if m.tendon_limited[t]:
                terms = []
                for w in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
                    if int(m.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_JOINT):
                        terms.append((int(m.jnt_qposadr[m.wrap_objid[w]]), float(m.wrap_prm[w])))
                if terms:
                    self.tendons.append((float(m.tendon_range[t][0]), terms))

    # ---- kinematic helpers -------------------------------------------------------------------------------
    def resolve(self, q: np.ndarray, driven_joint: str | None = None) -> np.ndarray:
        """Apply joint-polynomial equalities and one-sided tendon couplings to make q consistent."""
        m, mujoco = self.m, self.mj
        pair=self.meta.get('paired_leaf_holds',[])
        if pair and driven_joint in {pair[0]['leaf_joint'],*(r['joint'] for r in pair)}:
            # Geometric service fixture: the first leaf exposes the bolts.
            # Native cycles independently prove release/retention; this pose
            # is only used for the complete envelope and mounting sweeps.
            q[m.jnt_qposadr[m.joint(pair[0]['primary_joint']).id]]=.8
            if driven_joint==pair[0]['leaf_joint']:
                for row in pair:
                    q[m.jnt_qposadr[m.joint(row['joint']).id]]=row['nominal_joint_range_m'][1]
        if self.rotary_release is not None:
            for row in self.meta['rotary_locksets']:
                if driven_joint==row['outside_joint']:
                    q[m.jnt_qposadr[m.joint(row['catch_joint']).id]]=self.rotary_release['positions'][row['catch_joint']]
        if self.meta.get("garage_tiltup_linkage"):
            from .geometry.garage_tiltup import resolve_garage_configuration
            resolve_garage_configuration(m, q, self.meta)
        if self.meta.get("hatch_support"):
            from .geometry.hatch_supports import resolve_hatch_configuration
            resolve_hatch_configuration(m, q, self.meta)
        if self.meta.get('sectional_track'):
            from .geometry.sectional import resolve_sectional_configuration
            resolve_sectional_configuration(m, q, self.meta)
        if self.meta.get('marine_dog_linkage'):
            from .geometry.marine_linkage import resolve_marine_configuration
            resolve_marine_configuration(m, q, self.meta)
        if self.meta.get('vault_boltwork'):
            from .geometry.vault_hardware import resolve_vault_configuration
            resolve_vault_configuration(m,q,self.meta)
        for _ in range(2):
            for e in range(m.neq):
                if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT) or not m.eq_active0[e]:
                    continue
                j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
                c = m.eq_data[e][:5]
                if j2 < 0:
                    q[m.jnt_qposadr[j1]] = m.qpos0[m.jnt_qposadr[j1]]+c[0]
                    continue
                x = q[m.jnt_qposadr[j2]]-m.qpos0[m.jnt_qposadr[j2]]
                q[m.jnt_qposadr[j1]] = m.qpos0[m.jnt_qposadr[j1]]+c[0]+c[1]*x+c[2]*x**2+c[3]*x**3+c[4]*x**4
            for lo, terms in self.tendons:
                length = sum(coef * q[adr] for adr, coef in terms)
                if length < lo - 1e-9:
                    # driven joint is the one with the positive unit coefficient (the bolt); push it to satisfy
                    for adr, coef in terms:
                        if coef > 0:
                            q[adr] += (lo - length) / coef
                            break
        if self.contact_preview and driven_joint in self.contact_preview.allowed:
            report = self.contact_preview.resolve(q, driven_joint)
            if not report['ok']:
                raise ValueError(f"Contact-driven inspection pose {driven_joint}: {report['failures']}")
            q[:] = report['qpos']
        if self.meta.get('closer_mounts'):
            from .geometry.closer_mounts import resolve_closer_configuration
            resolve_closer_configuration(m, q, self.meta)
        if self.track_preview:
            report=self.track_preview.resolve(q,driven_joint)
            if not report['ok']:
                raise ValueError(f"Track contact inspection pose {driven_joint}: {report['failures']}")
            q[:]=report['qpos']
        if self.turnstile_preview:
            report=self.turnstile_preview.resolve(q,driven_joint)
            if not report['ok']:
                raise ValueError(f"Turnstile contact inspection pose: {report['failures']}")
            q[:]=report['qpos']
        return q

    def _locked(self, jname: str) -> bool:
        j = self.jid[jname]
        lo, hi = self.m.jnt_range[j]
        return bool(self.m.jnt_limited[j]) and (hi - lo) < 0.006

    def released_qpos(self) -> np.ndarray:
        q = self.m.qpos0.copy()
        for name, info in self.joints.items():
            if name in self.turnstile_drop_held:
                # Arms, catches and release shoe form a contact mechanism.
                # Use its observed manual-reset state for rotation; forcing
                # every coordinate to its upper limit penetrates physical stops.
                continue
            if self.turnstile_preview and name==self.meta['turnstile_locks']['pawl_joint']:
                # A permanent directional pawl is not a release actuator.
                continue
            if name not in self.jid or info.get("role") not in MECH_ROLES or name in self.material_flexures or name in self.multipoint_released or name in self.elevator_released or name in self.security_controlled:
                continue
            j = self.jid[name]
            if self._locked(name) or not self.m.jnt_limited[j]:
                continue
            q[self.m.jnt_qposadr[j]] = self._operating_range(name,*self.m.jnt_range[j])[1]
        for name,value in self.multipoint_released.items():
            q[self.m.jnt_qposadr[self.jid[name]]] = value
        for name,value in self.turnstile_drop_held.items():
            q[self.m.jnt_qposadr[self.jid[name]]] = value
        for name,value in self.elevator_released.items():
            q[self.m.jnt_qposadr[self.jid[name]]] = value
        return self.resolve(q)

    def _operating_range(self,jname,lo,hi):
        for row in self.meta.get('paired_leaf_holds',[]):
            if jname==row['joint']:return tuple(row['nominal_joint_range_m'])
        holder=self.meta.get('ship_holdback')
        if holder:
            if jname==holder['leaf_joint']:
                if not hasattr(self,'_ship_holdback_open_stop'):
                    from .geometry.ship_holdback import first_ship_holdback_stop_angle
                    self._ship_holdback_open_stop=first_ship_holdback_stop_angle(self.m,self.meta)['angle_rad']
                return 0.,self._ship_holdback_open_stop
            if jname==holder['hook_joint']:
                return tuple(holder['inspection_hook_range_rad'])
        for row in self.meta.get('rotary_locksets',[]):
            if jname==row['catch_joint']:return 0.,row['catch_stroke_m']
        if self.meta.get('vault_boltwork'):
            if jname==self.meta['primary_joint']:return tuple(self.meta['vault_primary_nominal_range'])
            for row in self.meta['vault_boltwork']['groups']:
                if jname==row['operator_joint']:return tuple(row['operator_nominal_range'])
        for row in self.meta.get('elevator_interlocks',{}).get('leaves',[]):
            if row['joint']==jname:return 0.,row['stroke_m']
        row=self.meta.get('turnstile_locks')
        if row and jname==row['rotor_joint'] and row['one_way']:return 0.,2*math.pi
        return lo,hi

    def _turnstile_steps(self,jname,n_steps):
        row=self.meta.get('turnstile_locks')
        if n_steps and row and row['rotor_joint']==jname:
            # Four samples per tooth avoid the former 15/30-degree grids
            # repeatedly missing a 10-degree ratchet's intermediate phases.
            return max(n_steps,4*len(row['ratchet_teeth']))
        return n_steps

    def _latched_range(self,jname,base,lo,hi):
        if self.vault_native_model is not None and jname==self.meta['primary_joint']:
            from .vault_hardware_qa import first_vault_contact_angle
            report=first_vault_contact_angle(self.vault_native_model,self.meta)
            if not report['ok']:raise ValueError(f"Vault has no actual initial bolt arrest: {report}")
            return 0.,report['angle_rad']
        if jname in self.elevator_locked:return 0.,self.elevator_locked[jname]
        row=self.meta.get('turnstile_locks')
        if not row or jname!=row['rotor_joint']:return lo,hi
        from .turnstile_contact_preview import first_turnstile_contact_angle
        for direction in (-1.,1.):
            locked=row['credential_locked_by_default'];reverse=row['one_way'] and direction<0
            if not (locked or reverse):continue
            result=first_turnstile_contact_angle(self.turnstile_preview.model,self.meta,base,
                direction=direction,bolt=locked,pawl=reverse)
            if not result['ok']:raise ValueError(f"Turnstile has no valid physical arrest: {result}")
            if direction<0:lo=result['angle_rad']
            else:hi=result['angle_rad']
        return lo,hi

    def contacts(self, q: np.ndarray, tol_fn) -> List[Tuple[str, str, float]]:
        m, d, mujoco = self.m, self.d, self.mj
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = self.gname[c.geom1], self.gname[c.geom2]
            tol = tol_fn(g1, g2)
            if c.dist < -tol:
                out.append((g1, g2, float(c.dist)))
        return out

    def _ancestor(self, b_desc: int, b_anc: int) -> bool:
        m = self.m
        b = b_desc
        for _ in range(m.nbody):
            b = int(m.body_parentid[b])
            if b == b_anc:
                return True
            if b == 0:
                return False
        return False

    def tol_for(self, g1: str, g2: str, ignore_blocking: bool = False) -> float:
        m = self.m
        s1, s2 = self.sem.get(g1, ""), self.sem.get(g2, "")
        for pa, pb in self.allow:
            if (fnmatch.fnmatch(g1, pa) and fnmatch.fnmatch(g2, pb)) or (fnmatch.fnmatch(g1, pb) and fnmatch.fnmatch(g2, pa)):
                return 1e9
        if s1 == "hinge" or s2 == "hinge":
            return TOL_HINGE
        b1, b2 = int(m.geom_bodyid[m.geom(g1).id]), int(m.geom_bodyid[m.geom(g2).id])
        # mortised hardware lives inside the leaf it is mounted on (bolt in its mortise, spindle through the door)
        for sh, sl, bh, bl in ((s1, s2, b1, b2), (s2, s1, b2, b1)):
            if sh in HARDWARE and sl in ("leaf", "glass") and self._ancestor(bh, bl):
                return 1e9
        if ignore_blocking and b1 != b2 and (s1 in BLOCKING or s2 in BLOCKING):
            return 1e9
        return TOL

    # ---- running clearance (moving part vs static structure) --------------------------------------------
    def required_gap(self, gm: str, gs: str) -> float:
        """Running clearance (m) required between MOVING geom ``gm`` and STATIC geom ``gs``; 0.0 = may touch.

        The allow-list is driven by geom semantics (a seal is meant to be squashed, a bearing is meant to carry the
        leaf, a bolt is meant to seat in its strike), by the names of the parts that are contact faces whatever
        semantic they carry (stops, bumpers, thresholds, rollers/glides in their tracks), and by the model's own
        ``meta["running_clearance_allow"]`` entries ``[geom_a, geom_b, reason]``.  Everything structural that is
        left over is a running fit and needs a real gap."""
        for row in self.meta.get('lock_stock',[]):
            bolts=set(row.get('bolt_geoms',[]));guides=set(row.get('guide_geoms',[]))
            if (gm in bolts and gs in guides) or (gs in bolts and gm in guides):
                return .00075  # measured internal cartridge fit; no penetration exemption
        if (gm,gs) in self.strip_bearing_pairs:
            return 0.
        for pa, pb in self.run_allow:
            if (fnmatch.fnmatch(gm, pa) and fnmatch.fnmatch(gs, pb)) or (fnmatch.fnmatch(gm, pb) and fnmatch.fnmatch(gs, pa)):
                return 0.0
        sm, ss = self.sem.get(gm, ""), self.sem.get(gs, "")
        if sm in RUN_TOUCH_SEM or ss in RUN_TOUCH_SEM:
            return 0.0
        if sm not in RUN_STRUCT_SEM or ss not in RUN_STRUCT_SEM:
            return 0.0
        for n in (gm, gs):
            if any(fnmatch.fnmatch(n, p) for p in RUN_TOUCH_NAME):
                return 0.0
        if sm == "track" and ss == "track":
            return 0.0          # running gear (roller / hanger / glide) rides in its own rail by design
        if sm == "floor" or ss == "floor":
            return RUN_MIN_FLOOR
        return self.run_min

    def gaps(self, q: np.ndarray) -> List[Tuple[str, str, float]]:
        """(moving geom, static geom, signed distance) for every moving-vs-static pair within ``RUN_MARGIN``.

        ``mj_geomDistance`` (libccd/GJK) rather than the contact set: a contact is only *generated* when the
        narrow-phase routine for that shape pair converges, and exactly-touching coaxial faces - precisely the
        defect this gate hunts - are the case it can miss (two full-height turnstiles whose rotor column ends flush
        on the cage roof produced no contact at all at margin 25 mm, while the other eight did).  A distance query
        has no such blind spot.  A cheap bounding-sphere prefilter keeps it to the pairs that can matter."""
        m, d, mujoco = self.m, self.d, self.mj
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        pos = np.asarray(d.geom_xpos)
        out = []
        for gi in self.moving_ids:
            sep = np.linalg.norm(pos[self.static_ids] - pos[gi], axis=1) - self.rbound[self.static_ids] - self.rbound[gi]
            for gj in self.static_ids[sep < RUN_MARGIN]:
                dist = float(mujoco.mj_geomDistance(m, d, int(gi), int(gj), RUN_MARGIN, None))
                if dist < RUN_MARGIN:
                    out.append((self.gname[gi], self.gname[int(gj)], dist))
        return out

    def run_running(self, n_steps: int = 12, record_all: bool = False) -> dict:
        """Sweep every leaf joint and measure the smallest gap each moving-vs-static pair ever reaches.

        A pair that comes closer than ``required_gap`` - at rest or anywhere in the travel - is a failure: MuJoCo
        with margin 0 shrugs at a 0.000 m touch, PhysX resolves it inside its contact offset and the door jams,
        drifts or explodes, and a real door has running clearance there."""
        m = self.m
        best: Dict[Tuple[str, str], Tuple[float, str, float]] = {}

        def record(config: str, qv: float, q: np.ndarray):
            for gm, gs, dist in self.gaps(q):
                key = (gm, gs)
                if key not in best or dist < best[key][0]:
                    best[key] = (dist, config, float(qv))

        base = m.qpos0.copy()
        record("rest", 0.0, self.resolve(base.copy()))
        for sample in self.security_samples:
            record('native_security:'+sample['phase'],sample['time_s'],np.asarray(sample['qpos']))
        for sample in self.paired_samples:
            record('native_inactive_bolts:'+sample['phase'],sample['time'],np.asarray(sample['qpos']))
        released = self.released_qpos()
        leaf_joints = [n for n, j in self.joints.items() if j.get("role") in LEAF_ROLES and n in self.jid and n not in self.material_flexures and n not in self.security_controlled]
        for jn in leaf_joints:
            j = self.jid[jn]
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-math.pi, math.pi))
            lo,hi=self._operating_range(jn,lo,hi)
            if hi - lo < 1e-6:
                continue
            steps=self._turnstile_steps(jn,n_steps)
            for k in range(steps + 1):
                qv = lo + (hi - lo) * k / steps if steps else lo   # n_steps=0: the closed pose only
                q = released.copy()
                q[m.jnt_qposadr[j]] = qv
                record(f"open:{jn}", qv, self.resolve(q,driven_joint=jn))
        pairs, fails = [], []
        for (gm, gs), (dist, config, qv) in best.items():
            need = self.required_gap(gm, gs)
            rec = {"moving": gm, "static": gs, "gap": round(dist, 5), "required": need, "config": config, "q": round(qv, 4),
                   "sem": [self.sem.get(gm, ""), self.sem.get(gs, "")],
                   "bodies": [self.bname[m.geom_bodyid[m.geom(gm).id]], self.bname[m.geom_bodyid[m.geom(gs).id]]]}
            if record_all:
                pairs.append(rec)
            if (need > 0 and dist < need - RUN_EPS) or ((gm,gs) in self.strip_bearing_pairs and dist < -1e-6):
                fails.append(rec)
        fails.sort(key=lambda f: f["gap"] - f["required"])
        out = {"ok": not fails, "n_failures": len(fails), "failures": fails[:40], "n_pairs": len(best)}
        if self.material_flexures:
            out['scope'] = 'Initial structural gaps; material bending travel requires native strip_mechanics proof.'
        if self.security_samples:
            out['security_scope']='Structural gaps sampled from successful native retention/release/reinsertion cycles; no independent chain or secured-leaf position sweep.'
            out['security_native_samples']=len(self.security_samples)
        if record_all:
            out["pairs"] = sorted(pairs, key=lambda p: p["gap"])
        return out

    # ---- the gate ---------------------------------------------------------------------------------------
    def run(self, n_steps: int = 24) -> dict:
        m = self.m
        failures: Dict[str, dict] = {}

        def record(config: str, jname: str, qv: float, cons):
            for g1, g2, dist in cons:
                key = tuple(sorted((g1, g2)))
                depth = -dist
                prev = failures.get(key)
                if prev is None or depth > prev["depth"]:
                    failures[key] = {"geoms": list(key), "depth": round(depth, 4), "config": config, "joint": jname, "q": round(float(qv), 4),
                                     "bodies": [self.bname[m.geom_bodyid[self.m.geom(g1).id]], self.bname[m.geom_bodyid[self.m.geom(g2).id]]]}

        base = m.qpos0.copy()
        record("initial", "", 0.0, self.contacts(self.resolve(base.copy()), lambda a, b: self.tol_for(a, b)))
        for sample in self.security_samples:
            record('native_security:'+sample['phase'],'',sample['time_s'],
                   self.contacts(np.asarray(sample['qpos']),lambda a,b:self.tol_for(a,b)))
        for sample in self.paired_samples:
            record('native_inactive_bolts:'+sample['phase'],'',sample['time'],
                   self.contacts(np.asarray(sample['qpos']),lambda a,b:self.tol_for(a,b)))
        if self.turnstile_preview:
            settled=self.turnstile_preview.default_qpos(base)
            if not settled['ok']:raise ValueError(f"Turnstile default electrical state does not settle: {settled['failures']}")
            base=np.asarray(settled['qpos'])
        released = self.released_qpos()
        leaf_joints = [n for n, j in self.joints.items() if j.get("role") in LEAF_ROLES and n in self.jid and n not in self.material_flexures and n not in self.security_controlled]
        mech_joints = [n for n, j in self.joints.items() if j.get("role") in MECH_ROLES and n in self.jid and n not in self.material_flexures]
        coupled_shoes={row.get('shoe_joint') for row in self.meta.get('closer_mounts',[])}
        mech_joints=[name for name in mech_joints if name not in coupled_shoes]
        # A cam follower cannot be swept independently through its rail.
        # These inputs and followers were tested together over two complete
        # native cycles above; missing pins/interlocks fail that prerequisite.
        mech_joints=[name for name in mech_joints if name not in self.multipoint_released]
        mech_joints=[name for name in mech_joints if name not in self.turnstile_drop_held]
        mech_joints=[name for name in mech_joints if name not in self.elevator_released]
        mech_joints=[name for name in mech_joints if name not in self.security_controlled]
        marine=self.meta.get('marine_dog_linkage',{})
        marine_followers={marine.get('output_joint'),*marine.get('dog_joints',[]),*marine.get('rod_joints',[])}
        mech_joints=[name for name in mech_joints if name not in marine_followers]
        vault=self.meta.get('vault_boltwork',{}).get('groups',[])
        vault_operators={row['operator_joint'] for row in vault}
        vault_followers={row[key] for row in vault for key in ('input_joint','carrier_joint','rod_joint')}-vault_operators
        mech_joints=[name for name in mech_joints if name not in vault_followers]
        for jn in leaf_joints:
            j = self.jid[jn]
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-math.pi, math.pi))
            lo,hi=self._operating_range(jn,lo,hi)
            if hi - lo < 1e-6:
                continue
            latched_lo, latched_hi = lo, hi
            latched_lo,latched_hi=self._latched_range(jn,base,latched_lo,latched_hi)
            if any(jn==row['leaf_joint'] for row in self.meta.get('paired_leaf_holds',[])):
                # Engaged bolts physically prevent this sweep. Their loaded
                # arrest was measured above; retain the authored closed check.
                latched_lo=latched_hi=0.
            if jn == self.meta.get('primary_joint') and any(r.get('kind') == 'gravity_fork' for r in self.meta.get('gate_hardware', [])):
                from .gate_hardware_qa import first_fork_contact_angle
                if hi > .01:
                    contact = first_fork_contact_angle(m, self.meta, direction=1.)
                    if not contact['ok']:
                        raise ValueError(f"Fork has no valid opening arrest: {contact}")
                    latched_hi = min(hi, contact['contact_angle_rad'])
                if lo < -.01:
                    contact = first_fork_contact_angle(m, self.meta, direction=-1.)
                    if not contact['ok']:
                        raise ValueError(f"Fork has no valid reverse arrest: {contact}")
                    latched_lo = max(lo, contact['contact_angle_rad'])
            steps=self._turnstile_steps(jn,n_steps)
            for k in range(steps + 1):
                qv = lo + (hi - lo) * k / steps if steps else lo   # n_steps=0: the closed pose only
                q = released.copy()
                q[m.jnt_qposadr[j]] = qv
                record(f"open:{jn}", jn, qv, self.contacts(self.resolve(q,driven_joint=jn), lambda a, b: self.tol_for(a, b, ignore_blocking=self.locked_shut)))
                q = base.copy()
                q_latched = latched_lo + (latched_hi-latched_lo)*k/steps if steps else latched_lo
                q[m.jnt_qposadr[j]] = q_latched
                record(f"latched:{jn}", jn, q_latched, self.contacts(self.resolve(q), lambda a, b: self.tol_for(a, b, ignore_blocking=True)))
        for jn in mech_joints:
            j = self.jid[jn]
            if not m.jnt_limited[j]:
                continue
            lo, hi = m.jnt_range[j]
            lo,hi=self._operating_range(jn,lo,hi)
            if hi - lo < 1e-6:
                continue
            for k in range(1, 13):
                qv = lo + (hi - lo) * k / 12
                q = base.copy()
                q[m.jnt_qposadr[j]] = qv
                support = self.meta.get('hatch_support')
                if support and jn == support.get('support_release_joint'):
                    # This pin is withdrawn/engaged at the fully extended
                    # stay slot. A closed lid deliberately blocks insertion.
                    q[m.jnt_qposadr[self.jid['hatch_hinge']]] = support['nominal_angle_rad']
                record(f"mech:{jn}", jn, qv, self.contacts(self.resolve(q, driven_joint=jn), lambda a, b: self.tol_for(a, b)))
        # couplings: a driven joint parked on a limit that its equality pushes against is a locked mechanism
        for c in coupling_range_failures(m):
            failures[("coupling", c["driven"])] = {"geoms": [c["driven"], c["driver"]], "depth": round(c["overshoot"], 4), "config": f"coupling:{c['driven']}",
                                                  "joint": c["driver"], "q": round(c["driver_q"], 4), "bodies": [self.bname[m.jnt_bodyid[self.jid[c["driven"]]]], self.bname[m.jnt_bodyid[self.jid[c["driver"]]]]],
                                                  "coupling": c}
        fails = sorted(failures.values(), key=lambda f: -f["depth"])
        result = {"ok": len(fails) == 0, "n_failures": len(fails), "failures": fails[:40], "leaf_joints": leaf_joints, "mech_joints": mech_joints}
        if self.multipoint_released:
            result['multipoint_scope']='Two bounded native operating cycles validate coupled hardware travel; leaf sweeps use the observed unlocked, depressed configuration.'
        if self.turnstile_drop_held:
            result['turnstile_drop_scope']='All three indexed drop/reset mechanisms pass two native cycles first; rotor sweeps retain the observed manually-reset arm/catch/shoe state.'
        if self.elevator_released:
            result['elevator_scope']='Two complete native cam/hook/leaf cycles and removed-contact controls pass first; geometric sweeps use actual rail travel and measured released/locked configurations.'
        if self.security_samples:
            result['security_scope']='All native steps pass the service penetration gate; full inspection geometry uses10Hz and exact phase-end states from those cycles. Initial stock and unrelated mechanism strokes are checked separately.'
            result['security_native_samples']=len(self.security_samples)
        if self.material_flexures:
            result['scope'] = 'Initial geometry and rigid joints; material bending travel requires native strip_mechanics proof.'
        return result


def run_clearance(door_dir: str, tier: str = "full", n_steps: int = 24, run_steps: int = 12) -> dict:
    """Both geometric gates off one compiled model: interpenetration (``run``) and running clearance (``run_running``).

    The running-clearance result is nested under ``["running"]`` and surfaced by qa.py as ``checks["running_clearance"]``."""
    try:
        c = Clearance(door_dir, tier)
        out = c.run(n_steps)
        out["running"] = c.run_running(run_steps)
        return out
    except Exception as e:  # a gate that cannot run is a failure, not a pass
        err = f"{type(e).__name__}: {e}"
        return {"ok": False, "n_failures": 1, "failures": [{"geoms": [], "depth": 0.0, "config": "error", "joint": "", "q": 0.0, "bodies": [], "error": err}],
                "running": {"ok": False, "n_failures": 1, "n_pairs": 0, "failures": [{"moving": "", "static": "", "gap": 0.0, "required": 0.0, "config": "error", "q": 0.0, "sem": [], "bodies": [], "error": err}]}}


def run_running_clearance(door_dir: str, tier: str = "full", n_steps: int = 12, record_all: bool = False) -> dict:
    try:
        return Clearance(door_dir, tier).run_running(n_steps, record_all=record_all)
    except Exception as e:
        return {"ok": False, "n_failures": 1, "n_pairs": 0,
                "failures": [{"moving": "", "static": "", "gap": 0.0, "required": 0.0, "config": "error", "q": 0.0, "sem": [], "bodies": [], "error": f"{type(e).__name__}: {e}"}]}
