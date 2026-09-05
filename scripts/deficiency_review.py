#!/usr/bin/env python
"""Deficiency review agent: one command that runs every deterministic gate over the dataset (or a selection), renders
every selected door in three states from three cameras into small contact sheets, and writes a Markdown report a
human or an LLM reviewer can scan for the "obvious deficiency" categories (floating parts, interpenetration,
mechanisms that do not actuate, missing hardware parts, flipped / mirrored parts, wrong-face hardware, wrongly scaled
parts, duplicate parts, degenerate geoms).  See docs/REVIEW_AGENT.md.

  python scripts/deficiency_review.py                                  # whole dataset -> docs/review/deficiency, docs/DEFICIENCY_REVIEW.md
  python scripts/deficiency_review.py --doors db0012_swing_single      # one door, large sheet
  python scripts/deficiency_review.py --families gate_swing,rollup     # families
  python scripts/deficiency_review.py --models norton_1600,slide_bolt  # every door using these hardware models
  python scripts/deficiency_review.py --doors db0012_swing_single --tag before --no-md   # "before" snapshot of an old build

Gates (deterministic, re-run here): clearance (doorbench/clearance.py), attachment (doorbench/attachment.py).  The
mass gate and the force-driven physics QA are read from each door's qa.json (written by generate_dataset.py); pass
--rerun-qa to re-simulate them.

Renders (MuJoCo offscreen, collision proxies hidden, kinematic loops solved so closer arms follow the door):
  states   closed (rest) | 45 % open (mechanisms at rest) | open (leaf at its limit, mechanisms released / actuated)
  cameras  iso (whole door, robot side) | structure (hinge line / track close-up) | hardware (operator / latch close-up)
  output   <out>/family_<family>_pNN.jpg   every door of the family, 3x3 mini sheet per door
           <out>/<door_id>[_<tag>].jpg     large 3x3 sheet for offenders and explicitly selected doors
           <out>/index.json                per-door gate results + sheet paths
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from doorbench.attachment import Attachment, run_attachment  # noqa: E402
from doorbench.clearance import run_clearance  # noqa: E402

STATES = ("closed", "open45", "open")
CAMS = ("iso", "structure", "hardware")
# "obvious deficiency" categories -> the rules / checks that detect them (also the checklist in the report)
CATEGORIES = [
    ("Floating parts", "attachment: detached, detached_in_motion, static_floating, intra_body", "a part hangs in the air, a body is not on its parent, hinges / hangers do not carry the leaf through the swing, an island of geoms is not connected to its body"),
    ("Interpenetration", "clearance (every joint swept, all geometry collidable)", "parts pass through each other in any configuration"),
    ("Mechanisms not actuating", "attachment: no_actuation, loop_open; qa: actuate_opens, latch_returns, relatch, closer_returns", "a joint never moves when its driver moves, a linkage cannot follow the door, the operator does not release the latch"),
    ("Missing hardware parts", "attachment: no_keeper; hardware review (docs/HARDWARE_REVIEW.md)", "an engaged bolt / hook has nothing capturing it; a lock 'that clearly does not exist'"),
    ("Flipped / mirrored parts", "attachment: degenerate/flipped_mesh", "an asymmetric mesh (handleset, knocker, hook, numbers, ring) mounted upside-down or sideways"),
    ("Wrong-face hardware", "attachment: degenerate/wrong_face", "keypad / card reader on the inside face"),
    ("Wrongly scaled parts", "attachment: degenerate/mesh_extent, proxy_mismatch", "a mesh with an implausible bounding box; a collision proxy far from its visual part"),
    ("Duplicate parts", "attachment: degenerate/duplicate", "two identical geoms at the same pose"),
    ("Degenerate geoms", "attachment: degenerate/zero_size, no_material, massive_empty_body", "zero-size geoms, undefined materials, massive bodies without geometry"),
    ("Mass / physics", "qa: mass, settle, hold, free_opens, actuate_opens, locked_holds, closer_returns, urdf/usd", "simulated mass off spec, door does not hold / open / relatch, exports do not load"),
]


# ------------------------------------------------------------------------------------------------------------------
def select_doors(man: dict, doors: str, families: str, models: str) -> list:
    rows = [d for d in man["doors"] if not d.get("error")]
    if doors:
        want = set(doors.split(","))
        rows = [d for d in rows if d["id"] in want]
    if families:
        want = set(families.split(","))
        rows = [d for d in rows if d["family"] in want]
    if models:
        want = set(models.split(","))
        rows = [d for d in rows if any(d.get(k) in want for k in ("operator", "latch", "lock", "closer", "hinge"))]
    return rows


# ------------------------------------------------------------------------------------------------------------------
class DoorScene:
    """Kinematics (couplings + loop closure) and cameras for one door, on the gate model (all geoms present)."""

    def __init__(self, door_dir: str):
        import mujoco
        self.mujoco = mujoco
        self.att = Attachment(door_dir, "full")
        self.m, self.d = self.att.m, self.att.d
        self.meta = self.att.meta
        self.spec = self.att.spec or {}
        self.u = float(self.meta.get("u", 1.0) or 1.0)
        self.v = float(self.meta.get("v", 1.0) or 1.0)
        self.horizontal = bool(self.meta.get("horizontal"))
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        for g in (0, 1, 2):
            self.opt.geomgroup[g] = 1

    # ---- configurations ---------------------------------------------------------------------------------------
    def _leaf_joint(self):
        pj = self.meta.get("primary_joint")
        return self.att.jid.get(pj) if pj else None

    def q_state(self, state: str) -> np.ndarray:
        m, att = self.m, self.att
        q = m.qpos0.copy()
        j = self._leaf_joint()
        if state == "open":
            # released mechanisms (bolts withdrawn, handles turned) like the clearance gate's open sweep
            for name, info in att.joints.items():
                jj = att.jid.get(name)
                if jj is None or info.get("role") not in ("operator", "latch", "lock", "mechanism") or name in att.loop_joint_names:
                    continue
                if att._locked(jj) or not m.jnt_limited[jj]:
                    continue
                q[m.jnt_qposadr[jj]] = m.jnt_range[jj][1]
        if j is not None and state != "closed":
            frac = 0.45 if state == "open45" else 1.0
            if m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]
                q[m.jnt_qposadr[j]] = lo + frac * (hi - lo)
            else:
                q[m.jnt_qposadr[j]] = frac * 1.2
        return att.pose(q)

    # ---- cameras ------------------------------------------------------------------------------------------------
    def camera(self, cam: str):
        m, d, mujoco = self.m, self.d, self.mujoco
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        u = self.u
        ext = float(self.meta.get("scene_extent", 1.5))
        target = np.array([float(self.meta.get("cam_target_x", 0.0)), float(self.meta.get("wall_y", 0.0)), float(self.meta.get("cam_target_z", 1.0))])
        if cam == "iso":
            c.lookat[:] = target
            c.distance = 2.4 * ext
            c.azimuth, c.elevation = (90.0 - 35.0 * u, -24.0) if not self.horizontal else (60.0, -50.0)
        elif cam == "structure":
            j = self._leaf_joint()
            if j is not None:
                anchor = d.xanchor[j].copy()
                if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
                    # sliding: the track / hangers along the top edge
                    look = np.array([anchor[0], anchor[1], float(self.spec.get("opening", {}).get("height", 2.0)) * 0.92])
                    c.distance, c.azimuth, c.elevation = 1.1, 90.0 - 25.0 * u, -12.0
                else:
                    look = np.array([anchor[0], anchor[1], target[2] * 0.75])
                    c.distance, c.azimuth, c.elevation = 1.0, 90.0 - 55.0 * u, -8.0
            else:
                look, c.distance, c.azimuth, c.elevation = target, 1.4, 60.0, -15.0
            if self.horizontal:
                c.azimuth, c.elevation = 60.0, -45.0
            c.lookat[:] = look
        else:
            t = self.meta.get("handle_cam_target") or target.tolist()
            c.lookat[:] = np.array(t, float)
            c.distance = 0.6
            c.azimuth, c.elevation = (90.0 - 30.0 * u, -14.0) if not self.horizontal else (90.0, -55.0)
        return c

    def render(self, renderer, state: str, cam: str):
        self.d.qpos[:] = self.q_state(state)
        self.mujoco.mj_forward(self.m, self.d)
        renderer.update_scene(self.d, camera=self.camera(cam), scene_option=self.opt)
        return renderer.render()


def _label(img, text):
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, img.size[0], 11], fill=(0, 0, 0))
    dr.text((2, 0), text, fill=(255, 255, 255))
    return img


def render_door_sheet(door_dir: str, out_path: str, cell=(120, 90), quality=60, title: str = "") -> str:
    """3 states x 3 cameras contact sheet for one door (rows = states, columns = cameras)."""
    import mujoco
    from PIL import Image
    sc = DoorScene(door_dir)
    w, h = max(cell[0], 160), max(cell[1], 120)
    r = mujoco.Renderer(sc.m, height=h, width=w)
    sheet = Image.new("RGB", (3 * cell[0], 3 * cell[1] + 12), (24, 24, 24))
    _label(sheet, title or os.path.basename(door_dir))
    for i, state in enumerate(STATES):
        for k, cam in enumerate(CAMS):
            try:
                img = Image.fromarray(sc.render(r, state, cam)).resize(cell)
            except Exception:
                img = Image.new("RGB", cell, (80, 0, 0))
            _label(img, f"{state} / {cam}")
            sheet.paste(img, (k * cell[0], 12 + i * cell[1]))
    r.close()
    sheet.save(out_path, quality=quality, optimize=True)
    return out_path


# ------------------------------------------------------------------------------------------------------------------
def gate_one(args):
    door_dir, rerun_qa = args
    did = os.path.basename(door_dir)
    out = {"id": did}
    try:
        cl = run_clearance(door_dir)
        out["clearance"] = {"ok": cl["ok"], "n": cl["n_failures"], "failures": cl["failures"][:5]}
    except Exception as e:
        out["clearance"] = {"ok": False, "n": 1, "failures": [{"error": str(e)[:200]}]}
    try:
        at = run_attachment(door_dir)
        out["attachment"] = {"ok": at["ok"], "ok_door": at.get("ok_door", at["ok"]), "n": at["n_findings"], "n_closer": at.get("n_closer_findings", 0), "counts": at["counts"],
                             "findings": at["findings"][:12], "joints_not_ok": [k for k, v in at.get("joints", {}).items() if not v.get("ok", True)]}
    except Exception as e:
        out["attachment"] = {"ok": False, "ok_door": False, "n": 1, "n_closer": 0, "counts": {"error": 1}, "findings": [{"rule": "error", "why": str(e)[:200], "domain": "door"}], "joints_not_ok": []}
    qa = {}
    try:
        with open(os.path.join(door_dir, "qa.json")) as f:
            qa = json.load(f)
    except Exception:
        pass
    if rerun_qa:
        try:
            from doorbench.qa import run_qa
            spec = json.load(open(os.path.join(door_dir, "spec.json")))
            meta = json.load(open(os.path.join(door_dir, "model.json")))["meta"]
            files = {"mjcf": {t: os.path.join(door_dir, fn) for t, fn in (("full", "door.xml"), ("simple", "door_simple.xml"), ("minimal", "door_minimal.xml"))}}
            qa = run_qa(spec, door_dir, meta, files, spec["physics"])
        except Exception as e:
            qa = {"checks": {"qa_error": False}, "metrics": {"error": str(e)[:200]}}
    checks = qa.get("checks", {})
    out["qa_failed"] = sorted(k for k, v in checks.items() if not v and k not in ("attachment", "attachment_closer", "clearance"))
    out["mass_ok"] = checks.get("mass", None)
    out["signed_off"] = qa.get("signed_off")
    return out


def render_one(args):
    door_dir, out_path, cell, quality, title = args
    try:
        return render_door_sheet(door_dir, out_path, cell, quality, title)
    except Exception as e:
        return f"ERROR {os.path.basename(door_dir)}: {e}"


# ------------------------------------------------------------------------------------------------------------------
def write_report(path: str, res: list, rows: list, sheets: dict, family_sheets: dict, out_rel: str, total_bytes: int, args):
    fam = {d["id"]: d["family"] for d in rows}
    n = len(res)
    by_rule = collections.Counter()
    by_rule_doors = collections.defaultdict(set)
    by_rule_fam = collections.defaultdict(collections.Counter)
    closer_by_rule = collections.Counter()
    qa_fail = collections.Counter()
    per_fam = collections.defaultdict(lambda: {"doors": 0, "clean": 0, "clearance": 0, "attachment": 0, "closer": 0, "qa": 0})
    offenders = []
    for r in res:
        f_ = fam[r["id"]]
        pf = per_fam[f_]
        pf["doors"] += 1
        a, c = r["attachment"], r["clearance"]
        for fd in a["findings"]:
            key = fd["rule"] + (f"/{fd['kind']}" if fd.get("kind") else "")
            if fd.get("domain") == "closer":
                closer_by_rule[key] += 1
            else:
                by_rule[key] += 1
                by_rule_doors[key].add(r["id"])
                by_rule_fam[key][f_] += 1
        if not c["ok"]:
            by_rule["clearance"] += 1
            by_rule_doors["clearance"].add(r["id"])
            by_rule_fam["clearance"][f_] += 1
            pf["clearance"] += 1
        if not a["ok_door"]:
            pf["attachment"] += 1
        if a["n_closer"]:
            pf["closer"] += 1
        for k in r["qa_failed"]:
            qa_fail[k] += 1
        if r["qa_failed"]:
            pf["qa"] += 1
        clean = c["ok"] and a["ok_door"] and not r["qa_failed"]
        if clean:
            pf["clean"] += 1
        score = (0 if c["ok"] else c["n"]) + (a["n"] - a["n_closer"]) + len(r["qa_failed"])
        if score > 0:
            offenders.append((score, r["id"]))
    offenders.sort(key=lambda t: (-t[0], t[1]))
    n_clean_all = sum(1 for r in res if r["clearance"]["ok"] and r["attachment"]["ok"] and not r["qa_failed"])
    n_clean_door = sum(1 for r in res if r["clearance"]["ok"] and r["attachment"]["ok_door"] and not r["qa_failed"])
    n_closer = sum(1 for r in res if r["attachment"]["n_closer"])
    L = []
    L.append("# Deficiency review\n")
    L.append(f"Generated by `scripts/deficiency_review.py` on {time.strftime('%Y-%m-%d %H:%M')} over **{n} doors** ({'whole dataset' if n == len(rows) and not (args.doors or args.families or args.models) else 'selection'}).  How to run / read / extend: [docs/REVIEW_AGENT.md](REVIEW_AGENT.md).\n")
    L.append("## Summary\n")
    L.append("| gate | result |\n|---|---|")
    L.append(f"| clearance (interpenetration, every joint swept) | **{sum(1 for r in res if r['clearance']['ok'])} / {n}** clean |")
    L.append(f"| attachment, door domain (floating parts, mechanisms, degenerate geometry) | **{sum(1 for r in res if r['attachment']['ok_door'])} / {n}** clean |")
    L.append(f"| attachment, closer domain (closer / power-operator / gas-strut parts, maintained with the closer mechanism model) | **{n - n_closer} / {n}** clean ({n_closer} doors with closer findings) |")
    L.append(f"| mass gate (qa.json) | **{sum(1 for r in res if r['mass_ok'])} / {n}** |")
    L.append(f"| physics QA (qa.json, other checks) | **{sum(1 for r in res if not r['qa_failed'])} / {n}** with no failed check |")
    L.append(f"| all of the above | **{n_clean_door} / {n}** clean outside the closer mechanism, **{n_clean_all} / {n}** clean including it |")
    L.append(f"| signed off in qa.json | **{sum(1 for r in res if r['signed_off'])} / {n}** |\n")
    L.append("## Findings per rule (door domain)\n")
    L.append("| rule | findings | doors | families (top) |\n|---|---|---|---|")
    for k, v in by_rule.most_common():
        L.append(f"| `{k}` | {v} | {len(by_rule_doors[k])} | {', '.join(f'{f} {c}' for f, c in by_rule_fam[k].most_common(5))} |")
    if not by_rule:
        L.append("| (none) | 0 | 0 | |")
    L.append("")
    L.append("## Findings on closer / power-operator / gas-strut parts (closer domain)\n")
    if closer_by_rule:
        L.append("| rule | findings |\n|---|---|")
        for k, v in closer_by_rule.most_common():
            L.append(f"| `{k}` | {v} |")
    else:
        L.append("None.")
    L.append("")
    L.append("## Physics QA checks failed (qa.json)\n")
    if qa_fail:
        L.append("| check | doors |\n|---|---|")
        for k, v in qa_fail.most_common():
            L.append(f"| `{k}` | {v} |")
    else:
        L.append("None.")
    L.append("")
    L.append("## Top offenders\n")
    if offenders:
        L.append("| door | family | clearance failures | attachment findings (door / closer) | qa failed | sheet |\n|---|---|---|---|---|---|")
        for score, did in offenders[: args.top]:
            r = next(x for x in res if x["id"] == did)
            why = "; ".join(f"{fd['rule']}: {fd['why'][:70]}" for fd in r["attachment"]["findings"][:2] if fd.get("domain") != "closer")
            sh = sheets.get(did)
            L.append(f"| `{did}` | {fam[did]} | {0 if r['clearance']['ok'] else r['clearance']['n']} | {r['attachment']['n'] - r['attachment']['n_closer']} / {r['attachment']['n_closer']} | {', '.join(r['qa_failed']) or '-'} | {'[sheet](' + out_rel + '/' + os.path.basename(sh) + ')' if sh else '-'} |")
            if why:
                L.append(f"| | | | {why} | | |")
    else:
        L.append("None: every selected door is clean outside the closer mechanism.")
    L.append("")
    L.append("## Per family\n")
    L.append("| family | doors | clean | clearance fails | attachment fails (door) | closer findings | qa fails | sheets |\n|---|---|---|---|---|---|---|---|")
    for f_ in sorted(per_fam):
        pf = per_fam[f_]
        links = " ".join(f"[p{i}]({out_rel}/{os.path.basename(p)})" for i, p in enumerate(family_sheets.get(f_, [])))
        L.append(f"| {f_} | {pf['doors']} | {pf['clean']} | {pf['clearance']} | {pf['attachment']} | {pf['closer']} | {pf['qa']} | {links} |")
    L.append("")
    L.append("## Checklist: obvious-deficiency categories and how they are caught\n")
    L.append("| category | detected by | what it means |\n|---|---|---|")
    for cat, det, what in CATEGORIES:
        L.append(f"| {cat} | {det} | {what} |")
    L.append("")
    L.append("## Reading the contact sheets\n")
    L.append("Each door is a 3 x 3 tile: rows = **closed**, **45 % open** (mechanisms at rest), **open** (leaf at its limit, bolts withdrawn / operator at full travel); columns = **iso** (whole door from the robot side), **structure** (hinge line or track close-up), **hardware** (operator / latch close-up).  Collision-only proxies are hidden; closer linkages are solved so the arms follow the leaf.  Scan for: a part whose position does not change with the leaf it should ride on, a part with air between it and everything else, a closer arm that leaves its shoe, a bolt that is still out in the *open* row, a handle plate on the far side of a leaf, text or a hook pointing down, a part much larger or smaller than its neighbours, two overlapping copies of a part, and anything that passes through a jamb, stop or wall.  Deterministic gates catch geometry; the sheets are for what geometry cannot say (a part in a plausible place that is still the wrong part).\n")
    L.append(f"Images: `{out_rel}/` ({total_bytes / 1e6:.1f} MB, {len(family_sheets)} families, {len(sheets)} large sheets).\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


# ------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--out", default="docs/review/deficiency")
    ap.add_argument("--md", default="docs/DEFICIENCY_REVIEW.md")
    ap.add_argument("--doors", default="", help="comma list of door ids (rendered as large sheets)")
    ap.add_argument("--families", default="")
    ap.add_argument("--models", default="", help="comma list of hardware model ids (operator / latch / lock / closer / hinge)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--top", type=int, default=40, help="offenders listed and rendered large")
    ap.add_argument("--tag", default="", help="suffix for the large per-door sheets (e.g. before / after)")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-md", action="store_true")
    ap.add_argument("--rerun-qa", action="store_true", help="re-simulate the physics QA instead of reading qa.json")
    ap.add_argument("--per-page", type=int, default=24, help="doors per family sheet page")
    ap.add_argument("--json", default="", help="write all gate results here (default <out>/index.json)")
    a = ap.parse_args()
    t0 = time.time()
    man = json.load(open(os.path.join(a.assets, "manifest.json")))
    rows = select_doors(man, a.doors, a.families, a.models)
    if not rows:
        print("no doors selected")
        return
    os.makedirs(a.out, exist_ok=True)
    dirs = [os.path.join(a.assets, "doors", d["id"]) for d in rows]
    # ---- gates
    jobs = [(p, a.rerun_qa) for p in dirs]
    if a.workers > 1 and len(jobs) > 1:
        with Pool(a.workers) as pool:
            res = pool.map(gate_one, jobs, chunksize=2)
    else:
        res = [gate_one(j) for j in jobs]
    print(f"gates: {len(res)} doors in {time.time() - t0:.0f}s: clearance {sum(1 for r in res if r['clearance']['ok'])}, attachment(door) {sum(1 for r in res if r['attachment']['ok_door'])}, qa-clean {sum(1 for r in res if not r['qa_failed'])}", flush=True)
    # ---- renders
    sheets, family_sheets, total_bytes = {}, {}, 0
    if not a.no_render:
        from PIL import Image
        fam = {d["id"]: d["family"] for d in rows}
        offenders = sorted(res, key=lambda r: -((0 if r["clearance"]["ok"] else r["clearance"]["n"]) + (r["attachment"]["n"] - r["attachment"]["n_closer"]) + len(r["qa_failed"])))
        large = set(a.doors.split(",")) if a.doors else set()
        large |= {r["id"] for r in offenders[: a.top] if (not r["clearance"]["ok"] or not r["attachment"]["ok_door"] or r["qa_failed"])}
        tmp = os.path.join(a.out, "_tiles")
        os.makedirs(tmp, exist_ok=True)
        jobs = []
        for d in rows:
            jobs.append((os.path.join(a.assets, "doors", d["id"]), os.path.join(tmp, d["id"] + ".jpg"), (120, 90), 60, f"{d['id']}  {d['family']}"))
        for did in sorted(large):
            fn = os.path.join(a.out, did + (f"_{a.tag}" if a.tag else "") + ".jpg")
            jobs.append((os.path.join(a.assets, "doors", did), fn, (240, 180), 72, f"{did}  {fam.get(did, '')}  {a.tag}"))
            sheets[did] = fn
        t1 = time.time()
        if a.workers > 1 and len(jobs) > 1:
            with Pool(a.workers, maxtasksperchild=64) as pool:
                outs = pool.map(render_one, jobs, chunksize=2)
        else:
            outs = [render_one(j) for j in jobs]
        errs = [o for o in outs if str(o).startswith("ERROR")]
        if errs:
            print("render errors:", errs[:5], flush=True)
        # family pages from the mini tiles (4 per row)
        by_fam = collections.defaultdict(list)
        for d in rows:
            by_fam[d["family"]].append(d["id"])
        for f_, ids in sorted(by_fam.items()):
            pages = [ids[i: i + a.per_page] for i in range(0, len(ids), a.per_page)]
            for pi, page in enumerate(pages):
                tiles = []
                for did in page:
                    try:
                        tiles.append(Image.open(os.path.join(tmp, did + ".jpg")).convert("RGB"))
                    except Exception:
                        continue
                if not tiles:
                    continue
                tw, th = tiles[0].size
                cols = 4
                nrows = math.ceil(len(tiles) / cols)
                sheet = Image.new("RGB", (cols * tw, nrows * th), (24, 24, 24))
                for i, im in enumerate(tiles):
                    sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
                fn = os.path.join(a.out, f"family_{f_}_p{pi:02d}.jpg")
                sheet.save(fn, quality=62, optimize=True)
                family_sheets.setdefault(f_, []).append(fn)
        for fn in os.listdir(tmp):
            os.remove(os.path.join(tmp, fn))
        os.rmdir(tmp)
        total_bytes = sum(os.path.getsize(os.path.join(a.out, fn)) for fn in os.listdir(a.out) if fn.endswith(".jpg"))
        print(f"renders: {len(jobs)} sheets in {time.time() - t1:.0f}s, {total_bytes / 1e6:.1f} MB under {a.out}", flush=True)
    # ---- index + report
    index = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "assets": a.assets, "n_doors": len(res), "doors": {r["id"]: dict(r, sheet=os.path.basename(sheets[r["id"]]) if r["id"] in sheets else None) for r in res},
             "family_sheets": {k: [os.path.basename(p) for p in v] for k, v in family_sheets.items()}}
    with open(a.json or os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    if not a.no_md:
        out_rel = os.path.relpath(a.out, os.path.dirname(os.path.abspath(a.md)))
        write_report(a.md, res, rows, sheets, family_sheets, out_rel, total_bytes, a)
        print(f"report: {a.md}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
