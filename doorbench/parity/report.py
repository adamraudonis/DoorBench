"""Join MuJoCo and Isaac parity results per door, aggregate, plot the offenders and render docs/ISAAC_PARITY.md.

Library half of ``scripts/isaaclab/parity_report.py`` (pure Python + Pillow; no simulator needed) so the tests can drive
it on synthetic result files.  Entry point: :func:`build_report`.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from collections import Counter, defaultdict
from typing import Any

from . import results as R

SCHEMA_VERSION = "1"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRADES = ("A", "B", "C", "X")
HARDWARE_AXES = ("latch", "lock", "closer", "operator")


# ---------------------------------------------------------------------------------------------------------------------
# plotting (Pillow only: matplotlib is not a dependency of the package)

_COLORS = {"mujoco": (35, 35, 40), "physx_full": (214, 84, 40), "physx_rl": (46, 110, 214), "other": (120, 120, 120)}


def _font(size: int = 11):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:   # Pillow < 10.1
        return ImageFont.load_default()


def _fmt_tick(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 0.1 or v == 0:
        return f"{v:.2f}"
    return f"{v:.3g}"


def plot_curves(path: str, title: str, series: list[dict], ylabel: str = "q [rad | m]", size: tuple[int, int] = (540, 220)) -> bool:
    """Small line plot of several (t, q) series -> PNG.  series: [{label, pts: [[t, q], ...], color: (r, g, b)}].  Returns False when nothing to draw."""
    from PIL import Image, ImageDraw
    series = [s for s in series if s.get("pts") and len(s["pts"]) >= 2]
    if not series:
        return False
    W, H = size
    L, Rm, T, B = 54, 14, 26, 30
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    f, fs = _font(11), _font(10)
    ts = [p[0] for s in series for p in s["pts"]]
    qs = [p[1] for s in series for p in s["pts"]]
    tmin, tmax = min(ts), max(ts)
    qmin, qmax = min(qs), max(qs)
    if qmax - qmin < 1e-6:
        qmin, qmax = qmin - 0.01, qmax + 0.01
    pad = 0.06 * (qmax - qmin)
    qmin, qmax = qmin - pad, qmax + pad
    if tmax - tmin < 1e-9:
        tmax = tmin + 1.0

    def X(t: float) -> float:
        return L + (t - tmin) / (tmax - tmin) * (W - L - Rm)

    def Y(q: float) -> float:
        return H - B - (q - qmin) / (qmax - qmin) * (H - T - B)

    # frame, grid, zero line
    d.rectangle([L, T, W - Rm, H - B], outline=(150, 150, 150), width=1)
    for i in range(1, 4):
        tv = tmin + (tmax - tmin) * i / 4
        qv = qmin + (qmax - qmin) * i / 4
        d.line([(X(tv), T), (X(tv), H - B)], fill=(232, 232, 232))
        d.line([(L, Y(qv)), (W - Rm, Y(qv))], fill=(232, 232, 232))
    if qmin < 0 < qmax:
        d.line([(L, Y(0)), (W - Rm, Y(0))], fill=(170, 170, 170))
    for i in range(5):
        tv = tmin + (tmax - tmin) * i / 4
        qv = qmin + (qmax - qmin) * i / 4
        d.text((X(tv) - 8, H - B + 4), _fmt_tick(tv), fill=(70, 70, 70), font=fs)
        d.text((4, Y(qv) - 6), _fmt_tick(qv), fill=(70, 70, 70), font=fs)
    d.text((W // 2 - 10, H - 13), "t [s]", fill=(70, 70, 70), font=fs)
    d.text((L, 4), f"{title[:80]}   {ylabel}", fill=(20, 20, 20), font=f)
    # series
    for s in series:
        pts = [(X(t), Y(q)) for t, q in s["pts"]]
        d.line(pts, fill=tuple(s.get("color") or _COLORS["other"]), width=2)
    # legend: one row just under the top edge of the frame, left to right
    lx, ly = L + 6, T + 4
    for s in series:
        d.rectangle([lx, ly + 2, lx + 10, ly + 10], fill=tuple(s.get("color") or _COLORS["other"]))
        d.text((lx + 14, ly - 1), s["label"], fill=(40, 40, 40), font=fs)
        lx += int(d.textlength(s["label"], font=fs)) + 30
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path, optimize=True)
    return True


def offender_plot(door_id: str, phase: str, mj: dict | None, px_by_kind: dict[str, dict | None], ctx: dict, media_dir: str) -> str | None:
    """MuJoCo vs PhysX (full, rl) primary-joint curve of the worst phase -> docs/media/parity/<door>_<phase>.png (relative path from docs/)."""
    hints = [ctx.get("primary_joint") or ""]
    series = []
    if mj:
        pts = R.pick_curve(mj.get("curves") or {}, "primary", hints, phase)
        if pts:
            series.append({"label": "MuJoCo", "pts": pts, "color": _COLORS["mujoco"]})
    for kind in R.KINDS:
        px = px_by_kind.get(kind)
        if px:
            pts = R.pick_curve(px.get("curves") or {}, "primary", hints, phase)
            if pts:
                series.append({"label": f"PhysX {kind}", "pts": pts, "color": _COLORS[f"physx_{kind}"]})
    if not series:
        return None
    fn = f"{door_id}_{phase}.png"
    ok = plot_curves(os.path.join(media_dir, fn), f"{door_id} - {phase}: primary joint", series)
    return f"media/parity/{fn}" if ok else None


# ---------------------------------------------------------------------------------------------------------------------
# aggregation

def _git_commit(root: str) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _engine_of(meta: dict, recs: dict[str, dict]) -> Any:
    for key in ("engine", "engines", "simulator", "version"):
        if meta.get(key):
            return meta[key]
    for rec in recs.values():
        if rec.get("engine"):
            return rec["engine"]
    return None


def _severity(v: dict) -> tuple:
    """Sort key for offenders: load errors and status disagreements first, then quantitative; more disagreeing phases first."""
    order = {"X": 0, "C": 1, "B": 2, "A": 3, None: 4}
    n_dis = sum(1 for k in R.KINDS for p in v["kinds"][k]["phases"].values() if p == "disagree")
    n_kinds_bad = sum(1 for k in R.KINDS if v["kinds"][k]["grade"] in ("C", "X"))
    return (order.get(v["grade"], 4), -n_kinds_bad, -n_dis, v["door_id"])


def _worst_phase(v: dict) -> str | None:
    if v.get("grade") == "X" or any(v["kinds"][k]["grade"] == "X" for k in R.KINDS):
        return "structure"
    for state in ("disagree", "quant"):
        for k in R.KINDS:
            for n in R.PHASES:
                if v["kinds"][k]["phases"].get(n) == state:
                    return n
    for k in R.KINDS:
        for n, st in v["kinds"][k]["phases"].items():
            if st in ("disagree", "quant"):
                return n
    return "operate_open"


def _group_stats(verdicts: list[dict]) -> dict:
    g: dict[str, Any] = {"n": len(verdicts), "tested": 0, "ok": 0, "fail": 0, "untested": 0}
    for k in R.KINDS:
        g[k] = {gr: 0 for gr in GRADES} | {"tested": 0, "untested": 0}
    cls: Counter = Counter()
    for v in verdicts:
        st = R.manifest_status(v)
        g[st] += 1
        if st != "untested":
            g["tested"] += 1
        for k in R.KINDS:
            kv = v["kinds"][k]
            if kv["status"] == "untested":
                g[k]["untested"] += 1
            else:
                g[k]["tested"] += 1
                if kv["grade"] in GRADES:
                    g[k][kv["grade"]] += 1
            for c in kv["classes"]:
                if c not in ("QUANT",):
                    cls[c] += 1
    g["top_classes"] = [f"{c} x{n}" for c, n in cls.most_common(3)]
    return g


def build_report(results_dir: str, assets_dir: str | None, media_dir: str | None = None, top_n: int = 20, plots: bool = True, root: str = ROOT) -> tuple[dict, str]:
    """Read results/parity/*.json (+ manifest / spec / qa.json when present) -> (summary dict, markdown text)."""
    files = R.collect_result_files(results_dir)
    mj_all, mj_meta = R.load_results(files["mujoco"]) if files["mujoco"] else ({}, {})
    px_all: dict[str, dict[str, dict]] = {}
    px_meta: dict[str, dict] = {}
    for kind, path in files["isaac"].items():
        px_all[kind], px_meta[kind] = R.load_results(path)
    variants: dict[str, dict[str, dict[str, dict]]] = {}   # kind -> variant -> door -> rec
    for kind, vs in files["isaac_variants"].items():
        for vname, path in vs.items():
            variants.setdefault(kind, {})[vname] = R.load_results(path)[0]
    manifest = R._read_json(os.path.join(assets_dir, "manifest.json")) if assets_dir else None
    man_doors = {d["id"]: d for d in (manifest or {}).get("doors", []) if isinstance(d, dict) and d.get("id")}
    ids = sorted(set(man_doors) | set(mj_all) | {d for recs in px_all.values() for d in recs})
    n_total = int((manifest or {}).get("n_doors") or len(ids) or 0)

    verdicts: dict[str, dict] = {}
    raw: dict[str, tuple] = {}
    for did in ids:
        mj = mj_all.get(did)
        pxk = {k: px_all.get(k, {}).get(did) for k in R.KINDS}
        if mj is None and all(v is None for v in pxk.values()):
            continue    # a manifest door nobody ran: untested, counted from n_total
        ctx = R.door_context(assets_dir, did, man_doors.get(did))
        vars_for = {k: {vn: recs[did] for vn, recs in variants.get(k, {}).items() if did in recs} for k in R.KINDS}
        v = R.door_verdict(did, mj, pxk, ctx, vars_for)
        verdicts[did] = v
        raw[did] = (mj, pxk, ctx)

    # ---- counts
    counts: dict[str, Any] = {"n_doors_total": n_total, "n_with_results": len(verdicts)}
    for k in R.KINDS:
        c = {gr: 0 for gr in GRADES}
        tested = [v for v in verdicts.values() if v["kinds"][k]["status"] != "untested"]
        for v in tested:
            if v["kinds"][k]["grade"] in GRADES:
                c[v["kinds"][k]["grade"]] += 1
        c["tested"] = len(tested)
        c["untested"] = n_total - len(tested)
        c["parity"] = c["A"]
        c["same_verdicts"] = c["A"] + c["B"]
        counts[k] = c
    door_c = Counter(R.manifest_status(v) for v in verdicts.values())
    counts["door"] = {"ok": door_c.get("ok", 0), "fail": door_c.get("fail", 0), "untested": n_total - door_c.get("ok", 0) - door_c.get("fail", 0)}

    # ---- by class / family / hardware / kinematics
    by_class: dict[str, dict] = {}
    for v in verdicts.values():
        for k in R.KINDS:
            for code in v["kinds"][k]["classes"]:
                e = by_class.setdefault(code, {"full": 0, "rl": 0, "doors": []})
                e[k] += 1
                if v["door_id"] not in e["doors"]:
                    e["doors"].append(v["door_id"])
    for e in by_class.values():
        e["n_doors"] = len(e["doors"])
        e["examples"] = e["doors"][:8]
        e["doors"] = e["doors"][:200]
    fam_groups: dict[str, list[dict]] = defaultdict(list)
    kin_groups: dict[str, list[dict]] = defaultdict(list)
    hw_groups: dict[str, dict[str, list[dict]]] = {ax: defaultdict(list) for ax in HARDWARE_AXES}
    for v in verdicts.values():
        fam_groups[str(v.get("family") or "unknown")].append(v)
        kin_groups[str(v.get("kinematics") or "unknown")].append(v)
        for ax in HARDWARE_AXES:
            hw_groups[ax][str(v["hardware"].get(ax) or "none")].append(v)
    fam_total = Counter(d.get("family") for d in man_doors.values()) if man_doors else Counter(v.get("family") for v in verdicts.values())
    by_family = {fam: _group_stats(vs) | {"n_family": fam_total.get(fam, len(vs))} for fam, vs in sorted(fam_groups.items())}
    by_kin = {kin: _group_stats(vs) for kin, vs in sorted(kin_groups.items())}
    by_hw = {ax: {kind: _group_stats(vs) for kind, vs in sorted(groups.items())} for ax, groups in hw_groups.items()}

    # ---- offenders + plots
    offenders = sorted([v for v in verdicts.values() if v["grade"] in ("B", "C", "X")], key=_severity)[:top_n]
    for v in offenders:
        v["worst_phase"] = _worst_phase(v)
        v["plot"] = None
        if plots and media_dir:
            mj, pxk, ctx = raw[v["door_id"]]
            try:
                v["plot"] = offender_plot(v["door_id"], v["worst_phase"], mj, pxk, ctx, media_dir)
            except Exception as e:  # a plotting problem must never sink the report
                v["plot_error"] = f"{type(e).__name__}: {e}"

    generated = _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    engines = {"mujoco": _engine_of(mj_meta, mj_all)} | {f"isaac_{k}": _engine_of(px_meta.get(k, {}), px_all.get(k, {})) for k in px_all}
    # which protocol produced the inputs (the runners write meta.protocol = {"name", "warning"?}; a legacy-probe adapter warns)
    metas = [("mujoco", mj_meta)] + [(f"isaac_{k}", px_meta.get(k, {})) for k in px_all]
    proto_names = {k: (m.get("protocol") if isinstance(m.get("protocol"), str) else (m.get("protocol") or {}).get("name")) for k, m in metas}
    warning = next((m["protocol"].get("warning") for _, m in metas if isinstance(m.get("protocol"), dict) and m["protocol"].get("warning")), None)
    protocol = {"names": proto_names, "warning": warning}
    summary = {
        "schema_version": SCHEMA_VERSION, "generated": generated, "date": generated[:10], "commit": _git_commit(root),
        "inputs": {"mujoco": _file_info(files["mujoco"], mj_all), "isaac": {k: _file_info(p, px_all.get(k, {})) for k, p in files["isaac"].items()},
                   "variants": {k: sorted(vs) for k, vs in files["isaac_variants"].items()}, "mujoco_variants": sorted(files["mujoco_variants"])},
        "engines": engines, "protocol": protocol,
        "counts": counts, "by_class": dict(sorted(by_class.items(), key=lambda kv: -(kv[1]["full"] + kv[1]["rl"]))),
        "by_family": by_family, "by_kinematics": by_kin, "by_hardware": by_hw,
        "top_offenders": [v["door_id"] for v in offenders],
        "class_info": R.CLASS_INFO,
        "doors": {did: {k: val for k, val in v.items() if not k.startswith("_")} for did, v in verdicts.items()},
    }
    md = render_markdown(summary, offenders, results_dir)
    return summary, md


def rel(path: str, root: str = ROOT) -> str:
    """Repo-relative when inside the repo, else the absolute path (never '../../..')."""
    ap = os.path.abspath(path)
    return os.path.relpath(ap, root) if ap.startswith(os.path.abspath(root) + os.sep) else ap


def _file_info(path: str | None, recs: dict) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    return {"path": rel(path), "n_doors": len(recs), "mtime": _dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%dT%H:%M:%S")}


# ---------------------------------------------------------------------------------------------------------------------
# markdown

def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.0f} %" if b else "-"


def _tol_table() -> list[str]:
    rows = ["| metric | hinge (rad, s) | slide (m, s) | relative |", "|---|---|---|---|"]
    for k, t in R.TOLERANCES.items():
        rows.append(f"| `{k}` | {t['abs_hinge']:.3g} | {t['abs_slide']:.3g} | {('%.0f %%' % (100 * t['rel'])) if t['rel'] else '-'} |")
    rows.append(f"| *(any other metric)* | {R.DEFAULT_TOLERANCE['abs_hinge']:.3g} | {R.DEFAULT_TOLERANCE['abs_slide']:.3g} | {100 * R.DEFAULT_TOLERANCE['rel']:.0f} % |")
    return rows


def render_markdown(summary: dict, offenders: list[dict], results_dir: str) -> str:
    c = summary["counts"]
    n = c["n_doors_total"] or 1
    eng = summary.get("engines") or {}
    L: list[str] = []
    L.append("# Isaac parity gate")
    L.append("")
    L.append(f"_Generated {summary['generated']} by `scripts/isaaclab/parity_report.py` from `{rel(results_dir)}/` "
             f"(commit `{summary.get('commit') or 'n/a'}`). Reference: MuJoCo {_short(eng.get('mujoco'))}; PhysX: {_short(eng.get('isaac_full') or eng.get('isaac_rl'))}._")
    proto = summary.get("protocol") or {}
    if proto.get("warning"):
        L.append("")
        L.append(f"> **Note.** {proto['warning']}")
    L.append("")
    L.append("Every door runs **one behavioural protocol** in MuJoCo (the reference physics, CPU) and in Isaac Sim / PhysX on the GPU pod, on both USD "
             "kinds (`door.usda` full fidelity, `door_rl.usda` canonical 8-link). The two runs are compared phase by phase: both simulators must reach the "
             "same pass / fail verdict (else grade **C**), and when they agree the metrics must be within tolerance (else grade **B**); **A** is parity, "
             "**X** means the door could not be compared (spawn / structure error). A disagreement is tagged with a discrepancy class whose likely root "
             "cause comes from the analysis of the first 40-door probe. The per-door verdict is published in `qa.json` (`isaac_parity`) and as a badge in the viewer.")
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append(f"| USD kind | tested | parity (A) | same verdicts (A + B) | disagree (C) | not comparable (X) | untested |")
    L.append("|---|---|---|---|---|---|---|")
    for k in R.KINDS:
        ck = c[k]
        L.append(f"| `{k}` | {ck['tested']} / {n} | **{ck['A']} / {n}** ({_pct(ck['A'], ck['tested'])} of tested) | {ck['A'] + ck['B']} / {n} ({_pct(ck['A'] + ck['B'], ck['tested'])}) | {ck['C']} | {ck['X']} | {ck['untested']} |")
    d = c["door"]
    L.append("")
    L.append(f"Door badge (`qa.json.isaac_parity.ok`; viewer chip *Isaac parity*): **{d['ok']} ok** (grade A or B in every tested kind), **{d['fail']} fail** (a status disagreement or not comparable), {d['untested']} untested.")
    L.append("")
    L.append("## What the gate runs")
    L.append("")
    L.append("| phase | what is compared | pass criterion (per simulator) |")
    L.append("|---|---|---|")
    crit = {"structure": "joint set, ranges (2e-3), stiffness / friction (1 %), spring target (1e-3), moving mass (20 % / 0.5 kg)",
            "pose0": "body origins, COMs, joint anchors within 5 mm; axes dot >= 0.999",
            "settle": "primary drift < 0.05 rad / 0.01 m, every other joint < 0.02 rad / 2 mm, no MuJoCo warnings, penetration > -12 mm",
            "hold": "has_holding: q < 2 deg / 15 mm under the adaptive push; else opens > 10 deg / 5 cm within 6 s",
            "operate_open": "q > min(20 deg, 0.5 max_open) / 5 cm after operator + push (chain: inside the slack window)",
            "release": "bolt < 6 mm after the operator is released",
            "relatch": "closed < 2 deg after 6 s closing drive, re-push < 2.5 deg",
            "closer_return": "abs(q) < 6 deg after 12 s from 60 deg",
            "locked_holds": "q < 2 deg / 15 mm (+ chain slack) with operator worked + push",
            "limits": "every limited joint inside lo - tol .. hi + tol (2 deg / 5 mm)",
            "sanity": "finite state, no velocity cap hit, no body displaced > 5 m, no MuJoCo warnings"}
    for p in R.PHASES:
        L.append(f"| `{p}` | {R.PHASE_LABELS[p]} | {crit[p]} |")
    L.append("")
    L.append("<details><summary>Metric tolerances (a delta passes when within either bound)</summary>")
    L.append("")
    L.extend(_tol_table())
    L.append("")
    L.append("</details>")
    L.append("")
    # ---- classes
    L.append("## Discrepancy classes")
    L.append("")
    L.append("| class | full | rl | doors | what it means | likely root cause | fix direction | examples |")
    L.append("|---|---|---|---|---|---|---|---|")
    if not summary["by_class"]:
        L.append("| *(none)* | 0 | 0 | 0 | every compared door agrees within tolerance | - | - | - |")
    for code, e in summary["by_class"].items():
        info = R.CLASS_INFO.get(code, {"meaning": "-", "root_cause": "-", "fix": "-"})
        L.append(f"| `{code}` | {e['full']} | {e['rl']} | {e['n_doors']} | {info['meaning']} | {info['root_cause']} | {info['fix']} | {', '.join(f'`{x}`' for x in e['examples'][:4])} |")
    L.append("")
    # ---- family
    L.append("## By family")
    L.append("")
    L.append("| family | doors | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |")
    L.append("|---|---|---|---|---|---|---|---|")
    for fam, g in summary["by_family"].items():
        L.append(f"| {fam} | {g['n_family']} | {g['tested']} | {g['ok']} | {g['fail']} | {g['full']['A']} / {g['full']['A'] + g['full']['B']} | {g['rl']['A']} / {g['rl']['A'] + g['rl']['B']} | {', '.join(g['top_classes']) or '-'} |")
    L.append("")
    # ---- hardware
    L.append("## By hardware")
    L.append("")
    for ax in HARDWARE_AXES:
        L.append(f"### {ax} kind")
        L.append("")
        L.append("| kind | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |")
        L.append("|---|---|---|---|---|---|---|")
        for kind, g in summary["by_hardware"].get(ax, {}).items():
            L.append(f"| {kind} | {g['tested']} | {g['ok']} | {g['fail']} | {g['full']['A']} / {g['full']['A'] + g['full']['B']} | {g['rl']['A']} / {g['rl']['A'] + g['rl']['B']} | {', '.join(g['top_classes']) or '-'} |")
        L.append("")
    # ---- kinematics
    L.append("## By kinematics")
    L.append("")
    L.append("| kinematics | tested | ok | fail | full A / A+B | rl A / A+B | most frequent classes |")
    L.append("|---|---|---|---|---|---|---|")
    for kin, g in summary["by_kinematics"].items():
        L.append(f"| {kin} | {g['tested']} | {g['ok']} | {g['fail']} | {g['full']['A']} / {g['full']['A'] + g['full']['B']} | {g['rl']['A']} / {g['rl']['A'] + g['rl']['B']} | {', '.join(g['top_classes']) or '-'} |")
    L.append("")
    # ---- offenders
    L.append(f"## Top offenders ({len(offenders)})")
    L.append("")
    if not offenders:
        L.append("No door disagrees.")
    else:
        L.append("| door | family | grade full / rl | phase | MuJoCo | PhysX full | PhysX rl | classes | likely root cause |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for v in offenders:
            ph = v["worst_phase"]
            mjv, fv, rv = _phase_metric_cells(v, ph)
            L.append(f"| `{v['door_id']}` | {v.get('family') or '-'} | {v['kinds']['full']['grade'] or '-'} / {v['kinds']['rl']['grade'] or '-'} | `{ph}` | {mjv} | {fv} | {rv} | {', '.join(f'`{c}`' for c in v['classes']) or '-'} | {_short(v.get('likely_root_cause'), 160)} |")
        L.append("")
        for v in offenders:
            L.append(f"### `{v['door_id']}` - grade {v['grade']} ({v.get('family')}, {v['hardware'].get('latch')} latch, {v['hardware'].get('lock')} lock{' engaged' if v['hardware'].get('lock_engaged') else ''}, {v['hardware'].get('closer')} closer)")
            L.append("")
            for k in R.KINDS:
                kv = v["kinds"][k]
                if kv["status"] == "untested":
                    L.append(f"* `{k}`: untested")
                    continue
                phases = ", ".join(f"{n} **{st}**" if st == "disagree" else f"{n} {st}" for n, st in kv["phases"].items())
                L.append(f"* `{k}` grade {kv['grade']}: {phases or 'no phases'}")
                for det in kv["details"][:4]:
                    L.append(f"  * {det}")
                if "LOAD_ERROR" not in kv["classes"]:
                    for err in kv.get("errors", [])[:2]:
                        L.append(f"  * error: {_short(err, 200)}")
            if v.get("plot"):
                L.append("")
                L.append(f"![{v['door_id']} {v['worst_phase']}]({v['plot']})")
            elif v.get("plot_error"):
                L.append(f"* plot error: {v['plot_error']}")
            L.append("")
    # ---- N/A
    L.append("## Known not-comparable categories")
    L.append("")
    L.append("* **Env-release locks** (mag lock, delayed egress, card reader, electric strike, interlock): MuJoCo holds them with a `<weld>` that has no PhysX counterpart; "
             "the runner emulates the hold or marks the phase `na_env_logic`. A door that *opens* here in PhysX is class `EXPORT_WELD`.")
    L.append("* **Panic doors with the robot outside and no far-side trim**: `operator_joint` is None, the exit device is welded in `door_rl.usda`; both simulators must *hold*.")
    L.append("* **Welded releases in `door_rl.usda`** (thumbturns, aux bolts, extra dogs): the RL expectation for `operate_open` flips to 'stays closed'; a `full` / `rl` disagreement there is `RL_CANON`.")
    L.append("* **Free-swing families** (saloon, bifold, accordion, bypass, pet door, strip curtain, revolving, turnstiles): no qa.py behavioural check exists, so their MuJoCo reference is itself unvalidated; their push phase is informational.")
    L.append("* **Closer-arm loop closures** (`connect` equalities) are not exported: the pinion / elbow joints swing freely in `door.usda` and are excluded from the limit check.")
    L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append("```bash")
    L.append("# CPU (local): MuJoCo reference for every door")
    L.append("PYTHONPATH=$PWD python -m doorbench.parity.protocol --sim mujoco --all            # -> results/parity/mujoco.json")
    L.append("# GPU pod: Isaac Sim / PhysX on both USD kinds")
    L.append("./isaaclab.sh -p scripts/isaaclab/parity_isaac.py --all --which both              # -> results/parity/isaac_full.json, isaac_rl.json")
    L.append("# join, classify, render this page + summary.json")
    L.append("PYTHONPATH=$PWD python scripts/isaaclab/parity_report.py")
    L.append("# publish per door: qa.json isaac_parity + manifest badge (idempotent)")
    L.append("PYTHONPATH=$PWD python scripts/merge_isaac_results.py")
    L.append("```")
    L.append("")
    return "\n".join(L)


def _short(s: Any, n: int = 60) -> str:
    if s is None:
        return "n/a"
    if isinstance(s, dict):
        s = ", ".join(f"{k} {v}" for k, v in s.items())
    s = str(s).replace("\n", " ").replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "..."


def _phase_metric_cells(v: dict, phase: str) -> tuple[str, str, str]:
    """The most telling metric of a phase as 'name=value' cells for MuJoCo / PhysX full / PhysX rl."""
    prefer = {"settle": ("settle_drift", "settle_drift_primary"), "hold": ("hold_displacement", "q_at_1s"), "operate_open": ("opened", "actuate_displacement", "bolt_retract_max_frac"),
              "release": ("bolt_after_release_m",), "relatch": ("relatch_closed_angle", "relatch_repush_angle"), "closer_return": ("closer_final_angle",), "locked_holds": ("locked_displacement",)}
    names = prefer.get(phase, ())
    mjv, cells = "-", {}
    for k in R.KINDS:
        kv = v["kinds"][k]
        st = kv["phases"].get(phase)
        if kv["status"] == "untested":
            cells[k] = "untested"
            continue
        if st is None:
            cells[k] = "-"
            continue
        picked = None
        for nm in names + tuple(sorted(m.split(".", 1)[1] for m in kv["metrics"] if m.startswith(phase + "."))):
            if f"{phase}.{nm}" in kv["metrics"]:
                picked = (nm, kv["metrics"][f"{phase}.{nm}"])
                break
        if picked:
            nm, (a, b, _ok) = picked
            mjv = f"{nm}={R._fmt(a)}"
            cells[k] = f"{R._fmt(b)} ({st})"
        else:
            cells[k] = st
    return mjv, cells.get("full", "-"), cells.get("rl", "-")


def write_outputs(summary: dict, md: str, summary_path: str, md_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    with open(md_path, "w") as f:
        f.write(md)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)
