"""Tests for the visual common-sense review (task G8): sheet rendering, prompt / verdict schema, mocked Claude API round
trip (single requests and the Message Batches API), cost estimate and report generation.

Run:  pytest -q tests/test_vision_review.py      (sheet rendering is skipped when assets/ has not been generated)
"""
from __future__ import annotations

import json
import os
import types

import pytest

from doorbench.review import vision as V

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.environ.get("DOORBENCH_ASSETS", os.path.join(ROOT, "assets"))
MANIFEST = os.path.join(ASSETS, "manifest.json")
SHEET_DOORS = ["db0024_swing_single", "db0079_sliding_single", "db0004_bifold"]      # hinged + closer/stop, slider, folding


def _manifest():
    if not os.path.exists(MANIFEST):
        return None
    with open(MANIFEST) as f:
        return json.load(f)


needs_assets = pytest.mark.skipif(_manifest() is None, reason=f"generated dataset not found at {ASSETS}")


# ---------------------------------------------------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------------------------------------------------
@needs_assets
@pytest.mark.parametrize("door_id", SHEET_DOORS)
def test_render_sheet(tmp_path, door_id):
    pytest.importorskip("mujoco")
    from PIL import Image
    door_dir = os.path.join(ASSETS, "doors", door_id)
    if not os.path.isdir(door_dir):
        pytest.skip(f"{door_id} not in this dataset")
    out = tmp_path / f"{door_id}.jpg"
    info = V.render_sheet(door_dir, str(out), cell=(200, 150), supersample=1)
    assert out.exists()
    im = Image.open(out)
    assert im.size[0] == 4 * 200
    assert im.size[1] > 3 * 150
    assert info["width"] == im.size[0] and info["height"] == im.size[1]
    assert len(info["panels"]) == 12                      # 3 rows x 4 columns for every family
    assert info["states"][:2] == ["closed", "open"]
    assert info["states"][2] in ("mid", "open-low")
    assert info["facts"]["door_id"] == door_id
    labels = [p["label"] for p in info["panels"]]
    assert any("hardware close-up" in l for l in labels)
    assert any("close-up" in l and "open" in l for l in labels)


@needs_assets
def test_states_and_open_configuration():
    mujoco = pytest.importorskip("mujoco")
    door_dir = os.path.join(ASSETS, "doors", "db0079_sliding_single")
    r = V.SheetRenderer(door_dir, cell=(100, 75), supersample=1)
    try:
        assert [s for s, _ in r.states()] == ["closed", "open", "mid"]
        q_closed, q_open, q_mid = r.q_state(0.0), r.q_state(1.0), r.q_state(0.5)
        j = mujoco.mj_name2id(r.m, mujoco.mjtObj.mjOBJ_JOINT, "leaf_slide")
        adr = r.m.jnt_qposadr[j]
        assert abs(q_closed[adr]) < 1e-9
        assert abs(q_open[adr] - r.m.jnt_range[j][1]) < 1e-9
        assert abs(q_mid[adr] - 0.5 * r.m.jnt_range[j][1]) < 1e-9
        c, rad, what = r.mechanism_target(q_open)
        assert what == "track hardware"                    # hangers + rail: where db0079's wheel leaves the rail
        assert 0.25 <= rad <= 1.0
    finally:
        r.close()


@needs_assets
def test_facts_and_prompt_mention_spec():
    door_dir = os.path.join(ASSETS, "doors", "db0024_swing_single")
    spec = json.load(open(os.path.join(door_dir, "spec.json")))
    mj = json.load(open(os.path.join(door_dir, "model.json")))
    facts = V.door_facts(spec, mj)
    assert facts["door_id"] == "db0024_swing_single"
    assert "butt_45_bb x 3" in facts["hinge"]
    assert facts["stop"] == "wall_bumper"
    text = V.user_prompt(facts)
    assert "wall_bumper" in text and "magnetic_hold" in text and "leaf_hinge" in text


def test_select_doors_seeded():
    man = _manifest()
    if man is None:
        man = {"doors": [{"id": f"db{i:04d}_{fam}", "index": i, "family": fam} for i, fam in enumerate(["a", "b", "c"] * 10)]}
    rows = V.select_doors(man, ids=[man["doors"][0]["id"]], per_family=2, seed=1)
    fams = {d["family"] for d in man["doors"]}
    assert {d["family"] for d in rows} == fams
    assert man["doors"][0]["id"] in {d["id"] for d in rows}
    again = V.select_doors(man, ids=[man["doors"][0]["id"]], per_family=2, seed=1)
    assert [d["id"] for d in rows] == [d["id"] for d in again]
    assert len(V.select_doors(man, per_family=1, limit=3)) == 3
    assert len(V.select_doors(man, families=[next(iter(fams))])) == sum(1 for d in man["doors"] if d["family"] == next(iter(fams)))


# ---------------------------------------------------------------------------------------------------------------------
# prompt + verdict schema
# ---------------------------------------------------------------------------------------------------------------------
FACTS = {"door_id": "db9999_swing_single", "family": "swing_single", "context": "test", "use_case": "unit test", "kinematics": "hinge_vertical",
         "travel": "90 deg", "leaf": "1 x 0.9 x 2.0 x 0.04 m", "opening": "0.92 x 2.05 m", "hinge": "butt x 3", "operator": "lever", "latch": "tubular",
         "lock": "none (engaged=False)", "closer": "none", "track": None, "roller": None, "stop": "wall_bumper", "extra_kinematics": {}, "seal": None,
         "condition": "new", "extras": [], "n_bodies": 3, "n_joints": 2, "leaf_joints": ["leaf_hinge (hinge, 0..90 deg)"], "mechanism_joints": [], "part_labels": ["Leaf"]}


def test_build_request_shape(tmp_path):
    from PIL import Image
    sheet = tmp_path / "s.jpg"
    Image.new("RGB", (64, 48), (0, 0, 0)).save(sheet)
    req = V.build_request(str(sheet), FACTS, model="claude-opus-5", effort="medium")
    assert req["model"] == "claude-opus-5" and req["max_tokens"] == V.MAX_TOKENS
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["output_config"]["format"]["schema"] is V.VERDICT_SCHEMA
    assert req["output_config"]["effort"] == "medium"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    content = req["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/jpeg"
    assert content[1]["type"] == "text" and "db9999_swing_single" in content[1]["text"]
    dry = V.build_request(str(sheet), FACTS, with_image=False)
    assert dry["messages"][0]["content"][0]["type"] == "text"
    # the rubric lists every category and severity so the model can only pick from the schema's enums
    sp = V.system_prompt()
    for c in V.CATEGORIES:
        assert c in sp
    for s in V.SEVERITIES:
        assert s in sp
    assert set(V.VERDICT_SCHEMA["properties"]["findings"]["items"]["properties"]["category"]["enum"]) == set(V.CATEGORIES)


def test_normalise_verdict():
    good = {"door_id": "db9999_swing_single", "ok": True, "summary": "looks fine",
            "findings": [{"category": "floating_part", "severity": "major", "part": "wall bumper stop", "description": "in mid-air", "where": "open / close-up"}]}
    v = V.normalise_verdict(good, "db9999_swing_single", V.REVIEWER_API, "claude-opus-5", extra={"usage": {"input_tokens": 10}})
    assert v["ok"] is False                      # a major finding overrides the model's ok=true
    assert v["reviewer"] == "claude-api" and v["model"] == "claude-opus-5" and v["usage"]["input_tokens"] == 10
    assert v["findings"][0]["severity"] == "major"
    with pytest.raises(V.VerdictError):
        V.normalise_verdict({"findings": [{"category": "nonsense", "severity": "major", "part": "x", "description": "y", "where": "z"}]}, "d", "r")
    with pytest.raises(V.VerdictError):
        V.normalise_verdict({"findings": [{"category": "floating_part", "severity": "huge", "part": "x", "description": "y", "where": "z"}]}, "d", "r")
    with pytest.raises(V.VerdictError):
        V.normalise_verdict({"findings": "none"}, "d", "r")
    empty = V.normalise_verdict({"door_id": "d", "ok": False, "summary": "", "findings": []}, "d", "r")
    assert empty["ok"] is True                   # no findings -> ok regardless of what the model said
    assert V.parse_verdict_text('```json\n{"findings": []}\n```') == {"findings": []}
    assert V.parse_verdict_text('Here: {"findings": []} done') == {"findings": []}


# ---------------------------------------------------------------------------------------------------------------------
# mocked API
# ---------------------------------------------------------------------------------------------------------------------
def _msg(text, stop="end_turn", usage=(1200, 300)):
    block = types.SimpleNamespace(type="text", text=text)
    u = types.SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1], cache_creation_input_tokens=0, cache_read_input_tokens=900)
    return types.SimpleNamespace(content=[block], stop_reason=stop, stop_details=None, usage=u, _request_id="req_test")


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _client(responses):
    return types.SimpleNamespace(messages=FakeMessages(responses))


VERDICT_JSON = json.dumps({"door_id": "db9999_swing_single", "ok": False, "summary": "bumper floats",
                           "findings": [{"category": "floating_part", "severity": "blocker", "part": "wall bumper stop",
                                         "description": "cylinder in mid-air 0.8 m from the wall", "where": "open / close-up (stop / hold-open)"}]})


def test_review_door_round_trip(tmp_path):
    from PIL import Image
    sheet = tmp_path / "s.jpg"
    Image.new("RGB", (64, 48)).save(sheet)
    client = _client([_msg(VERDICT_JSON)])
    verdict, usage = V.review_door(client, str(sheet), FACTS, model="claude-opus-5", effort="high", sleep=lambda s: None)
    assert verdict["door_id"] == "db9999_swing_single" and verdict["ok"] is False
    assert verdict["findings"][0]["category"] == "floating_part" and verdict["findings"][0]["severity"] == "blocker"
    assert verdict["reviewer"] == V.REVIEWER_API and verdict["model"] == "claude-opus-5" and verdict["request_id"] == "req_test"
    assert usage == {"input_tokens": 1200, "output_tokens": 300, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 900}
    kw = client.messages.calls[0]
    assert kw["model"] == "claude-opus-5" and kw["messages"][0]["content"][0]["type"] == "image"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    json.dumps(verdict)   # serialisable


def test_review_door_retries_invalid_then_truncated(tmp_path):
    from PIL import Image
    sheet = tmp_path / "s.jpg"
    Image.new("RGB", (64, 48)).save(sheet)
    bad = json.dumps({"door_id": "x", "ok": True, "summary": "", "findings": [{"category": "weird", "severity": "major", "part": "a", "description": "b", "where": "c"}]})
    client = _client([_msg(bad), _msg('{"door_id": "x", "ok": true', stop="max_tokens"), _msg(VERDICT_JSON)])
    verdict, _ = V.review_door(client, str(sheet), FACTS, sleep=lambda s: None)
    assert verdict["ok"] is False and len(client.messages.calls) == 3
    # the second call quoted the validation error back to the model; the third doubled max_tokens
    assert "rejected" in client.messages.calls[1]["messages"][-1]["content"]
    assert client.messages.calls[2]["max_tokens"] == 2 * V.MAX_TOKENS
    client = _client([_msg("not json at all"), _msg("still not"), _msg("nope")])
    with pytest.raises(V.VerdictError):
        V.review_door(client, str(sheet), FACTS, attempts=3, sleep=lambda s: None)


def test_review_door_refusal_and_transport_retry(tmp_path):
    anthropic = pytest.importorskip("anthropic")
    import httpx2 as httpx
    from PIL import Image
    sheet = tmp_path / "s.jpg"
    Image.new("RGB", (64, 48)).save(sheet)
    refused = _msg("", stop="refusal")
    refused.stop_details = types.SimpleNamespace(category="other", explanation="no")
    with pytest.raises(V.VerdictError, match="refused"):
        V.review_door(_client([refused]), str(sheet), FACTS, sleep=lambda s: None)
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.APIConnectionError(request=req)
    slept = []
    client = _client([err, _msg(VERDICT_JSON)])
    verdict, _ = V.review_door(client, str(sheet), FACTS, sleep=slept.append)
    assert verdict["ok"] is False and slept == [2.0]


def test_run_batch_mocked(tmp_path):
    pytest.importorskip("anthropic")
    from PIL import Image
    sheet = tmp_path / "s.jpg"
    Image.new("RGB", (64, 48)).save(sheet)
    items = [{"door_id": "db9999_swing_single", "sheet": str(sheet), "facts": FACTS},
             {"door_id": "db9998_swing_single", "sheet": str(sheet), "facts": dict(FACTS, door_id="db9998_swing_single")},
             {"door_id": "db9997_swing_single", "sheet": str(sheet), "facts": dict(FACTS, door_id="db9997_swing_single")}]
    ok_json = json.dumps({"door_id": "db9998_swing_single", "ok": True, "summary": "fine", "findings": []})
    results = [types.SimpleNamespace(custom_id="db9998_swing_single", result=types.SimpleNamespace(type="succeeded", message=_msg(ok_json))),
               types.SimpleNamespace(custom_id="db9999_swing_single", result=types.SimpleNamespace(type="succeeded", message=_msg(VERDICT_JSON))),
               types.SimpleNamespace(custom_id="db9997_swing_single", result=types.SimpleNamespace(type="errored", error=types.SimpleNamespace(type="api_error", message="boom")))]
    statuses = iter(["in_progress", "ended"])

    class Batches:
        def __init__(self):
            self.created = None

        def create(self, requests):
            self.created = requests
            return types.SimpleNamespace(id="msgbatch_1", processing_status="in_progress")

        def retrieve(self, bid):
            return types.SimpleNamespace(id=bid, processing_status=next(statuses), request_counts=types.SimpleNamespace(processing=1))

        def results(self, bid):
            return iter(results)

    client = types.SimpleNamespace(messages=types.SimpleNamespace(batches=Batches()))
    out = V.run_batch(client, items, model="claude-sonnet-5", sleep=lambda s: None, log=lambda *a: None)
    assert len(client.messages.batches.created) == 3
    assert client.messages.batches.created[0]["custom_id"] == "db9999_swing_single"
    assert client.messages.batches.created[0]["params"]["model"] == "claude-sonnet-5"
    assert out["db9999_swing_single"][0]["ok"] is False and out["db9999_swing_single"][0]["batch_id"] == "msgbatch_1"
    assert out["db9998_swing_single"][0]["ok"] is True
    assert out["db9997_swing_single"][0] is None and "boom" in out["db9997_swing_single"][1]


# ---------------------------------------------------------------------------------------------------------------------
# cost + report
# ---------------------------------------------------------------------------------------------------------------------
def test_estimate_cost():
    sheets = [{"width": 1600, "height": 1012, "facts": FACTS} for _ in range(10)]
    est = V.estimate_cost(sheets, "claude-opus-5")
    assert est["n_doors"] == 10 and est["image_tokens"] == 10 * V.image_tokens(1600, 1012)
    assert 0.02 < est["usd_per_door"] < 0.10           # sanity: cents per door, tens of dollars for 1000 doors
    assert abs(V.estimate_cost(sheets, "claude-opus-5", batch=True)["usd"] - est["usd"] / 2) < 1e-3
    cheaper = V.estimate_cost(sheets, "claude-sonnet-5")
    assert cheaper["usd"] < est["usd"]
    with pytest.raises(KeyError):
        V.estimate_cost(sheets, "claude-unknown")
    assert V.estimate_cost(sheets, "claude-unknown", prices={"claude-unknown": (1.0, 1.0)})["usd"] > 0
    usd = V.cost_from_usage([{"input_tokens": 1_000_000, "output_tokens": 0}], "claude-opus-5")
    assert abs(usd - 5.0) < 1e-6


def test_write_report(tmp_path):
    vdir = tmp_path / "vision"
    vdir.mkdir()
    from PIL import Image
    v1 = V.normalise_verdict(json.loads(VERDICT_JSON), "db9999_swing_single", V.REVIEWER_AGENT)
    v1["findings"][0]["triage"] = {"class": "geometry_bug", "owner": "agent:attachment", "note": "wall bumper placed where no wall exists"}
    v2 = V.normalise_verdict({"door_id": "db9998_sliding_single", "ok": True, "summary": "fine", "findings": [
        {"category": "wrong_scale", "severity": "minor", "part": "standoff", "description": "a bit big", "where": "closed / top"}]}, "db9998_sliding_single", V.REVIEWER_AGENT)
    for v in (v1, v2):
        json.dump(v, open(vdir / f"{v['door_id']}.json", "w"))
        Image.new("RGB", (32, 24)).save(vdir / f"{v['door_id']}.jpg")
    json.dump({"note": "not a verdict"}, open(vdir / "_triage.json", "w"))
    verdicts = V.load_verdicts(str(vdir))
    assert [v["door_id"] for v in verdicts] == ["db9998_sliding_single", "db9999_swing_single"]
    out = tmp_path / "VISION_REVIEW.md"
    est = V.estimate_cost([{"width": 1600, "height": 1012, "facts": FACTS}], "claude-opus-5")
    text = V.write_report(verdicts, str(out), assets=str(tmp_path), sheets_dir=str(vdir), cost_estimates=[est], handoff="- fix the bumper", intro="Intro.")
    assert out.exists() and text.startswith("# Visual common-sense review")
    assert "## How to run it" in text and "--dry-run" in text and "Expected cost" in text
    assert "floating_part" in text and "wall bumper stop" in text and "geometry bug" in text
    assert "| swing_single |" not in text or "sliding_single" in text
    assert "1 / 2 ok" in text
    assert "## Handoff" in text and "fix the bumper" in text
    assert "vision/db9999_swing_single.jpg" in text          # relative image link
    assert "## Minor findings" in text and "standoff" in text
