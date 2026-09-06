"""The vision review must not be able to turn a broken answer into a clean door.

Four things are worth locking down, because each of them silently degrades into "0 findings":
sheet rendering (a sheet with a missing panel is a review of nothing), the verdict schema (prose, an
invented category or a mismatched door id must raise, not parse), the API round trip (the request
body, the retry ladder and the batch results), and the report (a finding that is not in the report
was not reported).  There is no ANTHROPIC_API_KEY on the development machine, so the transport is
tested against a mocked client - these tests are the only thing standing behind the live path.
"""
import json
import os

import pytest

from doorbench.review import api, report, sheet
from doorbench.review.prompt import CATEGORIES, prompt_for, system_prompt
from doorbench.review.verdict import VerdictError, counts, extract_json, parse_verdict

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
SHEET_DOORS = ["db0079_sliding_single", "db0024_swing_single", "db0121_hatch_ceiling"]

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(ASSETS, "doors")),
                                reason="generated assets are required")


# ---------------------------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("door_id", SHEET_DOORS)
def test_sheet_renders_all_twelve_panels(tmp_path, door_id):
    from PIL import Image

    out = tmp_path / f"{door_id}.jpg"
    rec = sheet.render_sheet(os.path.join(ASSETS, "doors", door_id), str(out))
    assert out.is_file() and out.stat().st_size > 20_000
    assert len(rec["panels"]) == 12, "the sheet is 3 poses x 3 views plus 3 close-ups"
    keys = [p["key"] for p in rec["panels"]]
    for pose in sheet.POSES:
        for view in sheet.VIEWS:
            assert f"{pose}_{view}" in keys
    assert {"hardware_front", "hardware_back", "mechanism_open"} <= set(keys)
    with Image.open(out) as im:
        assert im.size[0] == 1200
        # 12 panels in a 3 x 4 grid under a caption band: the sheet has to be taller than it is wide
        assert im.size[1] > im.size[0]
    # the caption is the statement of intent the reviewer judges completeness against
    joined = "\n".join(rec["caption"])
    assert door_id in joined and rec["family"] in joined
    for word in ("KINEMATICS", "HINGES", "OPERATOR", "LATCH", "LOCK", "CLOSER", "LEAF"):
        assert word in joined
    assert rec["spec_facts"]["door_id"] == door_id
    assert rec["sheet_size_px"] == [1200, im.size[1]]


def test_sheet_shares_one_camera_per_column():
    """The three poses in a column must be the same shot, or rows cannot be compared."""
    r = sheet.SheetRenderer(os.path.join(ASSETS, "doors", SHEET_DOORS[0]))
    try:
        panels, _ = r.panels()
    finally:
        r.close()
    by_view = {}
    for p in panels[:9]:
        by_view.setdefault(p["view"], []).append(p["image"].shape)
    assert set(by_view) == set(sheet.VIEWS)
    assert all(len(v) == 3 for v in by_view.values())


def test_caption_states_hinge_count_and_travel():
    spec = json.load(open(os.path.join(ASSETS, "doors", "db0079_sliding_single", "spec.json")))
    meta = json.load(open(os.path.join(ASSETS, "doors", "db0079_sliding_single", "model.json")))["meta"]
    cap = "\n".join(sheet.caption_lines(spec, meta))
    assert "travel 1.067 m" in cap
    assert "surface_flat_track" in cap and "barn_hanger" in cap
    facts = sheet.spec_facts(spec, meta)
    assert facts["travel_m"] == pytest.approx(1.067)
    assert facts["hinge_count"] == 0


# ---------------------------------------------------------------------------------------------
# verdict schema
# ---------------------------------------------------------------------------------------------
GOOD = {
    "door_id": "db0079_sliding_single", "ok": False, "summary": "barn door on a flat track",
    "findings": [{"category": "guide_too_short", "severity": "blocker", "part": "flat track",
                  "description": "the trailing hanger is past the end of the rail at full open",
                  "where": "panels 7, 12", "confidence": 0.9}],
}


def test_valid_verdict_round_trips():
    v = parse_verdict(json.dumps(GOOD), "db0079_sliding_single", "unit-test", "m")
    assert v["ok"] is False and len(v["findings"]) == 1
    assert v["findings"][0]["category"] == "guide_too_short"
    assert v["reviewer"] == "unit-test" and v["model"] == "m"


def test_ok_is_recomputed_not_trusted():
    """A model that returns ok:true next to a blocker has contradicted itself."""
    bad = dict(GOOD, ok=True)
    v = parse_verdict(json.dumps(bad), "db0079_sliding_single", "unit-test")
    assert v["ok"] is False
    assert v["ok_claimed_by_model"] is True
    clean = parse_verdict('{"door_id": "d", "ok": false, "findings": []}', "d", "unit-test")
    assert clean["ok"] is True


@pytest.mark.parametrize("text", [
    "the door looks fine to me",                                    # prose, no JSON
    '{"door_id": "d"}',                                             # no findings key
    '{"door_id": "d", "findings": "none"}',                         # findings not a list
    '{"door_id": "d", "findings": [{"category": "gremlins", "severity": "blocker", "part": "p",'
    ' "description": "x", "where": "1"}]}',                          # invented category
    '{"door_id": "d", "findings": [{"category": "floating_part", "severity": "catastrophic",'
    ' "part": "p", "description": "x", "where": "1"}]}',             # invented severity
    '{"door_id": "d", "findings": [{"category": "floating_part", "severity": "major"}]}',  # short
    '{"door_id": "other", "ok": true, "findings": []}',              # wrong door
])
def test_malformed_verdicts_raise(text):
    with pytest.raises(VerdictError):
        parse_verdict(text, "d", "unit-test")


def test_fenced_and_preambled_json_is_recovered():
    fenced = "```json\n" + json.dumps(GOOD) + "\n```"
    assert parse_verdict(fenced, "db0079_sliding_single", "t")["findings"]
    chatty = "Here is my verdict.\n" + json.dumps(GOOD) + "\nHope that helps."
    assert parse_verdict(chatty, "db0079_sliding_single", "t")["findings"]
    assert extract_json('{"a": "}{"} ')["a"] == "}{"      # braces inside strings do not confuse it


def test_counts_aggregate():
    c = counts([parse_verdict(json.dumps(GOOD), "db0079_sliding_single", "t"),
                parse_verdict('{"door_id": "x", "findings": []}', "x", "t")])
    assert c == {"n_doors": 2, "n_clean": 1, "n_findings": 1,
                 "by_category": {"guide_too_short": 1},
                 "by_severity": {"blocker": 1, "major": 0, "minor": 0}}


def test_rubric_lists_every_category_and_the_artefact_exclusions():
    s = system_prompt()
    for cat in CATEGORIES:
        assert cat in s
    for excluded in ("Lighting", "shadows", "JPEG artefacts", "aliasing"):
        assert excluded in s


# ---------------------------------------------------------------------------------------------
# mocked API round trip
# ---------------------------------------------------------------------------------------------
class _Usage:
    input_tokens = 3000
    output_tokens = 400
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 1200


class _Msg:
    stop_reason = "end_turn"

    def __init__(self, text):
        self.content = [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": text}]
        self.usage = _Usage()


class _Messages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Msg(item)


class _Client:
    def __init__(self, script):
        self.messages = _Messages(script)


class _Status(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.status_code = code


@pytest.fixture
def rendered(tmp_path):
    door = SHEET_DOORS[0]
    path = tmp_path / f"{door}.jpg"
    rec = sheet.render_sheet(os.path.join(ASSETS, "doors", door), str(path))
    return rec, str(path)


def test_request_body_carries_image_rubric_and_cache_breakpoint(rendered):
    rec, path = rendered
    p = api.request_params(rec, path, model="claude-opus-5")
    assert p["model"] == "claude-opus-5"
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}, "the rubric is the cache prefix"
    assert p["system"][0]["text"] == system_prompt()
    content = p["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert len(content[0]["source"]["data"]) > 10_000
    assert rec["door_id"] in content[1]["text"]
    assert p["thinking"] == {"type": "adaptive"}
    assert p["output_config"]["effort"] == "high"
    assert api.request_params(rec, path, thinking=False).get("thinking") is None


def test_review_door_parses_and_records_usage(rendered):
    rec, path = rendered
    client = _Client([json.dumps(dict(GOOD, door_id=rec["door_id"]))])
    v = api.review_door(client, rec, path, sleep=lambda s: None)
    assert v["ok"] is False and v["findings"][0]["category"] == "guide_too_short"
    assert v["family"] == rec["family"] and v["sheet"] == rec["sheet"]
    assert v["usage"]["cache_read_input_tokens"] == 1200
    assert v["reviewer"].startswith("anthropic-api:")


def test_review_door_retries_transient_errors_then_succeeds(rendered):
    rec, path = rendered
    client = _Client([_Status(529), _Status(429),
                      json.dumps(dict(GOOD, door_id=rec["door_id"], findings=[]))])
    v = api.review_door(client, rec, path, sleep=lambda s: None)
    assert v["ok"] is True
    assert len(client.messages.calls) == 3


def test_review_door_does_not_retry_a_client_error(rendered):
    rec, path = rendered
    client = _Client([_Status(400)])
    with pytest.raises(_Status):
        api.review_door(client, rec, path, sleep=lambda s: None)
    assert len(client.messages.calls) == 1


def test_review_door_reasks_once_on_a_malformed_verdict(rendered):
    rec, path = rendered
    client = _Client(["I think it looks fine.", json.dumps(dict(GOOD, door_id=rec["door_id"]))])
    v = api.review_door(client, rec, path, sleep=lambda s: None)
    assert v["findings"]
    assert len(client.messages.calls) == 2
    # the re-ask replays the bad answer and says what was wrong
    assert "JSON verdict object only" in client.messages.calls[1]["messages"][-1]["content"][0]["text"]


def test_review_door_gives_up_on_a_persistently_malformed_verdict(rendered):
    rec, path = rendered
    client = _Client(["nope", "still nope"])
    with pytest.raises(VerdictError):
        api.review_door(client, rec, path, sleep=lambda s: None)


class _Batches:
    def __init__(self, results, statuses=("ended",)):
        self._results, self._statuses, self.created = results, list(statuses), None

    def create(self, requests):
        self.created = requests
        return type("B", (), {"id": "msgbatch_test"})()

    def retrieve(self, bid):
        s = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return type("B", (), {"id": bid, "processing_status": s})()

    def results(self, bid):
        return self._results


def test_batch_round_trip(rendered):
    rec, path = rendered
    ok = {"custom_id": rec["door_id"],
          "result": {"type": "succeeded", "message": _Msg(json.dumps(dict(GOOD, door_id=rec["door_id"])))}}
    bad = {"custom_id": "db0000_x", "result": {"type": "errored", "error": "boom"}}
    client = type("C", (), {})()
    client.messages = type("M", (), {})()
    client.messages.batches = _Batches([ok, bad], statuses=["in_progress", "ended"])
    bid = api.submit_batch(client, [(rec, path)])
    assert bid == "msgbatch_test"
    body = client.messages.batches.created[0]
    assert body["custom_id"] == rec["door_id"]
    assert body["params"]["messages"][0]["content"][0]["type"] == "image"
    verdicts, errors = api.collect_batch(client, bid, {rec["door_id"]: rec}, sleep=lambda s: None)
    assert len(verdicts) == 1 and verdicts[0]["door_id"] == rec["door_id"]
    assert errors and errors[0]["door_id"] == "db0000_x"


# ---------------------------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------------------------
def test_image_tokens_respect_the_api_downscale():
    assert api.image_tokens(1200, 1324) == pytest.approx(1200 * 1324 / 750, rel=0.01)
    # anything longer than 1568 px on its long edge is resized by the API before tokenising
    big = api.image_tokens(4000, 3000)
    assert big == api.image_tokens(1568, 1176)
    assert big < api.image_tokens(1568, 1568)


def test_cost_estimate_uses_the_real_sheets_and_batching_halves_it(rendered):
    rec, _ = rendered
    sheets = [rec] * 10
    single = api.estimate_cost(sheets, "claude-opus-5")
    batched = api.estimate_cost(sheets, "claude-opus-5", batch=True)
    assert single["n_doors"] == 10
    assert single["image_tokens"] == 10 * api.image_tokens(*rec["sheet_size_px"])
    assert batched["est_cost_usd"] == pytest.approx(single["est_cost_usd"] / 2, rel=1e-3)
    cheap = api.estimate_cost(sheets, "claude-haiku-4-5")
    assert cheap["est_cost_usd"] < single["est_cost_usd"]
    # the cached rubric is billed once at 1.25x and then at 0.1x, not 10 full times
    uncached = api.estimate_cost(sheets, "claude-opus-5", cached_system=False)
    assert uncached["system_tokens_billed"] > single["system_tokens_billed"] * 3


def test_actual_cost_prices_cache_reads_lower():
    full = api.actual_cost([{"input_tokens": 10_000, "output_tokens": 1000}], "claude-opus-5")
    cached = api.actual_cost([{"input_tokens": 0, "cache_read_input_tokens": 10_000,
                               "output_tokens": 1000}], "claude-opus-5")
    assert cached["cost_usd"] < full["cost_usd"]


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------
def _verdict(door, family, findings, **kw):
    return {"door_id": door, "family": family, "ok": not findings, "findings": findings,
            "sheet": f"{door}.jpg", "reviewer": "unit-test", "summary": "a door", **kw}


def test_report_contains_every_finding_and_the_gate_comparison(tmp_path):
    verdicts = [
        _verdict("db0079_sliding_single", "sliding_single",
                 [dict(GOOD["findings"][0])]),
        _verdict("db0024_swing_single", "swing_single",
                 [{"category": "floating_part", "severity": "minor", "part": "door stop",
                   "description": "sits 4 mm off the floor", "where": "panel 3", "confidence": 0.4}]),
        _verdict("db0002_swing_single", "swing_single", []),
    ]
    md = report.render(verdicts, ASSETS, run={"selection": "unit test"},
                       triage_md="## Triage\n\nhand written", how_to_run="RUNME")
    assert "db0079_sliding_single" in md and "db0024_swing_single" in md
    assert "guide_too_short" in md and "floating_part" in md
    assert "![db0079_sliding_single](review/vision/db0079_sliding_single.jpg)" in md
    assert "RUNME" in md and "hand written" in md
    assert "3 doors reviewed" in md.replace("**", "")
    # the gate comparison must name the doors that pass every deterministic check
    assert "all gates pass" in md
    assert "sits 4 mm off the floor" in md               # minors are tabulated, not dropped
    assert "db0002_swing_single" not in md.split("## Blockers")[1]   # a clean door is not in the gallery


def test_report_round_trips_verdicts_from_disk(tmp_path):
    d = tmp_path / "vision"
    d.mkdir()
    (d / "db0079_sliding_single.json").write_text(json.dumps(
        _verdict("db0079_sliding_single", "sliding_single", [dict(GOOD["findings"][0])])))
    (d / "db0079_sliding_single.sheet.json").write_text('{"door_id": "x", "sheet": "y.jpg"}')
    (d / "index.json").write_text('{"run": {}}')
    loaded = report.load_verdicts(str(d))
    assert [v["door_id"] for v in loaded] == ["db0079_sliding_single"], "sheet records are not verdicts"


def test_category_by_family_table_totals():
    verdicts = [_verdict("a", "swing_single", [dict(GOOD["findings"][0])]),
                _verdict("b", "swing_single", [dict(GOOD["findings"][0])]),
                _verdict("c", "rollup", [])]
    t = report.category_by_family(verdicts)
    assert "swing_single" in t and "guide_too_short" in t and "**2**" in t
    assert "Families with no findings at all: rollup" in t


def test_prompt_for_is_deterministic(rendered):
    rec, _ = rendered
    assert prompt_for(rec) == prompt_for(rec), "the rubric must be byte-stable or caching never hits"


# ---------------------------------------------------------------------------------------------
# the one dataset defect this review fixed
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("extra,pattern", [("louver_vent", "louver_vent_"),
                                           ("weather_drip_cap", "weather_drip_cap"),
                                           ("hold_open_kickdown", "kickdown_")])
def test_declared_extra_is_actually_drawn(extra, pattern):
    """A door that says it has a louvre vent must have one.

    These three extras are in ``taxonomy.EXTRAS``, are sampled into specs, and are charged
    0.9 / 0.3 / 0.3 kg of hardware mass in ``physics.py`` - and until the vision review looked at the
    pictures, no builder drew any of them.  156 declared extras across 153 doors had no geometry at
    all; these three were 48 of them.

    The check builds the door from its spec rather than reading ``assets/``, so it holds whether or
    not the shipped dataset has been regenerated since the generator changed.
    """
    from doorbench.build import build_model
    from doorbench.spec import generate_all

    specs = [s for s in generate_all() if extra in (s.get("extras") or [])]
    assert len(specs) >= 5, f"expected the sampler to place {extra} on several doors, saw {len(specs)}"
    checked = 0
    for spec in specs:
        if extra == "louver_vent" and (spec["leaf"].get("pet_flap")
                                       or spec["leaf"].get("panel_style") in ("louver_full", "louver_half")):
            continue          # a pet flap wants the same bottom third of the leaf; a louvred leaf already has slats
        model = build_model(spec)
        names = [g.name for b in model.bodies for g in b.geoms]
        assert any(pattern in n for n in names), f"{spec['id']} declares {extra} and draws none of it"
        checked += 1
    assert checked >= 5
