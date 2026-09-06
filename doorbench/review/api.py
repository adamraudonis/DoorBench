"""Anthropic transport for the vision review: cost estimate, retries, batching, resumability.

Two paths, the same request body:

* ``review_door`` - one Messages request per door.  Use it for a handful of doors, or when you want
  the verdicts as they arrive.
* ``submit_batch`` / ``collect_batch`` - the Message Batches API, half price, up to 24 h to return.
  Use it for the whole dataset.

The rubric is identical for every door, so it is sent as a cached system block: after the first door
each request re-reads it at a tenth of the input price.  The per-door text and the image are what vary.

**There is no ANTHROPIC_API_KEY on the machine this was written on.**  Every line below is exercised by
``tests/test_vision_review.py`` against a mocked client - the request body, the retry ladder, the
cost estimate, the batch round trip and the verdict parsing - but none of it has been run against the
live API.  Treat the first live run as a smoke test: start with ``--limit 3``.
"""
from __future__ import annotations

import base64
import json
import math
import os
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .prompt import prompt_for, system_prompt
from .verdict import VerdictError, parse_verdict

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 8000
DEFAULT_EFFORT = "high"

# USD per million tokens, from the Anthropic pricing table cached in the `claude-api` skill
# (2026-06-24).  Used only for the pre-run estimate and the post-run actuals; ``--price-in`` /
# ``--price-out`` override it if the table has moved.
PRICING: Dict[str, Tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-fable-5-1": (10.0, 50.0),
}
BATCH_DISCOUNT = 0.5
API_MAX_EDGE_PX = 1568        # the API downscales anything longer on its long edge before tokenising
IMAGE_TOKENS_PER_PX = 1.0 / 750.0
CHARS_PER_TOKEN = 3.7         # conservative for English prose with JSON in it
# Adaptive thinking is on by default on Opus 5, and thinking tokens are billed as output.  Measured
# against nothing (no key here); this is a planning figure, and --est-output-tokens overrides it.
DEFAULT_EST_OUTPUT_TOKENS = 1400


# ---------------------------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------------------------
def image_tokens(width: int, height: int) -> int:
    """Tokens the API charges for one image, after its own long-edge downscale."""
    w, h = float(width), float(height)
    longest = max(w, h)
    if longest > API_MAX_EDGE_PX:
        s = API_MAX_EDGE_PX / longest
        w, h = w * s, h * s
    return int(math.ceil(w * h * IMAGE_TOKENS_PER_PX))


def text_tokens(text: str) -> int:
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


def estimate_cost(sheets: Iterable[Dict[str, Any]], model: str = DEFAULT_MODEL, batch: bool = False,
                  est_output_tokens: int = DEFAULT_EST_OUTPUT_TOKENS,
                  price_in: Optional[float] = None, price_out: Optional[float] = None,
                  cached_system: bool = True) -> dict:
    """Pre-run cost estimate from the ACTUAL rendered sheets.

    ``sheets`` are sheet records (``sheet_size_px`` + the prompt text), so the estimate is computed
    from the images that will really be sent, not from a nominal size.
    """
    p_in, p_out = PRICING.get(model, (5.0, 25.0))
    p_in = price_in if price_in is not None else p_in
    p_out = price_out if price_out is not None else p_out
    sys_tokens = text_tokens(system_prompt())
    n = 0
    img_tok = 0
    txt_tok = 0
    for rec in sheets:
        n += 1
        w, h = rec.get("sheet_size_px") or (1200, 1324)
        img_tok += image_tokens(int(w), int(h))
        txt_tok += text_tokens(prompt_for(rec)["user_text"])
    # the rubric is a cached prefix: written once at 1.25x, re-read at 0.1x
    if cached_system and n:
        sys_cost_tok = sys_tokens * 1.25 + sys_tokens * 0.1 * (n - 1)
    else:
        sys_cost_tok = sys_tokens * n
    in_tok = img_tok + txt_tok
    out_tok = est_output_tokens * n
    mult = BATCH_DISCOUNT if batch else 1.0
    cost = mult * ((in_tok + sys_cost_tok) / 1e6 * p_in + out_tok / 1e6 * p_out)
    return {
        "n_doors": n, "model": model, "batch": bool(batch),
        "image_tokens": img_tok, "text_tokens": txt_tok,
        "system_tokens_billed": int(sys_cost_tok), "est_output_tokens": out_tok,
        "price_in_per_mtok": p_in, "price_out_per_mtok": p_out,
        "est_cost_usd": round(cost, 4),
        "est_cost_usd_per_door": round(cost / n, 5) if n else 0.0,
    }


def actual_cost(usages: Iterable[Any], model: str = DEFAULT_MODEL, batch: bool = False,
                price_in: Optional[float] = None, price_out: Optional[float] = None) -> dict:
    """Dollars actually spent, from the ``usage`` blocks the API returned."""
    p_in, p_out = PRICING.get(model, (5.0, 25.0))
    p_in = price_in if price_in is not None else p_in
    p_out = price_out if price_out is not None else p_out
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
           "cache_read_input_tokens": 0}
    for u in usages:
        for k in tot:
            tot[k] += int(getattr(u, k, 0) or 0) if not isinstance(u, dict) else int(u.get(k, 0) or 0)
    mult = BATCH_DISCOUNT if batch else 1.0
    cost = mult * ((tot["input_tokens"] + 1.25 * tot["cache_creation_input_tokens"]
                    + 0.1 * tot["cache_read_input_tokens"]) / 1e6 * p_in
                   + tot["output_tokens"] / 1e6 * p_out)
    return {**tot, "cost_usd": round(cost, 4), "batch": bool(batch), "model": model}


# ---------------------------------------------------------------------------------------------
# request construction
# ---------------------------------------------------------------------------------------------
def image_block(jpeg_path: str) -> dict:
    with open(jpeg_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def request_params(record: dict, sheet_path: str, model: str = DEFAULT_MODEL,
                   max_tokens: int = DEFAULT_MAX_TOKENS, effort: str = DEFAULT_EFFORT,
                   thinking: bool = True) -> dict:
    """The Messages request body for one door.  Shared by the single and the batch path."""
    p = prompt_for(record)
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        # one cached breakpoint on the rubric: it is byte-identical for every door in the run
        "system": [{"type": "text", "text": p["system"], "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [image_block(sheet_path),
                                                  {"type": "text", "text": p["user_text"]}]}],
        "output_config": {"effort": effort},
    }
    if thinking:
        params["thinking"] = {"type": "adaptive"}
    return params


def response_text(message: Any) -> str:
    """The text blocks of a Message, concatenated (thinking blocks are skipped)."""
    out = []
    for block in getattr(message, "content", None) or []:
        btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
        if btype == "text":
            out.append(getattr(block, "text", None) if not isinstance(block, dict) else block.get("text"))
    return "\n".join(t for t in out if t)


# ---------------------------------------------------------------------------------------------
# single-request path
# ---------------------------------------------------------------------------------------------
RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 504, 529)


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return int(status) in RETRYABLE_STATUS
    return exc.__class__.__name__ in ("APIConnectionError", "APITimeoutError", "RateLimitError",
                                      "InternalServerError")


def call_with_retry(fn: Callable[[], Any], attempts: int = 5, base_delay: float = 1.5,
                    max_delay: float = 60.0, sleep=time.sleep, rng=random.random) -> Any:
    """Exponential backoff over the SDK's own retries, for the errors worth retrying."""
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:                       # noqa: BLE001 - re-raised below when not retryable
            if not _is_retryable(e) or i == attempts - 1:
                raise
            last = e
            sleep(min(base_delay * (2 ** i) + rng(), max_delay))
    raise last  # pragma: no cover - unreachable, the loop either returns or raises


def review_door(client: Any, record: dict, sheet_path: str, model: str = DEFAULT_MODEL,
                max_tokens: int = DEFAULT_MAX_TOKENS, effort: str = DEFAULT_EFFORT,
                thinking: bool = True, attempts: int = 5, parse_retries: int = 1,
                sleep=time.sleep) -> dict:
    """One door, one verdict.  Raises on a malformed answer after ``parse_retries`` re-asks."""
    params = request_params(record, sheet_path, model, max_tokens, effort, thinking)
    last_err: Optional[Exception] = None
    for attempt in range(parse_retries + 1):
        msg = call_with_retry(lambda: client.messages.create(**params), attempts=attempts, sleep=sleep)
        text = response_text(msg)
        if getattr(msg, "stop_reason", None) == "refusal":
            raise VerdictError(f"{record['door_id']}: model declined the request")
        try:
            v = parse_verdict(text, record["door_id"], reviewer=f"anthropic-api:{model}", model=model,
                              extra={"usage": _usage_dict(msg), "stop_reason": getattr(msg, "stop_reason", None)})
            v["sheet"] = record.get("sheet")
            v["family"] = record.get("family")
            return v
        except VerdictError as e:
            last_err = e
            if attempt < parse_retries:
                params = dict(params)
                params["messages"] = list(params["messages"]) + [
                    {"role": "assistant", "content": [{"type": "text", "text": text[:2000] or "(no text)"}]},
                    {"role": "user", "content": [{"type": "text", "text":
                        f"That was not a valid verdict ({e}).  Reply with the JSON verdict object only, "
                        f"nothing else."}]}]
    raise last_err  # type: ignore[misc]


def _usage_dict(msg: Any) -> dict:
    u = getattr(msg, "usage", None)
    if u is None:
        return {}
    return {k: int(getattr(u, k, 0) or 0) for k in
            ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}


# ---------------------------------------------------------------------------------------------
# batch path (50 % cheaper, up to 24 h)
# ---------------------------------------------------------------------------------------------
def submit_batch(client: Any, jobs: List[Tuple[dict, str]], model: str = DEFAULT_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS, effort: str = DEFAULT_EFFORT,
                 thinking: bool = True) -> str:
    """Submit (record, sheet_path) pairs as one batch; returns the batch id."""
    requests = [{"custom_id": rec["door_id"],
                 "params": request_params(rec, path, model, max_tokens, effort, thinking)}
                for rec, path in jobs]
    batch = call_with_retry(lambda: client.messages.batches.create(requests=requests))
    return batch.id if not isinstance(batch, dict) else batch["id"]


def collect_batch(client: Any, batch_id: str, records: Dict[str, dict], model: str = DEFAULT_MODEL,
                  poll_s: float = 30.0, timeout_s: float = 24 * 3600, sleep=time.sleep,
                  now=time.time) -> Tuple[List[dict], List[dict]]:
    """Poll until the batch ends, then parse every result.  Returns (verdicts, errors)."""
    t0 = now()
    while True:
        batch = call_with_retry(lambda: client.messages.batches.retrieve(batch_id))
        status = getattr(batch, "processing_status", None) or (batch.get("processing_status") if isinstance(batch, dict) else None)
        if status == "ended":
            break
        if now() - t0 > timeout_s:
            raise TimeoutError(f"batch {batch_id} still {status} after {timeout_s:.0f}s")
        sleep(poll_s)
    verdicts, errors = [], []
    for result in client.messages.batches.results(batch_id):
        cid = getattr(result, "custom_id", None) or result["custom_id"]
        res = getattr(result, "result", None) or result["result"]
        rtype = getattr(res, "type", None) or (res.get("type") if isinstance(res, dict) else None)
        if rtype != "succeeded":
            errors.append({"door_id": cid, "error": rtype, "detail": str(res)[:400]})
            continue
        msg = getattr(res, "message", None) or res["message"]
        rec = records.get(cid, {"door_id": cid})
        try:
            v = parse_verdict(response_text(msg), cid, reviewer=f"anthropic-api:{model}", model=model,
                              extra={"usage": _usage_dict(msg), "batch_id": batch_id})
            v["sheet"] = rec.get("sheet")
            v["family"] = rec.get("family")
            verdicts.append(v)
        except VerdictError as e:
            errors.append({"door_id": cid, "error": "verdict_parse", "detail": str(e)})
    return verdicts, errors


def make_client(api_key: Optional[str] = None):
    """The Anthropic client, or a clear error saying exactly what is missing."""
    try:
        import anthropic
    except ImportError as e:                                   # pragma: no cover
        raise RuntimeError("the `anthropic` package is not installed (pip install anthropic)") from e
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        # the SDK also resolves an `ant auth login` profile; let it try, and surface its error
        try:
            return anthropic.Anthropic(max_retries=3)
        except Exception as e:                                 # pragma: no cover
            raise RuntimeError(
                "no Anthropic credentials: set ANTHROPIC_API_KEY, or run `ant auth login`. "
                "Use --dry-run to render sheets and prompts without calling the API.") from e
    return anthropic.Anthropic(api_key=key, max_retries=3) if key else anthropic.Anthropic(max_retries=3)
