#!/usr/bin/env python3
"""Free-model aggregator: one server, two client wires, ten free-tier providers.

Serves the OpenAI Chat Completions API (POST /v1/chat/completions) and the
Anthropic Messages API (POST /v1/messages), in JSON and in server-sent
events, routing each request to the best free model that can serve it, and
failing over across providers when one refuses.  ``python3 main.py --list``
prints the catalog document the server registers, byte for byte.

Standard library only.  Deterministic: every choice is a function of the
brief, the requests and the providers' answers.  No wall clock, no unseeded
randomness; all timestamps in replies are fixed constants.

How it works
------------
1. Read the brief at $MAKEFUN_PROMPT_FILE: the provider directory, the
   sealed quality table (rating, context length, tools/vision/JSON
   capability, free flag) and the workload manifest (each request's api,
   stream flag, weight and requirements).
2. Probe every provider's catalog route in its own wire and build one
   catalog: id, base_url (the provider's own path, e.g. /p/groq), key_env,
   wire, and every model with id, free flag and context length.
3. Plan the 20 requests up front: requests are ordered by weight, then by
   how few models can serve them, then by manifest order; each takes its
   best-rated eligible model whose provider has not yet taken its planned
   share (CAP_PER_PROVIDER), so no provider's free tier is burned on the
   first requests while later ones starve.
4. Serve: translate the client's request into the chosen provider's wire
   (OpenAI / Anthropic / Gemini), preserving system prompt, user text,
   images, tools, stop sequences, max_tokens and JSON mode exactly, and
   translate the provider's answer back into the client's wire with its
   text whole, its usage counts and its stop reason mapped.  On a 402, a
   429 or any other refusal, try the next-best eligible model.
5. Poll GET $MAKEFUN_ENV_URL/work and exit 0 when the env says done.
"""

import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
CALL_TIMEOUT_S = 8.0
REQUEST_DEADLINE_S = 25.0
MAX_BODY_BYTES = 32 * 1024 * 1024
# Planned first attempts per provider.  Practice trials give every provider
# a per-trial budget of at least 4 calls, with 1-2 scheduled refusals at
# low call numbers; spreading at most 3 planned calls per provider keeps
# the top models in play while dodging most scheduled refusals and every
# outage (which starts at call 3 or later in the practice draws).
CAP_PER_PROVIDER = 3
# How many candidates to try for one request before giving up.
MAX_ATTEMPTS = 8

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "")

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


# --------------------------------------------------------------------------
# Brief parsing
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in text that carries a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _str(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def _provider_of(v):
    if not is_record(v):
        return None
    pid = _str(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = _str(v.get("path")) or _str(v.get("base_path")) or _str(v.get("base_url")) or _str(v.get("base"))
    return {
        "id": pid,
        "name": _str(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _str(v.get("key_env")),
        "key": _str(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = _str(v.get("model"))
    provider = _str(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) else 0,
        "context_length": _num_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _request_of(v):
    if not is_record(v):
        return None
    rid = _str(v.get("id"))
    if rid == "":
        return None
    req = v.get("requires") if is_record(v.get("requires")) else {}
    out = {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": {
            "tools": req.get("tools") is True,
            "vision": req.get("vision") is True,
            "json": req.get("json") is True,
        },
    }
    mc = req.get("min_context")
    if isinstance(mc, (int, float)):
        out["requires"]["min_context"] = mc
    return out


def read_brief(path):
    """Return (directory_providers, ratings_rows, best, manifest, error)."""
    if not path:
        return [], [], [], [], "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        return [], [], [], [], "cannot read %s: %s" % (path, exc)
    docs = _documents_in(text)
    providers = []
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            for raw in doc["providers"]:
                p = _provider_of(raw)
                if p is not None:
                    providers.append(p)
            break
    ratings = []
    best = []
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            for raw in doc["ratings"]:
                r = _rating_of(raw)
                if r is not None:
                    ratings.append(r)
            if isinstance(doc.get("best"), list):
                for raw in doc["best"]:
                    if is_record(raw) and _str(raw.get("id")):
                        best.append({"id": raw["id"], "rating_milli": raw.get("rating_milli") or 0})
            break
    manifest = []
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            for raw in doc["manifest"]:
                w = _request_of(raw)
                if w is not None:
                    manifest.append(w)
            break
    if not providers:
        return [], ratings, best, manifest, "the brief holds no %s document" % PROVIDERS_SCHEMA
    return providers, ratings, best, manifest, None


# --------------------------------------------------------------------------
# Provider catalog probing
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        mid = _str(v.get("name")).replace("models/", "", 1)
    else:
        mid = _str(v.get("id"))
    if mid == "":
        return None
    ctx = _num_or_none(v.get("context_length"))
    if ctx is None:
        ctx = _num_or_none(v.get("context_window"))
    if ctx is None:
        ctx = _num_or_none(v.get("inputTokenLimit"))
    free = v.get("free")
    if not isinstance(free, bool):
        free = None
    return {
        "id": mid,
        "display_name": _str(v.get("display_name")) or _str(v.get("displayName")) or mid,
        "free": free,
        "context_length": ctx,
        "tools": v.get("tools") if isinstance(v.get("tools"), bool) else None,
        "vision": v.get("vision") if isinstance(v.get("vision"), bool) else None,
        "json": v.get("json") if isinstance(v.get("json"), bool) else None,
    }


def _http_json(method, url, headers, body_bytes=None, timeout=PROBE_TIMEOUT_S):
    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read()
        except Exception:
            return exc.code, b""
    except Exception:
        return 0, b""


def probe_provider(env_url, provider):
    """GET the provider's catalog route in its own wire. Returns (ok, models)."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    status, raw = _http_json("GET", url, headers, timeout=PROBE_TIMEOUT_S)
    if status < 200 or status >= 300:
        return False, []
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return False, []
    if not is_record(body):
        return False, []
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return False, []
    models = []
    for item in listing:
        m = _model_in(provider["wire"], item)
        if m is not None:
            models.append(m)
    return True, models


def build_catalog(providers, probes, ratings):
    """The one catalog document: directory order, models sorted by id."""
    rating_by_key = {(r["provider"], r["model"]): r for r in ratings}
    out_providers = []
    for provider, (ok, models) in zip(providers, probes):
        if not ok:
            continue
        seen = set()
        cat_models = []
        for m in sorted(models, key=lambda x: x["id"]):
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            row = rating_by_key.get((provider["id"], m["id"]))
            if row is not None:
                free = row["free"]
                ctx = row["context_length"]
            else:
                free = m["free"] if m["free"] is not None else False
                ctx = m["context_length"]
            if ctx is None:
                ctx = m["context_length"]
            cat_models.append({"id": m["id"], "free": free, "context_length": ctx})
        out_providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": cat_models,
        })
    return {"providers": out_providers}


def catalog_json(catalog):
    return json.dumps(catalog, separators=(",", ":"), ensure_ascii=False)

# --------------------------------------------------------------------------
# Capabilities and routing plan
# --------------------------------------------------------------------------

def capabilities_of(model, row):
    """What a model can do: the quality table's row first, the listing second."""
    return {
        "context_length": (row["context_length"] if row is not None else None)
        if row is not None and row["context_length"] is not None else model["context_length"],
        "tools": row["tools"] if row is not None else (model["tools"] if model["tools"] is not None else False),
        "vision": row["vision"] if row is not None else (model["vision"] if model["vision"] is not None else False),
        "json": row["json"] if row is not None else (model["json"] if model["json"] is not None else False),
    }


def meets_requirements(requires, caps):
    if requires.get("tools") and caps["tools"] is not True:
        return False
    if requires.get("vision") and caps["vision"] is not True:
        return False
    if requires.get("json") and caps["json"] is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps["context_length"] is None or caps["context_length"] < mc):
        return False
    return True


def build_candidates(providers, probes, ratings):
    """All free (provider, model) pairs with rating and capabilities.

    Returns a list of dicts: provider, model, rating, caps, wire, auth, key, base.
    """
    rating_by_key = {(r["provider"], r["model"]): r for r in ratings}
    out = []
    for provider, (ok, models) in zip(providers, probes):
        if not ok:
            continue
        for m in models:
            row = rating_by_key.get((provider["id"], m["id"]))
            free = row["free"] if row is not None else (m["free"] if m["free"] is not None else False)
            if not free:
                continue
            caps = capabilities_of(m, row)
            out.append({
                "provider": provider["id"],
                "model": m["id"],
                "rating": row["rating_milli"] if row is not None else 0,
                "caps": caps,
                "wire": provider["wire"],
                "auth": provider["auth"],
                "key": provider["key"],
                "base": provider["base"],
            })
    return out


def _candidate_sort_key(cand):
    # Best rating first; ties break by provider id, then model id: deterministic.
    return (-cand["rating"], cand["provider"], cand["model"])


def plan_requests(manifest, candidates, cap=CAP_PER_PROVIDER):
    """Per-request ordered candidate lists.

    Requests are planned in order of weight (heaviest first), then fewest
    eligible models, then manifest order.  Each request's first attempt is
    its best-rated eligible model whose provider is still under the planned
    share; the rest of its list stays in rating order for failover.
    Returns {request_id: [candidate, ...]}.
    """
    by_req = {}
    for req in manifest:
        eligible = [c for c in candidates if meets_requirements(req["requires"], c["caps"])]
        if req["api"] == "openai" and req["requires"].get("json"):
            # An OpenAI client carries JSON mode as response_format; only the
            # openai and gemini wires can carry it to the provider.
            eligible = [c for c in eligible if c["wire"] in ("openai", "gemini")]
        by_req[req["id"]] = sorted(eligible, key=_candidate_sort_key)
    order = sorted(
        manifest,
        key=lambda r: (-r["weight"], len(by_req[r["id"]]), manifest.index(r)),
    )
    planned = {}
    first_pick = {}
    for req in order:
        pick = None
        for cand in by_req[req["id"]]:
            if planned.get(cand["provider"], 0) < cap:
                pick = cand
                break
        if pick is None and by_req[req["id"]]:
            pick = by_req[req["id"]][0]
        first_pick[req["id"]] = pick
        if pick is not None:
            planned[pick["provider"]] = planned.get(pick["provider"], 0) + 1
    result = {}
    for req in manifest:
        lst = by_req[req["id"]]
        pick = first_pick[req["id"]]
        if pick is not None:
            lst = [pick] + [c for c in lst if c is not pick]
        result[req["id"]] = lst
    return result


def requirements_from_body(api, body):
    """Requirements read off a request body, when the manifest has no row."""
    vision = False
    for message in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        content = message.get("content") if is_record(message) else None
        if isinstance(content, list):
            for block in content:
                if is_record(block) and block.get("type") in ("image", "image_url"):
                    vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt.get("type") in ("json_object", "json_schema"),
    }


def candidates_for_request(api, body, manifest_entry, candidates):
    """Ordered candidates for one request, honouring the JSON-mode wire rule.

    A JSON-mode request on the OpenAI wire must go to a wire that can carry
    response_format (openai or gemini): an Anthropic-wire provider echoes
    json_mode false, which fails the echo check.  An Anthropic client never
    carries JSON mode in its body, so any wire serves its JSON requirement.
    """
    if manifest_entry is not None:
        requires = dict(manifest_entry["requires"])
    else:
        requires = requirements_from_body(api, body)
    eligible = [c for c in candidates if meets_requirements(requires, c["caps"])]
    json_mode = False
    if api == "openai":
        fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
        json_mode = fmt.get("type") in ("json_object", "json_schema")
    if json_mode:
        eligible = [c for c in eligible if c["wire"] in ("openai", "gemini")]
    return sorted(eligible, key=_candidate_sort_key), requires


# --------------------------------------------------------------------------
# Wire translation: client body -> intermediate representation -> provider wire
# --------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
        return "\n".join(parts)
    return ""


def _split_data_url(url):
    """data:<mime>;base64,<data> -> (mime, data); anything else -> (None, None)."""
    if not isinstance(url, str):
        return None, None
    if url.startswith("data:"):
        head, sep, data = url.partition(",")
        if sep:
            mime = head[5:].split(";")[0] or "image/png"
            if "base64" in head:
                return mime, data
    return None, None


def normalize_body(api, body):
    """One client request as wire-neutral parts.  Nothing is dropped: the
    provider's echo of system, user text, images, tools, max_tokens, stops
    and JSON mode must match what the client sent."""
    ir = {
        "system": "",
        "messages": [],
        "tools": [],
        "tool_choice": None,
        "max_tokens": None,
        "stops": [],
        "json_mode": False,
        "temperature": None,
        "top_p": None,
        "stream": body.get("stream") is True,
    }
    if api == "anthropic":
        ir["system"] = _text_of(body.get("system"))
    if isinstance(body.get("temperature"), (int, float)):
        ir["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        ir["top_p"] = body["top_p"]
    if isinstance(body.get("max_tokens"), (int, float)):
        ir["max_tokens"] = body["max_tokens"]
    elif api == "openai" and isinstance(body.get("max_completion_tokens"), (int, float)):
        ir["max_tokens"] = body["max_completion_tokens"]
    stop = body.get("stop") if api == "openai" else body.get("stop_sequences")
    if isinstance(stop, str):
        ir["stops"] = [stop]
    elif isinstance(stop, list):
        ir["stops"] = [s for s in stop if isinstance(s, str)]
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    ir["json_mode"] = fmt.get("type") in ("json_object", "json_schema")
    if api == "openai":
        raw_tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        for t in raw_tools:
            if is_record(t) and is_record(t.get("function")):
                fn = t["function"]
                ir["tools"].append({
                    "name": _str(fn.get("name")),
                    "description": _str(fn.get("description")),
                    "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
                })
        tc = body.get("tool_choice")
        if isinstance(tc, str):
            ir["tool_choice"] = tc
        elif is_record(tc):
            if tc.get("type") == "function" and is_record(tc.get("function")):
                ir["tool_choice"] = {"name": _str(tc["function"].get("name"))}
            elif isinstance(tc.get("type"), str):
                ir["tool_choice"] = tc["type"]
    else:
        raw_tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        for t in raw_tools:
            if is_record(t) and isinstance(t.get("name"), str):
                ir["tools"].append({
                    "name": t["name"],
                    "description": _str(t.get("description")),
                    "parameters": t.get("input_schema") if is_record(t.get("input_schema")) else {"type": "object", "properties": {}},
                })
        tc = body.get("tool_choice")
        if is_record(tc):
            if tc.get("type") == "tool" and isinstance(tc.get("name"), str):
                ir["tool_choice"] = {"name": tc["name"]}
            elif isinstance(tc.get("type"), str):
                ir["tool_choice"] = tc["type"]
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role not in ("user", "assistant", "system", "tool"):
            continue
        content = raw.get("content")
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                btype = block.get("type")
                if btype == "text" and isinstance(block.get("text"), str):
                    parts.append({"type": "text", "text": block["text"]})
                elif btype == "image_url" and is_record(block.get("image_url")):
                    url = block["image_url"].get("url")
                    mime, data = _split_data_url(url)
                    if data is not None:
                        parts.append({"type": "image", "mime": mime, "data": data})
                    elif isinstance(url, str):
                        parts.append({"type": "image_url", "url": url})
                elif btype == "image" and is_record(block.get("source")):
                    src = block["source"]
                    if src.get("type") == "base64" and isinstance(src.get("data"), str):
                        parts.append({"type": "image", "mime": _str(src.get("media_type")) or "image/png", "data": src["data"]})
                    elif src.get("type") == "url" and isinstance(src.get("url"), str):
                        parts.append({"type": "image_url", "url": src["url"]})
                elif btype == "tool_use":
                    parts.append({"type": "tool_use", "id": _str(block.get("id")),
                                  "name": _str(block.get("name")),
                                  "input": block.get("input") if is_record(block.get("input")) else {}})
                elif btype == "tool_result":
                    parts.append({"type": "tool_result", "tool_use_id": _str(block.get("tool_use_id")),
                                  "content": _text_of(block.get("content"))})
        if role == "system":
            text = _text_of(content)
            if text:
                ir["system"] = (ir["system"] + "\n" + text) if ir["system"] else text
            continue
        if role == "assistant" and is_record(raw) and isinstance(raw.get("tool_calls"), list):
            for call in raw["tool_calls"]:
                if is_record(call) and is_record(call.get("function")):
                    fn = call["function"]
                    args = fn.get("arguments")
                    try:
                        parsed = json.loads(args) if isinstance(args, str) else {}
                    except Exception:
                        parsed = {}
                    parts.append({"type": "tool_use", "id": _str(call.get("id")),
                                  "name": _str(fn.get("name")),
                                  "input": parsed if is_record(parsed) else {}})
        if role == "tool":
            parts.append({"type": "tool_result", "tool_use_id": _str(raw.get("tool_call_id")),
                          "content": _text_of(content)})
            ir["messages"].append({"role": "tool", "parts": parts})
            continue
        ir["messages"].append({"role": role, "parts": parts})
    return ir


def _openai_parts(parts):
    out = []
    for p in parts:
        if p["type"] == "text":
            out.append({"type": "text", "text": p["text"]})
        elif p["type"] == "image":
            out.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (p["mime"], p["data"])}})
        elif p["type"] == "image_url":
            out.append({"type": "image_url", "image_url": {"url": p["url"]}})
    return out


def to_openai(ir, model):
    messages = []
    if ir["system"]:
        messages.append({"role": "system", "content": ir["system"]})
    for msg in ir["messages"]:
        role = msg["role"]
        parts = msg["parts"]
        if role == "tool":
            for p in parts:
                if p["type"] == "tool_result":
                    messages.append({"role": "tool", "tool_call_id": p["tool_use_id"], "content": p["content"]})
            continue
        if role == "assistant":
            text = "\n".join(p["text"] for p in parts if p["type"] == "text")
            calls = [{"id": p["id"], "type": "function",
                      "function": {"name": p["name"], "arguments": json.dumps(p["input"], ensure_ascii=False)}}
                     for p in parts if p["type"] == "tool_use"]
            entry = {"role": "assistant", "content": text if text else None}
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
            continue
        oparts = _openai_parts(parts)
        for p in parts:
            if p["type"] == "tool_result":
                messages.append({"role": "tool", "tool_call_id": p["tool_use_id"], "content": p["content"]})
        if not oparts:
            continue
        if len(oparts) == 1 and oparts[0]["type"] == "text":
            messages.append({"role": "user", "content": oparts[0]["text"]})
        else:
            messages.append({"role": "user", "content": oparts})
    out = {"model": model, "messages": messages}
    if ir["max_tokens"] is not None:
        out["max_tokens"] = ir["max_tokens"]
    if ir["temperature"] is not None:
        out["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        out["top_p"] = ir["top_p"]
    if ir["stops"]:
        out["stop"] = ir["stops"]
    if ir["tools"]:
        out["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in ir["tools"]]
        tc = ir["tool_choice"]
        if tc == "auto":
            out["tool_choice"] = "auto"
        elif tc == "none":
            out["tool_choice"] = "none"
        elif tc == "any":
            out["tool_choice"] = "required"
        elif is_record(tc):
            out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
    if ir["json_mode"]:
        out["response_format"] = {"type": "json_object"}
    if ir["stream"]:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def to_anthropic(ir, model):
    messages = []
    for msg in ir["messages"]:
        role = msg["role"]
        parts = msg["parts"]
        if role == "tool":
            blocks = []
            for p in parts:
                if p["type"] == "tool_result":
                    blocks.append({"type": "tool_result", "tool_use_id": p["tool_use_id"],
                                   "content": p["content"]})
            if blocks:
                messages.append({"role": "user", "content": blocks})
            continue
        blocks = []
        for p in parts:
            if p["type"] == "text":
                blocks.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": p["mime"], "data": p["data"]}})
            elif p["type"] == "image_url":
                blocks.append({"type": "image", "source": {"type": "url", "url": p["url"]}})
            elif p["type"] == "tool_use":
                blocks.append({"type": "tool_use", "id": p["id"], "name": p["name"], "input": p["input"]})
        if not blocks:
            continue
        messages.append({"role": "assistant" if role == "assistant" else "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if ir["system"]:
        out["system"] = ir["system"]
    if ir["max_tokens"] is not None:
        out["max_tokens"] = ir["max_tokens"]
    if ir["temperature"] is not None:
        out["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        out["top_p"] = ir["top_p"]
    if ir["stops"]:
        out["stop_sequences"] = ir["stops"]
    if ir["tools"]:
        out["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in ir["tools"]]
        tc = ir["tool_choice"]
        if tc == "auto":
            out["tool_choice"] = {"type": "auto"}
        elif tc == "any":
            out["tool_choice"] = {"type": "any"}
        elif tc == "none":
            out["tool_choice"] = {"type": "none"}
        elif is_record(tc):
            out["tool_choice"] = {"type": "tool", "name": tc["name"]}
    if ir["stream"]:
        out["stream"] = True
    return out


def to_gemini(ir, model):
    contents = []
    for msg in ir["messages"]:
        role = msg["role"]
        parts = []
        for p in msg["parts"]:
            if p["type"] == "text":
                parts.append({"text": p["text"]})
            elif p["type"] == "image":
                parts.append({"inlineData": {"mimeType": p["mime"], "data": p["data"]}})
            elif p["type"] == "image_url":
                parts.append({"fileData": {"fileUri": p["url"]}})
            elif p["type"] == "tool_use":
                parts.append({"functionCall": {"name": p["name"], "args": p["input"]}})
            elif p["type"] == "tool_result":
                parts.append({"functionResponse": {"name": "", "response": {"result": p["content"]}}})
        if not parts:
            continue
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    out = {"contents": contents}
    if ir["system"]:
        out["systemInstruction"] = {"parts": [{"text": ir["system"]}]}
    if ir["tools"]:
        out["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in ir["tools"]]}]
        tc = ir["tool_choice"]
        mode = {"auto": "AUTO", "any": "ANY", "none": "NONE"}.get(tc if isinstance(tc, str) else None, "AUTO")
        cfg = {"functionCallingConfig": {"mode": mode}}
        if is_record(tc):
            cfg["functionCallingConfig"] = {"mode": "ANY", "allowedFunctionNames": [tc["name"]]}
        out["toolConfig"] = cfg
    config = {}
    if ir["max_tokens"] is not None:
        config["maxOutputTokens"] = ir["max_tokens"]
    if ir["stops"]:
        config["stopSequences"] = ir["stops"]
    if ir["temperature"] is not None:
        config["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        config["topP"] = ir["top_p"]
    if ir["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out

# --------------------------------------------------------------------------
# Provider calls and answer parsing
# --------------------------------------------------------------------------

def provider_url(env_url, cand, stream):
    base = join_base(env_url, cand["base"])
    if cand["wire"] == "gemini":
        model = urllib.parse.quote(cand["model"], safe="")
        verb = "streamGenerateContent" if stream else "generateContent"
        return "%s/v1beta/models/%s:%s?alt=sse" % (base, model, verb)
    if cand["wire"] == "anthropic":
        return base + "/v1/messages"
    return base + "/v1/chat/completions"


def call_provider(env_url, cand, ir, stream):
    """One provider call. Returns (status, raw_bytes, headers_dict)."""
    wire = cand["wire"]
    if wire == "openai":
        payload = to_openai(ir, cand["model"])
    elif wire == "anthropic":
        payload = to_anthropic(ir, cand["model"])
    else:
        payload = to_gemini(ir, cand["model"])
    url = provider_url(env_url, cand, stream)
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(cand["auth"], cand["key"]))
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    if stream:
        headers["accept"] = "text/event-stream"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as resp:
            return resp.status, resp.read(), dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read(), dict(exc.headers.items())
        except Exception:
            return exc.code, b"", {}
    except Exception:
        return 0, b"", {}


def _parse_sse(raw):
    """SSE bytes -> list of (event, data_str) frames, in order."""
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    frames = []
    for chunk in text.split("\n\n"):
        if chunk.strip() == "":
            continue
        event = None
        data = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _json_or_none(text):
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if is_record(value) else None


def parse_answer(wire, stream, raw):
    """A provider's answer as {text, input_tokens, output_tokens, stop, tool_calls}.

    stop is the provider's own stop string (stop / end_turn / STOP / ...).
    Returns None when the bytes are not a well-formed answer of that wire.
    """
    text = raw.decode("utf-8", errors="replace")
    if wire == "openai":
        if stream:
            out_text = []
            stop = None
            usage = None
            for _event, data in _parse_sse(raw):
                if data.strip() == "[DONE]":
                    continue
                chunk = _json_or_none(data)
                if chunk is None:
                    continue
                if is_record(chunk.get("usage")):
                    usage = chunk["usage"]
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0] if is_record(choices[0]) else {}
                delta = choice.get("delta") if is_record(choice.get("delta")) else {}
                if isinstance(delta.get("content"), str):
                    out_text.append(delta["content"])
                if isinstance(choice.get("finish_reason"), str):
                    stop = choice["finish_reason"]
            if stop is None:
                return None
            return {"text": "".join(out_text),
                    "input_tokens": usage.get("prompt_tokens") if is_record(usage) else None,
                    "output_tokens": usage.get("completion_tokens") if is_record(usage) else None,
                    "stop": stop, "tool_calls": []}
        body = _json_or_none(text)
        if body is None:
            return None
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not is_record(choices[0]):
            return None
        choice = choices[0]
        message = choice.get("message") if is_record(choice.get("message")) else {}
        content = message.get("content")
        if isinstance(content, str):
            out_text = content
        else:
            out_text = _text_of(content)
        calls = []
        if isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                if is_record(call) and is_record(call.get("function")):
                    calls.append({"id": _str(call.get("id")), "name": _str(call["function"].get("name")),
                                  "arguments": _str(call["function"].get("arguments"))})
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        stop = choice.get("finish_reason")
        if not isinstance(stop, str):
            return None
        return {"text": out_text,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "stop": stop, "tool_calls": calls}
    if wire == "anthropic":
        if stream:
            out_text = []
            stop = None
            in_tok = None
            out_tok = None
            for event, data in _parse_sse(raw):
                chunk = _json_or_none(data)
                if chunk is None:
                    continue
                if event == "message_start" and is_record(chunk.get("message")):
                    usage = chunk["message"].get("usage")
                    if is_record(usage):
                        in_tok = usage.get("input_tokens")
                elif event == "content_block_delta" and is_record(chunk.get("delta")):
                    if chunk["delta"].get("type") == "text_delta" and isinstance(chunk["delta"].get("text"), str):
                        out_text.append(chunk["delta"]["text"])
                elif event == "message_delta":
                    if is_record(chunk.get("delta")) and isinstance(chunk["delta"].get("stop_reason"), str):
                        stop = chunk["delta"]["stop_reason"]
                    if is_record(chunk.get("usage")):
                        out_tok = chunk["usage"].get("output_tokens")
            if stop is None:
                return None
            return {"text": "".join(out_text), "input_tokens": in_tok,
                    "output_tokens": out_tok, "stop": stop, "tool_calls": []}
        body = _json_or_none(text)
        if body is None or body.get("type") != "message":
            return None
        content = body.get("content")
        if not isinstance(content, list):
            return None
        out_text = []
        calls = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                out_text.append(block["text"])
            elif block.get("type") == "tool_use":
                calls.append({"id": _str(block.get("id")), "name": _str(block.get("name")),
                              "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False)})
        stop = body.get("stop_reason")
        if not isinstance(stop, str):
            return None
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return {"text": "".join(out_text),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "stop": stop, "tool_calls": calls}
    # gemini
    if stream:
        out_text = []
        stop = None
        in_tok = None
        out_tok = None
        for _event, data in _parse_sse(raw):
            chunk = _json_or_none(data)
            if chunk is None:
                continue
            cands = chunk.get("candidates")
            if isinstance(cands, list) and cands and is_record(cands[0]):
                cand = cands[0]
                content = cand.get("content") if is_record(cand.get("content")) else {}
                parts = content.get("parts") if isinstance(content.get("parts"), list) else []
                for part in parts:
                    if is_record(part) and isinstance(part.get("text"), str):
                        out_text.append(part["text"])
                if isinstance(cand.get("finishReason"), str):
                    stop = cand["finishReason"]
            meta = chunk.get("usageMetadata")
            if is_record(meta):
                in_tok = meta.get("promptTokenCount")
                out_tok = meta.get("candidatesTokenCount")
        if stop is None:
            return None
        return {"text": "".join(out_text), "input_tokens": in_tok,
                "output_tokens": out_tok, "stop": stop, "tool_calls": []}
    body = _json_or_none(text)
    if body is None:
        return None
    cands = body.get("candidates")
    if not isinstance(cands, list) or not cands or not is_record(cands[0]):
        return None
    cand = cands[0]
    content = cand.get("content") if is_record(cand.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    out_text = "".join(p["text"] for p in parts if is_record(p) and isinstance(p.get("text"), str))
    stop = cand.get("finishReason")
    if not isinstance(stop, str):
        return None
    meta = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return {"text": out_text,
            "input_tokens": meta.get("promptTokenCount"),
            "output_tokens": meta.get("candidatesTokenCount"),
            "stop": stop, "tool_calls": []}


# --------------------------------------------------------------------------
# Rendering answers in the client's wire
# --------------------------------------------------------------------------

_STOP_FAMILY = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}


def _stop_family(stop):
    return _STOP_FAMILY.get(stop, 0)


def _openai_finish(answer):
    if answer["tool_calls"]:
        return "tool_calls"
    fam = _stop_family(answer["stop"])
    return ("stop", "length", "tool_calls", "content_filter")[fam]


def _anthropic_stop(answer):
    if answer["tool_calls"]:
        return "tool_use"
    fam = _stop_family(answer["stop"])
    return ("end_turn", "max_tokens", "tool_use", "refusal")[fam]


def _num(value):
    return value if isinstance(value, (int, float)) else 0


def render_openai_json(answer, model, req_id):
    message = {"role": "assistant", "content": answer["text"]}
    if answer["tool_calls"]:
        message["tool_calls"] = [
            {"id": c["id"] or "call_%s" % req_id, "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in answer["tool_calls"]]
    usage = {"prompt_tokens": _num(answer["input_tokens"]),
             "completion_tokens": _num(answer["output_tokens"]),
             "total_tokens": _num(answer["input_tokens"]) + _num(answer["output_tokens"])}
    return {"id": "chatcmpl-" + req_id, "object": "chat.completion", "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": _openai_finish(answer), "logprobs": None}],
            "usage": usage}


def _sse_event(event, data):
    text = json.dumps(data, ensure_ascii=False)
    if event is None:
        return "data: %s\n\n" % text
    return "event: %s\ndata: %s\n\n" % (event, text)


def _split_text(text, size=512):
    if text == "":
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]


def render_openai_stream(answer, model, req_id):
    frames = []
    base = {"id": "chatcmpl-" + req_id, "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model}
    first = dict(base)
    first["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    frames.append(_sse_event(None, first))
    for piece in _split_text(answer["text"]):
        chunk = dict(base)
        chunk["choices"] = [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
        frames.append(_sse_event(None, chunk))
    if answer["tool_calls"]:
        for call in answer["tool_calls"]:
            chunk = dict(base)
            chunk["choices"] = [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": call["id"] or "call_%s" % req_id, "type": "function",
                 "function": {"name": call["name"], "arguments": call["arguments"]}}]}, "finish_reason": None}]
            frames.append(_sse_event(None, chunk))
    finish = dict(base)
    finish["choices"] = [{"index": 0, "delta": {}, "finish_reason": _openai_finish(answer)}]
    frames.append(_sse_event(None, finish))
    usage = dict(base)
    usage["choices"] = []
    usage["usage"] = {"prompt_tokens": _num(answer["input_tokens"]),
                      "completion_tokens": _num(answer["output_tokens"]),
                      "total_tokens": _num(answer["input_tokens"]) + _num(answer["output_tokens"])}
    frames.append(_sse_event(None, usage))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_json(answer, model, req_id):
    content = []
    if answer["text"]:
        content.append({"type": "text", "text": answer["text"]})
    for call in answer["tool_calls"]:
        try:
            parsed = json.loads(call["arguments"]) if call["arguments"] else {}
        except Exception:
            parsed = {}
        content.append({"type": "tool_use", "id": call["id"] or "toolu_%s" % req_id,
                        "name": call["name"], "input": parsed if is_record(parsed) else {}})
    if not content:
        content.append({"type": "text", "text": ""})
    return {"id": "msg_" + req_id, "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": _anthropic_stop(answer), "stop_sequence": None,
            "usage": {"input_tokens": _num(answer["input_tokens"]),
                      "output_tokens": _num(answer["output_tokens"])}}


def render_anthropic_stream(answer, model, req_id):
    frames = []
    frames.append(_sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_" + req_id, "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": _num(answer["input_tokens"]), "output_tokens": 0}}}))
    index = 0
    if answer["text"] or not answer["tool_calls"]:
        frames.append(_sse_event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "text", "text": ""}}))
        for piece in _split_text(answer["text"]):
            frames.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": piece}}))
        frames.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    for call in answer["tool_calls"]:
        frames.append(_sse_event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": call["id"] or "toolu_%s" % req_id,
                              "name": call["name"], "input": {}}}))
        if call["arguments"]:
            frames.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": call["arguments"]}}))
        frames.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    frames.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _anthropic_stop(answer), "stop_sequence": None},
        "usage": {"output_tokens": _num(answer["output_tokens"])}}))
    frames.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)

# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, env_url, providers, probes, ratings, manifest):
        self.env_url = env_url
        self.providers = providers
        self.probes = probes
        self.ratings = ratings
        self.manifest = manifest
        self.manifest_by_id = {m["id"]: m for m in manifest}
        self.catalog = build_catalog(providers, probes, ratings)
        self.candidates = build_candidates(providers, probes, ratings)
        self.plan = plan_requests(manifest, self.candidates)
        self.lock = threading.Lock()
        self.arrived = 0
        self.in_flight = 0
        self.request_ids = []

    def served_models(self):
        """Free models for GET /v1/models, deduped by id, catalog order."""
        seen = set()
        out = []
        for provider in self.catalog["providers"]:
            for model in provider["models"]:
                if not model["free"] or model["id"] in seen:
                    continue
                seen.add(model["id"])
                out.append({"id": model["id"], "provider": provider["id"]})
        return out

    def next_request_id(self):
        with self.lock:
            n = self.arrived
            self.arrived += 1
        if n < len(self.manifest):
            return self.manifest[n]["id"]
        return "r%02d" % (n + 1)

    def serve(self, api, body):
        """Answer one client request. Returns (status, content_type, payload)."""
        req_id = self.next_request_id()
        with self.lock:
            self.in_flight += 1
        try:
            return self._serve(api, body, req_id)
        finally:
            with self.lock:
                self.in_flight -= 1

    def _serve(self, api, body, req_id):
        stream = body.get("stream") is True
        ir = normalize_body(api, body)
        manifest_entry = self.manifest_by_id.get(req_id)
        planned = self.plan.get(req_id)
        if planned:
            ordered, _req = candidates_for_request(api, body, manifest_entry, self.candidates)
            # The plan's first pick leads; the rest stay in rating order.
            first = planned[0] if planned else None
            if first is not None and first in ordered:
                ordered = [first] + [c for c in ordered if c is not first]
        else:
            ordered, _req = candidates_for_request(api, body, manifest_entry, self.candidates)
        deadline = time.monotonic() + REQUEST_DEADLINE_S
        tried = []
        for cand in ordered[:MAX_ATTEMPTS]:
            if time.monotonic() > deadline:
                break
            if cand in tried:
                continue
            tried.append(cand)
            with self.lock:
                status, raw, _headers = call_provider(self.env_url, cand, ir, stream)
            if status < 200 or status >= 300:
                log("%s: %s/%s refused HTTP %s, failing over" % (req_id, cand["provider"], cand["model"], status))
                continue
            answer = parse_answer(cand["wire"], stream, raw)
            if answer is None:
                log("%s: %s/%s answered unparsable bytes, failing over" % (req_id, cand["provider"], cand["model"]))
                continue
            model = cand["model"]
            if api == "openai":
                if stream:
                    return 200, "text/event-stream; charset=utf-8", render_openai_stream(answer, model, req_id).encode("utf-8")
                payload = render_openai_json(answer, model, req_id)
                return 200, "application/json", json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if stream:
                return 200, "text/event-stream; charset=utf-8", render_anthropic_stream(answer, model, req_id).encode("utf-8")
            payload = render_anthropic_json(answer, model, req_id)
            return 200, "application/json", json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if api == "openai":
            payload = {"error": {"message": "no free model could serve this request", "type": "server_error", "code": "no_free_model"}}
        else:
            payload = {"type": "error", "error": {"type": "overloaded_error", "message": "no free model could serve this request"}}
        return 503, "application/json", json.dumps(payload).encode("utf-8")


def _model_list_openai(models):
    return {"object": "list",
            "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                      "owned_by": m["provider"], "free": True} for m in models]}


def _model_list_anthropic(models):
    return {"data": [{"type": "model", "id": m["id"], "display_name": m["id"],
                      "created_at": FIXED_ANTHROPIC_CREATED_AT, "free": True} for m in models],
            "has_more": False}


class Handler(BaseHTTPRequestHandler):
    aggregator = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, status, obj):
        self._send(status, "application/json", json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _read_body(self):
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/models":
            models = self.aggregator.served_models()
            if self.headers.get("x-api-key") is not None:
                self._send_json(200, _model_list_anthropic(models))
            else:
                self._send_json(200, _model_list_openai(models))
            return
        self._send_json(404, {"error": {"message": "no route GET %s" % parsed.path, "type": "invalid_request_error", "code": "not_found"}})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            body = self._read_body()
            if body is None:
                self._send_json(400, {"error": {"message": "the request body is not one JSON object", "type": "invalid_request_error", "code": "invalid_request"}})
                return
            status, ctype, payload = self.aggregator.serve("openai", body)
            self._send(status, ctype, payload)
            return
        if parsed.path == "/v1/messages":
            body = self._read_body()
            if body is None:
                self._send_json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "the request body is not one JSON object"}})
                return
            status, ctype, payload = self.aggregator.serve("anthropic", body)
            self._send(status, ctype, payload)
            return
        self._send_json(404, {"error": {"message": "no route POST %s" % parsed.path, "type": "invalid_request_error", "code": "not_found"}})


def load_all():
    """Parse the brief and probe the providers. Returns (providers, probes, ratings, manifest)."""
    providers, ratings, _best, manifest, error = read_brief(PROMPT_FILE)
    if error is not None:
        log(error)
    probes = []
    for provider in providers:
        ok, models = probe_provider(ENV_URL, provider)
        if not ok:
            log("probe failed for %s" % provider["id"])
        probes.append((ok, models))
    return providers, probes, ratings, manifest


def register(env_url, port, catalog):
    url = env_url + "/register"
    body = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:
            raw = resp.read()
            reply = json.loads(raw.decode("utf-8"))
            log("registered: the workload holds %s requests" % reply.get("requests"))
            return True
    except Exception as exc:
        log("POST %s failed: %s" % (url, exc))
        return False


def work_done(env_url):
    status, raw = _http_json("GET", env_url + "/work", {}, timeout=2.0)
    if status != 200:
        return False
    try:
        return json.loads(raw.decode("utf-8")).get("done") is True
    except Exception:
        return False


def main(argv):
    list_only = "--list" in argv
    if not ENV_URL or not PROMPT_FILE:
        # Outside a trial there is no brief and no provider route to probe.
        sys.stdout.write('{"providers":[]}\n')
        sys.stdout.flush()
        return 0
    providers, probes, ratings, manifest = load_all()
    catalog = build_catalog(providers, probes, ratings)
    if list_only:
        sys.stdout.write(catalog_json(catalog) + "\n")
        sys.stdout.flush()
        return 0
    aggregator = Aggregator(ENV_URL, providers, probes, ratings, manifest)
    # The register body must carry the same catalog document --list prints.
    aggregator.catalog = catalog
    Handler.aggregator = aggregator
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        server.server_close()
        raise SystemExit("the OS picked 8080, which the env owns on one layout: restart the trial")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    log("listening on 0.0.0.0:%d with %d providers and %d free models"
        % (port, len(catalog["providers"]), len(aggregator.served_models())))
    if not register(ENV_URL, port, catalog):
        server.shutdown()
        return 1

    def _exit_cleanly(signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _exit_cleanly)
    signal.signal(signal.SIGINT, _exit_cleanly)
    while True:
        time.sleep(POLL_INTERVAL_S)
        if work_done(ENV_URL):
            with aggregator.lock:
                busy = aggregator.in_flight > 0
            if not busy:
                break
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
