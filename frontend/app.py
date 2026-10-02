"""Gradio showcase for Clef-Flash.

Clef-Flash turns a state + a schema of typed questions into calibrated
probabilities for every option of every question, in one forward pass. The tabs
are built around what that makes possible:

  Decide        free-form state + schema, with a table-based schema builder
  Vision/Video  the same typed decisions over images and clips
  Live webcam   the browser's camera, queried continuously against a schema
  Tool router   agent routing: pick a tool, decide whether to call it, fill enum args
  Batch triage  many items, one schema, GPU-batched; confidence threshold -> human review
  What-if       two states side by side, showing how the probabilities move
  API           raw access to every endpoint, with a matching curl command

It only talks to the API container over HTTP.
"""
from __future__ import annotations

import base64
import copy
import csv
import html
import io
import json
import mimetypes
import os
import threading
import time
from collections import deque
from pathlib import Path

import gradio as gr
import requests

import presets

API_BASE = os.environ.get("API_BASE", "http://api:8000").rstrip("/")
PUBLIC_API = os.environ.get("PUBLIC_API_BASE", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("SERVED_MODEL_NAME", "clef-flash")
TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "300"))
HERE = Path(__file__).parent


# ------------------------------------------------------------------ helpers
def j(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def parse_state(text: str):
    """States may be free text or JSON; the model reads both (JSON is rendered compactly)."""
    s = (text or "").strip()
    if s[:1] in "{[":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def parse_questions(text: str) -> dict:
    try:
        q = json.loads(text or "")
    except json.JSONDecodeError as e:
        raise gr.Error(f"Questions JSON is invalid: {e}")
    if not isinstance(q, dict) or not q:
        raise gr.Error("Questions must be a non-empty JSON object of question_id -> question.")
    return q


def encode_file(path: str | None) -> str | None:
    if not path:
        return None
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def call(method: str, path: str, body=None):
    t0 = time.perf_counter()
    try:
        r = requests.request(method, API_BASE + path, json=body, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise gr.Error(f"API unreachable: {e}. Check `docker compose logs api`.")
    rtt = (time.perf_counter() - t0) * 1000
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text}
    if r.status_code >= 400:
        detail = data.get("detail") if isinstance(data, dict) else data
        raise gr.Error(f"API {r.status_code}: {detail}")
    return data, rtt


def curl_for(path: str, body: dict) -> str:
    shown = copy.deepcopy(body)
    for key in ("images", "videos"):
        if shown.get(key):
            shown[key] = [f"<base64 {key[:-1]}>" for _ in shown[key]]
    return (f"curl -s {PUBLIC_API}{path} \\\n  -H 'Content-Type: application/json' \\\n"
            f"  -d '{json.dumps(shown, ensure_ascii=False)}'")


def is_video(path: str | None) -> bool:
    return bool(path) and (mimetypes.guess_type(path)[0] or "").startswith("video")


# ---------------------------------------------------------------- rendering
def _label(qtype: str, q: dict, opt: str) -> str:
    if qtype == "score":
        return f"{opt} · {html.escape(str(q['criteria'][int(opt)]))}"
    if qtype == "choice":
        desc = q["criteria"].get(opt)
        return f"<b>{html.escape(opt)}</b>" + (f" <span class='cl-dim'>{html.escape(str(desc))}</span>" if desc else "")
    return html.escape(opt)


def _order(qtype: str, probs: dict) -> list[str]:
    if qtype == "choice":
        return sorted(probs, key=probs.get, reverse=True)
    if qtype == "noul":
        return ["true", "false"]
    return sorted(probs, key=int)


def _headline(qtype: str, q: dict, ans: dict, probs: dict) -> str:
    if qtype == "noul":
        p = probs["true"]
        verdict = "TRUE" if p >= 0.5 else "FALSE"
        return f"<span class='cl-verdict {'yes' if p >= .5 else 'no'}'>{verdict}</span> P(true) = {p:.3f}"
    if qtype == "choice":
        return f"<span class='cl-verdict'>{html.escape(ans['choice'])}</span> confidence {ans['confidence']:.3f}"
    top = max(probs, key=probs.get)
    n = len(q["criteria"]) - 1
    return (f"<span class='cl-verdict'>{ans['score']:.2f} / {n}</span> expected level · "
            f"most likely “{html.escape(str(q['criteria'][int(top)]))}”")


def _score_gauge(ans: dict, q: dict) -> str:
    n = len(q["criteria"]) - 1
    pct = 100 * ans["score"] / n if n else 0
    ticks = "".join(f"<span style='left:{100 * i / n:.1f}%'></span>" for i in range(n + 1))
    return (f"<div class='cl-gauge'><div class='cl-gauge-fill' style='width:{pct:.1f}%'></div>{ticks}"
            f"<div class='cl-gauge-dot' style='left:{pct:.1f}%'></div></div>")


def render_answers(questions: dict, response: dict) -> str:
    probs_all = response["x_clef"]["probabilities"]
    parts = ["<div class='cl-out'>"]
    for qid, q in questions.items():
        qtype, probs, ans = q["type"], probs_all[qid], response["answers"][qid]
        best = max(probs, key=probs.get)
        parts.append(
            f"<div class='cl-card'><div class='cl-qh'><span class='cl-id'>{html.escape(qid)}</span>"
            f"<span class='cl-type t-{qtype}'>{qtype}</span></div>"
            f"<div class='cl-instr'>{html.escape(str(q.get('instructions') or qid))}</div>"
            f"<div class='cl-head'>{_headline(qtype, q, ans, probs)}</div>"
        )
        if qtype == "score":
            parts.append(_score_gauge(ans, q))
        for opt in _order(qtype, probs):
            pct = 100 * probs[opt]
            parts.append(
                f"<div class='cl-row'><div class='cl-lbl'>{_label(qtype, q, opt)}</div>"
                f"<div class='cl-track'><div class='cl-bar{' best' if opt == best else ''}' style='width:{pct:.1f}%'></div></div>"
                f"<div class='cl-pct'>{pct:.1f}%</div></div>"
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def render_meta(response: dict, rtt: float) -> str:
    t = response["x_clef"].get("timing", {})
    u = response["usage"]
    chips = [
        ("forward pass", f"{t.get('forward_ms', 0):.0f} ms"),
        ("server total", f"{t.get('total_ms', 0):.0f} ms"),
        ("round trip", f"{rtt:.0f} ms"),
        ("input tokens", f"{u['input_tokens']:,}"),
        ("options scored", str(response["x_clef"]["options_scored"])),
        ("output tokens", "0"),
    ]
    return "<div class='cl-chips'>" + "".join(
        f"<div class='cl-chip'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in chips
    ) + "</div>"


EMPTY = "<div class='cl-empty'>Run a decision to see per-option probabilities here.</div>"


# --------------------------------------------------------- schema builder
OPTION_SEP = "|"


def table_to_questions(rows) -> str:
    """Rows are [id, type, instructions, options]; options use 'key: desc | key: desc'."""
    rows = rows.values.tolist() if hasattr(rows, "values") else (rows or [])
    out = {}
    for row in rows:
        row = [("" if c is None else str(c)).strip() for c in list(row) + [""] * 4][:4]
        qid, qtype, instr, opts = row
        if not qid:
            continue
        qtype = (qtype or "noul").lower()
        if qtype not in ("noul", "choice", "score"):
            raise gr.Error(f"{qid}: type must be noul, choice or score")
        q: dict = {"type": qtype}
        if instr:
            q["instructions"] = instr
        parts = [p.strip() for p in opts.split(OPTION_SEP) if p.strip()]
        if qtype == "score":
            if len(parts) < 2:
                raise gr.Error(f"{qid}: score needs at least 2 levels, e.g. 'Low | Medium | High'")
            q["criteria"] = parts
        elif qtype == "choice":
            crit = {}
            for p in parts:
                key, _, desc = p.partition(":")
                crit[key.strip().replace(" ", "_")] = desc.strip() or key.strip()
            if len(crit) < 2:
                raise gr.Error(f"{qid}: choice needs at least 2 options, e.g. 'yes: Approve | no: Reject'")
            q["criteria"] = crit
        elif parts:
            crit = {}
            for p in parts:
                key, _, desc = p.partition(":")
                if key.strip() in ("true", "false") and desc.strip():
                    crit[key.strip()] = desc.strip()
            if crit:
                q["criteria"] = crit
        out[qid] = q
    if not out:
        raise gr.Error("Add at least one row with an id.")
    return j(out)


def questions_to_table(text: str):
    qs = parse_questions(text)
    rows = []
    for qid, q in qs.items():
        crit = q.get("criteria")
        if isinstance(crit, dict):
            opts = f" {OPTION_SEP} ".join(f"{k}: {v}" for k, v in crit.items())
        elif isinstance(crit, list):
            opts = f" {OPTION_SEP} ".join(str(c) for c in crit)
        else:
            opts = ""
        rows.append([qid, q.get("type", ""), q.get("instructions", ""), opts])
    return rows


# --------------------------------------------------------------- Decide tab
def decide(state_text, questions_text, media):
    questions = parse_questions(questions_text)
    body = {"model": MODEL, "state": parse_state(state_text), "questions": questions}
    if media:
        body["videos" if is_video(media) else "images"] = [encode_file(media)]
    resp, rtt = call("POST", "/v1/systemone", body)
    return render_answers(questions, resp), render_meta(resp, rtt), resp, curl_for("/v1/systemone", body)


def load_decide_preset(name):
    p = presets.DECIDE[name]
    state = p["state"] if isinstance(p["state"], str) else j(p["state"])
    return state, j(p["questions"]), questions_to_table(j(p["questions"]))


# --------------------------------------------------------------- Vision tab
def vision_decide(image, video, state_text, questions_text, frames):
    questions = parse_questions(questions_text)
    body = {"model": MODEL, "state": parse_state(state_text), "questions": questions}
    if video:
        body["videos"] = [encode_file(video)]
        body["video_frames"] = int(frames)
    elif image:
        body["images"] = [encode_file(image)]
    else:
        raise gr.Error("Upload an image or a video first (or pick a preset).")
    resp, rtt = call("POST", "/v1/systemone", body)
    return render_answers(questions, resp), render_meta(resp, rtt), resp


def load_vision_preset(name):
    p = presets.VISION[name]
    media = str(HERE / p["media"]) if p["media"] else None
    state = p["state"] if isinstance(p["state"], str) else j(p["state"])
    img = media if media and not is_video(media) else None
    vid = media if is_video(media) else None
    return img, vid, state, j(p["questions"])


# ---------------------------------------------------------- Tool router tab
def build_router_schema(tools, extra_text, allow_none):
    rows = tools.values.tolist() if hasattr(tools, "values") else (tools or [])
    criteria = {}
    for row in rows:
        name = str(row[0] or "").strip()
        if name:
            criteria[name] = str(row[1] or "").strip() or name
    if len(criteria) < 2:
        raise gr.Error("Define at least two tools.")
    if allow_none:
        criteria["no_tool"] = "None of the tools fits; reply directly or decline"
    schema = {
        "tool": {"type": "choice", "instructions": "Which tool best serves the user's latest request?",
                 "criteria": criteria},
        "call_now": {"type": "noul",
                     "instructions": "Should the assistant call a tool right now, rather than reply in text?"},
        "needs_clarification": {"type": "noul",
                                "instructions": "Is required information missing, so the assistant must ask a "
                                                "clarifying question before acting?"},
    }
    extra = (extra_text or "").strip()
    if extra:
        try:
            schema.update(json.loads(extra))
        except json.JSONDecodeError as e:
            raise gr.Error(f"Extra questions JSON is invalid: {e}")
    return schema


def route(message, tools, extra_text, allow_none, history_text):
    schema = build_router_schema(tools, extra_text, allow_none)
    state = {"conversation": [], "latest_user_message": message}
    for line in (history_text or "").splitlines():
        role, _, text = line.partition(":")
        if text.strip():
            state["conversation"].append({"role": role.strip().lower(), "content": text.strip()})
    if not state["conversation"]:
        del state["conversation"]
    body = {"model": MODEL, "state": state, "questions": schema}
    resp, rtt = call("POST", "/v1/systemone", body)
    a = resp["answers"]
    tool, conf = a["tool"]["choice"], a["tool"]["confidence"]
    call_now, clarify = a["call_now"]["noul"], a["needs_clarification"]["noul"]
    args = {k: v.get("choice", v.get("score", v.get("noul"))) for k, v in a.items()
            if k not in ("tool", "call_now", "needs_clarification")}
    missing = [k for k, v in args.items() if v == "unspecified"]
    enum_args = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items()
                          if a[k]["type"] == "choice" and v not in ("unspecified", "none"))
    if clarify >= 0.5:
        verdict, cls = "Ask a clarifying question", "warn"
    elif tool == "no_tool" or call_now < 0.5:
        verdict, cls = "Reply directly, no tool call", "neutral"
    else:
        verdict, cls = f"Call {tool}({enum_args})", "warn" if missing else "go"
    args_html = "".join(f"<code>{html.escape(k)}={html.escape(json.dumps(v))}</code> " for k, v in args.items())
    if missing and cls != "neutral":
        args_html += f"<div class='cl-route-m'>⚠ unresolved: {html.escape(', '.join(missing))} — consider asking the user</div>"
    card = (f"<div class='cl-route {cls}'><div class='cl-route-v'>{html.escape(verdict)}</div>"
            f"<div class='cl-route-d'>tool <b>{html.escape(tool)}</b> @ {conf:.1%} · call now {call_now:.1%} · "
            f"needs clarification {clarify:.1%}</div>"
            + (f"<div class='cl-route-a'>{args_html}</div>" if args else "")
            + f"<div class='cl-route-t'>routing decided in {resp['x_clef']['timing']['forward_ms']:.0f} ms of GPU time</div></div>")
    return card + render_answers(schema, resp), render_meta(resp, rtt), j(body)


def load_toolset(name):
    p = presets.TOOLSETS[name]
    return p["message"], p["tools"], j(p.get("extra") or {}), ""


# --------------------------------------------------------------- Batch tab
def _items_from_file(file):
    if not file:
        return gr.update()
    text = Path(file).read_text(errors="replace")
    if file.lower().endswith(".csv"):
        rows = list(csv.reader(io.StringIO(text)))
        if rows and rows[0]:
            header = [h.strip().lower() for h in rows[0]]
            col = next((header.index(h) for h in ("text", "message", "body", "content", "review") if h in header), 0)
            items = [r[col] for r in rows[1:] if len(r) > col and r[col].strip()]
            return "\n".join(i.replace("\n", " ") for i in items)
    return text


def batch_run(items_text, questions_text, threshold, gate_scores):
    questions = parse_questions(questions_text)
    items = [line.strip() for line in (items_text or "").splitlines() if line.strip()]
    if not items:
        raise gr.Error("Enter at least one item (one per line).")
    body = {"model": MODEL, "state": [parse_state(i) for i in items], "questions": questions}
    data, rtt = call("POST", "/v1/systemone/batch", body)
    timing = data["timing"]
    # Decisions first, free text last: a long item column would push the answers off-screen.
    headers = ["#"]
    for qid, q in questions.items():
        headers += [qid, f"{qid} conf"] if q["type"] != "noul" else [f"{qid} P(true)"]
    headers += ["route", "item"]
    rows, flagged, dist = [], 0, {}
    for n, (item, resp) in enumerate(zip(items, data["responses"]), 1):
        row, low = [n], []
        for qid, q in questions.items():
            a = resp["answers"][qid]
            if q["type"] == "noul":
                p = a["noul"]
                row.append(round(p, 3))
                if max(p, 1 - p) < threshold:
                    low.append(qid)
            else:
                val = a["choice"] if q["type"] == "choice" else round(a["score"], 2)
                row += [val, round(a["confidence"], 3)]
                # Ordinal scores spread mass over neighbouring levels by design, so their
                # top-level confidence is a poor gate unless the user opts in.
                if a["confidence"] < threshold and (q["type"] == "choice" or gate_scores):
                    low.append(qid)
                if q["type"] == "choice":
                    dist.setdefault(qid, {}).setdefault(a["choice"], 0)
                    dist[qid][a["choice"]] += 1
        route_ = "👤 human review (" + ", ".join(low) + ")" if low else "✅ auto"
        flagged += bool(low)
        row += [route_, item]
        rows.append(row)

    n = len(items)
    stats = (
        "<div class='cl-chips'>"
        + "".join(f"<div class='cl-chip'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in [
            ("items", n), ("GPU batches", timing["gpu_batches"]), ("forward time", f"{timing['forward_ms']:.0f} ms"),
            ("throughput", f"{timing['records_per_second']:.1f} items/s"),
            ("per item", f"{timing['total_ms'] / n:.1f} ms"),
            ("auto-routed", f"{n - flagged}/{n}"), ("to humans", f"{flagged}/{n}"),
        ]) + "</div>"
    )
    dist_html = ""
    for qid, counts in dist.items():
        total = sum(counts.values())
        bars = "".join(
            f"<div class='cl-row'><div class='cl-lbl'><b>{html.escape(k)}</b></div><div class='cl-track'>"
            f"<div class='cl-bar best' style='width:{100 * v / total:.1f}%'></div></div><div class='cl-pct'>{v}</div></div>"
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        dist_html += f"<div class='cl-card'><div class='cl-qh'><span class='cl-id'>{html.escape(qid)}</span>" \
                     f"<span class='cl-type t-choice'>distribution</span></div>{bars}</div>"

    out_csv = HERE / "outputs"
    out_csv.mkdir(exist_ok=True)
    path = out_csv / f"clef_batch_{int(time.time())}.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)
    return gr.update(value=rows, headers=headers), stats, f"<div class='cl-out'>{dist_html}</div>", str(path)


def load_batch_preset(name):
    p = presets.BATCH[name]
    return "\n".join(p["items"]), j(p["questions"])


# -------------------------------------------------------------- Compare tab
def compare(a_text, b_text, questions_text):
    questions = parse_questions(questions_text)
    body = {"model": MODEL, "requests": [
        {"model": MODEL, "state": parse_state(a_text), "questions": questions},
        {"model": MODEL, "state": parse_state(b_text), "questions": questions},
    ]}
    data, rtt = call("POST", "/v1/systemone/batch", body)
    pa, pb = (r["x_clef"]["probabilities"] for r in data["responses"])
    parts = ["<div class='cl-out'>"]
    for qid, q in questions.items():
        qtype = q["type"]
        parts.append(
            f"<div class='cl-card'><div class='cl-qh'><span class='cl-id'>{html.escape(qid)}</span>"
            f"<span class='cl-type t-{qtype}'>{qtype}</span></div>"
            f"<div class='cl-instr'>{html.escape(str(q.get('instructions') or qid))}</div>"
            "<div class='cl-cmp-h'><span></span><span>A</span><span>B</span><span>Δ</span></div>")
        for opt in _order(qtype, pb[qid]):
            a, b = pa[qid][opt], pb[qid][opt]
            d = b - a
            arrow = "▲" if d > 0.005 else ("▼" if d < -0.005 else "·")
            cls = "up" if d > 0.005 else ("down" if d < -0.005 else "")
            parts.append(
                f"<div class='cl-cmp'><div class='cl-lbl'>{_label(qtype, q, opt)}</div>"
                f"<div class='cl-track'><div class='cl-bar a' style='width:{100 * a:.1f}%'></div></div>"
                f"<div class='cl-track'><div class='cl-bar best' style='width:{100 * b:.1f}%'></div></div>"
                f"<div class='cl-delta {cls}'>{arrow} {100 * d:+.1f}pp</div></div>")
        if qtype == "score":
            sa = data["responses"][0]["answers"][qid]["score"]
            sb = data["responses"][1]["answers"][qid]["score"]
            parts.append(f"<div class='cl-dim'>expected level {sa:.2f} → {sb:.2f}</div>")
        parts.append("</div>")
    parts.append("</div>")
    t = data["timing"]
    meta = (f"<div class='cl-chips'><div class='cl-chip'><div class='k'>both states, one batch</div>"
            f"<div class='v'>{t['forward_ms']:.0f} ms</div></div><div class='cl-chip'><div class='k'>round trip</div>"
            f"<div class='v'>{rtt:.0f} ms</div></div></div>")
    return "".join(parts), meta


def load_compare_preset(name):
    p = presets.COMPARE[name]
    f = lambda s: s if isinstance(s, str) else j(s)  # noqa: E731
    return f(p["a"]), f(p["b"]), j(p["questions"])


# --------------------------------------------------------------- Webcam tab
# The browser streams frames at STREAM_EVERY; each one lands in a per-session
# buffer. At most one API query is in flight per session, always on the newest
# frame (or the newest clip), so the loop runs as fast as the GPU allows and
# never builds a backlog. Benchmarked on a 3090: ~3.6 queries/s at 448px with 5
# questions, ~6/s with 1 question; parallel queries add latency, not throughput.
STREAM_EVERY = 0.1
BUFFER_SECONDS = 8
BUFFER_SIDE = 1024
LIVE: dict[str, dict] = {}
LIVE_LOCK = threading.Lock()
PALETTE = ["#f97316", "#6366f1", "#22c55e", "#eab308", "#ec4899", "#06b6d4", "#a855f7", "#94a3b8"]


def _session(request: gr.Request) -> dict:
    with LIVE_LOCK:
        return LIVE.setdefault(request.session_hash, _fresh_session())


def _fresh_session() -> dict:
    return {"frames": deque(maxlen=int(BUFFER_SECONDS / STREAM_EVERY) + 8), "busy": False, "last_start": 0.0,
            "lock": threading.Lock(), "params": None, "last_tick": 0.0, "last_frame_t": 0.0, "pending": {},
            "result": None, "result_id": 0, "rendered_id": -1, "error": None, "history": deque(maxlen=120),
            "log": deque(maxlen=40), "argmax": {}, "done": deque(maxlen=24), "queries": 0, "schema_key": None}


def _jpeg(img, side: int, quality: int = 85) -> bytes:
    from PIL import Image

    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    im = im.convert("RGB")
    if max(im.size) > side:
        im.thumbnail((side, side))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=quality)
    return out.getvalue()


def _resized_b64(jpeg: bytes, side: int) -> str:
    from PIL import Image

    im = Image.open(io.BytesIO(jpeg))
    if max(im.size) > side:
        jpeg = _jpeg(im, side)
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _primary(qtype: str, q: dict, ans: dict):
    """One comparable value per question for the timeline and the change log."""
    if qtype == "noul":
        return ans["noul"], ("true" if ans["noul"] >= 0.5 else "false")
    if qtype == "choice":
        return ans["confidence"], ans["choice"]
    n = len(q["criteria"]) - 1
    top = max(ans["probabilities"], key=ans["probabilities"].get)
    return (ans["score"] / n if n else 0.0), str(q["criteria"][int(top)])


def _live_worker(s: dict) -> None:
    """Query back-to-back on the newest frame while the stream is alive.

    Chaining here (rather than waiting for the next streamed frame to trigger a
    query) removes up to STREAM_EVERY of idle GPU time per detection.
    """
    try:
        while True:
            with s["lock"]:
                params = s["params"]
                alive = time.time() - s["last_tick"] < 1.5
                if not params or not alive:
                    s["busy"] = False
                    return
            wait = s["last_start"] + 1.0 / max(params["max_rate"], 0.1) - time.time()
            if wait > 0:
                time.sleep(wait)
            if not s["frames"] or s["frames"][-1][0] <= s["last_frame_t"]:
                time.sleep(0.01)  # never score the same frame twice
                continue
            s["last_start"] = time.time()
            _live_query(s, params)
    except Exception:  # noqa: BLE001
        with s["lock"]:
            s["busy"] = False
        raise


def _debounced(s: dict, qid: str, label: str, need: int) -> str | None:
    """Return the previous label if `label` has now held for `need` detections in a row."""
    current = s["argmax"].get(qid)
    if current is None:
        s["argmax"][qid] = label
        return None
    if label == current:
        s["pending"].pop(qid, None)
        return None
    plabel, count = s["pending"].get(qid, (label, 0))
    count = count + 1 if plabel == label else 1
    if count >= need:
        s["pending"].pop(qid, None)
        s["argmax"][qid] = label
        return current
    s["pending"][qid] = (label, count)
    return None


def _live_query(s: dict, params: dict) -> None:
    try:
        frames = list(s["frames"])
        questions = params["questions"]
        body = {"model": MODEL, "state": params["state"], "questions": questions}
        if params["mode"] == "clip":
            horizon = frames[-1][0] - params["clip_secs"]
            window = [f for f in frames if f[0] >= horizon] or frames[-1:]
            n = max(2, min(int(params["clip_frames"]), len(window)))
            picked = [window[round(i * (len(window) - 1) / (n - 1))] for i in range(n)] if len(window) > 1 else window * 2
            span = picked[-1][0] - picked[0][0]
            body["videos"] = [[_resized_b64(j, params["size"]) for _, j in picked]]
            body["video_frames"] = len(picked)
            body["video_fps"] = round((len(picked) - 1) / span, 3) if span > 0 else 2.0
            newest_t = picked[-1][0]
        else:
            newest_t, jpeg = frames[-1]
            body["images"] = [_resized_b64(jpeg, params["size"])]
        s["last_frame_t"] = frames[-1][0]
        t0 = time.perf_counter()
        r = requests.post(API_BASE + "/v1/systemone", json=body, timeout=60)
        rtt = (time.perf_counter() - t0) * 1000
        data = r.json()
        if r.status_code >= 400:
            raise RuntimeError(f"API {r.status_code}: {data.get('detail', data)}")
        now = time.time()
        s["done"].append(now)
        s["queries"] += 1
        values, stamp = {}, time.strftime("%H:%M:%S")
        for qid, q in questions.items():
            v, label = _primary(q["type"], q, data["answers"][qid])
            values[qid] = (v, label)
            prev = _debounced(s, qid, label, params["debounce"])
            if prev is not None:
                s["log"].appendleft((stamp, qid, prev, label, v))
        s["history"].append(values)
        s["result"] = (questions, data, {"rtt": rtt, "age": (now - newest_t) * 1000, "mode": params["mode"],
                                         "frames": body.get("video_frames", 1)})
        s["error"] = None
        s["result_id"] += 1
    except Exception as e:  # noqa: BLE001 - surfaced in the UI, loop keeps going
        s["error"] = f"{type(e).__name__}: {e}"
        s["result_id"] += 1
        time.sleep(0.5)  # back off rather than hammering a failing API


def _rate(s: dict) -> float:
    d = list(s["done"])
    return (len(d) - 1) / (d[-1] - d[0]) if len(d) > 1 and d[-1] > d[0] else 0.0


def _live_meta(s: dict) -> str:
    if not s["result"]:
        return ""
    _, data, m = s["result"]
    t = data["x_clef"]["timing"]
    chips = [("detections / s", f"{_rate(s):.1f}"), ("forward pass", f"{t['forward_ms']:.0f} ms"),
             ("frame → answer", f"{m['age']:.0f} ms"), ("input tokens", f"{data['usage']['input_tokens']:,}"),
             ("mode", "clip ×%d" % m["frames"] if m["mode"] == "clip" else "frame"), ("queries", str(s["queries"]))]
    return "<div class='cl-chips'>" + "".join(
        f"<div class='cl-chip'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in chips) + "</div>"


def _live_timeline(s: dict, questions: dict) -> str:
    hist = list(s["history"])
    if not hist:
        return ""
    n = len(hist)
    rows = []
    for qid, q in questions.items():
        pts = [h.get(qid) for h in hist]
        if q["type"] == "choice":
            opts = list(q["criteria"])
            rects = "".join(
                f"<rect x='{i * 300 / n:.2f}' y='4' width='{300 / n + 0.3:.2f}' height='22' "
                f"fill='{PALETTE[opts.index(p[1]) % len(PALETTE)] if p and p[1] in opts else 'transparent'}'>"
                f"<title>{html.escape(p[1]) if p else ''}</title></rect>" for i, p in enumerate(pts))
            legend = " ".join(f"<span class='cl-leg'><i style='background:{PALETTE[i % len(PALETTE)]}'></i>"
                              f"{html.escape(o)}</span>" for i, o in enumerate(opts))
            svg = f"<svg viewBox='0 0 300 30' preserveAspectRatio='none' class='cl-spark'>{rects}</svg>"
            extra = f"<div class='cl-legend'>{legend}</div>"
        else:
            coords = " ".join(f"{(i + 0.5) * 300 / n:.2f},{28 - 24 * (p[0] if p else 0):.2f}" for i, p in enumerate(pts))
            svg = (f"<svg viewBox='0 0 300 30' preserveAspectRatio='none' class='cl-spark'>"
                   f"<line x1='0' y1='16' x2='300' y2='16' class='cl-mid'/>"
                   f"<polyline points='{coords}' class='cl-line'/></svg>")
            extra = ""
        cur = pts[-1]
        now = f"{html.escape(cur[1])} · {cur[0]:.2f}" if cur else ""
        rows.append(f"<div class='cl-tl'><div class='cl-tl-h'><span class='cl-id'>{html.escape(qid)}</span>"
                    f"<span class='cl-type t-{q['type']}'>{q['type']}</span><span class='cl-tl-now'>{now}</span></div>"
                    f"{svg}{extra}</div>")
    span = ""
    d = list(s["done"])
    if len(d) > 1:
        span = f" · last {min(n, len(hist))} detections"
    return (f"<div class='cl-card'><div class='cl-qh'><span class='cl-id'>timeline</span>"
            f"<span class='cl-dim'>P(true) / confidence / expected level{span}</span></div>{''.join(rows)}</div>")


def _live_log(s: dict) -> str:
    if not s["log"]:
        return "<div class='cl-dim'>Changes in any answer will be logged here.</div>"
    items = "".join(f"<div class='cl-log'><code>{t}</code> <b>{html.escape(q)}</b> {html.escape(a)} → "
                    f"<b>{html.escape(b)}</b> <span class='cl-dim'>({v:.2f})</span></div>" for t, q, a, b, v in list(s["log"])[:12])
    return f"<div class='cl-card'><div class='cl-qh'><span class='cl-id'>changes</span></div>{items}</div>"


def live_tick(frame, questions_text, state_text, mode, size, clip_frames, clip_secs, max_rate, debounce, enabled,
              request: gr.Request):
    """Runs for every streamed frame: buffer it, maybe start a query, render new results."""
    s = _session(request)
    if frame is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    now = time.time()
    s["frames"].append((now, _jpeg(frame, BUFFER_SIDE, 80)))
    try:
        questions = json.loads(questions_text or "")
        assert isinstance(questions, dict) and questions
    except (json.JSONDecodeError, AssertionError):
        questions = None
        if s["error"] != INVALID_SCHEMA:
            s["error"] = INVALID_SCHEMA
            s["result_id"] += 1  # force a re-render so the message appears
    else:
        if s["error"] == INVALID_SCHEMA:
            s["error"] = None
    key = json.dumps(questions, sort_keys=True) if questions else None
    if key != s["schema_key"]:  # new schema: old timeline/log no longer comparable
        s.update(history=deque(maxlen=120), log=deque(maxlen=40), argmax={}, pending={}, schema_key=key)
    params = None
    if enabled and questions:
        params = {"questions": questions, "state": parse_state(state_text), "mode": mode, "size": int(size),
                  "clip_frames": int(clip_frames), "clip_secs": float(clip_secs), "max_rate": float(max_rate),
                  "debounce": int(debounce)}
    with s["lock"]:  # frame handlers can overlap; only one may start the worker
        s["params"], s["last_tick"] = params, now
        start = bool(params and not s["busy"])
        if start:
            s["busy"] = True
    if start:
        threading.Thread(target=_live_worker, args=(s,), daemon=True).start()
    with s["lock"]:
        result, error = s["result"], s["error"]
        if result is None:  # nothing scored yet: show a pending error, otherwise wait
            if error and s["rendered_id"] != ("err", error):
                s["rendered_id"] = ("err", error)
                return f"<div class='cl-empty'>{html.escape(error)}</div>", "", "", ""
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        if s["result_id"] == s["rendered_id"]:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        s["rendered_id"] = s["result_id"]
    qs, data, _ = result
    meta = _live_meta(s)
    if error:
        meta = f"<div class='cl-status warn'>● {html.escape(error)}</div>" + meta
    return render_answers(qs, data), meta, _live_timeline(s, qs), _live_log(s)


def live_reset(request: gr.Request):
    with LIVE_LOCK:
        LIVE[request.session_hash] = _fresh_session()
    return EMPTY_LIVE, "", "", ""


def live_close(request: gr.Request):
    with LIVE_LOCK:
        LIVE.pop(request.session_hash, None)


def load_webcam_preset(name):
    p = presets.WEBCAM[name]
    state = p["state"] if isinstance(p["state"], str) else j(p["state"])
    return j(p["questions"]), state, p["mode"], p.get("size", 448)


INVALID_SCHEMA = "Questions JSON is invalid - fix it to resume detections."
EMPTY_LIVE = ("<div class='cl-empty'>Allow camera access, then press <b>● Record</b> under the preview. Detections "
              "stream in here until you press Stop.</div>")


# ------------------------------------------------------------------ API tab
ENDPOINTS = {
    "GET /health": None,
    "GET /info": None,
    "GET /v1/models": None,
    "POST /v1/systemone": {"model": MODEL, **{k: presets.DECIDE["Support ticket triage"][k] for k in ("state", "questions")}},
    "POST /v1/systemone/batch": {"model": MODEL, "state": presets.BATCH["Support inbox"]["items"][:4],
                                 "questions": presets.DECIDE["Support ticket triage"]["questions"]},
}


def api_template(name):
    body = ENDPOINTS[name]
    return j(body) if body is not None else ""


def api_send(name, body_text):
    method, path = name.split(" ", 1)
    body = None
    if method == "POST":
        try:
            body = json.loads(body_text or "{}")
        except json.JSONDecodeError as e:
            raise gr.Error(f"Body is not valid JSON: {e}")
    t0 = time.perf_counter()
    try:
        r = requests.request(method, API_BASE + path, json=body, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise gr.Error(f"API unreachable: {e}")
    ms = (time.perf_counter() - t0) * 1000
    try:
        out = r.json()
    except ValueError:
        out = {"raw": r.text}
    curl = curl_for(path, body) if body is not None else f"curl -s {PUBLIC_API}{path}"
    return out, f"HTTP {r.status_code} · {ms:.0f} ms", curl


# --------------------------------------------------------------- status bar
def status_html():
    try:
        info = requests.get(API_BASE + "/info", timeout=5).json()
    except (requests.RequestException, ValueError):
        return "<div class='cl-status bad'>● API unreachable</div>"
    if not info.get("ready"):
        return f"<div class='cl-status warn'>● {html.escape(info.get('error') or 'Model loading…')}</div>"
    g = info.get("gpu") or {}
    bits = [
        "● ready",
        html.escape(g.get("name", info.get("device", "?"))),
        f"{g.get('vram_used_mb', 0) / 1024:.1f} / {g.get('vram_total_mb', 0) / 1024:.1f} GB VRAM",
        info.get("dtype", ""),
        f"{(info.get('parameters') or 0) / 1e9:.2f}B params",
        "rev " + (info.get("revision") or "?")[:7],
        "fla fast path" if info.get("fast_path") else "torch fallback",
        f"{info['stats']['records_scored']} decisions served",
    ]
    return "<div class='cl-status ok'>" + " · ".join(bits) + "</div>"


# --------------------------------------------------------------------- UI
CSS = (HERE / "style.css").read_text()

INTRO = f"""
<div class='cl-hero'>
  <div class='cl-hero-t'>Clef-Flash <span>typed decisions in one forward pass</span></div>
  <div class='cl-hero-s'>Give <a href='https://huggingface.co/Cloudflare/clef-flash' target='_blank'>Cloudflare/clef-flash</a>
  a <b>state</b> (text, JSON, images or video) and a <b>schema of typed questions</b>. It returns a calibrated
  probability for <i>every allowed option of every question</i>. There's no text generation, no output parsing, and
  no invalid answers. Question types: <code>noul</code> (true/false) · <code>choice</code> (named options) ·
  <code>score</code> (ordered levels).</div>
</div>
"""

SCHEMA_HELP = (
    "One row per question. **options**: for `choice` write `key: description | key: description`; "
    "for `score` write levels low→high `Low | Medium | High`; for `noul` leave blank (or `true: … | false: …`)."
)

with gr.Blocks(title="Clef-Flash local") as demo:
    gr.HTML(INTRO)
    status = gr.HTML(status_html)
    gr.Timer(10).tick(status_html, outputs=status, show_progress="hidden")

    with gr.Tabs():
        # ---------------------------------------------------------- Decide
        with gr.Tab("🎯 Decide", id="decide"):
            with gr.Row():
                with gr.Column(scale=5):
                    d_preset = gr.Dropdown(list(presets.DECIDE), value="Support ticket triage", label="Load an example")
                    d_state = gr.Textbox(label="State (free text or JSON)", lines=6)
                    with gr.Tabs():
                        with gr.Tab("Schema JSON"):
                            d_questions = gr.Code(label="Questions", language="json", lines=16)
                        with gr.Tab("Schema builder"):
                            gr.Markdown(SCHEMA_HELP)
                            d_table = gr.Dataframe(headers=["id", "type", "instructions", "options"],
                                                   datatype=["str", "str", "str", "str"], column_count=4,
                                                   row_count=4, interactive=True, wrap=True)
                            with gr.Row():
                                d_t2j = gr.Button("Table → JSON", size="sm")
                                d_j2t = gr.Button("JSON → Table", size="sm")
                    d_media = gr.File(label="Optional image or video in the state",
                                      file_types=["image", "video"], type="filepath")
                    d_run = gr.Button("Decide", variant="primary")
                with gr.Column(scale=6):
                    d_meta = gr.HTML()
                    d_out = gr.HTML(EMPTY)
                    with gr.Accordion("SystemOne JSON response", open=False):
                        d_json = gr.JSON()
                    with gr.Accordion("Same request as curl", open=False):
                        d_curl = gr.Code(language="shell", interactive=False)
            d_preset.change(load_decide_preset, d_preset, [d_state, d_questions, d_table])
            d_t2j.click(table_to_questions, d_table, d_questions)
            d_j2t.click(questions_to_table, d_questions, d_table)
            d_run.click(decide, [d_state, d_questions, d_media], [d_out, d_meta, d_json, d_curl])

        # ---------------------------------------------------------- Vision
        with gr.Tab("🖼️ Vision & video", id="vision"):
            gr.Markdown("The same typed questions, asked about pixels. Images are downscaled server-side; "
                        "videos are sampled into frames so a long clip still costs a bounded number of tokens.")
            with gr.Row():
                with gr.Column(scale=5):
                    v_preset = gr.Dropdown(list(presets.VISION), value="Restaurant photo moderation", label="Preset")
                    with gr.Row():
                        v_image = gr.Image(label="Image", type="filepath", height=260)
                        v_video = gr.Video(label="…or video (takes precedence)", height=260)
                    v_frames = gr.Slider(4, 32, value=16, step=2, label="Video frames sampled")
                    v_state = gr.Textbox(label="State / context", lines=2)
                    v_questions = gr.Code(label="Questions", language="json", lines=14)
                    v_run = gr.Button("Decide", variant="primary")
                with gr.Column(scale=6):
                    v_meta = gr.HTML()
                    v_out = gr.HTML(EMPTY)
                    with gr.Accordion("SystemOne JSON response", open=False):
                        v_json = gr.JSON()
            v_preset.change(load_vision_preset, v_preset, [v_image, v_video, v_state, v_questions])
            v_run.click(vision_decide, [v_image, v_video, v_state, v_questions, v_frames], [v_out, v_meta, v_json])

        # ----------------------------------------------------- Live webcam
        with gr.Tab("📹 Live webcam", id="webcam"):
            gr.Markdown("Your browser's webcam, asked the same questions over and over. Only one query is in flight "
                        "at a time and it always uses the newest frame, so this runs as fast as the GPU allows "
                        "without falling behind. **Frame** mode scores the latest frame. **Clip** mode sends the "
                        "last few seconds as a short video with real timestamps, for questions about motion. "
                        "Webcam access needs the page opened on `localhost` (or HTTPS).")
            with gr.Row():
                with gr.Column(scale=5):
                    w_cam = gr.Image(label="Webcam", sources=["webcam"], streaming=True, type="numpy", height=360)
                    w_preset = gr.Dropdown(list(presets.WEBCAM), value="Room watch", label="Question set")
                    with gr.Row():
                        w_mode = gr.Radio([("Frame", "frame"), ("Clip", "clip")], value="frame", label="Mode")
                        w_enabled = gr.Checkbox(value=True, label="Detections on")
                    with gr.Row():
                        w_size = gr.Slider(224, 1024, value=448, step=32, label="Frame size (px, longest side)")
                        w_rate = gr.Slider(0.5, 10, value=10, step=0.5, label="Max detections / s")
                    with gr.Row():
                        w_frames = gr.Slider(2, 16, value=8, step=2, label="Clip frames")
                        w_secs = gr.Slider(1, 6, value=3, step=0.5, label="Clip length (s)")
                    w_debounce = gr.Slider(1, 10, value=3, step=1, label="Log a change after N detections in a row",
                                           info="Debounces flicker when an answer sits near 50/50")
                    w_state = gr.Textbox(label="State / context", lines=1)
                    w_questions = gr.Code(label="Questions", language="json", lines=12)
                    w_reset = gr.Button("Reset timeline", size="sm")
                with gr.Column(scale=6):
                    w_meta = gr.HTML()
                    w_out = gr.HTML(EMPTY_LIVE)
                    w_timeline = gr.HTML()
                    w_log = gr.HTML()
            w_preset.change(load_webcam_preset, w_preset, [w_questions, w_state, w_mode, w_size])
            w_cam.stream(live_tick, [w_cam, w_questions, w_state, w_mode, w_size, w_frames, w_secs, w_rate, w_debounce, w_enabled],
                         [w_out, w_meta, w_timeline, w_log], stream_every=STREAM_EVERY, time_limit=None,
                         show_progress="hidden", concurrency_limit=None)
            w_reset.click(live_reset, None, [w_out, w_meta, w_timeline, w_log])

        # ----------------------------------------------------- Tool router
        with gr.Tab("🧰 Tool router", id="router"):
            gr.Markdown("Agent routing without generating a single token: the tool list becomes a `choice` "
                        "question, plus `noul` gates for *call now?* and *ask first?*, plus any enum arguments "
                        "you want filled. (Clef-Flash scores 98.8 on BFCL and 97.7 on the home appliance simulator.)")
            with gr.Row():
                with gr.Column(scale=5):
                    r_preset = gr.Dropdown(list(presets.TOOLSETS), value="Smart home", label="Toolset")
                    r_msg = gr.Textbox(label="User message", lines=2)
                    r_hist = gr.Textbox(label="Earlier conversation (optional, one 'role: text' per line)", lines=2,
                                        placeholder="user: turn on the lights\nassistant: which room?")
                    r_tools = gr.Dataframe(headers=["tool", "description"], datatype=["str", "str"],
                                           column_count=2, row_count=6, interactive=True, label="Tools", wrap=True)
                    r_none = gr.Checkbox(value=True, label="Allow 'no_tool' option")
                    r_extra = gr.Code(label="Enum arguments / extra questions (JSON)", language="json", lines=8)
                    r_run = gr.Button("Route", variant="primary")
                with gr.Column(scale=6):
                    r_meta = gr.HTML()
                    r_out = gr.HTML(EMPTY)
                    with gr.Accordion("Request sent to /v1/systemone", open=False):
                        r_req = gr.Code(language="json", interactive=False)
            r_preset.change(load_toolset, r_preset, [r_msg, r_tools, r_extra, r_hist])
            r_run.click(route, [r_msg, r_tools, r_extra, r_none, r_hist], [r_out, r_meta, r_req])

        # ----------------------------------------------------------- Batch
        with gr.Tab("⚡ Batch triage", id="batch"):
            gr.Markdown("One schema, many items, packed into padded GPU batches by the API. Because the outputs "
                        "are calibrated probabilities, a single **confidence threshold** turns the model into a "
                        "policy: confident items are auto-routed, uncertain ones go to a human.")
            with gr.Row():
                with gr.Column(scale=5):
                    b_preset = gr.Dropdown(list(presets.BATCH), value="Support inbox", label="Dataset")
                    b_items = gr.Textbox(label="Items (one per line; JSON lines allowed)", lines=12)
                    b_file = gr.File(label="…or upload .txt / .csv (uses a 'text' column if present)",
                                     file_types=[".txt", ".csv"], type="filepath")
                    b_questions = gr.Code(label="Shared schema", language="json", lines=12)
                    with gr.Row():
                        b_thresh = gr.Slider(0.5, 0.99, value=0.8, step=0.01, label="Auto-route confidence threshold",
                                             scale=3)
                        b_gate = gr.Checkbox(value=False, label="Also gate on score questions", scale=1)
                    b_run = gr.Button("Run batch", variant="primary")
                with gr.Column(scale=7):
                    b_stats = gr.HTML()
                    b_table = gr.Dataframe(label="Decisions", wrap=True, interactive=False)
                    b_csv = gr.File(label="Download CSV")
                    b_dist = gr.HTML()
            b_preset.change(load_batch_preset, b_preset, [b_items, b_questions])
            b_file.change(_items_from_file, b_file, b_items)
            b_run.click(batch_run, [b_items, b_questions, b_thresh, b_gate], [b_table, b_stats, b_dist, b_csv])

        # --------------------------------------------------------- Compare
        with gr.Tab("🔀 What-if", id="compare"):
            gr.Markdown("Change the state, keep the schema, and watch the probabilities move. Both states are "
                        "scored in one batched forward pass. Light bars = A, accent bars = B, Δ = B − A in "
                        "percentage points.")
            c_preset = gr.Dropdown(list(presets.COMPARE), value="Same ticket, different severity", label="Scenario")
            with gr.Row():
                c_a = gr.Textbox(label="State A", lines=5)
                c_b = gr.Textbox(label="State B", lines=5)
            with gr.Row():
                with gr.Column(scale=5):
                    c_questions = gr.Code(label="Shared schema", language="json", lines=12)
                    c_run = gr.Button("Compare", variant="primary")
                with gr.Column(scale=6):
                    c_meta = gr.HTML()
                    c_out = gr.HTML(EMPTY)
            c_preset.change(load_compare_preset, c_preset, [c_a, c_b, c_questions])
            c_run.click(compare, [c_a, c_b, c_questions], [c_out, c_meta])

        # ------------------------------------------------------------- API
        with gr.Tab("🛠️ API playground", id="api"):
            gr.Markdown(f"Direct access to the API container at `{PUBLIC_API}`. `/v1/systemone` takes a "
                        "Jev/SystemOne request body and returns the same response body, plus an `x_clef` block "
                        "with every option's probability and timings. Images/videos go in `images` / `videos` "
                        "as base64 or data: URLs. Interactive docs: "
                        f"[{PUBLIC_API}/docs]({PUBLIC_API}/docs).")
            with gr.Row():
                with gr.Column():
                    a_ep = gr.Dropdown(list(ENDPOINTS), value="POST /v1/systemone", label="Endpoint")
                    a_body = gr.Code(label="Body", language="json", lines=20, value=api_template("POST /v1/systemone"))
                    a_send = gr.Button("Send", variant="primary")
                with gr.Column():
                    a_status = gr.Markdown()
                    a_out = gr.JSON(label="Response")
                    a_curl = gr.Code(label="curl", language="shell", interactive=False)
            a_ep.change(api_template, a_ep, a_body)
            a_send.click(api_send, [a_ep, a_body], [a_out, a_status, a_curl])

    gr.HTML("<div class='cl-foot'>Local stack · weights in <code>./models</code> · "
            "<a href='https://blog.cloudflare.com/clef-decision-models' target='_blank'>Clef announcement</a></div>")

    demo.load(load_decide_preset, d_preset, [d_state, d_questions, d_table])
    demo.load(load_vision_preset, v_preset, [v_image, v_video, v_state, v_questions])
    demo.load(load_toolset, r_preset, [r_msg, r_tools, r_extra, r_hist])
    demo.load(load_webcam_preset, w_preset, [w_questions, w_state, w_mode, w_size])
    demo.unload(live_close)
    demo.load(load_batch_preset, b_preset, [b_items, b_questions])
    demo.load(load_compare_preset, c_preset, [c_a, c_b, c_questions])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=7860,
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="slate"),
        css=CSS, show_error=True, allowed_paths=[str(HERE / "examples"), str(HERE / "outputs")],
    )
