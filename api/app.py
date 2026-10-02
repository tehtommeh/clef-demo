"""FastAPI server for Cloudflare/clef-flash.

Clef-Flash is not a text generator. It is a Qwen3.5-9B backbone plus a small
"joint schema head": given a *state* (text / JSON / images / video) and a schema
of typed *questions*, one forward pass yields a logit for every allowed option of
every question. So there is no chat endpoint here - the API surface is the
Jev/SystemOne one the model card prescribes:

    POST /v1/systemone          one request body -> one SystemOne response body
    POST /v1/systemone/batch    many request bodies, scored in padded GPU batches
    GET  /health /info /v1/models

Loading follows the model card exactly: the repo's own ``joint_schema_model.py``
(shipped next to the weights) does record encoding, batching and the head, and
``load_release_model`` builds the model. Nothing here re-implements that logic.

Design notes:
  * The model loads once in a background thread, so /health answers (503) while
    the ~19GB of weights stream in, instead of the container looking dead.
  * Inference runs in a worker thread behind a lock: one GPU, one forward pass at
    a time. Throughput comes from the batch endpoint, not from racing requests.
  * Responses carry an extra ``x_clef`` block (full per-option probabilities,
    timings, token counts). Clients that only understand SystemOne ignore it.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


def _writable_dir(path: str | None, fallback: str) -> str:
    """Triton must write its kernel cache; a root-owned bind mount would break fla."""
    for candidate in (path, fallback):
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=candidate):
                return candidate
        except OSError:
            continue
    return tempfile.mkdtemp()


os.environ["TRITON_CACHE_DIR"] = _writable_dir(os.environ.get("TRITON_CACHE_DIR"), "/tmp/triton")

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clef-api")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/Cloudflare__clef-flash"))
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "clef-flash")
DTYPE_NAME = os.environ.get("DTYPE", "auto")
# No KV cache is ever allocated (single forward, use_cache=False), so the only
# memory beyond the 18.6GB of weights is activations, which scale with tokens.
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "16384"))
MAX_BATCH_TOKENS = int(os.environ.get("MAX_BATCH_TOKENS", "16384"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))
MAX_BATCH_REQUESTS = int(os.environ.get("MAX_BATCH_REQUESTS", "512"))
# Images are downscaled so the longest side is at most this many pixels; a raw
# 12MP photo would otherwise become ~45k vision tokens.
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "8"))
VIDEO_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", "16"))
VIDEO_MAX_SIDE = int(os.environ.get("VIDEO_MAX_SIDE", "448"))

STATE: dict[str, Any] = {"ready": False, "error": None, "loading_since": None}
STATS = {"requests": 0, "records": 0, "forward_ms_total": 0.0, "errors": 0, "started": time.time()}
GPU_LOCK = threading.Lock()


# --------------------------------------------------------------------- loading
def resolve_dtype() -> torch.dtype:
    """bf16 is what the model was trained in; fall back only if the GPU lacks it."""
    if not torch.cuda.is_available():
        log.warning("No CUDA device visible - running on CPU in float32. This will be very slow.")
        return torch.float32
    supports_bf16 = torch.cuda.get_device_capability()[0] >= 8
    wanted = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        DTYPE_NAME, torch.bfloat16 if supports_bf16 else torch.float16
    )
    if wanted is torch.bfloat16 and not supports_bf16:
        log.warning("bfloat16 requested but this GPU lacks it; using float16.")
        wanted = torch.float16
    return wanted


def read_revision() -> str | None:
    lock = MODEL_DIR.parent / "download.lock.json"
    try:
        for meta in json.loads(lock.read_text())["models"].values():
            if Path(meta.get("path", "")).name == MODEL_DIR.name:
                return meta.get("revision")
    except Exception:  # noqa: BLE001 - the revision is informational only
        return None
    return None


def load_model() -> None:
    STATE["loading_since"] = time.time()
    try:
        if not (MODEL_DIR / "joint_schema_model.py").exists():
            raise FileNotFoundError(
                f"{MODEL_DIR}/joint_schema_model.py not found - run `python3 scripts/download.py "
                "Cloudflare/clef-flash` on the host first."
            )
        # The model's own code lives beside the weights; import it from there so
        # an upstream update to the repo is picked up by re-downloading only.
        sys.path.insert(0, str(MODEL_DIR))
        import joint_schema_model as jsm  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = resolve_dtype()
        log.info("Loading %s on %s as %s", MODEL_DIR, device, dtype)
        t0 = time.time()
        model, processor = jsm.load_release_model(
            MODEL_DIR, device=device, dtype=dtype,
            attn_implementation="sdpa", local_files_only=True,
        )
        STATE.update(
            jsm=jsm, model=model, processor=processor, device=device, dtype=str(dtype).replace("torch.", ""),
            load_seconds=round(time.time() - t0, 1), revision=read_revision(),
            params=sum(p.numel() for p in model.parameters()),
            head_params=sum(p.numel() for p in model.head.parameters()),
        )
        try:
            import fla  # noqa: F401
            # transformers warns "fast path not available" unless causal-conv1d is
            # also installed, but that only affects the tiny depthwise conv; the
            # gated-delta-rule kernels that matter come from fla.
            STATE["fast_path"] = "flash-linear-attention " + getattr(fla, "__version__", "?")
        except ImportError:
            STATE["fast_path"] = None
            log.warning("flash-linear-attention missing: linear-attention layers use the slow torch path")
        warmup()
        STATS.update(records=0, forward_ms_total=0.0)  # don't count the warmup as traffic
        STATE["ready"] = True
        log.info("Ready in %.1fs (load %.1fs)", time.time() - STATE["loading_since"], STATE["load_seconds"])
    except Exception as exc:  # noqa: BLE001 - surfaced via /health and /info
        log.exception("Model failed to load")
        STATE["error"] = f"{type(exc).__name__}: {exc}"


def warmup() -> None:
    """One tiny forward pass so Triton kernels compile before the first real request."""
    t0 = time.time()
    score_records([{
        "model": SERVED_MODEL_NAME, "state": "warmup",
        "questions": {"ok": {"type": "noul", "instructions": "Is this a warmup?"}},
    }])
    STATE["warmup_seconds"] = round(time.time() - t0, 1)
    log.info("Warmup forward pass took %.1fs", STATE["warmup_seconds"])


# ----------------------------------------------------------------- media input
def _b64_bytes(value: str) -> bytes:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        return base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"media is not valid base64: {exc}") from exc


def decode_image(value: Any):
    from PIL import Image

    if not isinstance(value, str):
        raise ValueError("images must be base64 strings or data: URLs")
    image = Image.open(io.BytesIO(_b64_bytes(value)))
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return image


def decode_video(value: Any, max_frames: int):
    """Decode a base64 video (or a list of base64 frames) into a T x H x W x C uint8 array.

    The model card takes "frame arrays"; sampling frames uniformly here keeps the
    vision token count bounded regardless of the clip's length.
    """
    import numpy as np
    from PIL import Image

    if isinstance(value, list):
        frames = [decode_image(v) for v in value]
    elif isinstance(value, dict) and "frames" in value:
        frames = [decode_image(v) for v in value["frames"]]
    elif isinstance(value, (str, dict)):
        import av

        raw = _b64_bytes(value["data"] if isinstance(value, dict) else value)
        with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
            fh.write(raw)
            fh.flush()
            with av.open(fh.name) as container:
                decoded = [f.to_image() for f in container.decode(video=0)]
        if not decoded:
            raise ValueError("video contained no decodable frames")
        count = min(max_frames, len(decoded))
        step = len(decoded) / count
        frames = [decoded[int(i * step)] for i in range(count)]
    else:
        raise ValueError("videos must be base64 video files or lists of base64 frames")
    out = []
    for frame in frames[:max_frames]:
        frame = frame.convert("RGB")
        frame.thumbnail((VIDEO_MAX_SIDE, VIDEO_MAX_SIDE), Image.BICUBIC)
        out.append(np.asarray(frame))
    # The processor wants equal-sized frames and an even count (temporal patch = 2).
    h, w = out[0].shape[:2]
    out = [np.asarray(Image.fromarray(f).resize((w, h))) if f.shape[:2] != (h, w) else f for f in out]
    if len(out) % 2:
        out.append(out[-1])
    return np.stack(out)


# ------------------------------------------------------------------ validation
def validate_request(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    if "state" not in body:
        raise ValueError("'state' is required")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("'questions' must be a non-empty object of question_id -> question")
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise ValueError(f"{qid}: question must be an object")
        qtype = q.get("type")
        if qtype not in ("noul", "choice", "score"):
            raise ValueError(f"{qid}: type must be noul, choice, or score")
        crit = q.get("criteria")
        if qtype == "choice" and not (isinstance(crit, dict) and len(crit) >= 2):
            raise ValueError(f"{qid}: choice needs 'criteria' as an object with at least 2 options")
        if qtype == "score" and not (isinstance(crit, list) and len(crit) >= 2):
            raise ValueError(f"{qid}: score needs 'criteria' as a list of at least 2 levels")
        if qtype == "noul" and crit is not None and not (
            isinstance(crit, dict) and set(crit) <= {"true", "false"}
        ):
            raise ValueError(f"{qid}: noul criteria may only describe 'true' and 'false'")

    record = {"model": body.get("model") or SERVED_MODEL_NAME, "state": body["state"], "questions": questions}
    images = body.get("images") or []
    videos = body.get("videos") or []
    if len(images) + len(videos) > MAX_IMAGES:
        raise ValueError(f"at most {MAX_IMAGES} images/videos per request")
    if images:
        record["images"] = [decode_image(v) for v in images]
    if videos:
        frames = int(body.get("video_frames") or VIDEO_MAX_FRAMES)
        record["videos"] = [decode_video(v, max(2, min(frames, 64))) for v in videos]
    if body.get("id") is not None:
        record["id"] = body["id"]
    return record


# ------------------------------------------------------------------- inference
def _plan_batches(encoded: list) -> list[list[int]]:
    """Group records by length so padding waste and peak memory stay bounded."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i].input_ids))
    batches, current, longest = [], [], 0
    for i in order:
        n = len(encoded[i].input_ids)
        has_media = encoded[i].media is not None
        # Padded cost is (batch size) x (longest sequence) once i joins.
        if current and (
            (len(current) + 1) * max(longest, n) > MAX_BATCH_TOKENS
            or len(current) >= MAX_BATCH_SIZE
            or has_media  # media records go alone: their vision tokens dominate memory
            or encoded[current[0]].media is not None
        ):
            batches.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, n)
    if current:
        batches.append(current)
    return batches


def score_records(bodies: list[dict[str, Any]]) -> dict[str, Any]:
    """Encode, batch and score records. Returns per-record SystemOne responses + timings."""
    jsm, model, processor = STATE["jsm"], STATE["model"], STATE["processor"]
    tokenizer = processor.tokenizer
    t_start = time.perf_counter()
    encoded = []
    for body in bodies:
        rec = validate_request(body)
        encoded.append(jsm.encode_record(tokenizer, rec, max_length=MAX_INPUT_TOKENS, processor=processor))
    t_encoded = time.perf_counter()

    device = next(model.parameters()).device
    logits: list[Any] = [None] * len(encoded)
    batches = _plan_batches(encoded)
    forward_ms = 0.0
    with GPU_LOCK:
        for idx in batches:
            batch = jsm.collate_records([encoded[i] for i in idx], tokenizer.pad_token_id, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_ms += (time.perf_counter() - t0) * 1000
            for i, rec_logits in zip(idx, out):
                logits[i] = [ql.float().softmax(-1).tolist() for ql in rec_logits]
    t_done = time.perf_counter()

    responses = []
    for body, enc, probs in zip(bodies, encoded, logits):
        per_q = {
            q.question_id: dict(zip(q.option_ids, p)) for q, p in zip(enc.questions, probs)
        }
        answers = {
            qid: jsm.systemone_answer(body["questions"][qid], p) for qid, p in per_q.items()
        }
        responses.append({
            "model": body.get("model") or SERVED_MODEL_NAME,
            "answers": answers,
            "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0},
            "x_clef": {
                "probabilities": {qid: {k: round(v, 6) for k, v in p.items()} for qid, p in per_q.items()},
                "options_scored": sum(len(p) for p in per_q.values()),
            },
        })
    STATS["records"] += len(bodies)
    STATS["forward_ms_total"] += forward_ms
    return {
        "responses": responses,
        "timing": {
            "encode_ms": round((t_encoded - t_start) * 1000, 2),
            "forward_ms": round(forward_ms, 2),
            "total_ms": round((t_done - t_start) * 1000, 2),
            "gpu_batches": len(batches),
        },
    }


async def run_scoring(bodies: list[dict[str, Any]]) -> dict[str, Any]:
    if not STATE["ready"]:
        raise HTTPException(503, STATE["error"] or "model is still loading")
    STATS["requests"] += 1
    try:
        return await asyncio.to_thread(score_records, bodies)
    except ValueError as exc:
        STATS["errors"] += 1
        raise HTTPException(422, str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        STATS["errors"] += 1
        torch.cuda.empty_cache()
        raise HTTPException(
            413, "GPU out of memory for this input - shorten the state, use fewer/smaller images, "
                 "or lower MAX_BATCH_TOKENS.",
        ) from exc


# ------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=load_model, name="model-loader", daemon=True).start()
    yield


app = FastAPI(title="Clef-Flash local API", version="1.0", lifespan=lifespan)


async def read_json(request: Request) -> Any:
    try:
        return await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"body is not valid JSON: {exc}") from exc


@app.get("/health")
def health():
    if STATE["ready"]:
        return {"status": "ok"}
    status = "error" if STATE["error"] else "loading"
    return JSONResponse({"status": status, "error": STATE["error"]}, status_code=503)


@app.get("/info")
def info():
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "vram_total_mb": total // 2**20,
            "vram_used_mb": (total - free) // 2**20,
            "torch_allocated_mb": torch.cuda.memory_allocated() // 2**20,
            "torch_peak_allocated_mb": torch.cuda.max_memory_allocated() // 2**20,
        }
    records = max(STATS["records"], 1)
    return {
        "model": SERVED_MODEL_NAME,
        "model_dir": str(MODEL_DIR),
        "revision": STATE.get("revision"),
        "ready": STATE["ready"],
        "error": STATE["error"],
        "device": STATE.get("device"),
        "dtype": STATE.get("dtype"),
        "fast_path": STATE.get("fast_path"),
        "parameters": STATE.get("params"),
        "head_parameters": STATE.get("head_params"),
        "load_seconds": STATE.get("load_seconds"),
        "warmup_seconds": STATE.get("warmup_seconds"),
        "gpu": gpu,
        "torch": torch.__version__,
        "limits": {
            "max_input_tokens": MAX_INPUT_TOKENS, "max_batch_tokens": MAX_BATCH_TOKENS,
            "max_batch_size": MAX_BATCH_SIZE, "max_batch_requests": MAX_BATCH_REQUESTS,
            "max_image_side": MAX_IMAGE_SIDE, "max_media_per_request": MAX_IMAGES,
            "video_max_frames": VIDEO_MAX_FRAMES, "video_max_side": VIDEO_MAX_SIDE,
        },
        "stats": {
            "requests": STATS["requests"], "records_scored": STATS["records"], "errors": STATS["errors"],
            "mean_forward_ms_per_record": round(STATS["forward_ms_total"] / records, 2),
            "uptime_s": round(time.time() - STATS["started"]),
        },
    }


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "cloudflare"}]}


@app.post("/v1/systemone")
async def systemone(request: Request):
    """Jev/SystemOne-compatible: state + typed questions -> per-question answers."""
    body = await read_json(request)
    result = await run_scoring([body])
    response = result["responses"][0]
    response["x_clef"]["timing"] = result["timing"]
    return response


@app.post("/v1/systemone/batch")
async def systemone_batch(request: Request):
    """Score many SystemOne requests at once; the server packs them into GPU batches.

    Body: {"requests": [<systemone body>, ...]}  or  {"state": [...many states...],
    "questions": {...shared schema...}} for the common "same schema, many items" case.
    """
    body = await read_json(request)
    if isinstance(body, dict) and isinstance(body.get("state"), list) and "requests" not in body:
        shared = {k: v for k, v in body.items() if k != "state"}
        bodies = [{**shared, "state": s} for s in body["state"]]
    elif isinstance(body, dict) and isinstance(body.get("requests"), list):
        bodies = body["requests"]
    else:
        raise HTTPException(422, "body needs 'requests': [...] or a list-valued 'state' with shared 'questions'")
    if not bodies:
        raise HTTPException(422, "no requests to score")
    if len(bodies) > MAX_BATCH_REQUESTS:
        raise HTTPException(413, f"at most {MAX_BATCH_REQUESTS} requests per batch call")
    result = await run_scoring(bodies)
    n = len(bodies)
    result["timing"]["records"] = n
    result["timing"]["records_per_second"] = round(n / max(result["timing"]["total_ms"], 1e-6) * 1000, 1)
    return result
