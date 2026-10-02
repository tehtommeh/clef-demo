# Clef-Flash, local

A self-contained local stack for [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash):
a GPU API container and a Gradio showcase container.

![Tool routing with Clef-Flash: the request is routed to play_music(room="living_room") at 97% confidence, then an off-topic question is routed to "reply directly"](docs/media/clip-router.gif)

▶ **[Watch the full 50-second walkthrough (MP4)](docs/media/walkthrough.mp4)**, recorded against
the running stack on an RTX 3090. Every number in it is a live result.

## What the model is

Clef-Flash is a 9.5B multimodal **decision model**. It is a Qwen3.5-9B backbone (with its vision
encoder) plus a 122M-parameter "joint schema head". You give it a **state** (text, JSON, images
or video) and a **schema of typed questions**. A single forward pass returns a probability for every
allowed option of every question:

| type | answers | example |
|---|---|---|
| `noul` | P(true) | "Is a service down?" |
| `choice` | one of named options + full distribution | department ∈ {billing, technical, …} |
| `score` | expected level over ordered options + distribution | urgency ∈ [Can wait, This week, Today] |

It never generates text, so there is no output parsing and no invalid answers. That's also why this
stack uses **FastAPI + transformers, not vLLM**: the custom head scores options and is not a causal LM.

## Start / stop

```bash
make up            # docker compose up -d --build
make down          # docker compose down
make logs          # follow both containers
make test          # endpoint smoke tests (scripts/smoke_test.py)
make model-test    # answer-level correctness + latency (scripts/model_test.py)
make stats         # GPU + container usage
```

| URL | What |
|---|---|
| http://localhost:7860 | Showcase UI |
| http://localhost:8000/docs | API (OpenAPI docs) |
| http://localhost:8000/info | Device, VRAM, revision, limits, stats |

The first start of a fresh checkout takes ~40 s: 8 s to load weights, plus ~30 s while Triton
compiles kernels. The compiled kernels are cached in `./cache/triton`, so later restarts are ready
in ~9 s. The first request at a new input shape (e.g. your first 16k-token state) is also slow once
while Triton autotunes for it.

## The frontend

### 🎯 Decide

Any state plus any schema. There are 11 presets that mirror the model's benchmark strengths:
support triage, invoice JSON, ContractNLI, FinEntity, phishing, SOC alerts, agent-trace auditing,
WinoGrande, CLadder and forecasting. Below, three entity-level sentiment questions are answered
in one 131 ms pass. A table-based **schema builder** means you don't have to write JSON by hand,
and every request is shown as the equivalent `curl`.

![Decide tab: three entity sentiment questions answered in one forward pass](docs/media/clip-decide.gif)

### 🖼️ Vision & video

The same typed questions, asked about an image or a video clip. The included clip cuts from a
living room to a plate of food. The model gets the opening scene (room, 96%), the closing scene
(food, 95%) and the cut itself (92.5%), so it reads the clip over time rather than one frame.

![Vision tab: questions about the start and end of a video clip](docs/media/clip-vision.gif)

### 🧰 Tool router

Agent routing without generating a single token. The tool list becomes a `choice` question, with
`noul` gates for "call now?" and "ask first?" and enum arguments such as room and direction. An
argument that resolves to `unspecified` is flagged as something to ask the user about. The clip
at the top of this page is this tab.

### ⚡ Batch triage

Many items, one schema, packed into padded GPU batches. The outputs are calibrated
probabilities, so a single confidence threshold turns the model into a policy: confident items
are auto-routed and uncertain ones go to a human. CSV in and out.

![Batch tab: 12 support tickets classified, with auto-routing versus human review](docs/media/clip-batch.gif)

### 🔀 What-if

Two states, one schema, scored side by side in one batch, with Δ in percentage points. One word
("too large" vs "too small") flips the referent by 96 points, and the severity of a ticket moves
urgency from "Can wait" to "Right now".

![What-if tab: changing one word flips the decision](docs/media/clip-compare.gif)

### 🛠️ API playground

Every endpoint, with an editable body, the raw response and the equivalent `curl`.

<details>
<summary><b>Full-page screenshots</b></summary>

| | |
|---|---|
| ![Decide](docs/media/decide.png) | ![Tool router](docs/media/router.png) |
| **Decide**: support triage | **Tool router**: smart home |
| ![Vision](docs/media/vision.png) | ![Video](docs/media/video.png) |
| **Vision**: restaurant photo moderation | **Video**: scene-cut clip |
| ![Batch](docs/media/batch.png) | ![What-if](docs/media/compare.png) |
| **Batch triage**: support inbox | **What-if**: severity comparison |
| ![API playground](docs/media/api.png) | |
| **API playground** | |

</details>

The media in `docs/media/` was captured with Playwright against the live stack. To regenerate it
after UI changes, see [`docs/capture/`](docs/capture/README.md).

## API

`POST /v1/systemone` is Jev/SystemOne-compatible (same request body, same response body). It adds
an `x_clef` block with every option's probability and timings, which plain SystemOne clients ignore.

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "clef-flash",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle the message?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"}
  }
}'
```

- **Images**: `"images": ["data:image/jpeg;base64,…"]`. These are downscaled server-side to `MAX_IMAGE_SIDE`.
- **Video**: `"videos": ["data:video/mp4;base64,…"]` (or a list of base64 frames per video), plus an
  optional `"video_frames": 16`. Frames are sampled uniformly.
- **Batch**: `POST /v1/systemone/batch` with `{"requests": [...]}`, or with a list-valued `state` and a
  shared `questions` object. Records are length-sorted and packed into padded GPU batches.
  Batched results match single requests to within 0.01 probability (verified by `model_test.py`).
- Errors: invalid schema → 422, GPU OOM → 413 (the server recovers), still loading → 503.

## Measured on this host (RTX 3090 24 GB, driver 595, bf16)

| | |
|---|---|
| Weights on disk | 17.8 GB in `./models/Cloudflare__clef-flash` (revision `17f0b0ad64ef`) |
| VRAM | 18.2 GB allocated at idle; 19.1 GB peak in normal use; 20.4 GB peak at 16k tokens or 8 images |
| Load | 7.5 s weights (from page cache) + 1.3 s warmup with a warm Triton cache |
| Support triage, 361 tokens | **124 ms** median forward pass, 128 ms HTTP round trip |
| Long state, 16k tokens | 3.9 s (scales linearly, ~0.25 ms/token) |
| One 1024-px image (~1k tokens) | 0.35 s forward, 0.6 s total including decode/preprocess |
| 16-frame video (~770 tokens) | 0.26 s forward, 0.4 s round trip |
| Batch, 48 support tickets | 10.6 items/s |

The model card quotes a 38.8 ms median latency on an H200. On a 3090 the forward pass is
compute-bound (~47 TFLOPS achieved). Most of each request is the schema itself, since state comes
before schema and so there is no prefix to share. For that reason batching gives a modest gain
(~25%: 4.5 s vs ~6 s sequential for 48 items) rather than a multiple.

## Memory and limits (`.env`)

The weights take 18.6 GB of 24 GB. There is no KV cache (one forward pass, `use_cache=False`), so all
remaining headroom goes to activations. The limits below were stress-tested on this card without OOM:

| Variable | Value | Meaning |
|---|---|---|
| `MAX_INPUT_TOKENS` | 16384 | Model card default; longer states are truncated (state only, never the schema) |
| `MAX_BATCH_TOKENS` | 8192 | Padded tokens per GPU batch (batch size × longest record) |
| `MAX_BATCH_SIZE` | 16 | Records per GPU batch |
| `MAX_IMAGE_SIDE` | 1024 | Longest side after downscaling (~1k vision tokens per image) |
| `VIDEO_MAX_FRAMES` / `VIDEO_MAX_SIDE` | 16 / 448 | Video sampling |

Media records are scored one per GPU batch, so a batch of image requests can't stack vision memory.

## Weights and updates

```bash
make check     # python3 scripts/download.py --check   → has upstream changed?
make verify    # local files still match models/download.lock.json?
python3 scripts/download.py --update Cloudflare/clef-flash && docker compose restart api
```

The API imports the repo's own `joint_schema_model.py` from the weights directory. An upstream fix
to encoding or the head arrives with the download, with no code change needed here. Weights are
mounted read-only and both containers run as UID/GID 1000 (set in `.env`).

## Model-specific notes

- **Pinned stack**: torch 2.11 + transformers 5.10.2 (what the model card was tested with),
  base image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`.
- **`flash-linear-attention` matters.** 24 of the 32 backbone layers are Gated DeltaNet (linear
  attention), and fla supplies their Triton kernels. transformers still logs "fast path is not
  available" because the optional `causal-conv1d` package is absent. That only affects a tiny
  depthwise conv, which falls back to `F.conv1d`. `/info` → `fast_path` confirms fla is loaded.
- Option order doesn't matter for `choice` questions: the encoder sorts them. `score` options are
  ordered and indexed from 0.
- `instructions` is optional; the question id is used when it's missing. Option descriptions are
  part of what the head scores, so write them as meaningful text.
- The Space's example images (`frontend/examples/*.jpg`) are CC0. `scene_cut.mp4` was generated
  from them with ffmpeg.

## Layout

```
api/            FastAPI server (Dockerfile, app.py, requirements.txt)
frontend/       Gradio app (app.py, presets.py, style.css, examples/)
scripts/        download.py, smoke_test.py, model_test.py, preflight.py, _bootstrap.py
models/         weights + download.lock.json (gitignored except the lock)
cache/triton/   compiled kernel cache (gitignored)
stack.json      smoke test definitions
docs/media/     README screenshots, clips and the walkthrough video
docs/capture/   Playwright scripts that regenerate docs/media
```
