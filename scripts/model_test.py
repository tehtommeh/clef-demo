#!/usr/bin/env python3
"""Answer-level checks against the running API (stdlib only).

smoke_test.py proves every endpoint responds; this proves the model is actually
right on cases with an unambiguous answer, that batching does not change results
(right-padding bugs would show up here), that image and video inputs reach the
vision encoder, and measures latency.

    python3 scripts/model_test.py                 # against http://localhost:8000
    python3 scripts/model_test.py --api http://host:8000 --runs 50
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "frontend"))
import presets  # noqa: E402  (plain data module, no dependencies)

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def post(api, path, body, timeout=300):
    req = urllib.request.Request(api + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e


def get(api, path):
    with urllib.request.urlopen(api + path, timeout=30) as r:
        return json.loads(r.read())


def data_url(path):
    mime = "video/mp4" if path.suffix == ".mp4" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def req(preset, **extra):
    return {"model": "clef-flash", "state": preset["state"], "questions": preset["questions"], **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()
    api = args.api.rstrip("/")
    D, V = presets.DECIDE, presets.VISION
    ex = ROOT / "frontend" / "examples"
    results = []

    def check(name, fn, soft=False):
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        mark = (GREEN + "PASS") if ok else ((YELLOW + "WARN") if soft else (RED + "FAIL"))
        print(f"{mark}{RESET}  {name:<46} {detail}")
        results.append(ok or soft)

    def ans(body):
        return post(api, "/v1/systemone", body)[0]["answers"]

    print(f"Answer-level checks against {api}\n")
    check("support: department = technical", lambda: (
        (a := ans(req(D["Support ticket triage"])))["department"]["choice"] == "technical",
        f"{a['department']['choice']} @ {a['department']['confidence']:.3f}"))
    check("support: outage P(true) > 0.5", lambda: (
        (a := ans(req(D["Support ticket triage"])))["outage"]["noul"] > 0.5, f"P={a['outage']['noul']:.3f}"))
    check("invoice: total > 1000 USD", lambda: (
        (a := ans(req(D["Invoice processing (JSON state)"])))["large"]["noul"] > 0.5, f"P={a['large']['noul']:.3f}"))
    check("invoice: status = overdue", lambda: (
        (a := ans(req(D["Invoice processing (JSON state)"])))["status"]["choice"] == "overdue", a["status"]["choice"]))
    check("winogrande: 'too large' -> trophy", lambda: (
        (a := ans(req(D["Commonsense (WinoGrande style)"])))["too_large"]["choice"] == "trophy",
        f"{a['too_large']['choice']} @ {a['too_large']['confidence']:.3f}"))
    c = presets.COMPARE["One word changes the referent"]
    check("winogrande: 'too small' -> suitcase", lambda: (
        (a := ans({"model": "clef-flash", "state": c["b"], "questions": c["questions"]}))["referent"]["choice"] == "suitcase",
        f"{a['referent']['choice']} @ {a['referent']['confidence']:.3f}"))
    check("finentity: northwind neg / contoso pos", lambda: (
        (a := ans(req(D["Financial entity sentiment (many fields, one pass)"])))["northwind"]["choice"] == "negative"
        and a["contoso"]["choice"] == "positive", f"{a['northwind']['choice']}, {a['contoso']['choice']}"))
    check("phishing detected", lambda: (
        (a := ans(req(D["Phishing email check"])))["phishing"]["noul"] > 0.5, f"P={a['phishing']['noul']:.3f}"))

    t = presets.TOOLSETS["Smart home"]
    router_q = {"tool": {"type": "choice", "instructions": "Which tool best serves the user's latest request?",
                         "criteria": {n: d for n, d in t["tools"]}}, **t["extra"]}
    check("tool router: set_thermostat + bedroom", lambda: (
        (a := ans({"model": "clef-flash", "state": {"latest_user_message": t["message"]}, "questions": router_q}))
        ["tool"]["choice"] == "set_thermostat" and a["room"]["choice"] == "bedroom",
        f"{a['tool']['choice']} @ {a['tool']['confidence']:.3f}, room={a['room']['choice']}"))

    check("image: burger -> food", lambda: (
        (a := ans(req(V["Restaurant photo moderation"], images=[data_url(ex / "burger.jpg")])))["category"]["choice"] == "food",
        f"{a['category']['choice']} @ {a['category']['confidence']:.3f}"))
    check("image: living room -> living_room", lambda: (
        (a := ans(req(V["Rental listing check"], images=[data_url(ex / "living_room.jpg")])))["room"]["choice"] == "living_room",
        f"{a['room']['choice']} @ {a['room']['confidence']:.3f}"))
    vid = req(V["Video: what happens in the clip"], videos=[data_url(ex / "scene_cut.mp4")])
    check("video: ends on food", lambda: (
        (a := ans(vid))["closing_scene"]["choice"] == "food" and a["food_visible"]["noul"] > 0.5,
        f"closing={a['closing_scene']['choice']}, food_visible={a['food_visible']['noul']:.3f}"))
    check("video: starts in a room", lambda: (
        (a := ans(vid))["opening_scene"]["choice"] == "room", f"opening={a['opening_scene']['choice']}"), soft=True)

    def batch_consistency():
        items = presets.BATCH["Support inbox"]["items"]
        q = presets.BATCH["Support inbox"]["questions"]
        batch, _ = post(api, "/v1/systemone/batch", {"model": "clef-flash", "state": items, "questions": q})
        worst = 0.0
        for item, br in zip(items, batch["responses"]):
            single = post(api, "/v1/systemone", {"model": "clef-flash", "state": item, "questions": q})[0]
            for qid, probs in single["x_clef"]["probabilities"].items():
                for opt, p in probs.items():
                    worst = max(worst, abs(p - br["x_clef"]["probabilities"][qid][opt]))
        # bf16 matmuls differ slightly with batch shape; a padding bug shows up as >> 0.05.
        return worst < 0.05, f"max |Δp| batch vs single = {worst:.4f} over {len(items)} items"
    check("batch == single (padding correctness)", batch_consistency)

    # ------------------------------------------------------------- latency
    body = req(D["Support ticket triage"])
    fwd, rtt = [], []
    for _ in range(args.runs):
        r, ms = post(api, "/v1/systemone", body)
        fwd.append(r["x_clef"]["timing"]["forward_ms"])
        rtt.append(ms)
    tok = r["usage"]["input_tokens"]
    items = presets.BATCH["Support inbox"]["items"] * 4
    b, _ = post(api, "/v1/systemone/batch", {"model": "clef-flash", "state": items,
                                              "questions": presets.BATCH["Support inbox"]["questions"]})
    img_r, img_ms = post(api, "/v1/systemone", req(V["Rental listing check"], images=[data_url(ex / "living_room.jpg")]))
    vid_r, vid_ms = post(api, "/v1/systemone", vid)
    info = get(api, "/info")
    g = info["gpu"] or {}
    print(f"\nLatency ({args.runs} runs, {tok}-token support triage):"
          f" forward median {statistics.median(fwd):.1f} ms (p95 {sorted(fwd)[int(.95 * len(fwd)) - 1]:.1f}),"
          f" HTTP round trip median {statistics.median(rtt):.1f} ms")
    print(f"Batch: {len(items)} items in {b['timing']['total_ms']:.0f} ms -> {b['timing']['records_per_second']} items/s"
          f" ({b['timing']['gpu_batches']} GPU batches)")
    print(f"Image ({img_r['usage']['input_tokens']} tokens): forward {img_r['x_clef']['timing']['forward_ms']:.0f} ms,"
          f" round trip {img_ms:.0f} ms")
    print(f"Video ({vid_r['usage']['input_tokens']} tokens): forward {vid_r['x_clef']['timing']['forward_ms']:.0f} ms,"
          f" round trip {vid_ms:.0f} ms")
    print(f"GPU: {g.get('name')} - {g.get('vram_used_mb')} / {g.get('vram_total_mb')} MB in use,"
          f" torch peak {g.get('torch_peak_allocated_mb')} MB; load {info['load_seconds']}s,"
          f" warmup {info['warmup_seconds']}s; fast path: {info['fast_path']}")

    failed = results.count(False)
    print(f"\n{GREEN if not failed else RED}{len(results) - failed}/{len(results)} checks passed{RESET}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
