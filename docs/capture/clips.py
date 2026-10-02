"""Record one short clip per frontend feature (each its own Playwright video)."""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:7860/"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1280, 800


def vis(page, sel):
    return page.locator(f"{sel} >> visible=true")


def button(page, name):
    page.get_by_role("button", name=name, exact=True).locator("visible=true").first.click()


def pick(page, label, option):
    page.get_by_label(label, exact=True).locator("visible=true").first.click()
    page.wait_for_timeout(400)
    page.get_by_role("option", name=option, exact=True).click()
    page.wait_for_timeout(700)


def scroll_to(page, sel):
    vis(page, sel).first.scroll_into_view_if_needed()


def results_into_view(page, sel=".cl-chips"):
    """Bring the output column to the top so the bars are in frame."""
    vis(page, sel).first.evaluate("e => window.scrollTo({top: e.getBoundingClientRect().top + window.scrollY - 70, behavior: 'smooth'})")
    page.wait_for_timeout(700)


def clip_decide(page):
    page.wait_for_timeout(800)
    pick(page, "Load an example", "Financial entity sentiment (many fields, one pass)")
    page.wait_for_timeout(600)
    button(page, "Decide")
    vis(page, ".cl-card").nth(2).wait_for(timeout=60000)
    page.wait_for_timeout(500)
    results_into_view(page)
    page.wait_for_timeout(2500)


def clip_vision(page):
    page.get_by_role("tab", name="🖼️ Vision & video").click()
    page.wait_for_timeout(600)
    pick(page, "Preset", "Video: what happens in the clip")
    page.wait_for_timeout(1200)
    button(page, "Decide")
    vis(page, ".cl-card").nth(3).wait_for(timeout=60000)
    page.wait_for_timeout(500)
    results_into_view(page)
    page.wait_for_timeout(2800)


def clip_router(page):
    page.get_by_role("tab", name="🧰 Tool router").click()
    page.wait_for_timeout(700)
    box = page.get_by_label("User message", exact=True)
    box.click()
    box.fill("")
    box.press_sequentially("put some jazz on in the living room please", delay=40)
    page.wait_for_timeout(400)
    button(page, "Route")
    vis(page, ".cl-route").first.wait_for(timeout=60000)
    page.wait_for_timeout(400)
    results_into_view(page)
    page.wait_for_timeout(1500)
    box = page.get_by_label("User message", exact=True)
    box.scroll_into_view_if_needed()
    box.fill("")
    box.press_sequentially("what's the capital of France?", delay=40)
    page.wait_for_timeout(300)
    button(page, "Route")
    page.wait_for_timeout(1200)
    results_into_view(page)
    page.wait_for_timeout(2500)


def clip_batch(page):
    page.get_by_role("tab", name="⚡ Batch triage").click()
    page.wait_for_timeout(700)
    scroll_to(page, "button:has-text('Run batch')")
    page.wait_for_timeout(500)
    button(page, "Run batch")
    vis(page, ".cl-chip").first.wait_for(timeout=120000)
    page.wait_for_timeout(500)
    results_into_view(page)
    page.wait_for_timeout(3000)


def clip_compare(page):
    page.get_by_role("tab", name="🔀 What-if").click()
    page.wait_for_timeout(700)
    pick(page, "Scenario", "One word changes the referent")
    page.wait_for_timeout(800)
    button(page, "Compare")
    vis(page, ".cl-card").first.wait_for(timeout=60000)
    page.wait_for_timeout(1800)
    pick(page, "Scenario", "Same ticket, different severity")
    button(page, "Compare")
    page.wait_for_timeout(1500)
    results_into_view(page)
    page.wait_for_timeout(2800)


CLIPS = {"decide": clip_decide, "vision": clip_vision, "router": clip_router,
         "batch": clip_batch, "compare": clip_compare}

with sync_playwright() as p:
    browser = p.chromium.launch()
    for name, fn in CLIPS.items():
        if len(sys.argv) > 2 and name not in sys.argv[2:]:
            continue
        # Load the page off-camera first so clips start on a ready UI.
        ctx = browser.new_context(viewport={"width": W, "height": H}, color_scheme="dark",
                                  record_video_dir=str(OUT / name), record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        t_page = time.time()
        page.goto(URL)
        page.wait_for_selector(".cl-status.ok", timeout=60000)
        page.wait_for_timeout(1200)
        trim = time.time() - t_page  # seconds of page loading to cut from the head
        fn(page)
        video = page.video
        ctx.close()
        src = Path(video.path())
        (OUT / f"{name}.webm").write_bytes(src.read_bytes())
        (OUT / f"{name}.trim").write_text(f"{trim:.2f}")
        print("clip", name, src.stat().st_size)
    browser.close()
