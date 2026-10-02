"""Capture README screenshots (and optionally a walkthrough video) of the Clef-Flash frontend."""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:7860/"
OUT = Path(sys.argv[1])
MODE = sys.argv[2] if len(sys.argv) > 2 else "shots"   # shots | video
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1440, 900


def visible(page, selector):
    return page.locator(f"{selector} >> visible=true")


def click_button(page, name):
    page.get_by_role("button", name=name, exact=True).locator("visible=true").first.click()


def open_tab(page, name):
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(600)


def pick(page, label, option):
    box = page.get_by_label(label, exact=True).locator("visible=true").first
    box.click()
    page.get_by_role("option", name=option, exact=True).click()
    page.wait_for_timeout(900)


def wait_cards(page, n=1, timeout=60000):
    visible(page, ".cl-card").nth(n - 1).wait_for(timeout=timeout)
    page.wait_for_timeout(700)  # let the bar width transitions settle


def shot(page, name, selector=None):
    page.wait_for_timeout(300)
    if selector:
        visible(page, selector).first.screenshot(path=str(OUT / name))
    else:
        page.screenshot(path=str(OUT / name))
    print("saved", name)


def scroll_top(page):
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(300)


def run(page, video):
    pause = (lambda ms: page.wait_for_timeout(ms)) if video else (lambda ms: None)
    page.goto(URL)
    page.wait_for_selector(".cl-status.ok", timeout=60000)
    page.wait_for_timeout(1500)
    pause(1500)

    # ---- Decide
    if video:
        pick(page, "Load an example", "Financial entity sentiment (many fields, one pass)")
        pause(800)
    click_button(page, "Decide")
    wait_cards(page, 2)
    scroll_top(page)
    if not video:
        shot(page, "decide.png", ".gradio-container")
    pause(2500)
    if video:
        page.mouse.wheel(0, 500); pause(1500)
        page.mouse.wheel(0, -500); pause(500)

    # ---- Vision & video
    open_tab(page, "🖼️ Vision & video")
    pause(800)
    click_button(page, "Decide")
    wait_cards(page, 2)
    scroll_top(page)
    if not video:
        shot(page, "vision.png", ".gradio-container")
    pause(2000)
    pick(page, "Preset", "Video: what happens in the clip")
    pause(800)
    click_button(page, "Decide")
    page.wait_for_timeout(1500)
    wait_cards(page, 4)
    scroll_top(page)
    if not video:
        shot(page, "video.png", ".gradio-container")
    pause(2500)

    # ---- Tool router
    open_tab(page, "🧰 Tool router")
    if video:
        box = page.get_by_label("User message", exact=True)
        box.click(); box.fill("")
        box.press_sequentially("put some jazz on in the living room please", delay=35)
        pause(500)
    click_button(page, "Route")
    visible(page, ".cl-route").first.wait_for(timeout=60000)
    page.wait_for_timeout(700)
    scroll_top(page)
    if not video:
        shot(page, "router.png", ".gradio-container")
    pause(3000)

    # ---- Batch
    open_tab(page, "⚡ Batch triage")
    click_button(page, "Run batch")
    visible(page, ".cl-chip").first.wait_for(timeout=120000)
    page.wait_for_timeout(1200)
    scroll_top(page)
    if not video:
        shot(page, "batch.png", ".gradio-container")
    pause(2000)
    if video:
        page.mouse.wheel(0, 450); pause(2500)
        page.mouse.wheel(0, -450); pause(400)

    # ---- What-if
    open_tab(page, "🔀 What-if")
    if video:
        pick(page, "Scenario", "One word changes the referent")
        pause(600)
    click_button(page, "Compare")
    wait_cards(page, 1)
    scroll_top(page)
    if not video:
        shot(page, "compare.png", ".gradio-container")
    pause(3000)

    # ---- API playground
    open_tab(page, "🛠️ API playground")
    click_button(page, "Send")
    page.get_by_text("HTTP 200").wait_for(timeout=60000)
    page.wait_for_timeout(800)
    if not video:
        shot(page, "api.png", ".gradio-container")
    pause(2000)


with sync_playwright() as p:
    browser = p.chromium.launch()
    kwargs = dict(viewport={"width": W, "height": H}, color_scheme="dark", device_scale_factor=1)
    if MODE == "video":
        kwargs["record_video_dir"] = str(OUT)
        kwargs["record_video_size"] = {"width": W, "height": H}
    ctx = browser.new_context(**kwargs)
    page = ctx.new_page()
    t0 = time.time()
    run(page, MODE == "video")
    path = page.video.path() if MODE == "video" else None
    ctx.close()
    browser.close()
    print(f"done in {time.time() - t0:.0f}s", path or "")
