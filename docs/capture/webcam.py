"""Record the Live webcam tab for the README, with a VIDEO FILE standing in for the camera.

Never point this at a real camera: anything captured here ends up in a public repo.
It always launches Chromium with a fake capture device fed from the given file.

    python docs/capture/webcam.py <out-dir> <video.y4m|video.mjpeg>
"""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
if len(sys.argv) < 3 or not Path(sys.argv[2]).is_file() or Path(sys.argv[2]).suffix not in (".y4m", ".mjpeg"):
    sys.exit(__doc__)
OUT = Path(sys.argv[1]); FAKE = sys.argv[2]; OUT.mkdir(parents=True, exist_ok=True)
Q = {"scene": {"type": "choice", "instructions": "What does the camera show?",
               "criteria": {"room": "A living room", "food": "A plate of food", "person": "A person"}},
     "food": {"type": "noul", "instructions": "Is food visible?"},
     "tv": {"type": "noul", "instructions": "Is a television visible?"}}
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                f"--use-file-for-fake-video-capture={FAKE}"])
    ctx = b.new_context(viewport={"width": 1280, "height": 860}, color_scheme="dark",
                        record_video_dir=str(OUT), record_video_size={"width": 1280, "height": 860})
    ctx.grant_permissions(["camera"], origin="http://localhost:7860")
    page = ctx.new_page(); t_page = time.time()
    page.goto("http://localhost:7860/"); page.wait_for_selector(".cl-status.ok")
    page.get_by_role("tab", name="📹 Live webcam").click(); page.wait_for_timeout(1200)
    for label in ("Click to Access Webcam", "Access webcam", "webcam"):
        el = page.get_by_text(label, exact=False).locator("visible=true")
        if el.count(): el.first.click(); break
    page.wait_for_timeout(1500)
    # put the questions editor in view and replace its contents
    editor = page.locator(".cm-content >> visible=true").last
    editor.click(); page.keyboard.press("Control+A"); page.keyboard.press("Delete")
    page.keyboard.insert_text(json.dumps(Q, indent=1))
    page.evaluate("window.scrollTo({top: document.querySelector('#component-0') ? 300 : 300})")
    page.wait_for_timeout(500)
    trim = time.time() - t_page
    page.get_by_text("Record", exact=True).locator("visible=true").first.click()
    page.wait_for_timeout(1500)
    page.evaluate("window.scrollTo({top: 330, behavior: 'smooth'})")
    page.wait_for_timeout(11000)
    page.screenshot(path=str(OUT / "webcam.png"), full_page=True)
    chips = page.locator(".cl-chip >> visible=true").all_inner_texts()
    print("chips:", [c.replace("\n", "=") for c in chips], "log:", page.locator(".cl-log >> visible=true").all_inner_texts()[:4])
    v = page.video; ctx.close(); b.close()
    Path(v.path()).rename(OUT / "webcam.webm"); (OUT / "webcam.trim").write_text(f"{trim:.2f}")
