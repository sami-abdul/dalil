"""Take the four product screenshots from the live Dalil screen on 127.0.0.1:8787.

Usage: uv run --with playwright python shots.py [question substring]
Picks the newest queue item whose text contains the substring (default: the Golden Visa question).
"""
import pathlib, sys
from playwright.sync_api import sync_playwright

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
needle = sys.argv[1] if len(sys.argv) > 1 else "Golden Visa holder sponsor"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
    pg.goto("http://127.0.0.1:8787/", wait_until="networkidle")
    pg.locator("button.queue-item", has_text=needle).first.click()
    pg.wait_for_selector("#draft .chip", timeout=20000)
    pg.wait_for_timeout(800)
    pg.screenshot(path=str(ASSETS / "shot-answer.png"))
    pg.locator("#draft .chip").first.click()
    pg.wait_for_timeout(1200)
    pg.screenshot(path=str(ASSETS / "shot-evidence.png"))
    if pg.locator("#cut-section .cut-item").count():
        pg.locator("#cut-section").scroll_into_view_if_needed()
        pg.wait_for_timeout(500)
        pg.locator("#answer-panel").screenshot(path=str(ASSETS / "shot-cut.png"))
    else:
        sys.exit("no cut sentence on this answer; pick one that has a cut")
    pg.locator("#tab-audit").click()
    pg.wait_for_timeout(800)
    pg.screenshot(path=str(ASSETS / "shot-audit.png"))
    b.close()
print("ok", sorted(x.name for x in ASSETS.glob("shot-*.png")))
