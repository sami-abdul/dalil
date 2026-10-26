"""Record a captioned demo video of the live Dalil screen on 127.0.0.1:8787.

Usage: uv run --with playwright python record_demo.py OUT_DIR [START_QUESTION_ID]
Writes OUT_DIR/dalil-demo.mp4: the screen with captions and a visible pointer, no voice.
The wait while Dalil drafts is cut out of the final video with ffmpeg.
"""
import pathlib, shutil, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = pathlib.Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "raw"; shutil.rmtree(RAW, ignore_errors=True)
START_ID = sys.argv[2] if len(sys.argv) > 2 else "e467b1af3d50"  # an answered question to show first
QUESTION = "Can a Golden Visa holder sponsor their parents, and what do they need?"

OVERLAY = """
(() => {
  const add = () => {
    if (document.getElementById('demo-cap')) return;
    const cap = document.createElement('div'); cap.id = 'demo-cap';
    cap.style.cssText = 'pointer-events:none;position:fixed;left:50%;bottom:165px;transform:translateX(-50%);max-width:1180px;background:rgba(15,42,61,.94);color:#fff;font:600 25px/1.4 "Noto Sans",Arial,sans-serif;padding:16px 26px;border-radius:12px;z-index:99999;box-shadow:0 10px 30px rgba(0,0,0,.35);display:none';
    const dot = document.createElement('div'); dot.id = 'demo-dot';
    dot.style.cssText = 'position:fixed;width:22px;height:22px;border-radius:50%;background:rgba(232,90,40,.85);border:3px solid #fff;z-index:100000;pointer-events:none;transform:translate(-50%,-50%);left:-50px;top:-50px;transition:transform .1s';
    document.body.append(cap, dot);
    document.addEventListener('mousemove', e => { dot.style.left = e.clientX + 'px'; dot.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', () => dot.style.transform = 'translate(-50%,-50%) scale(.7)', true);
    document.addEventListener('mouseup', () => dot.style.transform = 'translate(-50%,-50%)', true);
  };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    ctx = b.new_context(viewport={"width": 1600, "height": 1000},
                        record_video_dir=str(RAW), record_video_size={"width": 1600, "height": 1000})
    t0 = time.monotonic()
    pg = ctx.new_page()
    pg.add_init_script(OVERLAY)

    def cap(text, hold):
        pg.evaluate("t => { const c = document.getElementById('demo-cap'); c.textContent = t; c.style.display = t ? 'block' : 'none'; }", text)
        pg.wait_for_timeout(int(hold * 1000))

    def point(loc, click=True):
        loc.scroll_into_view_if_needed()
        bb = loc.bounding_box()
        pg.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2, steps=30)
        pg.wait_for_timeout(300)
        if click:
            pg.mouse.down(); pg.mouse.up()
        pg.wait_for_timeout(500)

    pg.goto("http://127.0.0.1:8787/#" + START_ID, wait_until="networkidle")
    pg.mouse.move(800, 500)
    cap("Dalil answers questions about UAE government services. Every sentence is checked against official sources, and a person approves it before it goes out.", 8)
    cap("Open Agent Hackathon 2026, Track 02: The Agent That Can Explain Why.", 4)

    point(pg.locator("#ask-input"))
    cap("A resident asks: can a Golden Visa holder sponsor their parents?", 1)
    pg.keyboard.type(QUESTION, delay=35)
    point(pg.locator("#ask-submit"))
    cap("Dalil screens the question, recalls past reviews from Meterless H-MEM, searches 1,024 passages from 45 official documents with BM25 and NVIDIA Nemotron embeddings, and drafts with NVIDIA Nemotron 3 Nano.", 9)
    cap("On this server's CPU an answer takes about 40 seconds. The wait is cut from this video.", 4)
    cut_from = time.monotonic() - t0
    pg.wait_for_function("document.getElementById('ask-submit').textContent.trim() === 'Ask Dalil'", timeout=240000)
    pg.wait_for_timeout(1500)
    if not pg.locator("#cut-section .cut-item").count() or not pg.locator("#draft .chip").count():
        ctx.close(); b.close()
        sys.exit("no cut sentence this run; the model output varies, so record again")
    cut_to = time.monotonic() - t0 - 3

    cap("The draft is back. Every sentence points at a numbered source.", 8)
    point(pg.locator("#draft .chip").first)
    cap("Click a citation and the quoted words are highlighted on the official page. The checker found them there word for word.", 10)

    cut = pg.locator("#cut-section .cut-item", has_text="parents")
    if cut.count():
        point(cut.first, click=False)
        cap("The model also wrote that Golden Visa holders can sponsor their parents. Its source names spouses and children, so the checker cut the sentence. It will never be sent.", 12)
    else:
        point(pg.locator("#cut-section .cut-item").first, click=False)
        cap("The checker cut a sentence the model wrote, because its source does not back it. It will never be sent.", 10)

    point(pg.locator("#tab-audit"))
    cap("The audit trail records every step: screening, the H-MEM recall, the search, the Nemotron draft and the check. Anyone can replay why an answer went out.", 11)

    point(pg.locator("button.queue-item", has_text="renew my Emirates ID").first)
    pg.wait_for_timeout(1200)
    if not pg.locator("#tab-audit").get_attribute("aria-selected") == "true":
        point(pg.locator("#tab-audit"))
    data_step = pg.locator(".audit-item.audit-data_looked_up")
    if data_step.count():
        point(data_step.first, click=False)
    cap("For numbers, Dalil queries open data through Zetaris. The documents, pages, passages and four Bayanat open datasets sit in one Zetaris Lightning database.", 10)
    cap("When an answer states a number from open data, it cites the Zetaris query, and the checker reruns that query before the sentence may stay.", 9)

    point(pg.locator("button.queue-item", has_text="Golden Visa holder sponsor").first)
    pg.wait_for_timeout(1000)
    point(pg.locator("#tab-answer"))
    point(pg.locator("#reviewer"))
    pg.keyboard.type("Demo reviewer", delay=40)
    cap("Nothing goes out until a person approves it. Only the sentences that passed the check are sent.", 6)
    point(pg.locator("#btn-approve"))
    pg.wait_for_selector("#review-done", state="visible", timeout=15000)
    pg.wait_for_timeout(1000)
    cap("The decision and the passages it cited go to Meterless H-MEM's trust ledger. When a reviewer rejects an answer, those passages stay out of the next answer to a similar question.", 11)
    cap("Dalil. Answers about UAE government services that you can check. github.com/sami-abdul/dalil", 7)
    end = time.monotonic() - t0
    video = pg.video.path()
    ctx.close(); b.close()

raw = pathlib.Path(video)
out = OUT / "dalil-demo.mp4"
filt = (f"[0:v]trim=0:{cut_from:.2f},setpts=PTS-STARTPTS[a];"
        f"[0:v]trim={cut_to:.2f},setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0,fps=30,format=yuv420p[v]")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-filter_complex", filt, "-map", "[v]",
                "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-movflags", "+faststart", str(out)], check=True)
dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                     capture_output=True, text=True).stdout.strip()
print(f"ok {out} duration={dur}s cut={cut_from:.1f}-{cut_to:.1f} raw_end={end:.1f}")
