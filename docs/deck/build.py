#!/usr/bin/env python3
"""Build the Dalil deck: fill deck.html from values.json, check assets, print to PDF.

Usage: python3 -I build.py [--out DIR]   (writes DIR/dalil-deck.html and DIR/dalil-deck.pdf)
"""
import argparse, json, pathlib, re, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "out"))
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

html = (HERE / "deck.html").read_text()
values = json.loads((HERE / "values.json").read_text())
for k, v in values.items():
    html = html.replace("{{" + k + "}}", v)
left = sorted(set(re.findall(r"\{\{(\w+)\}\}", html)))
if left:
    sys.exit(f"unfilled placeholders: {left}")
missing = [s for s in re.findall(r'src="([^"]+)"', html) if not (HERE / s).exists()]
if missing:
    sys.exit(f"missing assets: {missing}")
if "—" in html:
    sys.exit("em dash in deck text")

if (out / "assets").exists():
    shutil.rmtree(out / "assets")
shutil.copytree(HERE / "assets", out / "assets")
(out / "dalil-deck.html").write_text(html)
pdf = out / "dalil-deck.pdf"
subprocess.run(["/usr/bin/google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
                "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                (out / "dalil-deck.html").as_uri()], check=True, capture_output=True, timeout=120)
pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
print(pdf, [l for l in pages.splitlines() if l.startswith(("Pages", "Page size"))])
