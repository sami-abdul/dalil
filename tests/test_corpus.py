"""Tests for dalil.corpus and the corpus scripts.

All fixtures are synthetic and built in tmp_path. No corpus text appears here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

from dalil import corpus
from dalil.corpus import (
    DOCUMENT_FIELDS,
    MANIFEST_FIELDS,
    CorpusError,
    doc_id,
    extract_document,
    extract_html_text,
    extract_pdf_pages,
    is_terms_page,
    load_manifest,
    normalise_text,
    resolve_corpus_dir,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# A page shaped like the u.ae template: chrome around a hero and tabbed content.
UAE_LIKE_HTML = """<!DOCTYPE html>
<html lang="en"><head><title>Widget permit | Portal</title>
<script>var tracking = "SCRIPT-TEXT";</script><style>.x{color:red}</style></head>
<body>
<div class="root position-relative">
  <header class="hero">
    <nav class="container"><a href="/">Home</a><a href="/menu">NAV-TEXT</a></nav>
    <section class="head-services">
      <div class="hero-text col-12">
        <ul class="breadcrumb"><li class="breadcrumb-item"><a>CRUMB-TEXT</a></li></ul>
        <h2>Widget permit</h2>
        <div class="text trimmed-text">A widget permit lets you   keep a widget.</div>
        <a class="read-more">READMORE-TEXT</a>
      </div>
    </section>
  </header>
  <main>
    <div class="container"><div class="row"><div class="col-md-9">
      <div class="sticky-wrapper"><div class="select-container">
        <select><option>SELECT-TEXT</option></select></div></div>
      <div class="tab-content">
        <div id="who" class="article-info">
          <h3>Who can apply</h3>
          <p>Residents with a <span>valid</span><span>ated</span> address.</p>
          <ul><li>first condition</li><li>second<br>condition</li></ul>
          <table><thead><tr><th>Kind</th><th>Fee</th></tr></thead>
            <tbody><tr><td>Small</td><td>10 dirhams</td></tr></tbody></table>
          <button>BUTTON-TEXT</button>
          <!-- COMMENT-TEXT -->
        </div>
      </div>
    </div></div></div>
    <div class="container pt-0"><div class="row"><div class="col-md-9 article-info">
      <div class="artical-info-link d-flex"><span class="artical-info-label">Updated on:</span><p>01 Jan 2026</p></div>
    </div></div></div>
    <div class="container"><form class="feedback"><p>FEEDBACK-TEXT</p></form></div>
    <div class="topic-navigation">TOPICNAV-TEXT</div>
  </main>
  <footer><p>FOOTER-TEXT</p></footer>
  <div id="cookie-alert" class="banner"><p>COOKIE-TEXT</p></div>
</div>
</body></html>
"""

CHROME_MARKERS = (
    "SCRIPT-TEXT", "NAV-TEXT", "CRUMB-TEXT", "READMORE-TEXT", "SELECT-TEXT", "BUTTON-TEXT",
    "COMMENT-TEXT", "FEEDBACK-TEXT", "TOPICNAV-TEXT", "FOOTER-TEXT", "COOKIE-TEXT", "color:red",
)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    return path


def manifest_row(file: str, data: bytes | None = None, **overrides: str) -> dict[str, str]:
    row = {
        "file": file,
        "title": f"Title of {file}",
        "publisher": "Test publisher",
        "source_url": f"https://example.invalid/{file}",
        "retrieved_utc": "2026-10-19T00:00:00Z",
        "sha256": hashlib.sha256(data).hexdigest() if data is not None else "0" * 64,
        "language": "en",
        "topic": "testing: synthetic",
        "licence_note": "synthetic fixture",
    }
    row.update(overrides)
    return row


def make_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Build a small text PDF; an empty list makes a blank page."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
    c.save()
    return path


# --------------------------------------------------------------------------
# Manifest, ids, paths


def test_doc_id_is_stable_slug_of_stem():
    assert doc_id("uae-golden-visa.html") == "uae-golden-visa"
    assert doc_id("Some File_v2 (final).PDF") == "some-file-v2-final"
    assert doc_id("a.b.pdf") == doc_id("a.b.pdf") == "a-b"


def test_is_terms_page_uses_licence_prefix():
    assert is_terms_page("licence: terms of use")
    assert is_terms_page("  Licence: open data")
    assert not is_terms_page("labour: worker rights")
    assert not is_terms_page("data protection: licence of processors")


def test_load_manifest_reads_rows(tmp_path):
    path = write_manifest(tmp_path / "m.csv", [manifest_row("a.html"), manifest_row("b.pdf")])
    rows = load_manifest(path)
    assert [r["file"] for r in rows] == ["a.html", "b.pdf"]
    assert set(rows[0]) == set(MANIFEST_FIELDS)


def test_load_manifest_rejects_missing_columns(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("file,title\na.html,A\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="lacks columns"):
        load_manifest(path)


@pytest.mark.parametrize("bad", ["../escape.html", "sub/dir.html", ".hidden.html", ""])
def test_load_manifest_rejects_unsafe_file_names(tmp_path, bad):
    path = write_manifest(tmp_path / "m.csv", [manifest_row(bad)])
    with pytest.raises(CorpusError, match="bad file name"):
        load_manifest(path)


def test_load_manifest_rejects_duplicate_ids(tmp_path):
    path = write_manifest(tmp_path / "m.csv", [manifest_row("a.html"), manifest_row("A.pdf")])
    with pytest.raises(CorpusError, match="duplicate doc_id"):
        load_manifest(path)


def test_real_manifest_shape():
    rows = load_manifest()
    assert len(rows) == 53
    assert sum(is_terms_page(r["topic"]) for r in rows) == 8
    assert all(len(r["sha256"]) == 64 for r in rows)
    assert len({doc_id(r["file"]) for r in rows}) == len(rows)


def test_resolve_corpus_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("DALIL_CORPUS_DIR", raising=False)
    assert resolve_corpus_dir() == Path("corpus")
    monkeypatch.setenv("DALIL_CORPUS_DIR", str(tmp_path))
    assert resolve_corpus_dir() == tmp_path
    assert resolve_corpus_dir(tmp_path / "x") == tmp_path / "x"


# --------------------------------------------------------------------------
# Normalisation


def test_normalise_text_whitespace_only():
    raw = "  Line one\u00a0\u00a0has\tspaces  \r\n\r\n\n  line two \x0c third\u2028fourth "
    assert normalise_text(raw) == "Line one has spaces\nline two\nthird\nfourth"


def test_normalise_text_drops_invisible_and_keeps_words():
    # Zero-width space, soft hyphen, bidi controls and tatweel go; letters stay.
    raw = "\u202bab\u200bc\u00add\u202c \u0628\u0640\u0640\u0627\u200f"
    assert normalise_text(raw) == "abcd \u0628\u0627"


def test_normalise_text_expands_ligatures_and_arabic_presentation_forms():
    assert normalise_text("arti\ufb01cial e\ufb00ect") == "artificial effect"
    # U+FEFB is the isolated lam-alef ligature.
    assert normalise_text("\ufefb") == "\u0644\u0627"


def test_repair_glyphs_only_for_listed_document():
    text = "\ufffde rule and di\ufffderent"
    fixed = corpus.repair_glyphs("ai-gov-uae-charter-development-use-of-ai-2024-en", text)
    assert fixed == "The rule and different"
    assert corpus.repair_glyphs("other-doc", text) == text


# --------------------------------------------------------------------------
# HTML


def test_html_keeps_article_and_drops_chrome():
    text = extract_html_text(UAE_LIKE_HTML.encode())
    for marker in CHROME_MARKERS:
        assert marker not in text, marker
    assert text.split("\n") == [
        "Widget permit",
        "A widget permit lets you keep a widget.",
        "Who can apply",
        "Residents with a validated address.",
        "first condition",
        "second",
        "condition",
        "Kind | Fee",
        "Small | 10 dirhams",
        "Updated on: 01 Jan 2026",
    ]


def test_html_generic_main_fallback_and_h1_title():
    html = b"""<html><body>
      <header><h1>Site title</h1><nav>NAV-TEXT</nav></header>
      <main><h2>Section</h2><p>Body text.</p><div role="navigation">ROLE-NAV</div></main>
      <footer>FOOTER-TEXT</footer></body></html>"""
    text = extract_html_text(html)
    assert text == "Site title\nSection\nBody text."


def test_html_body_fallback_without_main():
    html = b"<html><body><nav>NAV-TEXT</nav><div><p>Only text.</p></div><footer>F</footer></body></html>"
    assert extract_html_text(html) == "Only text."


def test_html_decodes_declared_charset():
    html = '<html><head><meta charset="windows-1252"></head><body><main><p>caf\xe9</p></main></body></html>'
    assert extract_html_text(html.encode("cp1252")) == "caf\xe9"


# --------------------------------------------------------------------------
# PDF


def test_pdf_one_entry_per_page_with_real_numbers(tmp_path):
    path = make_pdf(tmp_path / "doc.pdf", [["Article 1", "First page text."], [], ["Article 2", "Third   page."]])
    pages = extract_pdf_pages(path)
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert pages[0]["text"] == "Article 1\nFirst page text."
    assert pages[1]["text"] == ""
    assert pages[2]["text"].startswith("Article 2\nThird")


# --------------------------------------------------------------------------
# Documents


def test_extract_document_record(tmp_path):
    html = UAE_LIKE_HTML.encode()
    (tmp_path / "Widget-Permit.html").write_bytes(html)
    pdf = make_pdf(tmp_path / "law.pdf", [["One"], ["Two"]]).read_bytes()

    rec = extract_document(manifest_row("Widget-Permit.html", html, topic="licence: terms of use"), tmp_path)
    assert tuple(rec) == DOCUMENT_FIELDS
    assert rec["doc_id"] == "widget-permit"
    assert rec["kind"] == "html"
    assert rec["is_terms_page"] is True
    assert rec["sha256_ok"] is True
    assert [p["page"] for p in rec["pages"]] == [1]
    assert rec["char_count"] == len(rec["pages"][0]["text"]) > 0

    rec = extract_document(manifest_row("law.pdf", pdf), tmp_path)
    assert rec["kind"] == "pdf"
    assert rec["is_terms_page"] is False
    assert [p["text"] for p in rec["pages"]] == ["One", "Two"]
    assert rec["char_count"] == 6


def test_extract_document_flags_hash_mismatch(tmp_path):
    (tmp_path / "a.html").write_bytes(b"<main><p>Text</p></main>")
    rec = extract_document(manifest_row("a.html", b"different bytes"), tmp_path)
    assert rec["sha256_ok"] is False


def test_extract_document_missing_file_is_hard_error(tmp_path):
    with pytest.raises(CorpusError, match="missing corpus file"):
        extract_document(manifest_row("gone.pdf"), tmp_path)


def test_extract_document_unreadable_pdf_is_hard_error(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf at all")
    with pytest.raises(CorpusError, match="cannot extract"):
        extract_document(manifest_row("bad.pdf"), tmp_path)


# --------------------------------------------------------------------------
# Scripts


def test_extract_corpus_script(tmp_path, capsys):
    import extract_corpus

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    html = UAE_LIKE_HTML.encode()
    (corpus_dir / "page.html").write_bytes(html)
    make_pdf(corpus_dir / "law.pdf", [["One"], ["Two"]])
    manifest = write_manifest(tmp_path / "m.csv", [manifest_row("page.html", html), manifest_row("law.pdf")])
    out = tmp_path / "build" / "documents.jsonl"

    status = extract_corpus.main(["--manifest", str(manifest), "--corpus-dir", str(corpus_dir), "--out", str(out)])
    assert status == 0  # a hash mismatch (law.pdf) is not a failure
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["doc_id"] for r in records] == ["page", "law"]
    assert [r["sha256_ok"] for r in records] == [True, False]
    printed = capsys.readouterr().out
    assert "hash mismatches: 1: law" in printed
    assert "documents: 2 of 2" in printed


def test_extract_corpus_script_fails_on_missing_file(tmp_path):
    import extract_corpus

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    manifest = write_manifest(tmp_path / "m.csv", [manifest_row("gone.html")])
    out = tmp_path / "out.jsonl"
    assert extract_corpus.main(["--manifest", str(manifest), "--corpus-dir", str(corpus_dir), "--out", str(out)]) == 1
    assert out.read_text(encoding="utf-8") == ""


def test_fetch_corpus_skips_matching_and_reports_mismatch(tmp_path, monkeypatch, capsys):
    import fetch_corpus

    good, changed = b"<p>same</p>", b"<p>new</p>"
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "present.html").write_bytes(good)
    rows = [
        manifest_row("present.html", good),
        manifest_row("changed.html", b"<p>old</p>"),
        manifest_row("broken.html", good),
    ]
    manifest = write_manifest(tmp_path / "m.csv", rows)
    calls = []

    def fake_download(url: str, dest: Path) -> str:
        calls.append(url)
        if "broken" in url:
            raise OSError("connection refused")
        dest.write_bytes(changed)
        return hashlib.sha256(changed).hexdigest()

    monkeypatch.setattr(fetch_corpus, "download", fake_download)
    status = fetch_corpus.main(["--manifest", str(manifest), "--corpus-dir", str(corpus_dir), "--delay", "0"])
    assert status == 0
    assert calls == ["https://example.invalid/changed.html", "https://example.invalid/broken.html"]
    assert (corpus_dir / "changed.html").read_bytes() == changed
    printed = capsys.readouterr().out
    assert "present: 1, fetched: 1, hash mismatches: 1, failed: 1" in printed


def test_fetch_corpus_rejects_non_http_urls(tmp_path):
    import fetch_corpus

    with pytest.raises(ValueError, match="not an http"):
        fetch_corpus.download("file:///etc/passwd", tmp_path / "x")
