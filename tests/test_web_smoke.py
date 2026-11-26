"""Smoke test for the reviewer screen in web/, against stubbed server responses.

It loads the real index.html, app.js and styles.css in a headless browser,
answers every /api call with made-up data, and checks that the key regions
render: the header, the stage strip with its ITU-T Y.3172 nodes, the answer
with its citations, a cut sentence with its reason, the evidence for a
citation, the audit trail and its raw record, and an Arabic question right to
left. No server, model or corpus is needed.

It needs Playwright for Python and a Chromium it can launch, so it is skipped
when they are missing. To run it:

    uv run --extra web-test playwright install --only-shell chromium   # once
    uv run --extra web-test pytest tests/test_web_smoke.py
"""

from __future__ import annotations

import copy
import json
import mimetypes
from pathlib import Path
from typing import Any

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

WEB = Path(__file__).resolve().parent.parent / "web"
BASE = "http://dalil.test"

PAGE_TEXT = (
    "Sponsoring a family\n"
    "A resident may sponsor a spouse and children.\n"
    "The sponsor must earn at least AED 4,000 a month.\n"
    "Parents may be sponsored under separate rules."
)
QUOTE = "The sponsor must earn at least AED 4,000 a month."
Q_START = PAGE_TEXT.index(QUOTE)

AUDIT = [
    {"at": "2026-11-25T10:00:00.010Z", "kind": "screened", "detail": {"flags": []}},
    {"at": "2026-11-25T10:00:00.020Z", "kind": "recalled",
     "detail": {"available": True, "similar_reviews": [], "passages_left_out": [], "notes": 0}},
    {"at": "2026-11-25T10:00:00.700Z", "kind": "retrieved",
     "detail": {"k": 8, "hits": 2, "top": [{"passage_id": "doc-a:p1:1", "score": 12.5}]}},
    {"at": "2026-11-25T10:00:09.700Z", "kind": "drafted",
     "detail": {"drafter": "LLMDrafter", "sentences": 2, "kinds": ["claim", "claim"]}},
    {"at": "2026-11-25T10:00:09.710Z", "kind": "verified", "detail": {"status": "pass_with_cuts", "kept": 1, "cut": 1}},
    {"at": "2026-11-25T10:00:09.720Z", "kind": "queued", "detail": {"status": "awaiting_review"}},
]

RECORD: dict[str, Any] = {
    "id": "q1", "question": "Can I sponsor my parents?", "flags": [], "status": "awaiting_review",
    "created_at": "2026-11-25T10:00:00.000Z",
    "retrieval": [
        {"passage_id": "doc-a:p1:1", "doc_id": "doc-a", "title": "Sponsoring a family", "page": 1, "score": 12.5,
         "heading": "Sponsoring a family", "text": PAGE_TEXT, "start": 0, "end": len(PAGE_TEXT)},
        {"passage_id": "doc-b:p2:1", "doc_id": "doc-b", "title": "Residence rules", "page": 2, "score": 9.1,
         "heading": None, "text": "Residence rules apply to every sponsor.", "start": 0, "end": 39},
    ],
    "data": [],
    "check": {
        "status": "pass_with_cuts", "answer": QUOTE,
        "sentences": [
            {"text": QUOTE, "kind": "claim", "kept": True, "reasons": [],
             "citations": [{"ok": True, "reason": "", "doc_id": "doc-a", "page": 1, "passage_id": "doc-a:p1:1",
                            "page_start": Q_START, "page_end": Q_START + len(QUOTE), "quote": QUOTE,
                            "title": "Sponsoring a family", "source_url": "https://example.ae/family"}]},
            {"text": "Parents can always be sponsored.", "kind": "claim", "kept": False,
             "reasons": ["claim is not supported by its sources (key words found: 20%)"],
             "citations": [{"ok": True, "reason": "", "doc_id": "doc-a", "page": 1, "passage_id": "doc-a:p1:1",
                            "page_start": 0, "page_end": 19, "quote": "Sponsoring a family",
                            "title": "Sponsoring a family", "source_url": "https://example.ae/family"}]},
        ],
    },
    "review": None,
    "audit": AUDIT,
}

ARABIC_Q = "هل يمكنني كفالة والديّ؟"
ARABIC_A = "يجوز للمقيم كفالة الزوج والأبناء."
AR_RECORD = copy.deepcopy(RECORD)
AR_RECORD.update({"id": "q2", "question": ARABIC_Q, "created_at": "2026-11-25T09:00:00.000Z"})
AR_RECORD["check"] = {"status": "pass", "answer": ARABIC_A, "sentences": [
    {"text": ARABIC_A, "kind": "claim", "kept": True, "reasons": [],
     "citations": [{"ok": True, "reason": "", "doc_id": "doc-a", "page": 1, "passage_id": "doc-a:p1:1",
                    "page_start": 0, "page_end": 19, "quote": "Sponsoring a family", "title": "Sponsoring a family"}]}]}

# A question still being answered: screened and searched, the writer running.
LIVE_RECORD = copy.deepcopy(RECORD)
LIVE_RECORD.update({"id": "q3", "question": "How do I renew my card?", "status": "screening", "check": None,
                    "retrieval": [], "audit": AUDIT[:3], "created_at": "2026-11-25T08:00:00.000Z"})

# The shapes dalil/datasource.py status() returns, with and without a source.
HEALTH = {"ok": True, "zetaris": False, "passages": 1024, "memory": True, "memories": 0,
          "knowledge_base": {"id": "uae", "name": "UAE government services", "documents": 53, "passages": 1024},
          "data_source": {"connected": False, "kind": None, "label": "Open data: not configured", "tables": []}}
HEALTH_LOCAL = {**HEALTH, "data_source": {"connected": True, "kind": "local", "label": "Open data: local database",
                                          "tables": ["dalil.open_a", "dalil.open_b"]}}

# An answer citing an open-data lookup, with a law's scope article added ahead of the ranked results.
SQL = "SELECT year, transactions FROM dalil.open_a"
DATA_RECORD = copy.deepcopy(RECORD)
DATA_RECORD.update({"id": "q4", "question": "How many transactions were there in 2025?",
                    "created_at": "2026-11-25T07:00:00.000Z",
                    "data": [{"table": "dalil.open_a", "sql": SQL, "rows": [{"year": "2025", "transactions": "4242"}],
                              "note": "", "source": "local", "source_label": "queried from the local database"}]})
DATA_RECORD["retrieval"][1]["score"] = 0.0
DATA_RECORD["check"] = {"status": "pass", "answer": "There were 4242 transactions in 2025.", "sentences": [
    {"text": "There were 4242 transactions in 2025.", "kind": "claim", "kept": True, "reasons": [],
     "citations": [{"ok": True, "reason": "value found", "table": "dalil.open_a", "sql": SQL, "value": "4242",
                    "title": "Example transactions", "source_url": None,
                    "source_label": "queried from the local database"}]}]}


def summary(r: dict[str, Any]) -> dict[str, Any]:
    s = (r["check"] or {}).get("sentences", [])
    return {"id": r["id"], "question": r["question"], "status": r["status"], "created_at": r["created_at"],
            "verifier_status": (r["check"] or {}).get("status"),
            "kept": sum(1 for x in s if x["kept"]), "cut": sum(1 for x in s if not x["kept"])}


def handle(route: Any, health: dict[str, Any]) -> None:
    path = route.request.url[len(BASE):].split("?")[0].split("#")[0]
    records = {r["id"]: r for r in (RECORD, AR_RECORD, LIVE_RECORD, DATA_RECORD)}
    body: Any = None
    if path == "/api/health":
        body = health
    elif path == "/api/questions":
        body = [summary(r) for r in (RECORD, AR_RECORD, LIVE_RECORD, DATA_RECORD)]
    elif path.startswith("/api/questions/"):
        body = records.get(path.rsplit("/", 1)[1])
    elif path.startswith("/api/documents/"):
        body = {"doc_id": "doc-a", "title": "Sponsoring a family", "publisher": "Example Authority",
                "source_url": "https://example.ae/family", "page": 1, "text": PAGE_TEXT}
    if path.startswith("/api/"):
        if body is None:
            route.fulfill(status=404, content_type="application/json", body='{"detail": "not found"}')
        else:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(body, ensure_ascii=False))
        return
    file = (WEB / (path.lstrip("/") or "index.html")).resolve()
    if WEB not in file.parents or not file.is_file():
        route.fulfill(status=404, body="")
        return
    route.fulfill(status=200, content_type=mimetypes.guess_type(file.name)[0] or "text/plain", body=file.read_bytes())


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as pw:
        try:
            b = pw.chromium.launch()
        except Exception as e:  # no browser installed for this Playwright version
            pytest.skip(f"no Chromium for Playwright: {type(e).__name__}")
        yield b
        b.close()


def open_page(browser: Any, fragment: str, health: dict[str, Any] = HEALTH, size=(1920, 1080)) -> Any:
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    outside: list[str] = []
    page.on("request", lambda r: None if r.url.startswith(BASE) or r.url.startswith("data:") else outside.append(r.url))
    page.route("**/*", lambda route: handle(route, health))
    page.goto(f"{BASE}/#{fragment}")
    page.wait_for_selector("#answer-body:not([hidden])")
    page.wait_for_timeout(300)
    page.errors, page.outside = errors, outside
    return page


def test_main_regions_render(browser):
    page = open_page(browser, "q1")
    health = page.locator("#health").inner_text()
    assert "UAE government services" in health and "53 documents" in health and "1,024 passages" in health
    assert "not configured" in health
    # A source that is simply not configured is not shown as an error.
    assert page.locator("#health .dot-off").count() == 1 and page.locator("#health .health-warn").count() == 0
    assert page.locator("#q-text").inner_text() == "Can I sponsor my parents?"
    assert page.locator("#queue-list .queue-item").count() == 4

    stages = page.locator("#stages .stage")
    assert stages.count() == 6
    names = [stages.nth(i).locator(".stage-name").inner_text() for i in range(6)]
    assert names == ["Screener", "Searcher", "Writer", "Checker", "Human reviewer", "Sent"]
    nodes = [stages.nth(i).locator(".stage-node abbr").inner_text() for i in range(6)]
    assert nodes == ["PP", "M", "M", "P", "P", "D"]
    assert "stage-done" in (stages.nth(2).get_attribute("class") or "")
    assert "9.0 s" in stages.nth(2).inner_text()  # drafted 9.0 s after the search finished
    assert "1 kept, 1 cut" in stages.nth(3).inner_text()
    assert "stage-current" in (stages.nth(4).get_attribute("class") or "")

    assert page.locator("#draft .chip").count() == 1
    cut = page.locator("#cut-section")
    assert cut.is_visible() and "key words found: 20%" in cut.inner_text()
    # Internal codes stay behind the details control.
    assert not page.locator("#q-id").is_visible()
    assert page.locator(".score").count() == 0
    assert page.errors == [] and page.outside == []
    page.close()


def test_citation_opens_the_exact_words(browser):
    page = open_page(browser, "q1")
    page.locator("#draft .chip").first.click()
    mark = page.locator("#source-view .quote-mark")
    mark.wait_for()
    assert mark.inner_text() == QUOTE
    assert "exact words" in page.locator("#source-view .cite-status").inner_text()
    page.close()


def test_audit_trail_is_readable_with_raw_record(browser):
    page = open_page(browser, "q1")
    page.click("#tab-audit")
    rows = page.locator("#audit-list .audit-item")
    assert rows.count() == len(AUDIT)
    parts = [rows.nth(i).locator(".audit-part").inner_text().split("\n")[0].strip() for i in range(rows.count())]
    assert parts[0].startswith("Screener") and parts[3].startswith("Writer") and parts[4].startswith("Checker")
    assert "Parents can always be sponsored." in rows.nth(4).inner_text()
    assert not page.locator("#audit-raw").is_visible()
    page.click(".raw > summary")
    raw = json.loads(page.locator("#audit-raw").inner_text())
    assert raw["id"] == "q1" and len(raw["audit"]) == len(AUDIT)
    page.close()


def test_arabic_is_right_to_left(browser):
    page = open_page(browser, "q2")
    q = page.locator("#q-text")
    assert q.get_attribute("lang") == "ar"
    assert q.evaluate("e => getComputedStyle(e).direction") == "rtl"
    para = page.locator("#draft .answer-para")
    assert para.get_attribute("dir") == "rtl"
    # The citation number sits after the sentence, which in Arabic is to its left.
    sentence = page.locator("#draft .sentence").bounding_box()
    chip = page.locator("#draft .chip").bounding_box()
    assert chip["x"] < sentence["x"] + sentence["width"] / 2
    page.close()


def test_live_question_shows_the_running_stage(browser):
    page = open_page(browser, "q3")
    stages = page.locator("#stages .stage")
    assert "stage-done" in (stages.nth(1).get_attribute("class") or "")
    assert "stage-running" in (stages.nth(2).get_attribute("class") or "")
    assert "stage-waiting" in (stages.nth(3).get_attribute("class") or "")
    assert page.locator("#working").is_visible()
    assert page.locator("#btn-approve").is_disabled()
    page.close()


def test_falls_back_to_todays_health_fields(browser):
    old = {"ok": True, "zetaris": False, "passages": 1024, "memory": True, "memories": 0}
    page = open_page(browser, "q1", health=old)
    health = page.locator("#health").inner_text()
    assert "1,024 passages" in health and "not connected" in health
    assert page.locator("#health .health-warn").count() == 0
    page.close()


def test_open_data_source_and_citation(browser):
    page = open_page(browser, "q4", health=HEALTH_LOCAL)
    health = page.locator("#health").inner_text()
    assert "local database, 2 tables" in health and page.locator("#health .dot-on").count() == 1
    page.locator("#draft .chip").first.click()
    view = page.locator("#source-view")
    view.locator(".quote-mark").first.wait_for()
    assert "Example transactions" in view.inner_text()
    assert "queried from the local database" in view.inner_text()
    assert page.locator("#source-view td.is-cited").inner_text().startswith("4242")
    # The added scope article is labelled as added, not ranked as a match.
    assert "scope article added" in page.locator("#searched").inner_text()
    assert page.locator("#searched .hit-added").count() == 1
    page.close()


def test_holds_at_1440_by_900(browser):
    page = open_page(browser, "q1", size=(1440, 900))
    width = page.evaluate("document.documentElement.scrollWidth")
    assert width <= 1440
    for sel in ("#q-text", "#stages", "#draft", "#btn-approve", "#searched"):
        box = page.locator(sel).bounding_box()
        assert box and box["x"] >= 0 and box["x"] + box["width"] <= 1440, sel
    page.close()
