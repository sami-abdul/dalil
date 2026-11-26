"""The knowledge-base menu on the reviewer screen, against stubbed responses.

Reuses the stubbed server in tests/test_web_smoke.py and adds the two
knowledge-base endpoints: picking the UK knowledge base posts the switch,
then the header, the ask box and the queue show the UK one. Skipped, like the
smoke test, when Playwright or its Chromium is missing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("playwright.sync_api")

from test_web_smoke import BASE, HEALTH, handle  # noqa: E402
from test_web_smoke import browser  # noqa: E402,F401  (the module's browser fixture)

UAE = {"id": "uae", "name": "UAE government services", "country": "United Arab Emirates", "documents": 53,
       "passages": 1024, "ask_label": "Ask about a UAE government service",
       "example_question": "For example: How is end of service gratuity worked out?"}
UK = {"id": "uk", "name": "UK government services (GOV.UK)", "country": "United Kingdom", "documents": 7,
      "passages": 92, "ask_label": "Ask about a UK government service",
      "example_question": "For example: How much notice do I have to give my employer?"}
NO_DATA = {"connected": False, "kind": None, "label": "Open data: none for this knowledge base", "tables": []}


def open_switchable(browser: Any) -> Any:
    state = {"active": "uae", "posts": []}

    def route(r: Any) -> None:
        path = r.request.url[len(BASE):].split("?")[0].split("#")[0]
        active = UAE if state["active"] == "uae" else UK

        def reply(body: Any, status: int = 200) -> None:
            r.fulfill(status=status, content_type="application/json", body=json.dumps(body, ensure_ascii=False))
        if path == "/api/knowledge-bases":
            reply([{**k, "active": k["id"] == state["active"], "loaded": True} for k in (UAE, UK)])
        elif path == "/api/knowledge-base" and r.request.method == "POST":
            state["posts"].append(json.loads(r.request.post_data or "{}"))
            state["active"] = state["posts"][-1]["id"]
            reply({"knowledge_base": UK if state["active"] == "uk" else UAE, "seconds": 0.0})
        elif path == "/api/questions" and state["active"] == "uk":
            reply([])
        elif path == "/api/health":
            health = {**HEALTH, "knowledge_base": active, "passages": active["passages"]}
            if state["active"] == "uk":
                health["data_source"] = NO_DATA
            reply(health)
        else:
            handle(r, HEALTH)

    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/*", route)
    page.goto(f"{BASE}/#q1")
    page.wait_for_selector("#answer-body:not([hidden])")
    page.wait_for_function("() => !document.querySelector('#kb-select').disabled")
    page.errors, page.state = errors, state
    return page


def test_menu_lists_both_and_marks_the_active_one(browser):
    page = open_switchable(browser)
    options = page.locator("#kb-select option")
    assert [options.nth(i).inner_text() for i in range(options.count())] == [UAE["name"], UK["name"]]
    assert page.locator("#kb-select").input_value() == "uae"
    assert page.locator("#ask-label").inner_text() == UAE["ask_label"]
    assert page.errors == []
    page.close()


def test_switching_shows_the_uk_knowledge_base(browser):
    page = open_switchable(browser)
    page.select_option("#kb-select", "uk")
    page.wait_for_function("() => document.querySelector('#health').innerText.includes('92 passages')")
    assert page.state["posts"] == [{"id": "uk"}]
    health = page.locator("#health").inner_text()
    assert UK["name"] in health and "7 documents" in health and "none for this knowledge base" in health
    assert page.locator("#ask-label").inner_text() == UK["ask_label"]
    assert page.locator("#ask-input").get_attribute("placeholder") == UK["example_question"]
    # The UAE question that was open is cleared, and the UK queue is its own.
    assert page.locator("#answer-body").is_hidden()
    assert page.locator("#queue-list .queue-item").count() == 0
    assert page.locator("#kb-select").input_value() == "uk"
    assert page.errors == []
    page.close()
