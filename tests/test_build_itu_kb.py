"""Tests for the ITU knowledge base builder's pure helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("build_itu_kb", ROOT / "scripts" / "build_itu_kb.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_categorise_maps_manifest_topics_to_itu_types():
    m = load()
    assert m.categorise("AI policy: National Strategy for AI 2031 full text") == ("national_strategy", "AI_Strategies")
    assert m.categorise("data protection: PDPL full text (English)") == ("policy_document", "AI_Strategies")
    assert m.categorise("visa: Golden visa eligibility") == ("government_service_page", "Government_Services")


def test_use_case_sections_split_on_second_level_headings():
    m = load()
    sections = m.use_case_sections("# Title\n\nintro\n\n## One\n\nbody one\n\n## Two\nbody two\n")
    assert [h for h, _ in sections] == ["One", "Two"]
    assert sections[0][1].startswith("One") and "body one" in sections[0][1]


def test_the_use_case_document_has_the_sections_the_build_expects():
    m = load()
    headings = [h for h, _ in m.use_case_sections(m.USE_CASE.read_text(encoding="utf-8"))]
    assert "Requirements, by ITU-T Y.3172 pipeline node" in headings and "Error scenarios" in headings


def test_tenancy_law_is_a_policy_document_not_a_service_page():
    m = load()
    topic = "housing: landlord and tenant law (Dubai); later amendments not included"
    assert m.categorise(topic) == ("policy_document", "AI_Strategies")


def test_out_inside_the_repo_is_refused(monkeypatch):
    m = load()
    monkeypatch.setattr("sys.argv", ["build_itu_kb.py", "--out", str(ROOT / "build" / "itu"), "--no-archive"])
    assert m.main() == 2


class FakeEmbedder:
    url, model = "http://127.0.0.1:1/v1", "fake"

    def __init__(self, cache_dir=None, **_):
        pass

    def available(self):
        return True

    def _vec(self, t):
        return [float(len(t) % 7 + 1), 1.0, 0.5]

    def embed_passages(self, texts, batch_size=16):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _corpus(tmp_path, passage_text="Late renewal costs AED 20 a day."):
    import json
    page = "Intro line.\nLate renewal costs AED 20 a day.\nEnd."
    start = page.index("Late")
    (tmp_path / "documents.jsonl").write_text(json.dumps(
        {"doc_id": "doc-a", "pages": [{"page": 1, "text": page}]}) + "\n", encoding="utf-8")
    (tmp_path / "passages.jsonl").write_text(json.dumps(
        {"passage_id": "doc-a:p1:1", "doc_id": "doc-a", "page": 1, "start": start, "end": start + 32,
         "heading": None, "kind": "html", "language": "en", "text": passage_text}) + "\n", encoding="utf-8")


def test_build_end_to_end_with_a_fake_embedder(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("chromadb")
    import dalil.embed
    m = load()
    _corpus(tmp_path)
    monkeypatch.setattr(dalil.embed, "HTTPEmbedder", FakeEmbedder)
    monkeypatch.setattr(m, "PASSAGES", tmp_path / "passages.jsonl")
    monkeypatch.setattr(m, "DOCUMENTS", tmp_path / "documents.jsonl")
    monkeypatch.setattr(m, "REVIEWS_DB", tmp_path / "none.db")
    out = tmp_path / "kb"
    monkeypatch.setattr("sys.argv", ["build_itu_kb.py", "--out", str(out), "--no-archive"])
    assert m.main() == 0
    assert (out / "chromadb").is_dir() and (out / "BUILD.json").is_file()
    built = (out / "BUILD.json").read_text()

    # A passage that no longer matches its page fails the build and replaces nothing.
    _corpus(tmp_path, passage_text="Late renewal costs AED 99 a day.")
    assert m.main() == 1
    assert (out / "BUILD.json").read_text() == built
