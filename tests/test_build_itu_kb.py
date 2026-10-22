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
