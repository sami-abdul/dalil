"""Tests for dalil.zetaris and the loader's row building. Only the last test
needs a running Zetaris; it is skipped unless DALIL_ZETARIS_LIVE=1."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from dalil import zetaris as z

ROOT = Path(__file__).resolve().parent.parent
PW_KEY = "ZETARIS_ADMIN_" + "PASSWORD"  # test values below are fakes
FAKE = "not-a-real-value"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- config --------------------------------------------------------------------


def test_load_config_from_file(tmp_path):
    env = tmp_path / "z.env"
    env.write_text(
        "# comment\n"
        "ZETARIS_UI_URL=http://127.0.0.1:13000/\n"
        "export ZETARIS_ADMIN_EMAIL='admin@example.org'\n"
        f'{PW_KEY}="{FAKE} x=y"\n'
    )
    cfg = z.load_config(env, environ={})
    assert cfg.ui_url == "http://127.0.0.1:13000"
    assert cfg.email == "admin@example.org"
    assert cfg.password == f"{FAKE} x=y"
    assert cfg.source == str(env)
    assert FAKE not in repr(cfg)  # the password stays out of repr and logs


def test_environment_overrides_file(tmp_path):
    env = tmp_path / "z.env"
    env.write_text(f"ZETARIS_ADMIN_EMAIL=a@x.org\n{PW_KEY}=one\n")
    cfg = z.load_config(env, environ={PW_KEY: "two"})
    assert cfg.password == "two"


def test_env_file_from_variable(tmp_path):
    env = tmp_path / "z.env"
    env.write_text("ZETARIS_API_KEY=k\nZETARIS_ORG_ID=1\n")
    cfg = z.load_config(environ={"DALIL_ZETARIS_ENV": str(env)})
    assert cfg.api_key == "k" and cfg.org_id == "1"


def test_missing_explicit_file_is_an_error(tmp_path):
    with pytest.raises(z.ZetarisError, match="not found"):
        z.load_config(tmp_path / "nope.env", environ={})


def test_missing_credentials_error_names_variables(tmp_path):
    env = tmp_path / "z.env"
    env.write_text("ZETARIS_UI_URL=http://127.0.0.1:13000\n")
    with pytest.raises(z.ZetarisError, match="ZETARIS_ADMIN_EMAIL"):
        z.load_config(env, environ={})


@pytest.mark.parametrize("url", ["127.0.0.1:13000", "http://host/api", "ftp://host"])
def test_bad_ui_url(url):
    with pytest.raises(z.ZetarisError, match="origin"):
        z.Config(ui_url=url, email="a", password=FAKE)


# --- SQL text --------------------------------------------------------------------


def test_quote_string_escapes_quote_and_backslash():
    assert z.quote_string("plain") == "'plain'"
    assert z.quote_string("it's") == "'it\\'s'"
    assert z.quote_string("a\\b") == "'a\\\\b'"


def test_quote_ident_doubles_backticks():
    assert z.quote_ident("energy-kcal") == "`energy-kcal`"
    assert z.quote_ident("a`b") == "`a``b`"


@pytest.mark.parametrize("bad", ["", "1abc", "a b", "x;DROP", "a.b", "a`"])
def test_check_name_rejects(bad):
    with pytest.raises(z.ZetarisError):
        z.check_name(bad)


def test_strip_statement():
    assert z.strip_statement("  SELECT 1 ;; \n") == "SELECT 1"
    with pytest.raises(z.ZetarisError):
        z.strip_statement(" ; ")


def test_filestore_table_sql():
    sql = z.filestore_table_sql("dalil", "pages", "file:///x/pages.json", "json", {"multiLine": "true"})
    assert sql == (
        "CREATE LIGHTNING FILESTORE TABLE pages FROM dalil FORMAT JSON "
        "OPTIONS (path 'file:///x/pages.json', multiLine 'true')"
    )
    with pytest.raises(z.ZetarisError):
        z.filestore_table_sql("dalil", "pages", "/x", "EXE")


def test_create_database_sql_rejects_bad_description():
    assert z.create_database_sql("dalil", "UAE docs, v1").startswith("CREATE LIGHTNING DATABASE dalil")
    with pytest.raises(z.ZetarisError):
        z.create_database_sql("dalil", "docs (draft)")


def test_citation_sql_is_rerunnable_and_quoted():
    cite = z.citation("dalil", "o'brien-law", 3, sha256="ab", source_url="https://x")
    assert cite["doc_id"] == "o'brien-law" and cite["page"] == 3
    assert "WHERE p.doc_id = 'o\\'brien-law' AND p.page = 3" in cite["sql"]
    assert cite["tables"] == ["dalil.documents", "dalil.pages"]


def test_json_array_bytes_round_trip():
    rows = [{"doc_id": "a", "text": "سطر\n\"two\""}, {"doc_id": "b", "text": ""}]
    data = z.json_array_bytes(rows)
    assert json.loads(data) == rows
    assert data.startswith(b"[")  # an array, since Lightning reads JSON as multi-line


def test_error_detail_reads_msg():
    assert z._error_detail(b'{"msg": "table not\\nfound"}') == "table not found"
    assert "HTML" in z._error_detail(b"<html><body>nope</body></html>")


def test_result_helpers():
    res = z.Result(headers=["n"], rows=[["7"]])
    assert res.scalar() == "7" and res.dicts() == [{"n": "7"}]
    with pytest.raises(z.ZetarisError):
        z.Result(headers=[], rows=[]).scalar()


# --- loader row building -----------------------------------------------------------


def test_build_rows_from_documents_jsonl(tmp_path):
    loader = _load_script("load_zetaris")
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps({
        "doc_id": "d1", "file": "d1.pdf", "title": "T", "language": "ar", "kind": "pdf",
        "pages": [{"page": 1, "text": "abc"}, {"page": 2, "text": "de"}],
    }) + "\n", encoding="utf-8")
    documents, pages = loader.build_rows(loader.read_documents(path), "documents.jsonl", "2026-10-22T00:00:00Z")
    assert documents[0]["page_count"] == 2 and documents[0]["char_count"] == 5
    assert documents[0]["publisher"] is None
    assert [p["page"] for p in pages] == [1, 2]
    assert set(pages[0]) == {"doc_id", "page", "text", "char_count"}


def test_build_rows_passes_extra_scalars_and_aligns_columns():
    loader = _load_script("load_zetaris")
    documents, _ = loader.build_rows(
        [{"doc_id": "a", "sha256_ok": True, "nested": {"x": 1}}, {"doc_id": "b"}], "x", "t")
    assert documents[0]["sha256_ok"] is True and documents[1]["sha256_ok"] is None
    assert "nested" not in documents[0]
    assert list(documents[0]) == list(documents[1])


def test_build_rows_rejects_duplicates():
    loader = _load_script("load_zetaris")
    with pytest.raises(z.ZetarisError, match="Duplicate"):
        loader.build_rows([{"doc_id": "a"}, {"doc_id": "a"}], "x", "t")
    with pytest.raises(z.ZetarisError, match="twice"):
        loader.build_rows([{"doc_id": "a", "pages": [{"page": 1}, {"page": 1}]}], "x", "t")


def test_standin_from_manifest(tmp_path):
    loader = _load_script("load_zetaris")
    manifest = tmp_path / "MANIFEST.csv"
    manifest.write_text(
        "file,title,publisher,source_url,retrieved_utc,sha256,language,topic,licence_note\n"
        "u-ae-visa.html,Visa,u.ae,https://u.ae/v,2026-10-19T00:00:00Z,ab,en,visas,reserved\n",
        encoding="utf-8",
    )
    docs = loader.standin_documents(manifest)
    assert docs[0]["doc_id"] == "u-ae-visa" and docs[0]["kind"] == "html"
    assert "licence_note" not in docs[0]
    assert docs[0]["pages"][0]["text"].startswith("[stand-in page")


# --- live ---------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("DALIL_ZETARIS_LIVE") != "1",
                    reason="set DALIL_ZETARIS_LIVE=1 to run against the local Zetaris")
def test_live_select_one():
    client = z.Client(z.load_config())
    assert client.sql("SELECT 1 AS one").scalar() == "1"
    assert client.org_id


def test_default_open_tables_reads_committed_map(tmp_path):
    mod = _load_script("load_zetaris")
    m = tmp_path / "tables.csv"
    m.write_text("table,file,licence,source_publisher\nopen_x,x.csv,CC BY 4.0,Somebody\n", encoding="utf-8")
    assert mod.default_open_tables(m) == [f"open_x={tmp_path / 'x.csv'}"]
    assert mod.default_open_tables(tmp_path / "missing.csv") == []


def test_committed_open_tables_point_at_real_files():
    mod = _load_script("load_zetaris")
    specs = mod.default_open_tables()
    assert len(specs) == 4
    for spec in specs:
        table, _, path = spec.partition("=")
        z.check_name(table)
        assert Path(path).is_file()
