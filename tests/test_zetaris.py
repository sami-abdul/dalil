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


def test_ui_url_with_userinfo_is_rejected_without_echoing_it():
    secret = "s3cret-" + FAKE
    for url in (f"http://admin:{secret}@127.0.0.1:13000", f"http://{secret}@host"):
        with pytest.raises(z.ZetarisError) as info:
            z.Config(ui_url=url, email="a", password=FAKE)
        assert "user name or password" in str(info.value)
        assert secret not in str(info.value)


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body, self.status = body, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def test_non_json_200_raises_zetaris_error(monkeypatch):
    monkeypatch.setattr(z, "urlopen", lambda req, timeout: _FakeResponse(b"<!doctype html><html>app</html>"))
    client = z.Client(z.Config(api_key=FAKE, org_id="1"))
    with pytest.raises(z.ZetarisError, match="HTML page; check ZETARIS_UI_URL") as info:
        client.sql("SELECT 1")
    assert info.value.status == 200
    monkeypatch.setattr(z, "urlopen", lambda req, timeout: _FakeResponse(b"\xff\xfe not json"))
    with pytest.raises(z.ZetarisError, match="not JSON; check ZETARIS_UI_URL"):
        client.sql("SELECT 1")


def test_http_error_detail_never_echoes_credentials(monkeypatch):
    import io
    from urllib.error import HTTPError

    secret = "pw-" + FAKE

    def fail(req, timeout):
        body = json.dumps({"msg": f"bad request for password {secret}"}).encode()
        raise HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(body))

    monkeypatch.setattr(z, "urlopen", fail)
    client = z.Client(z.Config(email="a@example.invalid", password=secret))
    with pytest.raises(z.ZetarisError) as info:
        client.login()
    assert secret not in str(info.value) and "[redacted]" in str(info.value)


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
    import csv as _csv
    with mod.OPEN_TABLES.open(newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    specs = mod.open_table_specs([], include_defaults=True)
    assert len(specs) == len(rows) > 0
    assert all(path.is_file() for _, path, _ in specs)


def test_open_table_specs_adds_cli_to_defaults_and_can_skip_them(tmp_path):
    mod = _load_script("load_zetaris")
    f = tmp_path / "rates.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    both = mod.open_table_specs([f"rates={f}"], include_defaults=True)
    assert both[-1][0] == "rates" and len(both) == len(mod.default_open_tables()) + 1
    only = mod.open_table_specs([f"rates={f}"], include_defaults=False)
    assert [t for t, _, _ in only] == ["rates"]


def test_open_table_specs_rejects_collisions_and_bad_paths(tmp_path):
    mod = _load_script("load_zetaris")
    f = tmp_path / "x.csv"
    f.write_text("a\n1\n", encoding="utf-8")
    for bad in ([f"pages={f}"], [f"x={f}", f"X={f}"], [f"x={tmp_path / 'missing.csv'}"], ["nosep"]):
        with pytest.raises(z.ZetarisError):
            mod.open_table_specs(bad, include_defaults=False)


def test_csv_record_count_counts_multiline_records(tmp_path):
    mod = _load_script("load_zetaris")
    f = tmp_path / "m.csv"
    f.write_text('name,address\nA,"line one\nline two"\nB,plain\n', encoding="utf-8")
    assert mod.csv_record_count(f) == 2


def test_standin_ids_match_the_corpus_builder(tmp_path):
    from dalil.corpus import doc_id

    loader = _load_script("load_zetaris")
    manifest = tmp_path / "MANIFEST.csv"
    manifest.write_text(
        "file,title,publisher,source_url,retrieved_utc,sha256,language,topic,licence_note\n"
        "UAE_Visa Rules.HTM,Visa,u.ae,https://u.ae/v,2026-10-19T00:00:00Z,ab,en,visas,reserved\n",
        encoding="utf-8",
    )
    doc = loader.standin_documents(manifest)[0]
    assert doc["doc_id"] == doc_id("UAE_Visa Rules.HTM") == "uae-visa-rules"
    assert doc["kind"] == "html"
    manifest.write_text(manifest.read_text().replace(".HTM,", ".txt,"), encoding="utf-8")
    with pytest.raises(z.ZetarisError, match="unsupported file type"):
        loader.standin_documents(manifest)
