# Tool log

Which tool wrote what, day by day. The sponsor write-up and the README are written from this log and claim only what it records.

Tools on this project: Claude Code (Anthropic) running on the team's Linux server, and any edits a person makes by hand. Cursor is listed only where a team member used it.

| Date (UTC) | Tool | What it wrote | What a person changed | What broke |
|---|---|---|---|---|
| 2026-10-22 | Claude Code | Repo scaffold: `.gitignore`, `pyproject.toml`, `.env.example`, this log | Nothing yet | Nothing |
| 2026-10-22 | Claude Code | `eval/questions.jsonl`: ten test questions and four error scenarios, with the documents each answer should cite | Nothing yet | Nothing |
| 2026-10-22 | Claude Code | `dalil/corpus.py`, `scripts/extract_corpus.py`, `scripts/fetch_corpus.py`, `tests/test_corpus.py`: manifest loading, page-by-page text extraction from the HTML and PDF documents, and a fetch script for people reproducing the corpus | Nothing yet | pdftotext garbled Arabic word order, so extraction uses pypdf; the AI Charter PDF's font drops its "Th" and "ff" ligatures, repaired for that one file |
| 2026-10-22 | Claude Code | `dalil/zetaris.py`, `scripts/load_zetaris.py`, `scripts/check_zetaris.py`, `tests/test_zetaris.py`: a small Zetaris client, and a loader that registers `dalil.documents` and `dalil.pages` as Lightning file tables | Nothing yet | A file table over JSON lines showed only its first row on Zetaris 2.4.3.1; the loader writes JSON arrays and checks every row count |
| 2026-10-22 | Claude Code | `data/open/`: four Bayanat open datasets as tidy CSVs with sources and licences, and `tables.csv` so the loader registers them in Zetaris by default | Nothing yet | Bayanat licences differ per dataset page; two combined files had to take CC BY-SA |
| 2026-10-22 | Claude Code | Fixes from a high-effort code review of the open-data loader: row counts checked for every CSV table, `--csv` adds to the committed tables and `--no-open` skips them, one set of CSV options for the plan and the run, name collision checks, clearer `SOURCES.csv` notes | Nothing yet | Nothing |
| 2026-10-22 | Claude Code | Fixes from a high-effort code review of the corpus and Zetaris client: headers inside article content kept, precise chrome stripping, NFKC before tatweel removal, line breaks kept in `<pre>`, safer `--force` fetches, credentials kept out of every error message | Nothing yet | Nothing; the rebuilt corpus is byte-identical, so the fixes guard future pages |
