"""Check the open-data source Dalil would use, whichever it is.

    uv run python scripts/check_open_data.py                  # the source DALIL_OPEN_DATA picks (auto)
    DALIL_OPEN_DATA=local uv run python scripts/check_open_data.py
    DALIL_OPEN_DATA=zetaris uv run python scripts/check_open_data.py

Prints which source answered and why, then for every table in tables.csv the
row count against the CSV, then every fixed query in dalil/opendata.py with
its row and the checker's rerun of the sentence it backs, then the rows
recorded from the server's Zetaris (tests/fixtures/zetaris_recorded_rows.json)
against this source. Every line reads PASS or FAIL; the exit code is 1 if any
line fails. Open data only, so nothing private is printed.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil import datasource as ds  # noqa: E402
from dalil.draft import data_claims  # noqa: E402
from dalil.opendata import RULES, run_rules  # noqa: E402
from dalil.verify import Verifier, data_lookup  # noqa: E402

RECORDED = ROOT / "tests" / "fixtures" / "zetaris_recorded_rows.json"


def main() -> int:
    failed = 0

    def line(ok: bool, text: str) -> None:
        nonlocal failed
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {text}")

    started = time.monotonic()
    source, why = ds.resolve()
    took = time.monotonic() - started
    status = ds.status(source)
    print(f"mode={ds.open_data_mode()} source={status['kind']} label={status['label']!r} ready_in={took:.2f}s")
    if why:
        print(f"note: {why}")
    line(status["connected"], f"source answers SELECT 1 ({status['label']})")
    if source is None:
        return 1

    data_dir = ds.open_data_dir()
    for tf in ds.table_files(data_dir):
        with tf.path.open(newline="", encoding="utf-8") as fh:
            want = sum(1 for _ in csv.DictReader(fh))
        try:
            got = int(source.sql(f"SELECT COUNT(*) AS n FROM dalil.{tf.table}").scalar())
        except Exception as e:
            line(False, f"dalil.{tf.table}: {type(e).__name__}: {str(e)[:160]}")
            continue
        line(got == want, f"dalil.{tf.table}: {got} rows, {want} in {tf.path.name}")

    lookup = data_lookup(source)
    for rule in RULES:
        results = run_rules([rule], lambda sql: source.sql(sql, limit=200).dicts(), source=status["kind"])
        res = results[0]
        if not res.rows:
            line(False, f"{rule.name}: no row ({res.note})")
            continue
        claims = data_claims(results)
        check = Verifier({}, lookup, allowed_sql=[rule.sql]).check_answer(claims)
        line(check.status == "pass", f"{rule.name}: {res.rows[0]} -> checker {check.status}: {claims[0]['text']}")

    if RECORDED.is_file():
        for rec in json.loads(RECORDED.read_text(encoding="utf-8"))["recorded"]:
            got = source.sql(rec["sql"], limit=200).dicts()
            line(got == rec["rows"], f"matches Zetaris rows recorded {rec['first'][:10]} to {rec['last'][:10]}: "
                                      f"{got} vs {rec['rows']}")
    print(f"summary: {'ALL PASS' if not failed else f'{failed} FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
