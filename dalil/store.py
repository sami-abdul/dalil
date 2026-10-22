"""The audit store: every question, what was found, drafted and checked, and
every decision, in SQLite at build/dalil.db (git-ignored).

Only the redacted question is ever written. audit_events is append-only:
triggers refuse UPDATE and DELETE on it, so a step, once recorded, stays.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "build" / "dalil.db"

STATUSES = ("held", "awaiting_review", "approved", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    question_redacted TEXT NOT NULL,
    flags TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    question_id TEXT PRIMARY KEY REFERENCES questions(id),
    retrieval TEXT NOT NULL,
    data TEXT NOT NULL,
    draft TEXT NOT NULL,
    "check" TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    question_id TEXT PRIMARY KEY REFERENCES questions(id),
    decision TEXT NOT NULL CHECK (decision IN ('approve', 'reject')),
    reviewer TEXT NOT NULL,
    note TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT NOT NULL REFERENCES questions(id),
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_events_question ON audit_events(question_id, id);
CREATE TRIGGER IF NOT EXISTS audit_events_no_update BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_events_no_delete BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
"""


class ReviewError(ValueError):
    """A review the record's state does not allow."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # --- writes ------------------------------------------------------------

    def create_question(self, question_redacted: str, flags: list[str]) -> str:
        qid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO questions (id, created_at, question_redacted, flags, status) VALUES (?, ?, ?, ?, ?)",
                (qid, now(), question_redacted, json.dumps(flags), "screening"),
            )
        return qid

    def event(self, question_id: str, kind: str, detail: dict[str, Any] | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO audit_events (question_id, at, kind, detail) VALUES (?, ?, ?, ?)",
                (question_id, now(), kind, json.dumps(detail or {}, ensure_ascii=False)),
            )

    def save_record(self, question_id: str, retrieval: list[dict[str, Any]], data: list[dict[str, Any]],
                    draft: list[dict[str, Any]], check: dict[str, Any], status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"bad status {status!r}")
        dump = lambda v: json.dumps(v, ensure_ascii=False)  # noqa: E731
        with self._conn() as c:
            c.execute(
                'INSERT OR REPLACE INTO records (question_id, retrieval, data, draft, "check") VALUES (?, ?, ?, ?, ?)',
                (question_id, dump(retrieval), dump(data), dump(draft), dump(check)),
            )
            c.execute("UPDATE questions SET status = ? WHERE id = ?", (status, question_id))

    def review(self, question_id: str, decision: str, reviewer: str, note: str = "") -> None:
        """Record a reviewer's decision. Approve needs status awaiting_review;
        a held answer can only be rejected; a decided answer cannot be
        reviewed again."""
        if decision not in ("approve", "reject"):
            raise ReviewError("decision must be approve or reject")
        if not reviewer.strip():
            raise ReviewError("reviewer is required")
        at = now()
        with self._conn() as c:
            row = c.execute("SELECT status FROM questions WHERE id = ?", (question_id,)).fetchone()
            if row is None:
                raise KeyError(question_id)
            status = row["status"]
            if decision == "approve" and status != "awaiting_review":
                raise ReviewError(f"only an answer awaiting review can be approved (status is {status})")
            if decision == "reject" and status not in ("awaiting_review", "held"):
                raise ReviewError(f"this answer cannot be rejected (status is {status})")
            new = "approved" if decision == "approve" else "rejected"
            # The status guard in the WHERE clause makes a concurrent second review a no-op.
            cur = c.execute("UPDATE questions SET status = ? WHERE id = ? AND status = ?", (new, question_id, status))
            if cur.rowcount != 1:
                raise ReviewError("the answer changed while it was being reviewed")
            c.execute("INSERT INTO reviews (question_id, decision, reviewer, note, at) VALUES (?, ?, ?, ?, ?)",
                      (question_id, decision, reviewer.strip(), note, at))
            detail = {"reviewer": reviewer.strip(), "note": note, "from_status": status}
            c.execute("INSERT INTO audit_events (question_id, at, kind, detail) VALUES (?, ?, ?, ?)",
                      (question_id, at, "approved" if decision == "approve" else "rejected",
                       json.dumps(detail, ensure_ascii=False)))
            if decision == "approve":
                c.execute("INSERT INTO audit_events (question_id, at, kind, detail) VALUES (?, ?, ?, ?)",
                          (question_id, now(), "delivered", json.dumps({"reviewer": reviewer.strip()})))

    # --- reads -------------------------------------------------------------

    def get(self, question_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            q = c.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
            if q is None:
                return None
            rec = c.execute("SELECT * FROM records WHERE question_id = ?", (question_id,)).fetchone()
            rev = c.execute("SELECT * FROM reviews WHERE question_id = ?", (question_id,)).fetchone()
            events = c.execute("SELECT at, kind, detail FROM audit_events WHERE question_id = ? ORDER BY id",
                               (question_id,)).fetchall()
        load = lambda key, default: json.loads(rec[key]) if rec else default  # noqa: E731
        return {
            "id": q["id"],
            "question": q["question_redacted"],
            "flags": json.loads(q["flags"]),
            "status": q["status"],
            "created_at": q["created_at"],
            "retrieval": load("retrieval", []),
            "data": load("data", []),
            "draft": load("draft", []),
            "check": load("check", None),
            "review": ({"decision": rev["decision"], "reviewer": rev["reviewer"], "note": rev["note"], "at": rev["at"]}
                       if rev else None),
            "audit": [{"at": e["at"], "kind": e["kind"], "detail": json.loads(e["detail"])} for e in events],
        }

    def list(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                'SELECT q.id, q.question_redacted, q.status, q.created_at, r."check" AS chk '
                "FROM questions q LEFT JOIN records r ON r.question_id = q.id "
                "ORDER BY q.created_at DESC, q.rowid DESC"
            ).fetchall()
        out = []
        for r in rows:
            check = json.loads(r["chk"]) if r["chk"] else None
            sentences = (check or {}).get("sentences") or []
            out.append({
                "id": r["id"], "question": r["question_redacted"], "status": r["status"],
                "created_at": r["created_at"],
                "verifier_status": check["status"] if check else None,
                "kept": sum(1 for s in sentences if s.get("kept")),
                "cut": sum(1 for s in sentences if not s.get("kept")),
            })
        return out

    def rejection_notes(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent rejections with the reviewer's note and the question, newest
        first, for a drafter to learn from."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT r.question_id, q.question_redacted, r.reviewer, r.note, r.at FROM reviews r "
                "JOIN questions q ON q.id = r.question_id WHERE r.decision = 'reject' AND r.note <> '' "
                "ORDER BY r.at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [{"question_id": r["question_id"], "question": r["question_redacted"], "reviewer": r["reviewer"],
                 "note": r["note"], "at": r["at"]} for r in rows]
