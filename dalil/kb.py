"""Knowledge-base profiles: everything that makes Dalil answer for one
country's services, in one folder, chosen by one setting.

A profile is kb/<id>/profile.json:

    {"id": "uk", "name": "UK government services (GOV.UK)", "country": "United Kingdom",
     "audience": "people in the UK",
     "manifest": "kb/uk/MANIFEST.csv",
     "documents": "kb/uk/index/documents.jsonl",
     "passages": "kb/uk/index/passages.jsonl",
     "corpus_dir": "build/kb/uk/corpus",
     "store": "build/kb/uk/dalil.db",
     "open_data": null,
     "review_memory": false,
     "screener": {"id_formats": [...], "refusal_text": "...", "refusal_follow_up": "...",
                  "personal_data_caveat": "...", "ask_for_detail": "..."},
     "licence": "...", "attribution": "..."}

Paths are relative to the repository root. corpus_dir null means
$DALIL_CORPUS_DIR, else ./corpus. open_data is null for none, or
{"dir": "data/open", "rules": true}: the folder the local database loads and
whether the fixed open-data rules in dalil/opendata.py apply. A screener with
no id_formats keeps the UAE formats in dalil/safety.py, and each text left
out keeps the UAE wording.

DALIL_KB picks the profile at start (default "uae"). The reviewer screen can
switch between profiles while the app runs; see dalil/app.py.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from dalil import safety

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_ROOT = REPO_ROOT / "kb"
DEFAULT_KB = "uae"
_ID = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


class ProfileError(ValueError):
    """A profile is missing, or its file is not usable."""


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    country: str
    audience: str
    manifest: Path
    documents: Path
    passages: Path
    corpus_dir: Path | None
    store: Path
    open_data_dir: Path | None
    open_data_rules: bool
    review_memory: bool
    screener: safety.Screener = field(default_factory=safety.Screener)
    description: str = ""
    licence: str = ""
    attribution: str = ""
    # The ask box's label and example question on the reviewer screen.
    ask_label: str = ""
    example_question: str = ""

    @property
    def labels(self) -> dict[str, str]:
        return {k: v for k, v in (("ask_label", self.ask_label), ("example_question", self.example_question)) if v}

    def summary(self, documents: int | None = None, passages: int | None = None) -> dict[str, Any]:
        """What the health endpoint and the screen show."""
        return {"id": self.id, "name": self.name, "country": self.country,
                "documents": documents, "passages": passages, **self.labels}


def _path(value: Any, what: str, profile_id: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ProfileError(f"{profile_id}: {what} must be a path inside the repository, not {value!r}")
    return REPO_ROOT / value


def parse(raw: Mapping[str, Any]) -> Profile:
    pid = raw.get("id")
    if not isinstance(pid, str) or not _ID.match(pid):
        raise ProfileError(f"profile id must be lower-case letters, digits and dashes: {pid!r}")
    for key in ("name", "country", "audience", "manifest", "documents", "passages", "store"):
        if not raw.get(key):
            raise ProfileError(f"{pid}: {key} is required")
    s = raw.get("screener") or {}
    try:
        redactions = safety.compile_redactions(s["id_formats"]) if s.get("id_formats") else tuple(safety._REDACTIONS)
    except (KeyError, re.error, ValueError) as e:
        raise ProfileError(f"{pid}: screener.id_formats: {e}") from None
    default = safety.Screener()
    screener = safety.Screener(
        redactions=redactions,
        refusal_text=s.get("refusal_text") or default.refusal_text,
        refusal_follow_up=s.get("refusal_follow_up") or default.refusal_follow_up,
        personal_data_caveat=s.get("personal_data_caveat") or default.personal_data_caveat,
        ask_for_detail=s.get("ask_for_detail") or default.ask_for_detail,
    )
    for text in screener.fixed_texts.values():
        if re.search(r"\d", text):
            raise ProfileError(f"{pid}: fixed sentences may not hold numbers: {text!r}")
    od = raw.get("open_data")
    if od is not None and not isinstance(od, dict):
        raise ProfileError(f"{pid}: open_data must be null or {{\"dir\": ..., \"rules\": ...}}")
    return Profile(
        id=pid, name=raw["name"], country=raw["country"], audience=raw["audience"],
        manifest=_path(raw["manifest"], "manifest", pid),
        documents=_path(raw["documents"], "documents", pid),
        passages=_path(raw["passages"], "passages", pid),
        corpus_dir=_path(raw["corpus_dir"], "corpus_dir", pid) if raw.get("corpus_dir") else None,
        store=_path(raw["store"], "store", pid),
        open_data_dir=_path(od["dir"], "open_data.dir", pid) if od else None,
        open_data_rules=bool(od and od.get("rules")),
        review_memory=bool(raw.get("review_memory")),
        screener=screener,
        description=raw.get("description", ""),
        licence=raw.get("licence", ""),
        attribution=raw.get("attribution", ""),
        ask_label=raw.get("ask_label", ""),
        example_question=raw.get("example_question", ""),
    )


def available(root: Path = KB_ROOT) -> list[str]:
    """Profile ids with a profile.json under kb/, sorted."""
    if not root.is_dir():
        return []
    return sorted(p.parent.name for p in root.glob("*/profile.json") if _ID.match(p.parent.name))


def load(profile_id: str, root: Path = KB_ROOT) -> Profile:
    if not isinstance(profile_id, str) or not _ID.match(profile_id):
        raise ProfileError(f"not a knowledge-base id: {profile_id!r}")
    path = root / profile_id / "profile.json"
    if not path.is_file():
        raise ProfileError(f"no knowledge base {profile_id!r}; have: {', '.join(available(root)) or 'none'}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ProfileError(f"{path}: not JSON ({e})") from None
    profile = parse(raw)
    if profile.id != profile_id:
        raise ProfileError(f"{path}: id {profile.id!r} does not match its folder")
    return profile


def active_id(environ: Mapping[str, str] | None = None) -> str:
    environ = os.environ if environ is None else environ
    return (environ.get("DALIL_KB") or DEFAULT_KB).strip().lower()


def count_documents(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
