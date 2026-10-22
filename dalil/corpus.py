"""Load the document manifest and extract per-page text from the private corpus.

The corpus files are untrusted input. HTML is parsed with BeautifulSoup and
lxml and never rendered or run. PDFs are read with pypdf, which only parses
them. Nothing here executes document content.

Text is normalised for whitespace and invisible characters only. Wording is
left as extracted, because the citation checker matches quoted passages word
for word against this text.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "MANIFEST.csv"
DEFAULT_CORPUS_DIR = Path("corpus")
CORPUS_DIR_ENV = "DALIL_CORPUS_DIR"

MANIFEST_FIELDS = (
    "file",
    "title",
    "publisher",
    "source_url",
    "retrieved_utc",
    "sha256",
    "language",
    "topic",
    "licence_note",
)

# Field order of each record in build/documents.jsonl.
DOCUMENT_FIELDS = (
    "doc_id",
    "file",
    "title",
    "publisher",
    "source_url",
    "retrieved_utc",
    "sha256",
    "language",
    "topic",
    "kind",
    "is_terms_page",
    "sha256_ok",
    "pages",
    "char_count",
)

TERMS_TOPIC_PREFIX = "licence:"


class CorpusError(Exception):
    """A hard failure: bad manifest, missing file or unreadable document."""


# --------------------------------------------------------------------------
# Manifest


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> list[dict[str, str]]:
    """Read MANIFEST.csv into a list of row dicts, checking columns and file names."""
    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in MANIFEST_FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            raise CorpusError(f"manifest {path} lacks columns: {', '.join(missing)}")
        rows = [{k: (row.get(k) or "").strip() for k in MANIFEST_FIELDS} for row in reader]
    seen: set[str] = set()
    for i, row in enumerate(rows, start=2):
        name = row["file"]
        if not name or Path(name).name != name or name.startswith("."):
            raise CorpusError(f"manifest line {i}: bad file name {name!r}")
        did = doc_id(name)
        if did in seen:
            raise CorpusError(f"manifest line {i}: duplicate doc_id {did!r}")
        seen.add(did)
    return rows


def doc_id(file_name: str) -> str:
    """Stable document id: the file name stem, lower-cased, non [a-z0-9] runs as '-'."""
    stem = Path(file_name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    if not slug:
        raise CorpusError(f"cannot derive doc_id from {file_name!r}")
    return slug


def is_terms_page(topic: str) -> bool:
    """Publisher terms-of-use and licence pages: kept, but not answerable content."""
    return topic.strip().lower().startswith(TERMS_TOPIC_PREFIX)


def doc_kind(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".html", ".htm"):
        return "html"
    raise CorpusError(f"unsupported file type: {file_name}")


def resolve_corpus_dir(explicit: Path | str | None = None) -> Path:
    """Corpus directory: explicit argument, else $DALIL_CORPUS_DIR, else ./corpus."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(CORPUS_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    return DEFAULT_CORPUS_DIR


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Text normalisation

# Characters that carry no wording: zero-width space, BOM, soft hyphen, word
# joiner, the bidi embedding and isolate controls that PDF and HTML tools
# insert around Arabic runs, and the Arabic tatweel (U+0640), a stretching
# stroke typesetters add for justification and sometimes run together as a rule.
_INVISIBLE = dict.fromkeys(
    map(
        ord,
        "\u200b\ufeff\u00ad\u2060\u200e\u200f"
        "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
        "\u0640",
    ),
    None,
)
# Latin ligatures (U+FB00-FB06) and Arabic presentation forms are glyph
# artefacts of PDF extraction; NFKC maps them back to the plain letters.
_COMPAT_RE = re.compile("[\ufb00-\ufb06\ufb50-\ufdff\ufe70-\ufefe]")
# The isolated and medial forms of the Arabic marks (U+FE70-FE7F) decompose to
# a space or a tatweel followed by the combining mark. The space or tatweel is
# only a carrier for the glyph, so it is dropped and the mark stays on its letter.
_MARK_FORMS_RE = re.compile("[\ufe70-\ufe7f]")
_HSPACE_RE = re.compile(r"[^\S\n]+")


def _compat(match: re.Match[str]) -> str:
    char = match.group()
    plain = unicodedata.normalize("NFKC", char)
    if _MARK_FORMS_RE.match(char):
        plain = plain.lstrip(" \u0640")
    return plain


def normalise_text(text: str) -> str:
    """Normalise whitespace and drop invisible characters; leave wording alone.

    Each line is stripped and inner runs of spaces, tabs and other horizontal
    whitespace become one space. Blank lines are dropped, so lines are joined
    by a single newline.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n").replace("\x0c", "\n")
    # Compatibility forms first, so a tatweel or invisible character that NFKC
    # produces is removed by the step after it.
    text = _COMPAT_RE.sub(_compat, text)
    text = text.translate(_INVISIBLE)
    lines = (_HSPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


# --------------------------------------------------------------------------
# HTML

# Elements removed everywhere before the main content is chosen.
_DROP_TAGS = (
    "script", "style", "noscript", "template", "svg", "math", "iframe", "object",
    "embed", "link", "meta", "canvas", "img", "picture", "video", "audio", "source",
    "map", "head",
)
# Elements removed inside the chosen content: navigation and form controls.
_CHROME_TAGS = frozenset(
    ("nav", "form", "select", "option", "button", "input", "textarea", "label", "dialog")
)
# <header> and <footer> are page chrome only at page level. Inside a selected
# content root, or inside sectioning content, they hold the document's own
# title, number, date or signature, so they are kept.
_LANDMARK_TAGS = frozenset(("header", "footer"))
_SECTIONING_TAGS = frozenset(("article", "aside", "main", "section"))
# Class or id tokens naming chrome, compared whole and case-insensitively.
# Each is a container seen in this corpus's templates: u.ae and Bayanat,
# the Dubai Legislation Portal (SharePoint), ICP (aegov) and GDRFA (Drupal),
# plus the usual accessibility helpers. A token that merely contains one of
# these words, such as "cookie-policy-content", is not chrome.
_CHROME_TOKENS = frozenset(
    t.lower()
    for t in (
        # breadcrumbs
        "breadcrumb", "breadcrumbs", "breadcrumb-area",
        # cookie banners (ICP)
        "cookie-alert",
        # feedback and rating widgets (u.ae, ICP)
        "feedback", "feedback-container", "feedback-section", "feedback-modal",
        "feedBack-Container", "FeedbackMsg", "happIndexForm",
        # modal dialogs (ICP aegov, GDRFA bootstrap)
        "modal", "aegov-modal", "aegov-modal-wrapper",
        # share and social links (ICP, GDRFA)
        "social-sharing", "Social-platforms-container",
        # navigation bars and menus
        "nav", "navbar", "nav-bar", "nav-container", "nav-links", "nav-menu",
        "topic-navigation", "pagination",
        # u.ae tab picker and hero links
        "sticky-wrapper", "select-container", "read-more",
        # skip links, screen-reader-only text, print-hidden widgets
        "skip-link", "skip-to-content", "sr-only", "visually-hidden", "no-print",
        # GDRFA field labels, SharePoint hidden ribbons and dialog chrome
        "field__label", "ms-hide", "s4-notdlg", "ms-dialoghidden",
    )
)
_ROLE_CHROME = {"navigation", "banner", "contentinfo", "search", "dialog", "alertdialog"}

# Main-content selectors, most specific first. The first one that holds text
# wins. u.ae, Bayanat (same template), SharePoint (Dubai Legislation Portal),
# ICP (aegov), GDRFA (Drupal), then generic fallbacks.
_CONTENT_SELECTORS = (
    "main .tab-content",
    ".main .tab-content",
    "#DeltaPlaceHolderMain",
    ".entry-article",
    "main .node__content",
    "main",
    "article",
    "[role=main]",
    "#content",
    "#main-content",
)
# u.ae puts the page title and lead paragraph in a hero block above <main>,
# and the "Updated on" date in a strip below the tabbed content.
_HERO_SELECTOR = ".head-services .hero-text"
_TRAILER_SELECTORS = ("main .artical-info-link",)

_BLOCK_TAGS = frozenset(
    "address article aside blockquote body caption center dd details dialog div dl dt "
    "fieldset figcaption figure footer form h1 h2 h3 h4 h5 h6 header hr html legend li "
    "main nav ol p pre section summary table tbody tfoot thead tr ul".split()
)
_CELL_SEP = " | "
_NEWLINE = "\n"


def _tokens(tag: Tag) -> list[str]:
    tokens = list(tag.get("class") or [])
    if tag.get("id"):
        tokens.append(tag["id"])
    return [t.lower() for t in tokens]


def _is_chrome(tag: Tag, page_level: bool) -> bool:
    """True when tag is site chrome. page_level is True when the tag is not
    inside a selected content root, so a bare <header> or <footer> is the
    page's banner or footer rather than part of the document."""
    if tag.name in _CHROME_TAGS:
        return True
    if (
        page_level
        and tag.name in _LANDMARK_TAGS
        and not any(p.name in _SECTIONING_TAGS for p in tag.parents)
    ):
        return True
    if (tag.get("role") or "").lower() in _ROLE_CHROME:
        return True
    return any(t in _CHROME_TOKENS for t in _tokens(tag))


def _strip_chrome(root: Tag, page_level: bool = False) -> None:
    """Remove chrome below root. Only descendants are visited, so root itself
    and its ancestors stay even when their class or tag looks like chrome
    (the Dubai Legislation Portal's content sits inside an ASP.NET <form>)."""
    for tag in list(root.find_all(True)):
        if tag.decomposed:
            continue
        if _is_chrome(tag, page_level):
            tag.decompose()


def _render(node: Tag, out: list[str], pre: bool = False) -> None:
    """Append the visible text of node to out, with '\\n' at block edges.

    Inside <pre> the source line breaks are kept; elsewhere HTML collapses
    source whitespace, so this does the same.
    """
    for child in node.children:
        if isinstance(child, (Comment, Declaration, Doctype, ProcessingInstruction)):
            continue
        if isinstance(child, NavigableString):
            text = str(child)
            out.append(text if pre else re.sub(r"\s+", " ", text))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name == "br":
            out.append(_NEWLINE)
        elif name == "tr":
            cells = child.find_all(["td", "th"], recursive=False)
            if cells:
                texts = []
                for cell in cells:
                    parts: list[str] = []
                    _render(cell, parts, pre)
                    texts.append(" ".join("".join(parts).split()))
                out.append(_NEWLINE + _CELL_SEP.join(t for t in texts if t) + _NEWLINE)
            else:
                out.append(_NEWLINE)
                _render(child, out, pre)
                out.append(_NEWLINE)
        elif name in _BLOCK_TAGS:
            out.append(_NEWLINE)
            _render(child, out, pre or name == "pre")
            out.append(_NEWLINE)
        else:
            _render(child, out, pre)


def _node_text(node: Tag) -> str:
    parts: list[str] = []
    _render(node, parts)
    return normalise_text("".join(parts))


def _outermost(tags: Iterable[Tag]) -> list[Tag]:
    """Drop matches nested inside other matches, keeping document order."""
    chosen: list[Tag] = []
    chosen_ids: set[int] = set()
    for tag in tags:
        if not any(id(parent) in chosen_ids for parent in tag.parents):
            chosen.append(tag)
            chosen_ids.add(id(tag))
    return chosen


def extract_html_text(raw: bytes) -> str:
    """Return the main article text of an HTML page, without site chrome."""
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    blocks: list[str] = []
    hero = soup.select_one(_HERO_SELECTOR)
    if hero is not None:
        _strip_chrome(hero)
        hero_text = _node_text(hero)
        if hero_text:
            blocks.append(hero_text)
        hero.decompose()

    body_text = ""
    for selector in _CONTENT_SELECTORS:
        roots = _outermost(soup.select(selector))
        if not roots:
            continue
        texts = []
        for root in roots:
            _strip_chrome(root)
            text = _node_text(root)
            if text:
                texts.append(text)
        if texts:
            body_text = "\n".join(texts)
            break
    if not body_text:
        root = soup.body or soup
        _strip_chrome(root, page_level=True)
        body_text = _node_text(root)
    if not blocks:
        # Some templates keep the page title in an <h1> above the content.
        h1 = soup.find("h1")
        title = _node_text(h1).replace("\n", " ") if h1 is not None else ""
        if title and len(title) <= 200 and not body_text.startswith(title):
            blocks.append(title)
    if body_text:
        blocks.append(body_text)
    for selector in _TRAILER_SELECTORS:
        for tag in soup.select(selector):
            line = " ".join(_node_text(tag).split())
            if line and line not in body_text:
                blocks.append(line)
    return "\n".join(blocks)


def extract_html_pages(path: Path) -> list[dict[str, Any]]:
    return [{"page": 1, "text": extract_html_text(path.read_bytes())}]


# --------------------------------------------------------------------------
# PDF


# Per-document repairs for glyphs a PDF font maps to no character, which both
# pypdf and pdftotext emit as U+FFFD. Keyed by doc_id. The AI Charter's Colfax
# font has no Unicode mapping for its "Th" and "ff" ligatures; the patterns put
# back the letters the page displays (word-initial "Th", word-internal "ff").
_GLYPH_REPAIRS: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "ai-gov-uae-charter-development-use-of-ai-2024-en": [
        (re.compile(r"(?<![^\W\d_])\ufffd(?=[a-z])"), "Th"),
        (re.compile(r"(?<=[a-z])\ufffd(?=[a-z])"), "ff"),
    ],
}


def repair_glyphs(doc: str, text: str) -> str:
    for pattern, replacement in _GLYPH_REPAIRS.get(doc, ()):
        text = pattern.sub(replacement, text)
    return text


def extract_pdf_pages(path: Path) -> list[dict[str, Any]]:
    """One entry per physical PDF page, numbered from 1, empty pages included.

    pypdf is used because it keeps Arabic letters and lam-alef ligatures in
    logical order, where pdftotext swaps letters inside ligatures.
    """
    from pypdf import PdfReader

    # pypdf logs font warnings for every page of some files; they are noise.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(path)
    if reader.is_encrypted:
        reader.decrypt("")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = repair_glyphs(doc_id(path.name), page.extract_text() or "")
        pages.append({"page": number, "text": normalise_text(text)})
    return pages


# --------------------------------------------------------------------------
# Documents


def extract_document(row: dict[str, str], corpus_dir: Path | str) -> dict[str, Any]:
    """Build the documents.jsonl record for one manifest row."""
    path = Path(corpus_dir) / row["file"]
    if not path.is_file():
        raise CorpusError(f"missing corpus file: {path}")
    kind = doc_kind(row["file"])
    try:
        pages = extract_pdf_pages(path) if kind == "pdf" else extract_html_pages(path)
    except CorpusError:
        raise
    except Exception as exc:  # pypdf and lxml raise many types on bad input
        raise CorpusError(f"cannot extract {row['file']}: {type(exc).__name__}: {exc}") from exc
    record = {
        "doc_id": doc_id(row["file"]),
        "file": row["file"],
        "title": row["title"],
        "publisher": row["publisher"],
        "source_url": row["source_url"],
        "retrieved_utc": row["retrieved_utc"],
        "sha256": row["sha256"].lower(),
        "language": row["language"],
        "topic": row["topic"],
        "kind": kind,
        "is_terms_page": is_terms_page(row["topic"]),
        "sha256_ok": sha256_file(path) == row["sha256"].lower(),
        "pages": pages,
        "char_count": sum(len(p["text"]) for p in pages),
    }
    return {k: record[k] for k in DOCUMENT_FIELDS}


def iter_documents(
    rows: Iterable[dict[str, str]], corpus_dir: Path | str
) -> Iterator[tuple[dict[str, str], dict[str, Any] | None, CorpusError | None]]:
    """Yield (row, record, error) per manifest row; record is None on error."""
    for row in rows:
        try:
            yield row, extract_document(row, corpus_dir), None
        except CorpusError as exc:
            yield row, None, exc
