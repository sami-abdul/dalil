"""Exercise the citation checker on every real passage, printing counts only.

For each passage long enough, take an eight-word slice, reflow its whitespace
across a line break, and check that the checker finds it and maps it to the
exact span on the page. Then change one word and check that it is caught.

    uv run python scripts/selftest_verify.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dalil.verify import Verifier, load_passages  # noqa: E402


def main() -> int:
    passages = load_passages(ROOT / "build" / "passages.jsonl")
    pages = {}
    with (ROOT / "build" / "documents.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            for page in doc["pages"]:
                pages[(doc["doc_id"], page["page"])] = page["text"]
    v = Verifier(passages)
    rng = random.Random(7)
    tested = found = mapped = caught = 0
    for pid, p in passages.items():
        words = p["text"].split()
        if len(words) < 12:
            continue
        tested += 1
        i = rng.randrange(0, len(words) - 8)
        quote = " ".join(words[i:i + 4]) + "  \n" + " ".join(words[i + 4:i + 8])
        s = v.check_sentence({"text": "x", "citations": [{"passage_id": pid, "quote": quote}]})
        if s.kept:
            found += 1
            c = s.citations[0]
            span = pages[(c.doc_id, c.page)][c.page_start:c.page_end]
            mapped += " ".join(span.split()) == " ".join(quote.split())
        altered = list(words[i:i + 8])
        altered[3] += "x"
        s = v.check_sentence({"text": "x", "citations": [{"passage_id": pid, "quote": " ".join(altered)}]})
        caught += not s.kept
    print(f"passages tested={tested} reflowed quotes found={found} "
          f"mapped exactly to the page={mapped} altered quotes caught={caught}")
    return 0 if tested == found == mapped == caught else 1


if __name__ == "__main__":
    sys.exit(main())
