# Reviewer screen, before and after

Captured on 25 November 2026 with headless Chromium (Playwright) at 1440 by 900, then saved as JPEG at 1280 pixels wide to keep the repository small. Every screen is the real app answering real questions: Dalil running on a laptop, fully offline, NVIDIA Nemotron 3 Nano 4B drafting and Nemotron Embed 1B searching, and no open-data service. Each capture set started from an empty store. The reviewer name is a placeholder.

`after/citation.jpg` and `after/arabic.jpg` were taken later the same day on the final pipeline, with `scripts/run_offline.sh`; `after/open-data-citation.jpg` and `after/uk-knowledge-base.jpg` by the open-data branch's session on its merged code. `before/` is the screen at commit `49e8e0e`, the code shown in the 25 November video. `after/` is the screen in this branch. The answers differ between the two because the model drafted them again; some are wrong in ways a reviewer should catch, and they are left as the model wrote them.

| State | Before | After |
|---|---|---|
| Main screen, an answer with citations | `before/main.jpg` | `after/main.jpg` |
| Citation 1 opened on its page: the Golden Visa question, answered from the Golden Visa page only | | `after/citation.jpg` |
| A sentence cut by the checker, with its reason | | `after/cut.jpg` |
| Audit trail | `before/audit.jpg` | `after/audit.jpg` |
| An Arabic question and answer, citation opened. After: the answer quotes Article 2(2) of the PDPL in full, in Arabic, and the source reads in order | `before/arabic.jpg` | `after/arabic.jpg` |
| The stage strip while the writer runs | | `after/live-writer-running.jpg` |
| The data-protection question: the two scope articles added ahead of the ranked results, labelled as added | | `after/hard-scope-articles.jpg` |
| A numbers question answered from the open data on the local database, with the rerun query and its row | | `after/open-data-citation.jpg` |
| The knowledge base switched to the UK base, answering from a GOV.UK guide | | `after/uk-knowledge-base.jpg` |

The scripts that took the full set, all seven ITU scenarios included, are kept outside the repository with the demo kit.
