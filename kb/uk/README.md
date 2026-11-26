# UK knowledge base (GOV.UK)

A second, small knowledge base that shows Dalil adapting to another country with no code change. Seven GOV.UK guides, cut into 92 passages by the same steps that build the UAE knowledge base.

| File | Title | Source |
|---|---|---|
| govuk-apply-national-insurance-number.html | Apply for a National Insurance number | https://www.gov.uk/apply-national-insurance-number |
| govuk-find-your-national-insurance-number.html | Find your National Insurance number | https://www.gov.uk/lost-national-insurance-number |
| govuk-renew-adult-passport.html | Renew or replace your adult passport | https://www.gov.uk/renew-adult-passport/print |
| govuk-handing-in-your-notice.html | Handing in your notice | https://www.gov.uk/handing-in-your-notice/print |
| govuk-holiday-entitlement.html | Holiday entitlement | https://www.gov.uk/holiday-entitlement-rights/print |
| govuk-statutory-sick-pay.html | Statutory Sick Pay (SSP) | https://www.gov.uk/statutory-sick-pay/print |
| govuk-redundancy-your-rights.html | Redundancy: your rights | https://www.gov.uk/redundant-your-rights/print |

Guides with several parts are taken from their printable version, which holds every part on one page.

## Licence and attribution

Contains public sector information licensed under the Open Government Licence v3.0.

Each page's footer states that its content is available under the Open Government Licence v3.0, except where otherwise stated, and GOV.UK's terms (https://www.gov.uk/help/terms-conditions) say content published under the OGL can be reproduced as long as the licence's conditions are followed. The licence (https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) allows copying, publishing and adapting the information with that attribution statement. It does not cover logos, crests or the Royal Arms, and the downloaded HTML carries the GOV.UK crown, so the HTML is not committed: only the text extracted from it is.

The Apache 2.0 licence on Dalil's code does not cover this folder's text.

## What is here

- `profile.json`: the profile. Its name, paths, the ID formats the screener removes (National Insurance, NHS and UK phone and passport numbers, and email), and the fixed refusal and caveat sentences in UK wording. No open data, and no review memory, so UAE reviews never shape UK answers.
- `MANIFEST.csv`: the seven documents, with source URL, retrieval time and the SHA-256 of the HTML as downloaded.
- `index/documents.jsonl`: page text extracted from the HTML by `scripts/extract_corpus.py`.
- `index/passages.jsonl`: the 92 citable passages cut by `scripts/build_passages.py`.
- `scenarios.json`: six questions with their expected outcomes, written before the first run.

## Rebuild

From the repository root:

```bash
uv run python scripts/fetch_corpus.py --manifest kb/uk/MANIFEST.csv --corpus-dir build/kb/uk/corpus
uv run python scripts/extract_corpus.py --manifest kb/uk/MANIFEST.csv --corpus-dir build/kb/uk/corpus --out kb/uk/index/documents.jsonl
uv run python scripts/build_passages.py --documents kb/uk/index/documents.jsonl --out kb/uk/index/passages.jsonl
```

GOV.UK pages change. When a download's hash differs from the manifest, the fetcher keeps it and says so, and the extracted text may differ from the committed copy. To take the new version, put the new hashes and times in `MANIFEST.csv` and commit the rebuilt `index/` files.

The original HTML lands in `build/kb/uk/corpus/`, which git ignores; the reviewer screen serves it from there when it is present.

## Use it

```bash
DALIL_KB=uk uv run python scripts/serve.py              # start on the UK knowledge base
uv run python scripts/run_scenarios.py kb/uk/scenarios.json --kb uk
```

Or start as usual and pick it from the knowledge-base menu at the top of the reviewer screen.
