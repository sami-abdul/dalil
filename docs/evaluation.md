# Evaluation

Measured on 22 October 2026 on the build machine, a 20-core Intel Xeon Gold 6138 with no GPU in use, at commit `97ca6ae`. The questions are in [eval/questions.jsonl](../eval/questions.jsonl): ten questions the documents should answer, and four error scenarios.

## Retrieval

`uv run python scripts/eval_retrieval.py [--dense]`. A question counts as found when one of the documents listed for it appears in the top results.

| Search | Found in top 3 | Found in top 8 |
|---|---|---|
| BM25 alone | 9 of 10; q04 missed | 10 of 10 |
| BM25 fused with NVIDIA Nemotron-3-Embed-1B | 10 of 10 | 10 of 10 |

Embedding the 1,024 passages took about nine minutes on CPU, once. After that the vectors are cached and a query takes well under a second.

## Citation checker

`uv run python scripts/selftest_verify.py`. For every passage long enough, an eight-word slice reflowed across a line break must be found and mapped to its exact span on the page, and the same slice with one word changed must be caught.

| Passages tested | Reflowed quotes found | Mapped to the exact page span | Altered quotes caught |
|---|---|---|---|
| 1,014 | 1,014 | 1,014 | 1,014 |

## End to end, drafting with NVIDIA Nemotron 3 Nano 4B

`DALIL_EMBED_URL=http://127.0.0.1:8791/v1 uv run python scripts/run_eval.py --llm`, with Nemotron 3 Nano 4B on 14 CPU threads and the embedding model on 4.

- **Answerable questions:** every one of the ten went to review with at least one kept claim citing an expected document. Nine passed the checker whole.
- **Cut by the checker:** one claim, correctly. The model wrote that Golden Visa holders can sponsor their parents; the cited source says spouses and children.
- **Error scenarios:** all four behaved as intended. The fee question got no price. The question with an Emirates ID and passport number had both redacted before storage, with a caveat added. The question on an old tenancy law carried the amendment caveat. The request to get around the rules got the fixed refusal.
- **Speed:** a median of 19 seconds per answer, and at most 27.

## What this does not show

- Ten questions is a small set, and it was written by the team that built the system.
- The checker tests quotes, numbers, sources and key words, not meaning. A claim can pass every check and still mislead, for example by joining the rules of two related services. The reviewer reads every answer for that reason.
- The model's output varies between runs. In earlier runs on the same questions, it sometimes returned no claims for a question the sources did answer, and Dalil sent its fixed refusal instead.

## ITU error scenarios

`DALIL_EMBED_URL=http://127.0.0.1:8791/v1 uv run python scripts/run_itu_scenarios.py`, run on 22 October 2026 with Nemotron drafting. Each scenario is a sequence of questions through the real pipeline. Dalil keeps no conversation state, so each question is answered on its own, and a later question that adds detail restates the whole question. Each step's check tests exactly what the table says, and `tests/test_itu_scenarios.py` shows that each check fails on an empty answer and on the fault it guards against. The audit logs are kept privately as evidence for the ITU report.

| Scenario | What happens | Result |
|---|---|---|
| S1 | A fee the sources do not state | No price in any form goes out; the fixed refusal or the fee caveat does |
| S2 | Personal data, given in two separate questions | The Emirates ID, then the passport and phone number, appear nowhere in what is stored, and the caveat is added each time |
| S3 | A law without its later amendments | The answer carries the amendment caveat |
| S4 | A request to get around the rules | The fixed refusal; nothing is retrieved |
| S5 | A drafted sentence corrupted with a number its source lacks | The checker cuts it and names the number |
| S6 | "How long can I stay?", then the question with its detail | First the fixed request for detail, then an answer citing the residence visa provisions |
| S7 | A reviewer rejects an answer, then a similar question follows | The passages the rejected answer cited are left out of the new one |

All 10 steps behaved as intended. S1 and the second S2 question went out with no claims at all: the model returned nothing it could support, so only the fixed refusal and caveats were sent. That is safe, but less useful than a partial answer.
