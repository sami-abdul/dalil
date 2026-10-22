# Dalil

Answers about UAE government services, with every sentence tied to the official text it came from. A person approves each answer before it goes out.

Built for the Open Agent Hackathon 2026, Track 02, "The Agent That Can Explain Why", during the build window of 22 to 27 October 2026.

## What it does

1. You ask a question about a UAE government service.
2. Dalil screens it. Emirates ID, passport, phone and email details are redacted before anything is stored, and requests to get around the rules get a refusal. Reviewers' names and notes are redacted the same way.
3. It recalls past reviews from H-MEM, Meterless's memory engine. For a question that shares most of its key words with an earlier one, it leaves out the passages a reviewer rejected there, unless an approved answer to a similar question used them.
4. It searches 1,024 passages from 45 official documents, and runs fixed open-data queries through Zetaris when the question needs numbers.
5. NVIDIA Nemotron drafts an answer. The model sees the sources as numbered sentences and may only point at those ids or at the queries Dalil ran, enforced by a JSON schema; Dalil then writes each quote itself, so quotes are exact. If the model is unavailable, an extractive drafter quotes source sentences instead, and the audit trail records which one ran.
6. A checker tests every citation. A quote must appear word for word in a passage the drafter was given. Every number in a sentence, in digits or words, must appear in its evidence, and values from open data must come from one row of a rerun query. Most of a claim's key words must be in its sources, and it may not add a negation they lack. Sentences that fail are cut, and an answer that loses too much is held. The checker cannot judge meaning, so the reviewer reads every answer.
7. A reviewer sees the answer, the exact passage behind each citation highlighted on its page, the sentences that were cut and why, and the audit trail. Nothing goes out until they approve it.
8. The decision is saved to H-MEM's trust ledger with the passages the answer cited. A rejection keeps those passages out of the next answer to a similar question, and its note is passed to the Nemotron drafter.

![Reviewer screen](docs/reviewer-screen.png)

## How it fits together

```mermaid
flowchart LR
    Q[Resident question] --> S[Screen: redact personal data,<br/>refuse rule-dodging]
    S --> R[Recall similar reviews<br/>Meterless H-MEM]
    R --> F[Search 1,024 passages<br/>BM25]
    S --> Z[Fixed open-data queries<br/>Zetaris]
    F --> D[Draft: every sentence cites<br/>a passage or a Zetaris query]
    Z --> D
    R -. reviewer notes .-> D
    D --> C[Checker: quotes word for word,<br/>numbers in evidence, queries rerun]
    C --> P[Reviewer screen:<br/>approve or reject]
    P -->|approve| O[Answer sent]
    P -->|decision and cited passages| L[(H-MEM trust ledger)]
    L --> R
    subgraph ZDB["Zetaris Lightning database: dalil"]
        T1[documents, pages, passages]
        T2[four Bayanat open datasets]
    end
    F -. same passages .- T1
    Z --- T2
    C -. reruns .- T2
```


## Run it

You need Python 3.13 with [uv](https://docs.astral.sh/uv/), Node 22.7 or later for the H-MEM bridge, and a Zetaris instance. This was built against Zetaris's free local version, 2.4.3.1.

```bash
uv sync
uv run python scripts/fetch_corpus.py            # download the documents into corpus/, for your own use
uv run python scripts/extract_corpus.py          # text per page -> build/documents.jsonl
uv run python scripts/build_passages.py          # passages with exact offsets -> build/passages.jsonl
cp .env.example .env                             # then fill in the Zetaris settings
uv run python scripts/load_zetaris.py            # documents, pages, passages and open data into Zetaris
uv run python scripts/check_zetaris.py           # counts, and a citation query rerun inside Zetaris
LLAMA_SERVER=... MODEL=... scripts/serve_nemotron.sh   # Nemotron 3 Nano 4B on 127.0.0.1:8790
uv run python scripts/serve.py                   # app on http://127.0.0.1:8787, H-MEM bridge on 8788
```

The screen also runs on invented sample data with no backend: serve `web/` and open it with `?mock=1`.

Checks:

```bash
uv run pytest                                    # unit tests, synthetic text only
uv run python scripts/selftest_verify.py         # the checker against every real passage
uv run python scripts/eval_retrieval.py          # retrieval on the ten test questions
uv run python scripts/run_eval.py                # every test question and error scenario end to end
```

## Built with

| Part | What |
|---|---|
| Zetaris | Federated SQL over the documents, pages, passages and four Bayanat open datasets, all in one Lightning database. Citations to open data are Zetaris queries that the checker reruns |
| Meterless | H-MEM, vendored unchanged from Meterless's reference implementation, keeps approved answers and reviewers' rejections with their provenance in its append-only trust ledger |
| NVIDIA | NVIDIA Nemotron 3 Nano 4B drafts the answers, running locally through llama.cpp from NVIDIA's own GGUF, under the NVIDIA Nemotron Open Model License. The drafter speaks the OpenAI-compatible API, so NVIDIA's hosted endpoint can replace the local server by setting `DALIL_LLM_URL`, `DALIL_LLM_KEY` and `DALIL_LLM_MODEL`; that path has not been tried yet |
| Claude Code | Wrote the code; see [docs/tool-log.md](docs/tool-log.md) |

## Not done yet

- **Dense retrieval.** Search is BM25 for now. `dalil/retrieve.py` has the hook for Nemotron embeddings.
- **Speed.** On the build machine's CPU, Nemotron 3 Nano 4B takes about 40 to 55 seconds per answer: it reads the prompt at 65 to 80 tokens a second and writes at 6 to 8. A GPU would be far faster.
- **Model size.** A 4B model sometimes picks a correct but off-topic sentence, or mixes the rules of two related services. Every quote and number is still checked, and the reviewer catches the rest.

## Data

See [data/README.md](data/README.md). The documents are public UAE government publications, but most of them may not be republished, so this repo holds the manifest and a fetch script, not the files.

## Pre-existing work

Declared under rule 4 of the hackathon rules:

- The document collection and its manifest, `data/MANIFEST.csv`, gathered from public sources before 22 October 2026.
- Planning notes with no code.
- A local install of Zetaris's free version, set up before the window from Zetaris's own bundle.

All code in this repo was written during the build window. [docs/tool-log.md](docs/tool-log.md) records which tool wrote what.

## Licence

Apache 2.0 for the code. The documents keep their publishers' terms.
