# ITU reference code, vendored

Copied unchanged from https://github.com/CrashingGuru/ITUAIReadiness at commit `84010da`, 20 October 2026, under its MIT licence; see `LICENSE`:

- `server/config.py`, `server/__init__.py`
- `server/knowledge/kb.py`, `server/knowledge/ingest.py`, `server/knowledge/__init__.py`
- `data/framework/dimensions.json`, ITU's 13 AI readiness dimensions and 6 factors

Dalil's `scripts/build_itu_kb.py` stores its knowledge base with ITU's `KnowledgeBase` class and loads the framework with ITU's own `ingest_framework_definitions`. The one thing it swaps at run time is ITU's `embed_texts`, which calls Ollama, for Dalil's local Nemotron embedder. Nothing in these files was modified.
