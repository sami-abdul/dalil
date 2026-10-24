"""Dense embeddings from an OpenAI-compatible /v1/embeddings endpoint.

By default this is NVIDIA Nemotron-3-Embed-1B served locally by llama.cpp's
llama-server with --embeddings, on 127.0.0.1. The model expects "query: " and
"passage: " prefixes, which this client adds. Passage vectors are cached on
disk under build/embeddings/, keyed by the model name and a hash of each text,
so the corpus is embedded once.

    DALIL_EMBED_URL    default http://127.0.0.1:8792/v1
    DALIL_EMBED_MODEL  default "nemotron-embed"
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Sequence
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "build" / "embeddings"


class HTTPEmbedder:
    def __init__(self, url: str | None = None, model: str | None = None, timeout: float = 300.0,
                 query_prefix: str = "query: ", passage_prefix: str = "passage: ",
                 cache_dir: Path | None = CACHE_DIR):
        self.url = (url or os.environ.get("DALIL_EMBED_URL") or "http://127.0.0.1:8792/v1").rstrip("/")
        self.model = model or os.environ.get("DALIL_EMBED_MODEL") or "nemotron-embed"
        self.timeout = timeout
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model)
        self.cache_file = None if cache_dir is None else Path(cache_dir) / f"{safe}.json"
        self._cache: dict[str, list[float]] | None = None

    def _post(self, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        req = Request(f"{self.url}/embeddings", data=body, headers={"content-type": "application/json"})
        with urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode("utf-8"))
        data = sorted(out["data"], key=lambda d: d["index"])
        if len(data) != len(texts):
            raise ValueError("embedding endpoint returned the wrong number of vectors")
        return [d["embedding"] for d in data]

    def available(self) -> bool:
        try:
            return len(self._post(["passage: ping"])) == 1
        except (OSError, ValueError, KeyError):
            return False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._post(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._post([self.query_prefix + text])[0]

    def embed_passages(self, texts: Sequence[str], batch_size: int = 16) -> list[list[float]]:
        cache = self._load()
        # The prefix is part of what was embedded, so it is part of the key.
        keys = [hashlib.sha256((self.passage_prefix + t).encode("utf-8")).hexdigest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            for i, vec in zip(batch, self._post([self.passage_prefix + texts[i] for i in batch])):
                cache[keys[i]] = vec
        if missing:
            self._save()
        return [cache[k] for k in keys]

    def _load(self) -> dict[str, list[float]]:
        if self._cache is None:
            self._cache = {}
            if self.cache_file is not None and self.cache_file.is_file():
                self._cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
        return self._cache

    def _save(self) -> None:
        if self.cache_file is None or self._cache is None:
            return
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache), encoding="utf-8")
        tmp.replace(self.cache_file)
