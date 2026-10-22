"""Tests for the embedding client, with the endpoint faked."""

from __future__ import annotations

from dalil.embed import HTTPEmbedder


class Fake(HTTPEmbedder):
    def __init__(self, cache_dir):
        super().__init__(url="http://127.0.0.1:1/v1", model="fake", cache_dir=cache_dir)
        self.sent = []

    def _post(self, texts):
        self.sent.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


def test_prefixes_are_added(tmp_path):
    e = Fake(tmp_path)
    e.embed_query("renew id")
    e.embed_passages(["a passage"])
    assert e.sent == [["query: renew id"], ["passage: a passage"]]


def test_passage_vectors_are_cached_on_disk(tmp_path):
    first = Fake(tmp_path)
    v1 = first.embed_passages(["one", "two"], batch_size=1)
    assert len(first.sent) == 2 and (tmp_path / "fake.json").is_file()
    again = Fake(tmp_path)
    assert again.embed_passages(["two", "one"]) == [v1[1], v1[0]]
    assert again.sent == []  # nothing re-embedded
    again.embed_passages(["one", "three"])
    assert again.sent == [["passage: three"]]
