from __future__ import annotations
from collections.abc import Iterator
import random

import bm25s
import fastjsonschema
import json


json_dtype = dict | list | str | int | float | bool | None


def remove_json_values(json_content: json_dtype) -> json_dtype:
    if json_content is None:
        return None

    if isinstance(json_content, (str, int, float, bool)):
        return ""

    if isinstance(json_content, dict):
        return {k: remove_json_values(v) for k, v in json_content.items()}
    elif isinstance(json_content, list):
        return [remove_json_values(v) for v in json_content]
    else:
        return json_content


def format_json_for_indexing(
    json_content: json_dtype,
) -> str:
    """Recursively traverse the thing, replace all values with \"\" and then json.dumps the result"""
    json_content = remove_json_values(json_content)
    return json.dumps(json_content, indent=1).lower()


class IndexedJsonSet:
    def __init__(
        self,
        json_contents: list[json_dtype] | None = None,
        ids: list[int] | None = None,
    ) -> None:
        self.json_contents: list[json_dtype] = json_contents or []
        self.ids: list[int] = ids or []
        self.corpus_text: list[str] = [
            format_json_for_indexing(json_content)
            for json_content in self.json_contents
        ]
        self.retriever = bm25s.BM25(backend="auto")
        if self.corpus_text:
            self.build_index()

    def build_index(self) -> None:
        tokenized_corpus = self.tokenize(self.corpus_text)
        self.retriever.index(tokenized_corpus, show_progress=False)

    def tokenize(self, text: str | list[str]) -> list[str]:
        return bm25s.tokenize(
            text,
            stemmer=None,
            show_progress=False,
            stopwords=None,
            token_pattern=r"(?u)((?:[a-zA-Z0-9]+)|(?:[^\w]))",
            allow_empty=False,
        )

    def add(self, json_content: json_dtype, id: int) -> None:
        self.json_contents.append(json_content)
        self.ids.append(id)
        json_text = format_json_for_indexing(json_content)
        self.corpus_text.append(json_text)
        self.build_index()

    def add_many(self, json_contents: list[json_dtype], ids: list[int]) -> None:
        self.ids.extend(ids)
        self.json_contents.extend(json_contents)
        json_texts = [
            format_json_for_indexing(json_content) for json_content in json_contents
        ]
        self.corpus_text.extend(json_texts)
        self.build_index()

    def query(
        self,
        query_json: json_dtype,
        k: int = 5,
        n_threads: int = 4,
    ) -> list[tuple[json_dtype, int, float]]:
        k = min(k, len(self.json_contents) - 1)
        query_text = format_json_for_indexing(query_json)
        tokenized_query = self.tokenize([query_text])

        json_idxs, scores = self.retriever.retrieve(
            tokenized_query,
            k=k,
            show_progress=False,
            n_threads=n_threads,
        )

        ranked_json_contents: list[tuple[json_dtype, int, float]] = []
        seen_ids = set()
        for i in range(json_idxs.shape[1]):
            json_idx = json_idxs[0, i]
            score = scores[0, i]
            json_content, json_id = self.json_contents[json_idx], self.ids[json_idx]
            if json_id not in seen_ids:
                seen_ids.add(json_id)
                ranked_json_contents.append((json_content, json_id, score))
        return ranked_json_contents

    def __len__(self) -> int:
        return len(self.json_contents)

    def __iter__(self) -> Iterator[json_dtype]:
        return iter(self.json_contents)
