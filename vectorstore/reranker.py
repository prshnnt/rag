"""BM25 reranker over retrieved chunks.

Pure Python, no extra deps. Tokenize on lowercase words; IDF = log((N - df + 0.5)/(df + 0.5) + 1).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_lens = [len(_tokenize(d)) for d in docs]
        self.avgdl = (sum(self.doc_lens) / len(docs)) if docs else 0.0
        self.df: dict[str, int] = defaultdict(int)
        self.tf: list[Counter] = []
        for d in docs:
            tokens = _tokenize(d)
            c = Counter(tokens)
            self.tf.append(c)
            for term in c:
                self.df[term] += 1
        self.N = len(docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(((self.N - df + 0.5) / (df + 0.5)) + 1.0)

    def score(self, query: str, doc_index: int) -> float:
        if self.avgdl == 0:
            return 0.0
        q_tokens = _tokenize(query)
        tf = self.tf[doc_index]
        dl = self.doc_lens[doc_index] or 1
        s = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            f = tf[t]
            idf = self._idf(t)
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * (num / den)
        return s

    def rank(self, query: str, top_k: int | None = None) -> list[int]:
        scores = [self.score(query, i) for i in range(self.N)]
        order = sorted(range(self.N), key=lambda i: scores[i], reverse=True)
        if top_k is not None:
            order = order[:top_k]
        return order


def rerank(query: str, chunks: list[dict], top_k: int | None = None,
           text_key: str = "text") -> list[dict]:
    """Rerank chunks by BM25 score. `chunks` items must have text_key."""
    if not chunks:
        return []
    docs = [c[text_key] for c in chunks]
    bm = BM25(docs)
    order = bm.rank(query, top_k=top_k)
    return [chunks[i] for i in order]
