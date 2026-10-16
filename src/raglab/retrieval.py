from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace

from raglab.schema import Document


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def retrieve(
    question: str, documents: tuple[Document, ...], top_k: int, method: str = "bm25"
) -> tuple[Document, ...]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if method == "all":
        return documents[:top_k]
    if method != "bm25":
        raise ValueError(f"Unknown retriever: {method}")
    tokens = [tokenize(f"{d.title} {d.text}") for d in documents]
    counts = [Counter(t) for t in tokens]
    average = sum(map(len, tokens)) / max(len(tokens), 1) or 1.0
    frequencies = Counter(token for t in tokens for token in set(t))
    scores = []
    for doc, words, counter in zip(documents, tokens, counts, strict=True):
        score = 0.0
        for token in set(tokenize(question)):
            tf = counter[token]
            df = frequencies[token]
            idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(words) / average))
        scores.append(replace(doc, score=score))
    return tuple(sorted(scores, key=lambda d: -(d.score or 0))[:top_k])
