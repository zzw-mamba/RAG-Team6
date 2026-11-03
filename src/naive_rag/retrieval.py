import math
import re
from collections import Counter


def tokens(text):
    return re.findall(r"\w+", text.lower(), re.UNICODE)


def retrieve(question, docs, top_k, mode="bm25"):
    """Select pre-retrieved candidates, optionally reranking with per-question BM25."""
    if mode == "input":
        return [{**d, "bm25_score": None, "source_index": i}
                for i, d in enumerate(docs[:top_k])]
    if mode != "bm25":
        raise ValueError("retrieval_mode must be bm25 or input")
    counts = [Counter(tokens(d["title"] + " " + d["text"])) for d in docs]
    lengths = [sum(c.values()) for c in counts]
    avg = sum(lengths) / len(lengths) or 1
    query = set(tokens(question))
    df = {t: sum(t in c for c in counts) for t in query}
    scored = []
    for i, (d, c, length) in enumerate(zip(docs, counts, lengths)):
        score = sum(
            math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
            * c[t] * 2.5 / (c[t] + 1.5 * (0.25 + 0.75 * length / avg))
            for t in query if c[t]
        )
        scored.append({**d, "bm25_score": score, "source_index": i})
    return sorted(scored, key=lambda d: (-d["bm25_score"], d["source_index"]))[:top_k]
