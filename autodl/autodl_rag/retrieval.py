from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

from .core import fingerprint, read_json, records, save_json
from .data import prepared

STOP = set(
    "a an the is are was were of in at on by for to and or which what who where when how did does do born".split()
)


def tokens(text):
    return [
        x for x in re.findall(r"\w+", text.casefold()) if x not in STOP and len(x) > 1
    ]


def build_index(cfg):
    folder = prepared(cfg)
    dest = Path(cfg["work_dir"]) / "corpus.sqlite"
    signature = fingerprint(read_json(folder / "manifest.json"))
    info_path = dest.with_suffix(".manifest.json")
    if dest.exists() and info_path.exists():
        info = read_json(info_path)
        if info["prepared_signature"] != signature:
            raise ValueError("Index has a different data signature")
        return info
    building = dest.with_suffix(".building")
    building.unlink(missing_ok=True)
    con = sqlite3.connect(building)
    try:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute(
            "CREATE TABLE docs(id TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL)"
        )
        con.execute(
            "CREATE VIRTUAL TABLE search USING fts5(title, text, content='docs', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2')"
        )
        count = 0
        for i, row in enumerate(records(folder / "train.jsonl")):
            for d in row["ctxs"]:
                con.execute(
                    "INSERT OR IGNORE INTO docs VALUES (?,?,?)",
                    (d["id"], d["title"], d["text"]),
                )
            if (i + 1) % 5000 == 0:
                con.commit()
                print(f"Indexed contexts from {i + 1:,} training questions", flush=True)
        con.execute("INSERT INTO search(search) VALUES ('rebuild')")
        con.execute("INSERT INTO search(search) VALUES ('optimize')")
        count = con.execute("SELECT count(*) FROM docs").fetchone()[0]
        con.commit()
    finally:
        con.close()
    building.replace(dest)
    info = {
        "documents": count,
        "prepared_signature": signature,
        "source": "ONLY prepared/train.jsonl contexts; no answers/rationale/test/held-out contexts",
    }
    save_json(info_path, info)
    return info


def bm25_scores(query, docs):
    if not docs:
        return []
    all_tokens = [tokens(d["title"] + " " + d["title"] + " " + d["text"]) for d in docs]
    counts = [Counter(ts) for ts in all_tokens]
    df = Counter(t for c in counts for t in c)
    avglen = max(1, sum(map(len, all_tokens)) / len(docs))
    result = []
    qtokens = set(tokens(query))
    for d, ts, c in zip(docs, all_tokens, counts):
        score = 0.0
        for t in qtokens:
            freq = c[t]
            idf = math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * len(ts) / avglen))
        title_overlap = len(qtokens & set(tokens(d["title"])))
        result.append(score + 0.4 * title_overlap)
    return result


def focus_text(text, query, limit):
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    qt = set(tokens(query))
    ranked = sorted(
        range(len(sentences)), key=lambda i: (-len(qt & set(tokens(sentences[i]))), i)
    )
    chosen, total = set(), 0
    # Preserve lead identity and question-related sentences, in original order.
    for i in [0] + ranked:
        if i in chosen:
            continue
        if total + len(sentences[i]) + 1 <= limit:
            chosen.add(i)
            total += len(sentences[i]) + 1
    if not chosen:
        return text[:limit]
    return " ".join(sentences[i] for i in sorted(chosen))


class Retriever:
    def __init__(self, cfg):
        self.cfg = cfg
        db = Path(cfg["work_dir"]) / "corpus.sqlite"
        if not db.exists() or not db.with_suffix(".manifest.json").exists():
            raise RuntimeError("Run index before retrieval")
        if read_json(db.with_suffix(".manifest.json"))[
            "prepared_signature"
        ] != fingerprint(read_json(prepared(cfg) / "manifest.json")):
            raise ValueError("Index does not match prepared data")
        self.db = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
        self.reranker = None
        if cfg.get("reranker_path"):
            from .models import Reranker

            self.reranker = Reranker(cfg)

    def search(self, query, k=None):
        k = self.cfg["global_top_k"] if k is None else k
        words = list(dict.fromkeys(tokens(query)))[:24]
        if not words or k <= 0:
            return []
        expression = " OR ".join('"' + w.replace('"', '""') + '"' for w in words)
        rows = self.db.execute(
            "SELECT d.id,d.title,d.text FROM search JOIN docs d ON d.rowid=search.rowid "
            "WHERE search MATCH ? ORDER BY bm25(search,3.0,1.0) LIMIT ?",
            (expression, k),
        ).fetchall()
        return [dict(zip(("id", "title", "text"), r)) for r in rows]

    def retrieve(self, question, contexts, extra_queries=(), previous=()):
        # Keep provided contexts in the candidate set; global retrieval supplies missing hops.
        pool = {}
        for d in [*contexts, *previous, *self.search(question)]:
            if d["id"] not in pool or len(d["text"]) > len(pool[d["id"]]["text"]):
                pool[d["id"]] = d
        scores = bm25_scores(question, list(pool.values()))
        leading = sorted(zip(scores, pool.values()), key=lambda x: -x[0])[:3]
        bridge_entities = []
        for _, d in leading:
            for ent in re.findall(
                r"\b[A-Z][\w’'\-]+(?:\s+[A-Z][\w’'\-]+){1,3}", d["text"]
            ):
                if (
                    ent.casefold() not in question.casefold()
                    and ent not in bridge_entities
                ):
                    bridge_entities.append(ent)
        queries = list(extra_queries)[:2] + bridge_entities[:2]
        for query in queries:
            for d in self.search(query, min(3, self.cfg["global_top_k"])):
                pool[d["id"]] = d
        docs = list(pool.values())
        scores = bm25_scores(question + " " + " ".join(extra_queries), docs)
        if self.reranker:
            scores = self.reranker.score(question, docs)
        ranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: -x[0])]
        # A reserved local quota protects bridge evidence with low question-word overlap.
        local_ids = {d["id"] for d in contexts}
        locals_ = [d for d in ranked if d["id"] in local_ids][: self.cfg["local_top_k"]]
        rest = [d for d in ranked if d["id"] not in {x["id"] for x in locals_}]
        chosen = locals_ + rest[: max(0, self.cfg["max_documents"] - len(locals_))]
        return [
            {
                **d,
                "text": focus_text(
                    d["text"],
                    question + " " + " ".join(extra_queries),
                    self.cfg["document_chars"],
                ),
            }
            for d in chosen
        ]
