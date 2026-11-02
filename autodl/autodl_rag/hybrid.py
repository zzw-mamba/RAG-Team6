"""Public-text-only title recall and per-hop rank fusion; no external corpus."""
from __future__ import annotations

import re
from collections import Counter

from .retrieval import bm25_scores, focus_text, tokens


def entity_queries(question, contexts):
    # Actual supplied titles preserve punctuation, accents and disambiguating years.
    found = [d['title'].strip() for d in contexts
             if d['title'].strip() and d['title'].casefold() in question.casefold()]
    found.sort(key=lambda x: (-len(x), x))
    quoted = re.findall(r'["“]([^"”]{2,100})["”]', question)
    names = re.findall(r"\b[A-ZÀ-Ž][\w’'\-]*(?:\s+(?:(?:of|the|de|van)\s+)?[A-ZÀ-Ž][\w’'\-]*){1,5}(?:\s*\(\d{4}[^)]*\))?", question)
    cleaned = []
    for name in [*found, *quoted, *names]:
        name = re.sub(r'^(?:Who|What|Which|Where|When|How|Was|Were|Is|Are|Did)\s+', '', name)
        if len(tokens(name)) and name not in cleaned:
            cleaned.append(name)
    return cleaned[:3]


def fuse(channels):
    scores, docs = {}, {}
    for weight, ranking in channels:
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            key = doc['id']
            if key in seen:
                continue
            seen.add(key)
            if key not in docs or len(doc['text']) > len(docs[key]['text']):
                docs[key] = doc
            scores[key] = scores.get(key, 0.0) + weight / (20 + rank)
    return [docs[k] for k in sorted(docs, key=lambda k: (-scores[k], k))]


def ranked(query, docs):
    return [d for _, d in sorted(zip(bm25_scores(query, docs), docs), key=lambda x: -x[0])]


def retrieve(retriever, question, contexts, extra_queries, previous):
    cfg = retriever.cfg
    queries = list(dict.fromkeys([question, *extra_queries]))[:4]
    entities = entity_queries(question, contexts)
    pools = []
    for query in queries:
        exact = retriever.title_search(query)
        broad = retriever.focused_search(query)
        title_hits = [d for name in (entities if query == question else entity_queries(query, []))
                      for d in retriever.title_search(name)]
        pool = list({d['id']: d for d in [*contexts, *previous, *exact, *title_hits, *broad]}.values())
        # Bound pairwise reranking cost. All local contexts stay in the lexical candidate set.
        lexical = ranked(query, pool)
        channels = [(1.0, lexical), (1.5, exact), (1.5, title_hits)]
        if retriever.reranker:
            candidates = fuse(channels)[:cfg.get('rerank_candidates', 48)]
            scores = retriever.reranker.score(query, candidates)
            cross = [d for _, d in sorted(zip(scores, candidates), key=lambda x: -x[0])]
            channels.append((2.0, cross))
        pools.append(fuse(channels))
    chosen, seen, titles = [], set(), Counter()
    cap = cfg['max_documents']

    def add(doc):
        title = doc['title'].strip().casefold() or doc['id']
        if doc['id'] not in seen and len(chosen) < cap and titles[title] < 2:
            seen.add(doc['id'])
            titles[title] += 1
            chosen.append(doc)

    # Reserve local evidence and each missing-hop branch before filling by fusion.
    for doc in ranked(question, contexts)[:4]:
        add(doc)
    for rank in range(5):
        for i, pool in enumerate(pools):
            if rank < (5 if i == 0 else 3) and rank < len(pool):
                add(pool[rank])
    for doc in fuse([(1.0, pool) for pool in pools]):
        add(doc)
    return [{**d, 'text': focus_text(d['text'], ' '.join(queries), cfg['document_chars'])}
            for d in chosen]
