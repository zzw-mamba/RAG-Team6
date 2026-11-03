"""Question-text routing only: no IDs, labels, cached answers or scorer access."""
from __future__ import annotations

import re

BOOLEAN_PREFIX = re.compile(
    r'^\s*(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b',
    re.IGNORECASE,
)


def route_config(cfg, question):
    if not cfg.get('question_router'):
        return cfg, 'uniform'
    if BOOLEAN_PREFIX.match(question):
        # Reproduce the previous planned policy, including answer order and retrieval.
        return {**cfg, 'retrieval_policy': 'coverage', 'max_documents': 32,
                'document_chars': 4000, 'global_top_k': 8, 'answer_tokens': 384,
                'answer_order': 'answer_first', 'allow_model_knowledge': True,
                'retrieval_plan_rounds': 1, 'retrieval_plan_trigger': 'always',
                'reranker_path': None}, 'boolean_legacy'
    return cfg, 'entity_hybrid'
