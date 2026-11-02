"""Synthetic policy/format regression tests, never quality claims."""
import unittest
from unittest.mock import Mock

from autodl_rag.optimize import profiles
from autodl_rag.pipeline import Pipeline, answer_prompt
from autodl_rag.retrieval import Retriever
from autodl_rag.structured import qwen_schema


class ScoreV3Tests(unittest.TestCase):
    def test_local_contexts_cannot_be_evicted_or_polluted_by_global_search(self):
        r = Retriever.__new__(Retriever)
        r.cfg = {'retrieval_policy': 'provided', 'max_documents': 1}
        r.search = Mock(side_effect=AssertionError('global search forbidden'))
        docs = [{'id': str(i), 'title': str(i), 'text': 'bridge fact'} for i in range(40)]
        self.assertEqual(r.retrieve('question', docs), docs)
        r.search.assert_not_called()

    def test_order_propagates_to_payload_and_bounded_schema(self):
        p = Pipeline.__new__(Pipeline)
        p.cfg = {'answer_order': 'evidence_first'}
        payload = p.payload({'question': 'Public question', 'answers': ['SECRET']}, [])
        self.assertNotIn('SECRET', str(payload))
        schema = qwen_schema(payload, set())
        self.assertEqual(schema['required'], ['explanation', 'answer', 'evidence_ids'])
        self.assertEqual(schema['properties']['explanation']['maxLength'], 320)
        self.assertEqual(qwen_schema({}, set())['required'][0], 'answer')

    def test_suite_ablation_keeps_reasoning_and_retrieval_effects_separate(self):
        suite = profiles({'optimization_suite': 'score-v3'})
        self.assertEqual(set(suite), {'baseline', 'local_direct', 'local_reasoned', 'local_knowledge'})
        a, b = suite['local_direct'], suite['local_reasoned']
        self.assertEqual({k: v for k, v in a.items() if k != 'answer_order'},
                         {k: v for k, v in b.items() if k != 'answer_order'})
        self.assertFalse(b['allow_model_knowledge'])
        self.assertNotIn('prior knowledge', answer_prompt(b))
        self.assertIn('prior knowledge', answer_prompt(suite['local_knowledge']))
        self.assertEqual(b['retrieval_plan_rounds'], 0)
