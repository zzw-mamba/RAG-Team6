"""Synthetic routing isolation tests, never factual quality claims."""
import unittest
from unittest.mock import Mock

from autodl_rag.optimize import profiles
from autodl_rag.pipeline import Pipeline
from autodl_rag.routing import route_config


class RoutingTests(unittest.TestCase):
    def test_question_wording_not_entity_names_controls_route(self):
        cfg = {'question_router': True}
        for question in ['Are Alice and Bob of the same nationality?',
                         'Was Alice born before Bob?', 'Did Alice direct Film?']:
            effective, route = route_config(cfg, question)
            self.assertEqual(route, 'boolean_legacy')
            self.assertEqual(effective['answer_order'], 'answer_first')
            self.assertEqual(effective['retrieval_policy'], 'coverage')
        for question in ['Who is younger, Alice or Bob?', 'Which film was released first?',
                         'Where was Alice born?', 'What does Alice do?']:
            self.assertEqual(route_config(cfg, question)[1], 'entity_hybrid')
        self.assertEqual(cfg, {'question_router': True})

    def test_restore_per_worker_state_even_after_failure_without_mutating_qwen(self):
        p = Pipeline.__new__(Pipeline)
        p.cfg = {'question_router': True, 'reranker_path': '/bge'}
        p.mode, p.signature = 'rag', 'original'
        reranker = object()
        p.retriever = Mock(cfg=p.cfg, reranker=reranker)
        p.qwen = Mock(cfg={'max_input_tokens': 16384})
        original = p.cfg

        def run(row, sample=False):
            self.assertEqual(p.cfg['answer_order'], 'answer_first')
            self.assertIsNone(p.retriever.reranker)
            self.assertNotEqual(p.signature, 'original')
            self.assertNotIn('answer_order', p.qwen.cfg)
            raise RuntimeError('controlled failure')

        p._run = run
        with self.assertRaisesRegex(RuntimeError, 'controlled failure'):
            p.run({'id': '1', 'question': 'Are A and B related?', 'ctxs': [], 'answers': ['SECRET']})
        self.assertIs(p.cfg, original)
        self.assertIs(p.retriever.cfg, original)
        self.assertIs(p.retriever.reranker, reranker)
        self.assertEqual(p.signature, 'original')

    def test_success_records_route_and_does_not_pass_labels(self):
        p = Pipeline.__new__(Pipeline)
        p.cfg, p.mode, p.signature = {'question_router': True}, 'rag', 'original'
        p.retriever = Mock(cfg=p.cfg, reranker=None)
        p._run = Mock(return_value={'final': {'answer': 'Synthetic answer'}})
        row = {'id': '1', 'question': 'Who is younger?', 'ctxs': []}
        self.assertEqual(p.run({**row, 'answers': ['SECRET']})['question_route'], 'entity_hybrid')
        p._run.assert_called_once_with(row, sample=False)

    def test_candidate_profiles_are_frozen_not_selected_by_confirmation(self):
        candidates = profiles({'optimization_suite': 'score-v5', 'reranker_path': None})
        self.assertEqual(set(candidates), {'baseline', 'hybrid_direct', 'routed_reasoned'})
        self.assertTrue(candidates['routed_reasoned']['question_router'])
        self.assertEqual(candidates['hybrid_direct']['answer_order'], 'answer_first')
