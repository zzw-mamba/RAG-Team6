"""Synthetic retrieval fixtures test contracts, not real benchmark scores."""
import sqlite3
import unittest
from unittest.mock import Mock

from autodl_rag.hybrid import entity_queries, fuse, retrieve
from autodl_rag.optimize import profiles
from autodl_rag.retrieval import Retriever


def doc(i, title, text):
    return {'id': str(i), 'title': title, 'text': text}


class HybridTests(unittest.TestCase):
    def test_title_search_is_bound_and_preserves_year_and_accents(self):
        r = Retriever.__new__(Retriever)
        r.cfg = {'global_top_k': 8}
        r.db = sqlite3.connect(':memory:')
        self.addCleanup(r.db.close)
        r.db.executescript("""CREATE TABLE docs(id,title,text);
            CREATE VIRTUAL TABLE search USING fts5(title,text,content='docs',content_rowid='rowid');""")
        r.db.executemany('INSERT INTO docs VALUES (?,?,?)', [
            ('right', 'Łukasz Film (1999)', 'A composer biography.'),
            ('wrong', 'Łukasz Film (2001)', 'Mentions 1999.'),
        ])
        r.db.execute("INSERT INTO search(search) VALUES ('rebuild')")
        self.assertEqual([d['id'] for d in r.title_search('Łukasz Film (1999)')], ['right'])
        r.title_search('" OR title:*')  # operators cannot escape quoted words

    def test_question_titles_preserve_disambiguation(self):
        docs = [doc(1, 'Example Film (1999)', 'Text')]
        self.assertEqual(entity_queries('Who composed Example Film (1999)?', docs)[0],
                         'Example Film (1999)')

    def test_rrf_does_not_double_count_duplicates_in_a_channel(self):
        a, b = doc('a', 'A', 'x'), doc('b', 'B', 'x')
        self.assertEqual(fuse([(1, [a, a]), (2, [b])]), [b, a])

    def test_low_original_overlap_second_hop_survives_separate_channel(self):
        r = Mock()
        r.cfg = {'max_documents': 8, 'document_chars': 4000}
        r.reranker = None
        local = [doc(i, f'Film {i}', 'Example film director Fictional Artist.') for i in range(10)]
        bridge = doc('hop', 'Fictional Artist', 'Born in Synthetic Harbor.')
        r.title_search.side_effect = lambda q: [bridge] if q == 'Fictional Artist' else []
        r.focused_search.side_effect = lambda q: [bridge] if q == 'Fictional Artist' else local
        out = retrieve(r, 'Where was the director of Example Film born?', local,
                       ['Fictional Artist'], [])
        self.assertIn('hop', [d['id'] for d in out])
        self.assertLessEqual(len(out), 8)
        self.assertEqual(len({d['id'] for d in out}), len(out))

    def test_reranker_scores_whole_candidate_pool_for_each_relation(self):
        r = Mock()
        r.cfg = {'max_documents': 8, 'document_chars': 4000, 'rerank_candidates': 48}
        docs = [doc(1, 'Film', 'Directed by Artist.'), doc(2, 'Artist', 'Born in Harbor.')]
        r.title_search.return_value = docs
        r.focused_search.return_value = docs
        r.reranker.score.side_effect = lambda q, ds: list(range(len(ds)))
        retrieve(r, 'Film director birthplace?', docs, ['Artist birthplace'], [])
        self.assertEqual(r.reranker.score.call_count, 2)
        self.assertEqual(set(d['id'] for d in r.reranker.score.call_args_list[0].args[1]), {'1', '2'})

    def test_v4_baseline_is_previous_planned_and_bge_is_explicit(self):
        configs = profiles({'optimization_suite': 'score-v4', 'reranker_path': None})
        self.assertEqual(set(configs), {'baseline', 'hybrid_reasoned'})
        self.assertEqual(configs['baseline']['retrieval_policy'], 'coverage')
        self.assertTrue(configs['baseline']['allow_model_knowledge'])
        self.assertEqual(configs['baseline']['retrieval_plan_rounds'], 1)
        bge = profiles({'optimization_suite': 'score-v4', 'reranker_path': '/local/bge'})
        self.assertEqual(bge['hybrid_bge']['reranker_path'], '/local/bge')
        self.assertIsNone(bge['baseline']['reranker_path'])

    def test_batch_workers_reuse_one_reranker(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        from scripts.batched_qwen import SharedBatchPipeline
        from test_shared_batch import SyntheticController, SyntheticEngine

        cfg = {'reranker_path': '/synthetic/bge'}
        shared, received = object(), []

        def controller(cfg, mode, qwen, reranker):
            received.append(reranker)
            return SyntheticController(cfg, qwen)

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'public.jsonl'
            rows = [{'id': str(i), 'question': str(i), 'ctxs': []} for i in range(4)]
            source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            local = SimpleNamespace(identity={'synthetic': True}, cfg=cfg)
            with patch('autodl_rag.models.Reranker', return_value=shared) as load, \
                    patch('scripts.batched_qwen.PlannerFirstPipeline', side_effect=controller):
                pipeline = SharedBatchPipeline(cfg, source, batch_size=4, local=local,
                                               engine_factory=SyntheticEngine)
                try:
                    out = [pipeline.run(r) for r in rows]
                finally:
                    pipeline.close()
                load.assert_called_once_with(cfg)
            self.assertEqual([r['id'] for r in out], ['0', '1', '2', '3'])
            self.assertEqual(len(received), 4)
            self.assertTrue(all(r is shared for r in received))
