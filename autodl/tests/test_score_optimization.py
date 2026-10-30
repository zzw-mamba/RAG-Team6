"""Regression fixtures for evidence coverage; no synthetic benchmark claims."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from autodl_rag.core import fingerprint, read_json, save_json
from autodl_rag.optimize import choose, profiles, verify_selection
from autodl_rag.pipeline import ANSWER, Pipeline, answer_prompt
from autodl_rag.retrieval import Retriever
from autodl_rag.structured import qwen_schema
from test_system import FakeQwen, configuration


def document(i, title=None, text="Synthetic evidence."):
    return {"id": str(i), "title": title or f"Title {i}", "text": text}


class ScoreOptimizationTests(unittest.TestCase):
    def retriever(self, cap=16):
        r = Retriever.__new__(Retriever)
        r.cfg = {"max_documents": cap, "document_chars": 4000, "retrieval_policy": "coverage"}
        r.reranker = None
        r.focused_search = Mock(return_value=[])
        return r

    def test_coverage_keeps_all_supplied_contexts_including_low_overlap_second_hop(self):
        r = self.retriever()
        local = [document(i) for i in range(10)]
        local[-1] = document(9, "Intermediate Artist", "Her mother was Synthetic Person.")
        r.focused_search.return_value = [document(i) for i in range(10, 40)]
        result = r.retrieve("Who is the mother of the film director?", local)
        self.assertTrue({d["id"] for d in local} <= {d["id"] for d in result})
        self.assertEqual(result[:10], local)
        self.assertEqual(len(result), 16)

    def test_query_channels_reserve_second_hop_without_original_question_overlap(self):
        r = self.retriever(cap=4)
        channels = {
            "first artist": [document("mother", "Mother", "Born in Fictional Harbor.")],
            "second artist": [document("father", "Father", "Born in Fictional Mountain.")],
            "film question": [document("noise", "Film question")],
        }
        r.focused_search.side_effect = lambda q: channels[q]
        result = r.retrieve("film question", [document("local")],
                            ["first artist", "second artist"])
        self.assertEqual([d["id"] for d in result], ["local", "mother", "father", "noise"])

    def test_title_diversity_does_not_evict_provided_passages(self):
        r = self.retriever()
        local = [document(i, "Same title") for i in range(5)]
        r.focused_search.return_value = [document(i, "Noisy title") for i in range(10, 30)]
        result = r.retrieve("query", local)
        self.assertEqual(sum(d["title"] == "Same title" for d in result), 5)
        self.assertEqual(sum(d["title"] == "Noisy title" for d in result), 3)

    def test_focused_search_uses_quoted_conjunction_and_broad_backup(self):
        r = self.retriever()
        r.cfg["global_top_k"] = 8
        del r.focused_search
        r.db = Mock()
        r.db.execute.return_value.fetchall.return_value = [("a", "Artist", "Specific.")]
        r.search = Mock(return_value=[document("b")])
        result = r.focused_search('Artist" OR title:*')
        expression = r.db.execute.call_args.args[1][0]
        self.assertEqual(expression, '"artist" AND "title"')
        self.assertEqual([d["id"] for d in result], ["a", "b"])

    def test_knowledge_prompt_has_no_contradictory_missing_hop_refusal(self):
        self.assertIs(answer_prompt({"allow_model_knowledge": False}), ANSWER)
        prompt = answer_prompt({"allow_model_knowledge": True})
        self.assertNotIn("If any required relation is missing, answer unknown", prompt)
        self.assertNotIn("Use only supplied documents", prompt)
        self.assertIn("prior knowledge", prompt)
        self.assertIn("Never invent a fact or a citation", prompt)

    def test_planner_schema_is_separate_and_bounded(self):
        schema = qwen_schema({"task": "retrieval_plan"}, {"d"})
        self.assertEqual(schema["required"], ["retrieval_queries"])
        self.assertEqual(schema["properties"]["retrieval_queries"]["maxItems"], 3)
        self.assertNotIn("answer", schema["properties"])

    def test_planning_never_sees_labels_and_records_real_call_count(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Pipeline.__new__(Pipeline)
            p.cfg = {"retrieval_plan_rounds": 1, "retrieval_plan_trigger": "always"}
            p.calls, p.signature, p.cache_dir = 0, "fixture", Path(temp)
            docs = [document("d")]
            p.retriever = Mock()
            p.retriever.retrieve.return_value = docs
            requests = []

            def generate(system, payload, **kwargs):
                requests.append(copy.deepcopy(payload))
                self.assertNotIn("SECRET_LABEL", json.dumps(payload))
                return {"raw": '{"retrieval_queries":["Intermediate Person mother"]}'}

            p.qwen = Mock()
            p.qwen.generate.side_effect = generate
            responses = iter(["unknown", "Synthetic Person"])

            def answer(payload):
                p.calls += 1
                value = next(responses)
                return {"answer": value}, "fixture", 0, docs

            p.answer = answer
            row = {"id": "x", "question": "Synthetic question", "ctxs": docs,
                   "answers": ["SECRET_LABEL"], "rationale": "SECRET_LABEL"}
            initial, cached = p.initial(row)
            self.assertFalse(cached)
            self.assertEqual(initial["answer"]["answer"], "Synthetic Person")
            self.assertEqual(initial["initial_calls"], 3)
            self.assertEqual(initial["retrieval_plans"][0]["queries"], ["Intermediate Person mother"])
            p.retriever.retrieve.assert_called_with(
                row["question"], docs, extra_queries=["Intermediate Person mother"], previous=docs
            )
            self.assertEqual(len(requests), 1)

    def test_selection_uses_accuracy_then_f1_then_time_not_unknown_rate(self):
        def metric(em, f1, seconds):
            return {"normalized_em": em, "token_f1": f1, "avg_logical_seconds": seconds}
        result = {"fast": metric(.4, .5, 1), "accurate": metric(.5, .5, 9),
                  "verbose": metric(.4, .8, 3)}
        self.assertEqual(choose(result), "accurate")

    def test_profiles_do_not_mutate_base_or_enable_training(self):
        base = {"max_input_tokens": 4096, "max_documents": 14, "allow_model_knowledge": False}
        original = dict(base)
        variants = profiles(base)
        self.assertEqual(base, original)
        self.assertEqual(variants["coverage"]["max_input_tokens"], 16384)
        self.assertFalse(variants["coverage"]["allow_model_knowledge"])
        self.assertTrue(variants["knowledge"]["allow_model_knowledge"])

    def test_changed_config_or_source_cannot_reuse_tune_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            save_json(folder / "selection.json", {
                "base_signature": fingerprint({"a": 1}), "source": "old", "tune_manifests": {},
            })
            with self.assertRaises(ValueError):
                verify_selection({"a": 2}, folder)
            with patch("autodl_rag.optimize.source_identity", return_value="old"):
                self.assertEqual(verify_selection({"a": 1}, folder), read_json(folder / "selection.json"))

    def test_real_runner_stages_resume_and_export_template_with_synthetic_backend(self):
        from autodl_rag.data import prepare
        from autodl_rag.optimize import main
        from autodl_rag.retrieval import build_index

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = configuration(root)
            prepare(cfg)
            build_index(cfg)
            config_file = root / "config.json"
            save_json(config_file, cfg)
            with patch("autodl_rag.models.LocalModel", side_effect=lambda *a, **k: FakeQwen()), \
                    contextlib.redirect_stdout(io.StringIO()):
                for stage in ("tune", "tune", "confirmation", "predict"):
                    with patch("sys.argv", ["optimize", "--config", str(config_file), "--stage", stage]):
                        main()
            folder = Path(cfg["work_dir"]) / "optimization/score-v2"
            selected = read_json(folder / "selection.json")["selected"]
            self.assertTrue(read_json(folder / "confirmation.json")["passed"])
            output = folder / selected / "test.csv"
            self.assertTrue(output.is_file())
            self.assertEqual(read_json(output.with_suffix(".validation.json"))["rows"], 3)


if __name__ == "__main__":
    unittest.main()
