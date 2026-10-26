"""Real control-flow recovery with explicitly synthetic responses, never model scores."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autodl_rag.core import InvalidOutput, read_json, records
from autodl_rag.data import prepare, prepared
from autodl_rag.experiment import probe_answer, run_dataset
from autodl_rag.models import LocalModel
from autodl_rag.pipeline import Pipeline
from test_system import configuration


class ReplySequence:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.budgets = []

    def generate(self, system, payload, max_new_tokens=None):
        self.requests.append(copy.deepcopy(payload))
        self.budgets.append(max_new_tokens)
        return {
            "raw": next(self.replies),
            "allowed": {d["id"] for d in payload["documents"]},
            "documents": payload["documents"],
        }


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def answer_pipeline(self, replies):
        p = Pipeline.__new__(Pipeline)
        p.cfg = {"work_dir": str(self.root), "allow_model_knowledge": False}
        p.calls = 0
        p.qwen = ReplySequence(replies)
        return p

    def test_citation_retry_uses_previous_output_and_keeps_strict_validation(self):
        bad = json.dumps(
            {"answer": "Synthetic City", "explanation": "fixture", "evidence_ids": ["invented"]}
        )
        good = json.dumps(
            {"answer": "Synthetic City", "explanation": "fixture", "evidence_ids": ["ctx:0"]}
        )
        p = self.answer_pipeline([bad, good])
        answer, raw, retries, docs = p.answer(
            {"question": "Synthetic question?", "documents": [{"id": "ctx:0", "text": "fixture"}]}
        )
        self.assertEqual(retries, 1)
        self.assertEqual(answer["evidence_ids"], ["ctx:0"])
        repair = p.qwen.requests[1]["format_repair"]
        self.assertEqual(repair["previous_output"], bad)
        self.assertIn("ctx:0", repair["error"])
        self.assertEqual(p.calls, 2)

    def test_three_invalid_replies_are_logged_and_never_become_placeholder_answers(self):
        p = self.answer_pipeline(['{"evidence_ids":["invented"]}'] * 3)
        with self.assertRaisesRegex(InvalidOutput, "Diagnostic:"):
            p.answer(
                {
                    "question": "Synthetic question?",
                    "documents": [{"id": "ctx:0", "text": "fixture"}],
                }
            )
        diagnostics = list((self.root / "diagnostics").glob("*.json"))
        self.assertEqual(len(diagnostics), 1)
        log = read_json(diagnostics[0])
        self.assertEqual(len(log["attempts"]), 3)
        self.assertEqual(log["attempts"][0]["raw"], '{"evidence_ids":["invented"]}')
        self.assertNotIn("answers", log)

    def test_incomplete_json_retries_increase_budget_instead_of_repeating_same_limit(self):
        truncated = '{"answer":"Synthetic", "explanation":"' + "repeating ancestor " * 100
        good = json.dumps(
            {"answer": "unknown", "evidence_ids": [], "explanation": "Missing second hop."}
        )
        p = self.answer_pipeline([truncated, truncated, good])
        answer, _, retries, _ = p.answer(
            {"question": "Synthetic?", "documents": [{"id": "d", "text": "fixture"}]}
        )
        self.assertEqual(answer["answer"], "unknown")
        self.assertEqual(retries, 2)
        self.assertEqual(p.qwen.budgets, [384, 768, 1536])

    def test_budgeted_citation_list_contains_only_remaining_documents(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1] * len(json.dumps(messages))

        model = LocalModel.__new__(LocalModel)
        model.cfg = {"max_input_tokens": 1000}
        model.tokenizer = Tokenizer()
        payload = {
            "question": "Synthetic?",
            "allowed_evidence_ids": ["stale"],
            "documents": [{"id": f"doc:{i}", "text": "a" * 250} for i in range(6)],
        }
        messages, ids, allowed = model.fit_messages("System", payload)
        visible = json.loads(messages[1]["content"])
        self.assertLess(len(visible["documents"]), 6)
        self.assertEqual(set(visible["allowed_evidence_ids"]), allowed)
        self.assertEqual(allowed, {d["id"] for d in visible["documents"]})
        self.assertEqual(payload["allowed_evidence_ids"], ["stale"])

    def test_code_restart_archives_predictions_and_csv_without_touching_checkpoints(self):
        cfg = configuration(self.root)
        prepare(cfg)
        output = self.root / "tune/group-000016.jsonl"
        weights = self.root / "checkpoints/adapter_model.safetensors"
        weights.parent.mkdir()
        weights.write_bytes(b"synthetic checkpoint preservation fixture")

        class CounterPipeline:
            def __init__(self):
                self.count = 0

            def run(self, row):
                self.count += 1
                return {
                    "id": row["id"],
                    "final": {"answer": "synthetic"},
                    "stop_reason": "critic_stop",
                }

        p = CounterPipeline()
        with patch("autodl_rag.experiment.source_identity", return_value="old-code"):
            run_dataset(cfg, "tune", "rag", output, limit=2, pipeline=p)
        original = output.read_bytes()
        output.with_suffix(".csv").write_text("old export")
        with patch("autodl_rag.experiment.source_identity", return_value="new-code"):
            with self.assertRaises(ValueError):
                run_dataset(cfg, "tune", "rag", output, limit=2, resume=True, pipeline=p)
            run_dataset(
                cfg,
                "tune",
                "rag",
                output,
                limit=2,
                resume=True,
                pipeline=p,
                restart_on_code_change=True,
            )
        self.assertEqual(p.count, 4)
        archives = list((output.parent / ".archive").iterdir())
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / output.name).read_bytes(), original)
        self.assertEqual((archives[0] / output.with_suffix(".csv").name).read_text(), "old export")
        self.assertEqual(weights.read_bytes(), b"synthetic checkpoint preservation fixture")
        self.assertEqual(
            read_json(output.with_suffix(".manifest.json"))["signature"]["implementation"],
            "new-code",
        )
        with patch("autodl_rag.experiment.source_identity", return_value="newest-code"):
            with self.assertRaises(ValueError):
                run_dataset(
                    cfg,
                    "tune",
                    "rag",
                    output,
                    limit=3,
                    resume=True,
                    pipeline=p,
                    restart_on_code_change=True,
                )

    def test_failure_records_sample_and_resume_retries_it_without_skipping(self):
        cfg = configuration(self.root)
        prepare(cfg)
        output = self.root / "predictions.jsonl"

        class Failing:
            def run(self, row):
                raise InvalidOutput("synthetic citation failure")

        with self.assertRaisesRegex(InvalidOutput, "Failed sample id="):
            run_dataset(cfg, "tune", "rag", output, limit=1, pipeline=Failing())
        failure = read_json(output.with_suffix(".failure.json"))
        first = next(iter(records(prepared(cfg) / "tune.jsonl")))
        self.assertEqual(failure["id"], first["id"])
        self.assertEqual(failure["completed_rows"], 0)
        self.assertEqual(output.read_text(), "")

        class Valid:
            def run(self, row):
                return {"id": row["id"], "final": {"answer": "synthetic"}, "stop_reason": "vanilla"}

        result = run_dataset(cfg, "tune", "rag", output, limit=1, resume=True, pipeline=Valid())
        self.assertEqual(result["rows"], 1)
        self.assertEqual(next(iter(records(output)))["id"], first["id"])

    def test_single_question_probe_passes_only_public_fields(self):
        cfg = configuration(self.root)
        prepare(cfg)
        row = next(iter(records(prepared(cfg) / "tune.jsonl")))
        case = self

        class Probe:
            def initial(self, public_row):
                case.assertEqual(set(public_row), {"id", "question", "ctxs"})
                return {
                    "answer": {
                        "answer": "unknown",
                        "explanation": "Synthetic missing hop",
                        "evidence_ids": [],
                    },
                    "raw": "synthetic fixture",
                    "format_retries": 0,
                }, False

        output = self.root / "probe.json"
        result = probe_answer(cfg, "tune", row["id"], output, pipeline=Probe())
        self.assertEqual(result["id"], row["id"])
        self.assertNotIn("SECRET_GOLD", output.read_text())


if __name__ == "__main__":
    unittest.main()
