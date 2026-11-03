import argparse
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from naive_rag.cli import run, submission
from naive_rag.data import normalize, records, validation_group
from naive_rag.generation import PROMPT_VERSIONS, DryRun, HuggingFace, clean_answer, messages
from naive_rag.metrics import scores
from naive_rag.retrieval import retrieve


class BaselineTests(unittest.TestCase):
    def test_stream_large_object_and_unicode(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.json"
            expected = [{"text": "汉字" * 40000}, {"text": "tail"}]
            path.write_text(json.dumps(expected, ensure_ascii=False))
            self.assertEqual(list(records(path)), expected)

    def test_reject_corrupt_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.json"
            for content in ('[{"a": 1},]', '[{"a": 1}', '[{}] garbage', '[{}, {}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    list(records(path))

    def test_gold_and_rationale_cannot_enter_prompt(self):
        row = {"question": "Where?", "answers": ["SECRET_GOLD"], "rationale": "SECRET_REASON",
               "qa_pairs": "SECRET_PAIRS", "ctxs": [{"title": "Place", "text": "Evidence."}]}
        sample = normalize(row, 0)
        prompt = json.dumps(messages(sample["question"], retrieve(sample["question"], sample["ctxs"], 10)))
        self.assertNotIn("SECRET", prompt)

    def test_retrieval_and_duplicate_passages(self):
        row = {"question": "Amber Bridge director?", "ctxs": [
            {"text": "noise noise", "title": "Noise"},
            {"text": "A film director", "title": "Amber Bridge"},
            {"text": "A film director", "title": "Amber Bridge"}]}
        sample = normalize(row, 1)
        self.assertEqual(len(sample["ctxs"]), 2)
        self.assertEqual(retrieve(sample["question"], sample["ctxs"], 1)[0]["title"], "Amber Bridge")

    def test_group_split_is_stable(self):
        self.assertEqual(validation_group(" Who is A?  ", 42, .1), validation_group("who is a?", 42, .1))

    def test_precomputed_order_does_not_guess_score_direction(self):
        docs = [{"title": "Noise", "text": "unrelated"}, {"title": "Target", "text": "relevant"}]
        selected = retrieve("Target", docs, 1, mode="input")
        self.assertEqual(selected[0]["title"], "Noise")
        self.assertIsNone(selected[0]["bm25_score"])
        with self.assertRaises(ValueError):
            retrieve("Target", docs, 1, mode="invalid")

    def test_substring_accuracy_is_separate_from_exact_match(self):
        result = scores("It was Willow Bay", ["Willow Bay"])
        self.assertEqual(result["substring_accuracy"], 1)
        self.assertEqual(result["normalized_em"], 0)
        self.assertEqual(scores("yes indeed", ["yes"])["token_f1"], 0)
        self.assertEqual(scores("yes", ["yes"])["token_f1"], 1)
        self.assertEqual(scores("something", [""])["substring_accuracy"], 0)

    def test_metric_alias_and_overlap(self):
        self.assertEqual(scores("The Willow Bay", ["other", "Willow Bay"])["normalized_em"], 1)
        self.assertAlmostEqual(scores("Willow", ["Willow Bay"])["token_f1"], 2 / 3)
        self.assertEqual(scores("no", ["yes"])["normalized_em"], 0)

    def test_unknown_is_not_a_correct_no_answer(self):
        for gold in ("no", "known"):
            result = scores("unknown", [gold])
            self.assertEqual(result["substring_accuracy"], 0)
            self.assertEqual(result["normalized_em"], 0)
            self.assertEqual(result["legacy_substring_accuracy"], 1)
        self.assertEqual(scores("yes and no", ["no"])["substring_accuracy"], 0)
        self.assertEqual(scores("nobody", ["no"])["substring_accuracy"], 0)
        self.assertEqual(scores("No.", ["no"])["substring_accuracy"], 1)

    def test_evidence_summary_is_not_exported_as_the_answer(self):
        output = "Evidence: Document 1 identifies Mira; document 2 gives her city.\n<answer>Willow Bay</answer>"
        self.assertEqual(clean_answer(output, "evidence_answer"), "Willow Bay")
        self.assertEqual(clean_answer("<answer>unknown</answer>", "evidence_answer"), "unknown")
        for malformed in ["Willow Bay", "<answer>unfinished", "<answer></answer>",
                          "<answer>a</answer><answer>b</answer>", "<answer>a</answer><answer>"]:
            with self.assertRaises(ValueError):
                clean_answer(malformed, "evidence_answer")

    def test_evidence_mode_keeps_one_generation_and_records_raw_summary(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config, source = root / "config.json", root / "input.json"
            config.write_text(json.dumps({"prompt_mode": "evidence_answer", "max_new_tokens": 256}))
            source.write_text(json.dumps([{"question": "Where?", "ctxs": [{"text": "Evidence."}]}]))
            args = argparse.Namespace(input=str(source), output=str(root / "run"), config=str(config),
                                      limit=None, dry_run=True, resume=False)
            run(args)
            trace = list(records(root / "run/traces.jsonl"))[0]
            self.assertEqual(trace["final_answer"], "MOCK_ONLY")
            self.assertIn("<answer>", trace["raw_output"])
            self.assertEqual(trace["prompt_version"], "evidence-answer-v1")
            self.assertEqual(trace["revision_count"], 0)
            self.assertEqual(trace["critic_calls"], 0)
            args.output = str(root / "bad-format")
            with patch("naive_rag.cli.DryRun.generate", return_value=["Incomplete evidence summary"]):
                with self.assertRaises(ValueError):
                    run(args)
            error = json.loads((root / "bad-format/generation_error.json").read_text())
            self.assertEqual(error["raw_output"], "Incomplete evidence summary")

    def test_prompt_ablation_is_explicit_and_remains_label_free(self):
        docs = [{"title": "Film", "text": "Directed by an intermediate person."}]
        direct = messages("Where?", docs)
        alternative = messages("Where?", docs, "best_supported")
        self.assertNotEqual(direct[0]["content"], alternative[0]["content"])
        self.assertEqual(direct[1]["content"], alternative[1]["content"])
        packed, _, _ = DryRun({"prompt_mode": "best_supported"}).pack("Where?", docs, 6000)
        self.assertEqual(packed, alternative)
        self.assertNotEqual(PROMPT_VERSIONS["direct"], PROMPT_VERSIONS["best_supported"])
        with self.assertRaises(ValueError):
            messages("Where?", docs, "invalid")

    def test_alternative_prompt_version_is_recorded_without_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps({"prompt_mode": "best_supported"}))
            source = root / "input.json"
            source.write_text(json.dumps([{"question": "Where?", "answers": ["City"],
                                           "ctxs": [{"text": "A person lives in City."}]}]))
            run(argparse.Namespace(input=str(source), output=str(root / "run"), config=str(config),
                                   limit=None, dry_run=True, resume=False))
            manifest = json.loads((root / "run/manifest.json").read_text())
            trace = list(records(root / "run/traces.jsonl"))[0]
            self.assertEqual(manifest["prompt_version"], "best-supported-v1")
            self.assertEqual(trace["prompt_version"], "best-supported-v1")
            self.assertEqual(manifest["metric_version"], "answer-metrics-v2")
            self.assertEqual(trace["critic_calls"], 0)

    def test_pack_retains_bridge_and_question(self):
        class CharacterTokenizer:
            def encode(self, text, **kwargs):
                return list(text)
            def decode(self, ids, **kwargs):
                return "".join(ids)
            def apply_chat_template(self, prompt, **kwargs):
                return list(json.dumps(prompt))
        backend = HuggingFace.__new__(HuggingFace)
        backend.tokenizer = CharacterTokenizer()
        docs = [{"title": "Film", "text": "a" * 1000}, {"title": "Bridge", "text": "b" * 1000}]
        prompt, packed, count = backend.pack("Where?", docs, 950)
        self.assertLessEqual(count, 950)
        self.assertTrue(all(d["text"] for d in packed))
        self.assertIn("Where?", prompt[1]["content"])

    def test_pack_preserves_all_text_when_full_prompt_fits(self):
        class CharacterTokenizer:
            def encode(self, text, **kwargs):
                return list(text)
            def decode(self, ids, **kwargs):
                return "".join(ids)
            def apply_chat_template(self, prompt, **kwargs):
                return list(json.dumps(prompt))
        backend = HuggingFace.__new__(HuggingFace)
        backend.tokenizer = CharacterTokenizer()
        docs = [{"title": "Long", "text": "a" * 500}, {"title": "Short", "text": "b"}]
        budget = len(backend.encode(messages("Where?", docs))) + 1
        _, packed, count = backend.pack("Where?", docs, budget)
        self.assertEqual([d["text"] for d in packed], [d["text"] for d in docs])
        self.assertTrue(all(not d["truncated"] for d in packed))
        self.assertLessEqual(count, budget)
        base = len(backend.encode(messages("Where?", [{**d, "text": ""} for d in docs])))
        _, packed, count = backend.pack("Where?", docs, base + 200)
        self.assertGreater(len(packed[0]["text"]), 100)
        self.assertEqual(packed[1]["text"], "b")
        self.assertLessEqual(count, base + 200)

    def test_run_resume_and_dry_export_guard(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            path, out = Path(d) / "input.json", Path(d) / "run"
            path.write_text(json.dumps([{"id": 9, "question": "Where?", "answers": ["secret"],
                                         "ctxs": [{"text": "Evidence"}]}]))
            args = argparse.Namespace(input=str(path), output=str(out), config=None,
                                      limit=None, dry_run=True, resume=False)
            run(args)
            trace_path = out / "traces.jsonl"
            trace = list(records(trace_path))[0]
            self.assertEqual(trace["critic_calls"], 0)
            self.assertEqual(trace["revision_count"], 0)
            self.assertEqual(trace["final_answer"], "MOCK_ONLY")
            before = trace_path.read_bytes()
            args.resume = True
            run(args)
            self.assertEqual(before, trace_path.read_bytes())
            template = Path(d) / "sample.csv"
            template.write_text("id,answer\n9,\n")
            with self.assertRaises(ValueError):
                submission(argparse.Namespace(traces=str(trace_path), template=str(template), output=str(Path(d)/"sub.csv")))
            path.write_text("[]")
            with self.assertRaises(ValueError):
                run(args)

    def test_submission_order_quoting_and_id_guards(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            trace, template, output = [Path(d) / n for n in ("traces.jsonl", "sample.csv", "sub.csv")]
            trace.write_text('\n'.join(json.dumps({"id": rid, "backend": "hf", "final_answer": ans})
                                      for rid, ans in [("2", "A, B"), ("1", 'a "quote"')]) + '\n')
            template.write_text("id,answer\n1,\n2,\n")
            args = argparse.Namespace(traces=str(trace), template=str(template), output=str(output))
            submission(args)
            with output.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["id"] for r in rows], ["1", "2"])
            self.assertEqual(rows[1]["answer"], "A, B")
            template.write_text("id,answer\n1,\n3,\n")
            with self.assertRaises(ValueError):
                submission(args)

    def test_batch_alignment_final_partial_batch_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source, config = root / "input.json", root / "config.json"
            source.write_text(json.dumps([{"id": i, "question": f"Q{i}",
                                           "answers": [f"answer-Q{i}"], "ctxs": [{"text": "Evidence"}]}
                                          for i in range(10)]))
            config.write_text(json.dumps({"batch_size": 8}))
            args = argparse.Namespace(input=str(source), output=str(root / "run"), config=str(config),
                                      limit=None, dry_run=True, resume=False)
            with patch("naive_rag.cli.DryRun") as constructor:
                backend = constructor.return_value
                backend.pack.side_effect = lambda question, docs, budget: (question, docs, None)
                backend.generate.side_effect = lambda prompts: [f"answer-{q}" for q in prompts]
                run(args)
                self.assertEqual([len(c.args[0]) for c in backend.generate.call_args_list], [8, 2])
                args.resume = True
                run(args)
                self.assertEqual(backend.generate.call_count, 2)
            traces = list(records(root / "run/traces.jsonl"))
            self.assertEqual([r["id"] for r in traces], [str(i) for i in range(10)])
            self.assertEqual([r["final_answer"] for r in traces], [f"answer-Q{i}" for i in range(10)])
            self.assertEqual([r["timing"]["batch_size"] for r in traces], [8] * 8 + [2] * 2)


if __name__ == "__main__":
    unittest.main()
