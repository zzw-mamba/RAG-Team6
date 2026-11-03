import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from naive_rag.data import records
from naive_rag.validation import evaluate, partition, prepare_eval, report, wilson_interval


def make_row(i):
    return {"id": str(i), "question": f"Where was fictional person {i} born?",
            "answers": [f"City {i}"], "rationale": "SECRET_REASONING",
            "type": "compositional", "ctxs": [{"title": f"Person {i}", "text": f"Born in City {i}."}]}


class ValidationTests(unittest.TestCase):
    def test_fixed_samples_are_disjoint_repeatable_and_order_independent(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data = [make_row(i) for i in range(200)]
            data.append({**data[0], "id": "duplicate-question"})
            source = root / "train.json"
            source.write_text(json.dumps(data))
            args = argparse.Namespace(input=str(source), output=str(root / "first"),
                                      seed=42, dev_size=5, holdout_size=5)
            prepare_eval(args)
            dev = list(records(root / "first/dev.jsonl"))
            heldout = list(records(root / "first/holdout.jsonl"))
            self.assertEqual(len(dev), 5)
            self.assertEqual(len(heldout), 5)
            self.assertFalse({r["question"] for r in dev} & {r["question"] for r in heldout})
            self.assertTrue(all(partition(r["question"], 42) == "dev" for r in dev))
            self.assertNotIn("SECRET_REASONING", (root / "first/dev.jsonl").read_text())
            self.assertEqual(list(records(root / "first/pilot.jsonl")), dev)
            # Remove duplicate before reversing: first-occurrence representative is explicit.
            source.write_text(json.dumps(list(reversed(data[:-1]))))
            args.output = str(root / "second")
            prepare_eval(args)
            self.assertEqual([r["question"] for r in dev],
                             [r["question"] for r in records(root / "second/dev.jsonl")])

    def test_preparation_refuses_missing_labels_and_insufficient_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.json"
            source.write_text(json.dumps([make_row(0)]))
            args = argparse.Namespace(input=str(source), output=str(root / "prepared"),
                                      seed=42, dev_size=5, holdout_size=5)
            with self.assertRaises(ValueError):
                prepare_eval(args)
            self.assertFalse((root / "prepared").exists())
            row = make_row(0)
            row.pop("answers")
            source.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):
                prepare_eval(args)

    def test_evaluation_reports_recomputed_scores_and_example_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = root / "dev.json"
            source.write_text(json.dumps([make_row(1), make_row(2)]))
            args = argparse.Namespace(input=str(source), output=str(root / "run"),
                                      config=None, resume=False, dry_run=True)
            evaluate(args)
            summary = json.loads((root / "run/report/summary.json").read_text())
            self.assertEqual(summary["samples"], 2)
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["normalized_em"], 0)
            self.assertIn("DRY-RUN", (root / "run/report/report.md").read_text())
            args.resume = True
            evaluate(args)
            trace_path = root / "run/traces.jsonl"
            traces = list(records(trace_path))
            traces[0]["final_answer"] = "City 1"
            trace_path.write_text("".join(json.dumps(r) + "\n" for r in traces))
            report(argparse.Namespace(input=str(source), run_dir=str(root / "run"), output=str(root / "recomputed")))
            summary = json.loads((root / "recomputed/summary.json").read_text())
            self.assertEqual(summary["normalized_em"], 0.5)
            cases = json.loads((root / "recomputed/cases.json").read_text())
            self.assertEqual(len(cases["correct"]), 1)
            self.assertEqual(len(cases["answer_mismatch_with_gold_string_present"]), 1)
            self.assertIn("SECRET_REASONING", source.read_text())
            self.assertNotIn("SECRET_REASONING", trace_path.read_text())
            self.assertNotIn('"answers"', json.dumps(traces[0]["messages"]))

    def test_evaluation_refuses_unlabeled_input_before_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = make_row(0)
            row["answers"] = []
            source = root / "test.json"
            source.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):
                evaluate(argparse.Namespace(input=str(source), output=str(root / "run"),
                                            config=None, resume=False, dry_run=True))
            self.assertFalse((root / "run").exists())

    def test_report_refuses_missing_predictions_and_changed_labels(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = root / "dev.json"
            source.write_text(json.dumps([make_row(1), make_row(2)]))
            evaluate(argparse.Namespace(input=str(source), output=str(root / "run"),
                                        config=None, resume=False, dry_run=True))
            trace_path = root / "run/traces.jsonl"
            trace_path.write_text(trace_path.read_text().splitlines()[0] + "\n")
            args = argparse.Namespace(input=str(source), run_dir=str(root / "run"), output=str(root / "report2"))
            with self.assertRaises(ValueError):
                report(args)
            source.write_text(json.dumps([make_row(3)]))
            with self.assertRaises(ValueError):
                report(args)

    def test_wilson_interval_is_sensible_at_boundaries(self):
        low, high = wilson_interval(0, 100)
        self.assertAlmostEqual(low, 0)
        self.assertLess(high, 0.05)
        low, high = wilson_interval(100, 100)
        self.assertGreater(low, 0.95)
        self.assertAlmostEqual(high, 1)

    def test_abstention_report_separates_legacy_false_hits(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = root / "dev.json"
            row = make_row(1)
            row["answers"] = ["no"]
            row["ctxs"] = [{"title": "Unrelated", "text": "The result is unknown."}]
            source.write_text(json.dumps([row]))
            evaluate(argparse.Namespace(input=str(source), output=str(root / "run"),
                                        config=None, resume=False, dry_run=True))
            trace_path = root / "run/traces.jsonl"
            trace = list(records(trace_path))[0]
            trace["final_answer"] = "unknown"
            # Older complete runs have no version fields or legacy score column.
            trace.pop("prompt_version", None)
            trace["metrics"].pop("legacy_substring_accuracy", None)
            trace_path.write_text(json.dumps(trace) + "\n")
            manifest_path = root / "run/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("metric_version", None)
            manifest.pop("prompt_version", None)
            manifest["identity"]["config"].pop("prompt_mode", None)
            manifest_path.write_text(json.dumps(manifest))
            report(argparse.Namespace(input=str(source), run_dir=str(root / "run"), output=str(root / "rescored")))
            summary = json.loads((root / "rescored/summary.json").read_text())
            self.assertEqual(summary["unknown_count"], 1)
            self.assertEqual(summary["answered_count"], 0)
            self.assertIsNone(summary["answered_normalized_em"])
            self.assertEqual(summary["substring_accuracy"], 0)
            self.assertEqual(summary["legacy_substring_accuracy"], 1)
            self.assertEqual(summary["legacy_substring_false_positive_count"], 1)
            self.assertEqual(summary["candidate_gold_phrase_rate"], 0)
            self.assertEqual(summary["by_answer_kind"]["yes_no"]["unknown_rate"], 1)


if __name__ == "__main__":
    unittest.main()
