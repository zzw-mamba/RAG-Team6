import csv
import json
import tempfile
import unittest
from pathlib import Path

from raglab.evaluation import evaluate, exact_match, token_f1
from raglab.schema import Document, Example
from raglab.submission import export_submission
from raglab.training.rewards import outcome_reward


def trace(sample_id=0, **kwargs):
    return {
        "id": sample_id,
        "question": "Q",
        "initial_answer": "wrong",
        "final_answer": "Harbor City",
        "backend": "http+http",
        "elapsed_seconds": 1.0,
        "revisions": 1,
        "stop_reason": "critic_stop",
        **kwargs,
    }


class EvaluationTests(unittest.TestCase):
    def test_alias_and_normalization(self):
        self.assertEqual(exact_match("The Harbor City!", ["Other", "Harbor City"]), 1.0)
        self.assertEqual(token_f1("Harbor", ["Harbor City"]), 2 / 3)
        self.assertEqual(token_f1("yes", ["no"]), 0)

    def test_correction_regression_and_coverage(self):
        examples = [
            Example(0, "Q", (Document("d", "", "E"),), ("Harbor City",)),
            Example(1, "Q", (Document("d", "", "E"),), ("Harbor City",)),
        ]
        metrics = evaluate(
            examples, [trace(), trace(1, initial_answer="Harbor City", final_answer="wrong")]
        )
        self.assertEqual(metrics["normalized_em"], 0.5)
        self.assertEqual(metrics["correction_rate"], 1)
        self.assertEqual(metrics["regression_rate"], 1)

    def test_partial_evaluation_is_explicit(self):
        examples = [Example(i, "Q", (), ("Harbor City",)) for i in range(2)]
        with self.assertRaises(ValueError):
            evaluate(examples, [trace()])
        self.assertEqual(evaluate(examples, [trace()], allow_partial=True)["coverage"], 0.5)

    def test_unlabeled_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([Example(0, "Q", ())], [trace()])

    def test_reward_prefers_fix_and_preservation(self):
        fixed = outcome_reward("wrong", "right", ["right"], True)["total"]
        stopped_wrong = outcome_reward("wrong", "wrong", ["right"], False)["total"]
        kept = outcome_reward("right", "right", ["right"], False)["total"]
        unnecessary = outcome_reward("right", "right", ["right"], True)["total"]
        broken = outcome_reward("right", "wrong", ["right"], True)["total"]
        self.assertGreater(fixed, stopped_wrong)
        self.assertGreater(kept, unnecessary)
        self.assertGreater(unnecessary, broken)

    def test_export_template_order_and_csv_escaping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template, traces, output = (
                root / "template.csv",
                root / "traces.jsonl",
                root / "out.csv",
            )
            template.write_text("id,answer\n1,unknown\n0,unknown\n")
            rows = [trace(final_answer='City, "East"'), trace(1)]
            traces.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self.assertEqual(export_submission(traces, template, output, expected_rows=2), 2)
            with output.open(newline="") as stream:
                result = list(csv.DictReader(stream))
            self.assertEqual([r["id"] for r in result], ["1", "0"])
            self.assertEqual(result[1]["answer"], 'City, "East"')

    def test_export_rejects_mock_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template, traces = root / "template.csv", root / "traces.jsonl"
            template.write_text("id,answer\n0,unknown\n")
            traces.write_text(json.dumps(trace(backend="mock+mock")) + "\n")
            with self.assertRaises(ValueError):
                export_submission(traces, template, root / "a.csv", expected_rows=1)
            traces.write_text(json.dumps(trace(2)) + "\n")
            with self.assertRaises(ValueError):
                export_submission(traces, template, root / "b.csv", expected_rows=1)

    def test_export_rejects_unresolved_format_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "template.csv").write_text("id,answer\n0,unknown\n")
            (root / "traces.jsonl").write_text(
                json.dumps(trace(stop_reason="invalid_critique")) + "\n"
            )
            with self.assertRaises(ValueError):
                export_submission(
                    root / "traces.jsonl", root / "template.csv", root / "a.csv", expected_rows=1
                )


if __name__ == "__main__":
    unittest.main()
