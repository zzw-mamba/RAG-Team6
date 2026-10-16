import json
import tempfile
import unittest
from pathlib import Path

from raglab.data import example_record, is_validation, iter_examples, normalize, prepare


class DataTests(unittest.TestCase):
    def raw(self, **kwargs):
        return {
            "id": 7,
            "question": "Where?",
            "ctxs": [{"title": "A", "text": "Evidence"}],
            "answers": ["Here"],
            **kwargs,
        }

    def test_competition_schema_and_round_trip(self):
        example = normalize(self.raw(rationale="unverified", qa_pairs=None))
        self.assertEqual(example.answers, ("Here",))
        self.assertEqual(normalize(example_record(example)), example)

    def test_original_schema(self):
        example = normalize(
            {
                "_id": "abc",
                "question": "Q",
                "answer": "X",
                "context": [["Title", ["Sentence one.", "Sentence two."]]],
            }
        )
        self.assertEqual(example.contexts[0].text, "Sentence one. Sentence two.")

    def test_contents_alias(self):
        self.assertEqual(normalize(self.raw(ctxs=[{"contents": "body"}])).contexts[0].text, "body")

    def test_unsupported_context_fails_loudly(self):
        with self.assertRaises(ValueError):
            normalize(self.raw(ctxs=[{"unknown_field": "data"}]))

    def test_duplicate_document_ids_rejected(self):
        with self.assertRaises(ValueError):
            normalize(self.raw(ctxs=[{"id": "same", "text": "A"}, {"id": "same", "text": "B"}]))

    def test_nonstring_answer_rejected(self):
        with self.assertRaises(ValueError):
            normalize(self.raw(answers=[123]))

    def test_fallback_training_id_is_marked(self):
        raw = self.raw()
        del raw["id"]
        example = normalize(raw, 8)
        self.assertEqual(example.id, "row:8")
        self.assertFalse(example.explicit_id)

    def test_question_group_split(self):
        self.assertEqual(
            is_validation("What is this?", 42, 0.2), is_validation("WHAT   is this!", 42, 0.2)
        )

    def test_duplicate_sample_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text(json.dumps([self.raw(), self.raw()]))
            with self.assertRaises(ValueError):
                list(iter_examples(path))

    def test_prepare_groups_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp) / "input.json", Path(tmp) / "split"
            raw = [self.raw(id=i, question=f"Question {i}") for i in range(80)]
            raw.append(self.raw(id=80, question="QUESTION 0!"))
            path.write_text(json.dumps(raw))
            manifest = prepare(path, output, fraction=0.25)
            train = {e.id for e in iter_examples(output / "train.jsonl")}
            validation = {e.id for e in iter_examples(output / "validation.jsonl")}
            self.assertTrue(train and validation)
            self.assertEqual(0 in train, 80 in train)
            self.assertFalse(train & validation)
            self.assertEqual(sum(manifest["counts"].values()), 81)
            self.assertEqual(manifest["status"], "complete")


if __name__ == "__main__":
    unittest.main()
