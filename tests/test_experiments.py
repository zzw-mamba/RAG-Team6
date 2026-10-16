import json
import tempfile
import unittest
from pathlib import Path

from raglab.config import PipelineConfig
from raglab.data import iter_records
from raglab.experiments import recover_trace_tail, run_inference
from raglab.training.annotations import check_disjoint, read_annotations
from raglab.training.sft import supervised_tokens

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "P" if add_generation_prompt else "PT!"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text]}


class ExperimentTests(unittest.TestCase):
    def test_resume_does_not_duplicate_and_refuses_changed_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            config = PipelineConfig()
            run_inference(FIXTURES / "questions.json", config, output)
            run_inference(FIXTURES / "questions.json", config, output, resume=True)
            self.assertEqual(len(list(iter_records(output / "traces.jsonl"))), 3)
            with self.assertRaises(ValueError):
                run_inference(
                    FIXTURES / "questions.json", PipelineConfig(top_k=1), output, resume=True
                )

    def test_interrupted_tail_recovery_preserves_complete_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_bytes(b'{"id": 0}\n{"id":')
            recover_trace_tail(path)
            self.assertEqual(path.read_bytes(), b'{"id": 0}\n')

    def test_sft_masks_prompt_and_padding_not_target(self):
        result = supervised_tokens(FakeTokenizer(), [], "target", 20)
        self.assertEqual(result["labels"], [-100, ord("T"), ord("!")])
        with self.assertRaises(ValueError):
            supervised_tokens(FakeTokenizer(), [], "target", 2)

    def test_train_validation_question_overlap_rejected(self):
        with self.assertRaises(ValueError):
            check_disjoint([{"question": "What?"}], [{"question": "WHAT!"}])

    def test_unreviewed_annotations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.jsonl"
            path.write_text(json.dumps({"id": 0, "split": "train", "reviewed": False}) + "\n")
            with self.assertRaises(ValueError):
                read_annotations(path, "train")


if __name__ == "__main__":
    unittest.main()
