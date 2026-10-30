"""Real tiny Qwen + synthetic scheduling tests; no pretrained accuracy/speed claims."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autodl_rag.core import InvalidOutput, parse_answer, records, save_json
from autodl_rag.data import prepare, prepared
from autodl_rag.experiment import export_csv, run_dataset
from scripts.batched_qwen import (
    BatchPrefix,
    Broker,
    HFBatchEngine,
    Request,
    SharedBatchPipeline,
    pack,
    trim_eos,
)
from scripts.score_predict_batch import require_pilot, validation_file
from test_system import configuration

HAS_TENSORS = all(importlib.util.find_spec(p) for p in ("torch", "transformers", "lmformatenforcer"))


class SyntheticEngine:
    def __init__(self, local, batch_size=4, token_cap=32768):
        self.local, self.batch_size, self.stats = local, batch_size, Counter()

    def generate(self, requests):
        self.stats[f"batch_size_{len(requests)}"] += 1
        return [{"raw": json.dumps({"answer": r.payload["question"]}), "allowed": set()}
                for r in requests]


class SyntheticController:
    def __init__(self, cfg, qwen):
        self.qwen = qwen
        owner = threading.get_ident()

        def close():
            if threading.get_ident() != owner:
                raise AssertionError("SQLite closed on wrong thread")

        self.retriever = SimpleNamespace(db=SimpleNamespace(close=close))

    def run(self, row):
        if set(row) != {"id", "question", "ctxs"}:
            raise AssertionError("Labels entered controller")
        if row["id"] == "fail":
            raise InvalidOutput("Synthetic controlled failure")
        reply = self.qwen.generate("Synthetic fixture", {"question": row["question"]})
        return {"id": row["id"], "final": json.loads(reply["raw"]), "stop_reason": "vanilla"}


class SharedBatchTests(unittest.TestCase):
    def test_production_requires_passing_pilot_for_the_exact_batch_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            report = validation_file(folder, 4, 32768)
            with patch("scripts.score_predict_batch.backend_identity", return_value={"fixture": "current"}):
                save_json(report, {"passed": False, "identity": {"fixture": "current"}})
                with self.assertRaises(ValueError):
                    require_pilot({}, folder, 4, 32768)
                save_json(report, {"passed": True, "identity": {"fixture": "old"}})
                with self.assertRaises(ValueError):
                    require_pilot({}, folder, 4, 32768)
                save_json(report, {"passed": True, "identity": {"fixture": "current"}})
                self.assertTrue(require_pilot({}, folder, 4, 32768)["passed"])

    def test_pack_buckets_budget_and_padded_token_cost_without_dropping_items(self):
        items = [{"index": i, "budget": budget, "prompt": [1] * length}
                 for i, (budget, length) in enumerate([(20, 90), (20, 10), (40, 15), (20, 20)])]
        groups = list(pack(items, 4, 120))
        self.assertEqual(sorted(x["index"] for g in groups for x in g), list(range(4)))
        for group in groups:
            self.assertEqual(len({x["budget"] for x in group}), 1)
            cost = (max(len(x["prompt"]) for x in group) + group[0]["budget"]) * len(group)
            self.assertTrue(len(group) == 1 or cost <= 120)

    def test_eos_trim_ignores_padding_after_first_eos(self):
        self.assertEqual(trim_eos([4, 5, 2, 2, 2], [2, 3]), [4, 5, 2])
        self.assertEqual(trim_eos([4, 5], [2, 3]), [4, 5])

    def test_broker_coalesces_requests_and_delivers_errors_to_all_waiters(self):
        local = SimpleNamespace(identity={"synthetic": True}, cfg={})
        engine = SyntheticEngine(local)
        broker = Broker(engine, wait_ms=100)
        barrier = threading.Barrier(4)

        def request(i):
            barrier.wait()
            return broker.generate("Synthetic", {"question": str(i)})

        try:
            with ThreadPoolExecutor(4) as pool:
                results = list(pool.map(request, range(4)))
            self.assertEqual([json.loads(r["raw"])["answer"] for r in results], ["0", "1", "2", "3"])
            self.assertEqual(engine.stats["batch_size_4"], 1)
            with patch.object(engine, "generate", side_effect=RuntimeError("Real backend failed")):
                with self.assertRaisesRegex(RuntimeError, "Real backend failed"):
                    broker.generate("Synthetic", {"question": "failure"})
            with self.assertRaises(RuntimeError):
                broker.generate("Synthetic", {}, sample=True)
        finally:
            broker.close()

    def test_one_loaded_model_services_multiple_controllers_and_resume_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = configuration(Path(temp))
            prepare(cfg)
            output = Path(temp) / "batch.jsonl"
            shared = SimpleNamespace(identity={"synthetic": True}, cfg=cfg)
            pipeline = SharedBatchPipeline(
                cfg, prepared(cfg) / "test.jsonl", 4,
                pipeline_factory=SyntheticController, engine_factory=SyntheticEngine,
            )
            with patch("autodl_rag.models.seed_everything"), \
                    patch("autodl_rag.models.LocalModel", return_value=shared) as load, \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    run_dataset(cfg, "test", "rag", output, pipeline=pipeline)
                finally:
                    pipeline.close()
                load.assert_called_once_with(cfg, "qwen")
            self.assertEqual([r["id"] for r in records(output)], ["0", "1", "2"])
            self.assertTrue(all(r["execution_backend"] == "hf_shared_batch" for r in records(output)))
            export_csv(cfg, output, output.with_suffix(".csv"))
            resumed = SharedBatchPipeline(cfg, prepared(cfg) / "test.jsonl", 4)
            with patch("autodl_rag.models.LocalModel", side_effect=AssertionError("Unnecessary load")):
                run_dataset(cfg, "test", "rag", output, pipeline=resumed, resume=True)
            self.assertIsNone(resumed.local)


@unittest.skipUnless(HAS_TENSORS, "Optional real tensor/JSON dependencies required")
class TinyQwenBatchTests(unittest.TestCase):
    def local(self):
        import string

        import torch
        from autodl_rag.models import LocalModel
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        torch.set_num_threads(1)
        torch.manual_seed(7)
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            [string.printable + "Ł-fixture Ž-fixture 你好 😀"] * 5,
            trainers.BpeTrainer(vocab_size=260, show_progress=False,
                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                               special_tokens=["<unk>", "<eos>"]),
        )
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>",
                                      pad_token="<eos>", clean_up_tokenization_spaces=False)
        fast.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        local = LocalModel.__new__(LocalModel)
        local.role, local.trainable, local.device, local.dtype = "qwen", False, "cpu", torch.float32
        local.tokenizer = fast
        local.cfg = {"max_input_tokens": 1024, "answer_tokens": 192, "temperature": .8}
        local.model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=len(fast), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=2048, eos_token_id=fast.eos_token_id,
            pad_token_id=fast.pad_token_id, attention_dropout=0.0,
        )).eval()
        return local

    def test_real_generate_left_padding_independent_citations_and_single_agreement(self):
        from autodl_rag.structured import qwen_schema

        local = self.local()
        requests = [Request("Synthetic.", {
            "question": "Synthetic question " + "long " * i,
            "fixture_answer": name,
            "documents": [{"id": f"d{i}", "title": "Fixture", "text": "Synthetic evidence."}],
        }) for i, name in enumerate(("A", "Beta", "Gamma"))]
        requests.append(Request("Synthetic planner.", {
            "task": "retrieval_plan", "question": "Synthetic missing relation",
            "documents": [{"id": "plan-doc", "title": "Fixture", "text": "Synthetic."}],
        }))

        def schema(payload, allowed):
            result = qwen_schema(payload, allowed)
            if payload.get("task") == "retrieval_plan":
                result["properties"]["retrieval_queries"] = {
                    "type": "array", "minItems": 1, "maxItems": 1,
                    "items": {"type": "string", "enum": ["Synthetic mother"]},
                }
                return result
            result["properties"]["answer"] = {"type": "string", "enum": [payload["fixture_answer"]]}
            result["properties"]["explanation"] = {"type": "string", "enum": ["Fixture."]}
            result["properties"]["evidence_ids"]["minItems"] = 1
            result["properties"]["evidence_ids"]["maxItems"] = 1
            return result

        with patch("autodl_rag.structured.qwen_schema", side_effect=schema):
            sequential = [local.generate(r.system, r.payload) for r in requests]
            engine = HFBatchEngine(local, 4, 4096)
            batch = engine.generate(requests)
        self.assertEqual(engine.stats["generation_batches"], 1)
        for i, (single, combined) in enumerate(zip(sequential[:3], batch[:3])):
            a = parse_answer(single["raw"], single["allowed"])
            b = parse_answer(combined["raw"], combined["allowed"])
            self.assertEqual(a, b)
            self.assertEqual(b["evidence_ids"], [f"d{i}"])
            self.assertEqual(combined["finish_reason"], "eos")
            self.assertEqual(combined["completion_ids"][-1], local.tokenizer.eos_token_id)
            self.assertEqual(combined["prompt_ids"], single["prompt_ids"])
        self.assertEqual(json.loads(batch[3]["raw"]), {"retrieval_queries": ["Synthetic mother"]})
        self.assertEqual(batch[3]["raw"], sequential[3]["raw"])

    def test_prefix_only_generated_tokens_matches_full_prompt_parser(self):
        import torch
        from autodl_rag.structured import QwenJsonGuard

        local = self.local()
        guard = QwenJsonGuard(local.tokenizer, len(local.tokenizer))
        full = guard.prefix({}, {"d1"})
        sliced = BatchPrefix([guard.prefix({}, {"d1"})], 20, local.tokenizer.eos_token_id)
        raw = '{"answer":"Ł-fixture","evidence_ids":["d1"],"explanation":"Fixture."}'
        prompt, generated = [7] * 20, []
        for token in local.tokenizer.encode(raw, add_special_tokens=False):
            values = torch.tensor(prompt + generated)
            self.assertEqual(set(full(0, values)), set(sliced(0, values)))
            self.assertIn(token, sliced(0, values))
            generated.append(token)
        self.assertIn(local.tokenizer.eos_token_id, sliced(0, torch.tensor(prompt + generated)))
        self.assertEqual(sliced(0, torch.tensor(prompt + generated + [local.tokenizer.eos_token_id])),
                         [local.tokenizer.eos_token_id])

    def test_oom_splits_real_requests_without_changing_budget(self):
        import torch

        engine = HFBatchEngine(self.local(), 4)
        chunk = [{"index": i, "budget": 123} for i in range(4)]

        def generate(items):
            if len(items) > 1:
                raise torch.cuda.OutOfMemoryError("Synthetic OOM control-flow fixture")
            return [(items[0]["index"], items[0]["budget"])]

        with patch.object(engine, "_generate_chunk", side_effect=generate):
            self.assertEqual(engine._run(chunk), [(i, 123) for i in range(4)])
        self.assertEqual(engine.stats["oom_splits"], 3)


if __name__ == "__main__":
    unittest.main()
