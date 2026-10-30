"""Always-plan optimization keeps public planner/final-answer inputs unchanged."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from autodl_rag.core import public, records
from autodl_rag.data import prepare, prepared
from autodl_rag.pipeline import Pipeline
from autodl_rag.retrieval import build_index
from scripts.batched_qwen import PlannerFirstPipeline
from test_system import FakeQwen, configuration


class PlannerQwen(FakeQwen):
    def __init__(self, query_reply):
        super().__init__()
        self.query_reply, self.requests = query_reply, []
        self.plan_count = 0

    def generate(self, system, payload, **kwargs):
        self.requests.append(copy.deepcopy(payload))
        if "SECRET_GOLD" in json.dumps(payload) or "rationale" in payload:
            raise AssertionError("Labels entered planner or answer model")
        if payload.get("task") == "retrieval_plan":
            self.plan_count += 1
            return {"raw": self.query_reply.replace("{n}", str(self.plan_count)),
                    "allowed": {d["id"] for d in payload["documents"]}}
        return super().generate(system, payload, **kwargs)


class PlannerFirstTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cfg = configuration(Path(self.temp.name))
        prepare(self.cfg)
        build_index(self.cfg)
        self.cfg.update(retrieval_policy="coverage", retrieval_plan_trigger="always",
                        retrieval_plan_rounds=1)
        self.row = next(iter(records(prepared(self.cfg) / "tune.jsonl")))

    def tearDown(self):
        self.temp.cleanup()

    def pair(self, reply):
        a, b = PlannerQwen(reply), PlannerQwen(reply)
        original = Pipeline(self.cfg, "rag", qwen=a)
        optimized = PlannerFirstPipeline(self.cfg, "rag", qwen=b)
        original.cache_dir = Path(tempfile.mkdtemp(dir=self.temp.name))
        optimized.cache_dir = Path(tempfile.mkdtemp(dir=self.temp.name))
        try:
            x, _ = original.initial(self.row)
            y, _ = optimized.initial(self.row)
        finally:
            original.retriever.db.close()
            optimized.retriever.db.close()
        return a, b, x, y

    def test_successful_plan_removes_only_discarded_answer_and_preserves_inputs(self):
        a, b, x, y = self.pair('{"retrieval_queries":["Synthetic Entity mother"]}')
        self.assertEqual(x["initial_calls"], 3)
        self.assertEqual(y["initial_calls"], 2)
        self.assertEqual(a.requests[1:], b.requests)
        self.assertEqual(x["answer"], y["answer"])
        self.assertEqual(x["documents"], y["documents"])
        self.assertEqual(x["prompt_documents"], y["prompt_documents"])
        self.assertNotEqual(x["cache_key"], y["cache_key"])
        self.assertEqual(y["execution_strategy"], "plan_before_answer_v1")

    def test_empty_or_invalid_plan_still_generates_a_real_answer_from_original_docs(self):
        for reply in ('{"retrieval_queries":[]}', 'invalid JSON'):
            with self.subTest(reply=reply):
                a, b, x, y = self.pair(reply)
                self.assertEqual(x["answer"], y["answer"])
                self.assertEqual(a.requests[0], b.requests[-1])
                self.assertEqual(x["initial_calls"], 2)
                self.assertEqual(y["initial_calls"], 2)

    def test_unknown_trigger_keeps_original_flow(self):
        self.cfg["retrieval_plan_trigger"] = "unknown"
        qwen = PlannerQwen('{"retrieval_queries":["Synthetic Entity mother"]}')
        pipeline = PlannerFirstPipeline(self.cfg, "rag", qwen=qwen)
        try:
            initial, _ = pipeline.initial(public(self.row))
        finally:
            pipeline.retriever.db.close()
        self.assertEqual(pipeline.execution_strategy, "original")
        self.assertEqual(initial["initial_calls"], 1)
        self.assertNotIn("task", qwen.requests[0])

    def test_two_plan_rounds_remove_two_unused_answers_without_changing_plan_inputs(self):
        self.cfg["retrieval_plan_rounds"] = 2
        a, b, x, y = self.pair('{"retrieval_queries":["Synthetic Entity mother{n}"]}')
        self.assertEqual(x["initial_calls"], 5)
        self.assertEqual(y["initial_calls"], 3)
        self.assertEqual([a.requests[i] for i in (1, 3, 4)], b.requests)
        self.assertEqual(x["answer"], y["answer"])


if __name__ == "__main__":
    unittest.main()
