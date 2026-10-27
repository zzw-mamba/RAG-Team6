"""Synthetic scheduling tests, never GPU inference or research results."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from autodl_rag.core import InvalidOutput, public, read_json, records
from autodl_rag.data import prepare, prepared
from autodl_rag.experiment import export_csv, run_dataset
from scripts.predict_parallel import ParallelPipeline, _initialize
from test_system import configuration


class SyntheticPipeline:
    def __init__(self, cfg, mode, adapter):
        self.cfg = cfg

    def run(self, row):
        if set(row) != {"id", "question", "ctxs"}:
            raise AssertionError("Labels entered the inference worker")
        if row["id"] == self.cfg.get("synthetic_failure"):
            raise InvalidOutput("Synthetic failure, no placeholder")
        time.sleep(0.03 if row["id"] == "0" else 0.001)
        jobs = Path(self.cfg["work_dir"]) / "synthetic-jobs"
        jobs.mkdir(exist_ok=True)
        (jobs / row["id"]).write_text("synthetic scheduling fixture")
        return {
            "id": row["id"],
            "final": {"answer": "Synthetic " + row["id"]},
            "stop_reason": "critic_stop",
        }


class ParallelPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = configuration(self.root)
        prepare(self.cfg)
        self.source = prepared(self.cfg) / "test.jsonl"
        self.rows = list(records(self.source))

    def tearDown(self):
        self.temp.cleanup()

    def pool(self, **kwargs):
        return ParallelPipeline(
            self.cfg, self.source, 2, mode="critic", factory=SyntheticPipeline, **kwargs
        )

    def test_out_of_order_work_keeps_source_order_and_public_fields(self):
        pool = self.pool()
        try:
            result = [pool.run(public(r)) for r in self.rows]
        finally:
            pool.close()
        self.assertEqual([r["id"] for r in result], [r["id"] for r in self.rows])
        self.assertEqual(pool.completed, 3)

    def test_resume_keeps_sequential_prefix_and_official_export_order(self):
        output = self.root / "predictions.jsonl"
        cfg = self.cfg

        class Interrupted:
            def run(self, row):
                if row["id"] != "0":
                    raise InvalidOutput("Synthetic interruption")
                return SyntheticPipeline(cfg, "critic", None).run(row)

        with self.assertRaises(InvalidOutput):
            run_dataset(cfg, "test", "critic", output, pipeline=Interrupted())
        prefix = output.read_bytes()
        pool = self.pool()
        try:
            result = run_dataset(cfg, "test", "critic", output, resume=True, pipeline=pool)
        finally:
            pool.close()
        self.assertEqual(result["rows"], 3)
        self.assertTrue(output.read_bytes().startswith(prefix))
        self.assertEqual(pool.completed, 2)
        csv = self.root / "submission.csv"
        export_csv(cfg, output, csv)
        self.assertEqual(
            csv.read_text().splitlines()[1:], ["2,Synthetic 2", "0,Synthetic 0", "1,Synthetic 1"]
        )

    def test_failure_stops_at_failed_row_without_writing_later_results(self):
        self.cfg["synthetic_failure"] = "1"
        output = self.root / "failed.jsonl"
        pool = self.pool()
        try:
            with self.assertRaises(InvalidOutput):
                run_dataset(self.cfg, "test", "critic", output, pipeline=pool)
        finally:
            pool.close()
        self.assertEqual([r["id"] for r in records(output)], ["0"])
        self.assertEqual(read_json(output.with_suffix(".failure.json"))["id"], "1")

    def test_benchmark_does_not_prefetch_beyond_limit(self):
        pool = self.pool(limit=1)
        try:
            pool.run(public(self.rows[0]))
        finally:
            pool.close()
        jobs = Path(self.cfg["work_dir"]) / "synthetic-jobs"
        self.assertEqual(sorted(p.name for p in jobs.iterdir()), ["0"])

    def test_model_load_failure_releases_other_workers_from_startup_barrier(self):
        barrier = Mock()

        def failed_load(*args, **kwargs):
            raise RuntimeError("Synthetic model-load failure")

        with self.assertRaisesRegex(RuntimeError, "model-load failure"):
            _initialize({}, "rl", None, barrier, failed_load)
        barrier.abort.assert_called_once()
        barrier.wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
