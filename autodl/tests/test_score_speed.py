"""Scheduling tests use explicit synthetic backends, never model speed claims."""

from __future__ import annotations

import contextlib
import io
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autodl_rag.core import InvalidOutput, read_json, records, write_line
from autodl_rag.data import prepare, prepared
from autodl_rag.experiment import run_dataset
from scripts.predict_parallel import ParallelPipeline
from scripts.score_predict_fast import BenchmarkPipeline, benchmark_rows, predict, stop_sequential
from test_parallel_prediction import SyntheticPipeline
from test_system import configuration


class ScoreSpeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = configuration(self.root)
        prepare(self.cfg)
        self.source = prepared(self.cfg) / "test.jsonl"
        self.output = self.root / "predictions.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def test_benchmark_skips_completed_prefix_and_strips_labels(self):
        rows = list(records(self.source))
        with self.output.open("w") as f:
            write_line(f, {"id": rows[0]["id"]})
        count, pending = benchmark_rows(self.source, self.output, 1)
        self.assertEqual(count, 1)
        self.assertEqual(pending[0]["id"], rows[1]["id"])
        self.assertEqual(set(pending[0]), {"id", "question", "ctxs"})

    def test_benchmark_refuses_nonprefix_production_results(self):
        with self.output.open("w") as f:
            write_line(f, {"id": "wrong"})
        with self.assertRaises(ValueError):
            benchmark_rows(self.source, self.output, 2)

    def test_benchmark_ignores_torn_tail_without_mutating_production(self):
        self.output.write_text('{"id":"0"}\n{"id":')
        original = self.output.read_bytes()
        done, pending = benchmark_rows(self.source, self.output, 1)
        self.assertEqual(done, 1)
        self.assertEqual(pending[0]["id"], "1")
        self.assertEqual(self.output.read_bytes(), original)

    def test_benchmark_uses_an_independent_cache_instead_of_production_answers(self):
        target = self.root / "fresh-benchmark-cache"
        with patch("autodl_rag.pipeline.Pipeline.__init__", return_value=None):
            pipeline = BenchmarkPipeline({"benchmark_cache_dir": str(target)}, "rag")
        self.assertEqual(pipeline.cache_dir, target)
        self.assertTrue(target.is_dir())

    def test_stop_only_matches_same_config_name_and_sequential_predict_stage(self):
        proc = self.root / "proc"
        proc.mkdir()
        config = str(self.root / "config.json")
        common = ["python", "-m", "autodl_rag.optimize", "--config", config]
        commands = {
            100: common + ["--stage", "predict"],
            101: common + ["--stage", "tune"],
            102: common + ["--stage", "confirmation"],
            103: common + ["--stage", "predict", "--name", "another-run"],
            104: ["python", "other_project.py"],
            105: ["python", "-m", "autodl_rag.optimize", "--config", "/other/config.json",
                  "--stage", "predict"],
        }
        for pid, args in commands.items():
            folder = proc / str(pid)
            folder.mkdir()
            (folder / "cmdline").write_bytes("\0".join(args).encode())
        with patch("scripts.score_predict_fast.Path",
                   side_effect=lambda p: proc if p == "/proc" else Path(p)), \
                patch("scripts.score_predict_fast.os.kill") as kill, \
                contextlib.redirect_stdout(io.StringIO()):
            stop_sequential(config, "score-v2")
        kill.assert_called_once_with(100, signal.SIGTERM)

    def test_parallel_rag_resumes_sequential_prefix_without_parent_model_and_exports(self):
        class Interrupted:
            def run(self, row):
                if row["id"] != "0":
                    raise InvalidOutput("Synthetic interruption")
                return {"id": row["id"], "final": {"answer": "Synthetic 0"},
                        "stop_reason": "vanilla"}

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(InvalidOutput):
                run_dataset(self.cfg, "test", "rag", self.output, pipeline=Interrupted())
            original = self.output.read_text()

            def scheduler(cfg, source, workers, mode):
                return ParallelPipeline(cfg, source, workers, mode=mode, factory=SyntheticPipeline)

            with patch("scripts.score_predict_fast.ParallelPipeline", side_effect=scheduler), \
                    patch("autodl_rag.models.LocalModel", side_effect=AssertionError("Parent loaded a model")):
                predict(self.cfg, self.source, self.output, 2)
        self.assertTrue(self.output.read_text().startswith(original))
        self.assertEqual([r["id"] for r in records(self.output)], ["0", "1", "2"])
        self.assertEqual(read_json(self.output.with_suffix(".validation.json"))["rows"], 3)
        self.assertEqual(read_json(self.output.with_suffix(".parallel-runtime.json"))["new_rows"], 2)


if __name__ == "__main__":
    unittest.main()
