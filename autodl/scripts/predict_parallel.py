"""Bounded, ordered inference workers; reuse the existing prediction protocol."""

from __future__ import annotations

import argparse
import fcntl
import itertools
import multiprocessing
import os
import signal
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from autodl_rag.core import digest_file, load_config, public, records, save_json, write_line
from autodl_rag.data import prepared
from autodl_rag.experiment import export_csv, run_dataset

_pipeline = None


def _initialize(cfg, mode, adapter, barrier, factory):
    global _pipeline
    try:
        if factory is None:
            from autodl_rag.models import seed_everything
            from autodl_rag.pipeline import Pipeline

            seed_everything(cfg["seed"])
            factory = Pipeline
        _pipeline = factory(cfg, mode=mode, adapter=adapter)
        barrier.wait(timeout=600)
    except BaseException:
        barrier.abort()
        raise


def _ready():
    return os.getpid()


def _run(row):
    return _pipeline.run(row)


class ParallelPipeline:
    def __init__(self, cfg, source, workers, mode="rl", adapter=None, factory=None, limit=0):
        self.cfg, self.source, self.workers = cfg, source, workers
        self.mode, self.adapter, self.factory = mode, adapter, factory
        self.limit = limit
        self.pool = None
        self.pending = deque()
        self.completed = 0
        self.startup_seconds = 0.0
        self.started = None

    def _fill(self):
        while len(self.pending) < self.workers * 2:
            row = next(self.iterator, None)
            if row is None:
                break
            self.pending.append((row["id"], self.pool.submit(_run, public(row))))

    def run(self, row):
        if self.pool is None:
            # run_dataset has already verified signature and completed prefix.
            self.iterator = iter(records(self.source))
            if self.limit:
                self.iterator = itertools.islice(self.iterator, self.limit)
            for candidate in self.iterator:
                if candidate["id"] == row["id"]:
                    break
            else:
                raise ValueError("Resume row missing from prepared test data")
            ctx = multiprocessing.get_context("spawn")
            started = time.perf_counter()
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=ctx,
                initializer=_initialize,
                initargs=(
                    self.cfg,
                    self.mode,
                    self.adapter,
                    ctx.Barrier(self.workers),
                    self.factory,
                ),
            )
            ready = [self.pool.submit(_ready) for _ in range(self.workers)]
            for future in ready:
                future.result()
            self.startup_seconds = time.perf_counter() - started
            self.started = time.perf_counter()
            self.pending.append((row["id"], self.pool.submit(_run, public(row))))
            self._fill()
        expected, future = self.pending.popleft()
        if expected != row["id"]:
            raise ValueError("Parallel output would violate official question order")
        trace = future.result()
        self.completed += 1
        self._fill()
        return trace

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)


def stop_existing(output):
    """Stop only the old sequential test command writing this exact output."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            if "autodl_rag" not in args:
                continue

            def value(flag):
                return args[args.index(flag) + 1] if flag in args else None

            if value("--split") != "test" or value("--mode") != "rl":
                continue
            target = value("--output")
            if target and Path(target).is_absolute() and Path(target).resolve() == output:
                print(f"Stopping old sequential test prediction PID {proc.name}", flush=True)
                os.kill(int(proc.name), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/autodl-qwen-image.json"),
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--run-name", default="pilot64")
    parser.add_argument(
        "--limit", type=int, default=0, help="Benchmark only: separate output required"
    )
    parser.add_argument("--output")
    parser.add_argument("--stop-existing", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    cfg = load_config(args.config)
    if not cfg["load_in_4bit"] and args.workers > 1:
        parser.error("Two workers require the existing 4-bit configuration")
    work = Path(cfg["work_dir"])
    output = Path(args.output or work / "test" / args.run_name / "rl.jsonl").resolve()
    if args.limit and not args.output:
        parser.error("A limited benchmark must use its own --output")
    production = (work / "test" / args.run_name / "rl.jsonl").resolve()
    if args.limit and output == production:
        parser.error("Benchmark output cannot be the submission prediction file")
    adapter = (work / "training" / args.run_name / "selection.adapter.txt").read_text().strip()
    lock_path = Path(__file__).resolve().parents[1] / ".cache/run-all.lock"
    lock_path.parent.mkdir(exist_ok=True)
    if args.stop_existing:
        stop_existing(production)
    with lock_path.open("a") as lock:
        deadline = time.monotonic() + 45
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Another job holds run-all.lock; inspect its log before retrying"
                    ) from None
                time.sleep(0.5)
        pipeline = ParallelPipeline(
            cfg, prepared(cfg) / "test.jsonl", args.workers, adapter=adapter, limit=args.limit
        )
        started = time.perf_counter()
        status = "failed"
        try:
            result = run_dataset(
                cfg,
                "test",
                "rl",
                output,
                adapter=adapter,
                limit=args.limit,
                resume=True,
                pipeline=pipeline,
            )
            if not args.limit and not output.with_suffix(".csv").exists():
                export_csv(cfg, output, output.with_suffix(".csv"))
            status = "complete"
            print(result, flush=True)
        finally:
            processing = time.perf_counter() - pipeline.started if pipeline.started else 0
            pipeline.close()
            runtime = {
                "status": status,
                "workers": args.workers,
                "scheduler_sha256": digest_file(Path(__file__)),
                "new_rows": pipeline.completed,
                "startup_seconds": pipeline.startup_seconds,
                "processing_seconds": processing,
                "seconds_per_new_row": processing / pipeline.completed
                if pipeline.completed
                else None,
                "total_wall_seconds": time.perf_counter() - started,
                "note": "Same Pipeline/config/adapter and ordered writer; runtime scheduling only. GPU throughput must be measured.",
            }
            save_json(output.with_suffix(".parallel-runtime.json"), runtime)
            with output.with_suffix(".parallel-runs.jsonl").open("a", encoding="utf-8") as history:
                write_line(history, runtime)


if __name__ == "__main__":

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Prediction interrupted; retaining completed output rows")

    signal.signal(signal.SIGTERM, interrupt)
    main()
