"""Parallel scheduler for the frozen score-v2 policy; inference source is unchanged."""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import signal
import sys
import time
from pathlib import Path

from autodl_rag.core import load_config, public, read_json, records, save_json, write_line
from autodl_rag.data import prepared
from autodl_rag.experiment import export_csv, run_dataset
from autodl_rag.optimize import profiles, verify_selection
from autodl_rag.pipeline import Pipeline
from scripts.predict_parallel import ParallelPipeline


def stop_sequential(config, name):
    """Only terminate this project's matching sequential *test* prediction."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            if "autodl_rag.optimize" not in args or "-m" not in args:
                continue

            def value(flag, default=None):
                return args[args.index(flag) + 1] if flag in args else default

            target = value("--config")
            if value("--stage") != "predict" or value("--name", "score-v2") != name or not target:
                continue
            target = Path(target)
            if not target.is_absolute():
                target = (proc / "cwd").resolve() / target
            if target.resolve() == Path(config).resolve():
                print(f"Stopping sequential score-v2 test PID {proc.name}", flush=True)
                os.kill(int(proc.name), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


class BenchmarkPipeline(Pipeline):
    def __init__(self, cfg, mode, adapter=None):
        super().__init__(cfg, mode=mode, adapter=adapter)
        # Separate fresh caches for every benchmark: otherwise later worker counts
        # would appear fast by simply reading answers from the first benchmark.
        self.cache_dir = Path(cfg["benchmark_cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def selected_config(base, folder):
    from autodl_rag.core import digest_file

    selection = verify_selection(base, folder)
    confirmation = read_json(folder / "confirmation.json")
    if confirmation["selection_sha256"] != digest_file(folder / "selection.json"):
        raise ValueError("Confirmation belongs to another selection")
    if not confirmation["passed"]:
        raise ValueError("Confirmation regressed; inspect it before full prediction")
    name = selection["selected"]
    return profiles(base)[name], folder / name / "test.jsonl"


def benchmark_rows(source, output, size):
    """Use the next uncompleted source rows, validating the existing ordered prefix."""
    iterator = iter(records(source))
    done = 0
    if output.exists():
        with output.open(encoding="utf-8") as f:
            for line in f:
                # A terminated writer may leave a torn last row. Benchmarking
                # ignores it read-only; production resume performs its own backup.
                if not line.endswith("\n"):
                    break
                if not line.strip():
                    continue
                old = json.loads(line)
                row = next(iterator, None)
                if row is None or row["id"] != old["id"]:
                    raise ValueError("Existing predictions are not an exact dataset prefix")
                done += 1
    return done, [public(r) for r in itertools.islice(iterator, size)]


def benchmark(cfg, source, output, folder, workers, size):
    done, rows = benchmark_rows(source, output, size)
    if not rows:
        raise ValueError("No uncompleted questions remain for a benchmark")
    dest = folder / "benchmarks" / f"workers-{workers}-{time.time_ns()}"
    dest.mkdir(parents=True)
    public_source = dest / "input.jsonl"
    with public_source.open("w", encoding="utf-8") as f:
        for row in rows:
            write_line(f, row)
    runtime_cfg = {**cfg, "benchmark_cache_dir": str(dest / "initial-cache")}
    pipeline = ParallelPipeline(runtime_cfg, public_source, workers, mode="rag",
                                factory=BenchmarkPipeline, limit=len(rows))
    cached = 0
    started = time.perf_counter()
    processing = 0
    try:
        with (dest / "traces.jsonl").open("w", encoding="utf-8") as f:
            for i, row in enumerate(rows, 1):
                trace = pipeline.run(row)
                cached += bool(trace["initial_cache_hit"])
                trace.pop("_samples", None)
                write_line(f, trace)
                print(f"benchmark {i}/{len(rows)} workers={workers}", flush=True)
        processing = time.perf_counter() - pipeline.started
    finally:
        pipeline.close()
    total = sum(1 for _ in records(source))
    seconds = processing / len(rows)
    result = {
        "workers": workers, "rows": len(rows), "cache_hits": cached,
        "startup_seconds": pipeline.startup_seconds,
        "processing_seconds": processing, "seconds_per_row": seconds,
        "rows_per_hour": 3600 / seconds,
        "already_saved": done,
        "estimated_remaining_hours": (total - done) * seconds / 3600,
        "total_wall_seconds": time.perf_counter() - started,
        "note": "Fresh independent benchmark cache; production predictions untouched. ETA is an estimate from this small question sample, excluding model startup.",
    }
    save_json(dest / "runtime.json", result)
    print(result, flush=True)
    print(f"Benchmark report: {dest / 'runtime.json'}", flush=True)


def predict(cfg, source, output, workers):
    pipeline = ParallelPipeline(cfg, source, workers, mode="rag")
    started = time.perf_counter()
    status = "failed"
    try:
        result = run_dataset(cfg, "test", "rag", output, resume=True, pipeline=pipeline)
        if not output.with_suffix(".csv").exists():
            export_csv(cfg, output, output.with_suffix(".csv"))
        status = "complete"
        print(result, flush=True)
        print(f"Submission: {output.with_suffix('.csv')}", flush=True)
    finally:
        processing = time.perf_counter() - pipeline.started if pipeline.started else 0
        pipeline.close()
        report = {
            "status": status, "workers": workers, "new_rows": pipeline.completed,
            "startup_seconds": pipeline.startup_seconds, "processing_seconds": processing,
            "seconds_per_new_row": processing / pipeline.completed if pipeline.completed else None,
            "total_wall_seconds": time.perf_counter() - started,
        }
        save_json(output.with_suffix(".parallel-runtime.json"), report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", default="score-v2")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--benchmark", type=int, default=0, metavar="N")
    parser.add_argument("--stop-sequential", action="store_true")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a single directory name")
    if args.benchmark < 0:
        parser.error("--benchmark must be nonnegative")
    base = load_config(args.config)
    if args.workers > 1 and not base["load_in_4bit"]:
        parser.error("Multiple workers require the validated 4-bit configuration")
    folder = Path(base["work_dir"]) / "optimization" / args.name
    if args.stop_sequential:
        if not sys.platform.startswith("linux"):
            parser.error("--stop-sequential requires Linux /proc")
        stop_sequential(args.config, args.name)
    with (folder / ".run.lock").open("a") as lock:
        deadline = time.monotonic() + (30 if args.stop_sequential else 0)
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as e:
                if time.monotonic() >= deadline:
                    raise RuntimeError("A score-v2 stage is still running; finish tune/confirmation or stop the previous prediction first") from e
                time.sleep(.5)
        cfg, output = selected_config(base, folder)
        source = prepared(cfg) / "test.jsonl"
        if args.benchmark:
            benchmark(cfg, source, output, folder, args.workers, args.benchmark)
        else:
            predict(cfg, source, output, args.workers)


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Interrupted; retaining the completed prediction prefix")

    signal.signal(signal.SIGTERM, interrupt)
    main()
