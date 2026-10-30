"""Validate and run score-v2 with one shared Qwen and real HF batched generation."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import itertools
import os
import signal
import sys
import time
from pathlib import Path

from autodl_rag.core import (
    digest_file,
    fingerprint,
    load_config,
    normalized,
    public,
    read_json,
    records,
    save_json,
    write_line,
)
from autodl_rag.data import prepared
from autodl_rag.experiment import evaluate, export_csv, run_dataset
from autodl_rag.models import model_identity
from autodl_rag.pipeline import Pipeline, source_identity
from scripts.batched_qwen import SharedBatchPipeline
from scripts.score_predict_fast import selected_config, stop_sequential


def backend_identity(cfg, batch_size, token_cap):
    import torch

    gpu = {"cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if gpu["cuda_available"]:
        gpu.update(name=torch.cuda.get_device_name(cfg["qwen_device"]),
                   capability=list(torch.cuda.get_device_capability(cfg["qwen_device"])))
    return {
        "config": fingerprint(cfg), "inference_source": source_identity(),
        "model": model_identity(cfg["qwen_path"]),
        "batch_size": batch_size, "token_cap": token_cap,
        "gpu": gpu,
        "backend_files": {name: digest_file(Path(__file__).with_name(name))
                          for name in ("batched_qwen.py", "score_predict_batch.py")},
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "bitsandbytes", "lm-format-enforcer")},
        "confirmation_sha256": digest_file(prepared(cfg) / "confirmation.jsonl"),
    }


def validation_file(folder, batch_size, token_cap):
    return folder / "batch-validation" / f"batch-{batch_size}-tokens-{token_cap}.json"


def require_pilot(cfg, folder, batch_size, token_cap):
    pilot = read_json(validation_file(folder, batch_size, token_cap))
    if not pilot["passed"] or pilot["identity"] != backend_identity(cfg, batch_size, token_cap):
        raise ValueError("Run a passing --benchmark with this exact backend/config/model/batch size first")
    return pilot


def stop_parallel(config, name):
    stop_sequential(config, name)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not any(Path(arg).name in ("score_predict_fast.py", "score_predict_batch.py") for arg in args):
                continue

            def value(flag, default=None):
                return args[args.index(flag) + 1] if flag in args else default

            target = value("--config")
            if not target or value("--name", "score-v2") != name:
                continue
            target = Path(target)
            if not target.is_absolute():
                target = (proc / "cwd").resolve() / target
            if target.resolve() == Path(config).resolve():
                print(f"Stopping previous prediction scheduler PID {proc.name}", flush=True)
                os.kill(int(proc.name), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def validate(cfg, folder, batch_size, token_cap, size):
    from autodl_rag.models import LocalModel, seed_everything
    from autodl_rag.structured import QwenJsonGuard

    identity = backend_identity(cfg, batch_size, token_cap)
    save_json(validation_file(folder, batch_size, token_cap), {
        "identity": identity, "passed": False, "status": "pilot_running",
    })
    seed_everything(cfg["seed"])
    # Exactly one Qwen is loaded, then reused by both sequential and batch pilots.
    local = LocalModel(cfg, "qwen")
    # A shared startup cost must not be charged only to sequential timing.
    local._json_guard = QwenJsonGuard(local.tokenizer, local.model.config.vocab_size)
    dest = folder / "batch-validation" / f"pilot-{time.time_ns()}"
    dest.mkdir(parents=True)
    rows = [public(r) for r in itertools.islice(records(prepared(cfg) / "confirmation.jsonl"), size)]
    if not rows:
        raise ValueError("No confirmation samples available")
    source = dest / "public-input.jsonl"
    with source.open("w", encoding="utf-8") as f:
        for row in rows:
            write_line(f, row)
    timings = {}
    memory = {}
    metrics = {}
    answers = {}
    for mode in ("sequential", "batch"):
        output = dest / f"{mode}.jsonl"
        if mode == "sequential":
            pipeline = Pipeline(cfg, mode="rag", qwen=local)
            pipeline.cache_dir = dest / "sequential-cache"
            pipeline.cache_dir.mkdir()
        else:
            pipeline = SharedBatchPipeline(cfg, source, batch_size, token_cap, local=local,
                                           cache_dir=dest / "batch-cache")
        import torch

        cuda = str(local.device).startswith("cuda")
        if cuda:
            torch.cuda.synchronize(local.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(local.device)
        started = time.perf_counter()
        try:
            with output.open("w", encoding="utf-8") as f:
                for i, row in enumerate(rows, 1):
                    trace = pipeline.run(row)
                    trace.pop("_samples", None)
                    write_line(f, trace)
                    os.fsync(f.fileno())
                    print(f"{mode} pilot: {i}/{len(rows)}", flush=True)
            timings[mode] = time.perf_counter() - started
            if cuda:
                memory[mode] = {
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(local.device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(local.device) / 2**30,
                }
        finally:
            if mode == "sequential":
                pipeline.retriever.db.close()
            else:
                pipeline.close()
        metrics[mode] = evaluate(cfg, output, "confirmation")
        answers[mode] = {r["id"]: normalized(r["final"]["answer"]) for r in records(output)}
        if mode == "batch":
            timings["batch_generation_stats"] = dict(pipeline.broker.engine.stats)
    changed = sum(answers["sequential"][sid] != answer for sid, answer in answers["batch"].items())
    seconds = timings["batch"] / len(rows)
    quality_passed = metrics["batch"]["normalized_em"] >= metrics["sequential"]["normalized_em"]
    speedup = timings["sequential"] / timings["batch"]
    passed = quality_passed and speedup > 1.0
    result = {
        "identity": identity, "passed": passed, "status": "pilot_complete",
        "quality_passed": quality_passed, "speed_passed": speedup > 1.0,
        "rows": len(rows), "model_copies": 1, "metrics": metrics, "timings": timings,
        "gpu_memory": memory,
        "changed_answers": changed, "answer_agreement": 1 - changed / len(rows),
        "seconds_per_row": seconds, "rows_per_hour": 3600 / seconds,
        "observed_speedup": speedup,
        "pilot_dir": str(dest),
        "note": "Real same-question paired pilot with separate empty caches. No claimed Kaggle score or statistical guarantee from this small sample.",
    }
    save_json(validation_file(folder, batch_size, token_cap), result)
    print(f"Pilot passed={passed}; sequential EM={metrics['sequential']['normalized_em']:.4f}; "
          f"batch EM={metrics['batch']['normalized_em']:.4f}; answer changes={changed}; "
          f"speedup={result['observed_speedup']:.2f}x; throughput={result['rows_per_hour']:.1f}/hour", flush=True)
    print(f"Validation: {validation_file(folder, batch_size, token_cap)}", flush=True)


def predict(cfg, source, output, batch_size, token_cap):
    identity = backend_identity(cfg, batch_size, token_cap)
    pipeline = SharedBatchPipeline(cfg, source, batch_size, token_cap)
    status = "failed"
    started = time.perf_counter()
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
        stats = dict(pipeline.broker.engine.stats) if pipeline.broker else {}
        save_json(output.with_suffix(".batch-runtime.json"), {
            "status": status, "backend": "hf_shared_batch", "model_copies": int(pipeline.local is not None),
            "batch_size": batch_size, "token_cap": token_cap, "new_rows": pipeline.completed,
            "startup_seconds": pipeline.startup_seconds, "processing_seconds": processing,
            "seconds_per_new_row": processing / pipeline.completed if pipeline.completed else None,
            "total_wall_seconds": time.perf_counter() - started, "generation_stats": stats,
            "backend_identity": identity,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", default="score-v2")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8, 16), default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=147456)
    parser.add_argument("--benchmark", type=int, default=0, metavar="N")
    parser.add_argument("--stop-existing", action="store_true")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a single directory name")
    if args.benchmark < 0 or args.max_batch_tokens < 1024:
        parser.error("--benchmark must be nonnegative and --max-batch-tokens >=1024")
    base = load_config(args.config)
    folder = Path(base["work_dir"]) / "optimization" / args.name
    cfg, output = selected_config(base, folder)
    if cfg.get("reranker_path"):
        parser.error("Shared Qwen batching currently requires reranker_path=null; it will not silently disable a configured reranker")
    if args.stop_existing:
        if not sys.platform.startswith("linux"):
            parser.error("--stop-existing requires Linux /proc")
        stop_parallel(args.config, args.name)
    with (folder / ".run.lock").open("a") as lock:
        deadline = time.monotonic() + (120 if args.stop_existing else 0)
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as e:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Previous score-v2 job still holds the lock; check its log") from e
                time.sleep(.5)
        if args.benchmark:
            validate(cfg, folder, args.batch_size, args.max_batch_tokens, args.benchmark)
        else:
            require_pilot(cfg, folder, args.batch_size, args.max_batch_tokens)
            predict(cfg, prepared(cfg) / "test.jsonl", output, args.batch_size, args.max_batch_tokens)


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Interrupted; retaining completed output rows")

    signal.signal(signal.SIGTERM, interrupt)
    main()
