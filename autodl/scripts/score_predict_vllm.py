"""Compare a local vLLM service with saved real HF pilots, then predict separately."""

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
from autodl_rag.pipeline import source_identity
from scripts.score_predict_batch import stop_parallel
from scripts.score_predict_fast import benchmark_rows, selected_config
from scripts.vllm_client import RemotePipeline, RemoteQwen, read_manifest


def reference_pilot(cfg, folder, size):
    candidates = []
    for path in (folder / "batch-validation").glob("batch-*-tokens-*.json"):
        report = read_json(path)
        identity = report.get("identity", {})
        if (report.get("passed") and report.get("rows") == size
                and identity.get("config") == fingerprint(cfg)
                and identity.get("inference_source") == source_identity()
                and identity.get("model") == model_identity(cfg["qwen_path"])
                and identity.get("confirmation_sha256") == digest_file(prepared(cfg) / "confirmation.jsonl")):
            trace = Path(report["pilot_dir"]) / "batch.jsonl"
            if trace.is_file() and report["timings"]["batch"] > 0:
                candidates.append((report["timings"]["batch"], path, trace))
    if not candidates:
        raise ValueError("No matching real HF batch pilot. Run the existing HF batch validation with the same sample count first")
    return min(candidates)


def stop_existing(config, name):
    stop_parallel(config, name)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not any(Path(a).name == "score_predict_vllm.py" for a in args):
                continue
            target = args[args.index("--config") + 1] if "--config" in args else None
            run = args[args.index("--name") + 1] if "--name" in args else "score-v2"
            if not target or run != name:
                continue
            target = Path(target) if Path(target).is_absolute() else (proc / "cwd").resolve() / target
            if target.resolve() == Path(config).resolve():
                print(f"Stopping previous vLLM client PID {proc.name}", flush=True)
                os.kill(int(proc.name), signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def proof(cfg, manifest, reference, concurrency):
    _, report, trace = reference
    return {
        "config": fingerprint(cfg), "deployment": fingerprint(manifest), "concurrency": concurrency,
        "source": source_identity(), "reference": digest_file(report), "reference_traces": digest_file(trace),
        "client_packages": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
        "files": {name: digest_file(Path(__file__).with_name(name))
                  for name in ("vllm_client.py", "score_predict_vllm.py", "batched_qwen.py")},
    }


def validate(cfg, original_cfg, folder, manifest, reference, concurrency, size, report_path, legacy_output, min_speedup):
    save_json(report_path, {"passed": False, "status": "running"})
    rows = [public(r) for r in itertools.islice(records(prepared(cfg) / "confirmation.jsonl"), size)]
    old = list(records(reference[2]))
    if [r["id"] for r in rows] != [r["id"] for r in old]:
        raise ValueError("Reference pilot questions differ; no fair paired accuracy comparison")
    dest = folder / "vllm-validation" / f"pilot-{time.time_ns()}"
    dest.mkdir(parents=True)
    source = dest / "public-input.jsonl"
    with source.open("w", encoding="utf-8") as f:
        for row in rows:
            write_line(f, row)
    remote = RemoteQwen(cfg, manifest, cache_salt=f"pilot-{time.time_ns()}")
    pipeline = RemotePipeline(cfg, source, remote, concurrency, cache_dir=dest / "initial-cache")
    output = dest / "vllm.jsonl"
    started = time.perf_counter()
    try:
        with output.open("w", encoding="utf-8") as f:
            for i, row in enumerate(rows, 1):
                trace = pipeline.run(row)
                trace.pop("_samples", None)
                write_line(f, trace)
                os.fsync(f.fileno())
                print(f"vLLM pilot {i}/{len(rows)}", flush=True)
        elapsed = time.perf_counter() - started
    finally:
        pipeline.close()
    baseline = evaluate(original_cfg, reference[2], "confirmation")
    metrics = evaluate(cfg, output, "confirmation")
    speedup = reference[0] / elapsed
    quality = metrics["normalized_em"] >= baseline["normalized_em"]
    changed = sum(normalized(a["final"]["answer"]) != normalized(b["final"]["answer"])
                  for a, b in zip(old, records(output)))
    done, _ = benchmark_rows(prepared(cfg) / "test.jsonl", legacy_output, 0)
    total = sum(1 for _ in records(prepared(cfg) / "test.jsonl"))
    result = {
        "passed": quality and speedup >= min_speedup, "min_speedup": min_speedup,
        "identity": proof(cfg, manifest, reference, concurrency),
        "rows": len(rows), "hf_metrics": baseline, "vllm_metrics": metrics,
        "seconds": elapsed, "reference_seconds": reference[0], "speedup": speedup,
        "rows_per_hour": len(rows) * 3600 / elapsed, "changed_answers": changed,
        "legacy_completed": done, "total_test_rows": total,
        "estimated_hf_remaining_hours": (total - done) * reference[0] / len(rows) / 3600,
        "estimated_vllm_full_test_hours": total * elapsed / len(rows) / 3600,
        "request_stats": dict(remote.stats), "reference_report": str(reference[1]),
        "precision": "vLLM BF16; original HF report uses its recorded precision",
        "note": "Real paired answers, historical HF timing from another run/load. Speed ratio is indicative, not a simultaneous controlled benchmark or Kaggle score guarantee.",
    }
    save_json(report_path, result)
    print(f"vLLM pilot passed={result['passed']}; HF EM={baseline['normalized_em']:.4f}; "
          f"vLLM EM={metrics['normalized_em']:.4f}; changes={changed}; "
          f"observed speedup={speedup:.2f}x; throughput={result['rows_per_hour']:.1f}/hour", flush=True)
    print(f"Report: {report_path}", flush=True)


def predict(cfg, folder, output, manifest, concurrency):
    remote = RemoteQwen(cfg, manifest)
    pipeline = RemotePipeline(cfg, prepared(cfg) / "test.jsonl", remote, concurrency)
    started = time.perf_counter()
    try:
        result = run_dataset(cfg, "test", "rag", output, resume=True, pipeline=pipeline)
        if not output.with_suffix(".csv").exists():
            export_csv(cfg, output, output.with_suffix(".csv"))
        print(result, flush=True)
        print(f"BF16 vLLM submission: {output.with_suffix('.csv')}", flush=True)
    finally:
        pipeline.close()
        save_json(output.with_suffix(".runtime.json"), {
            "backend": "vllm_http_bf16", "concurrency": concurrency, "new_rows": pipeline.completed,
            "wall_seconds": time.perf_counter() - started, "stats": dict(remote.stats),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("stop", "validate", "predict", "all"), default="all")
    parser.add_argument("--name", default="score-v2")
    parser.add_argument("--concurrency", type=int, choices=(8, 16, 32, 64), default=32)
    parser.add_argument("--pilot-size", type=int, default=32)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    args = parser.parse_args()
    if (not args.name or Path(args.name).name != args.name or args.name in (".", "..")
            or args.pilot_size < 1 or args.min_speedup <= 1):
        parser.error("Invalid experiment name or pilot size")
    base = load_config(args.config)
    folder = Path(base["work_dir"]) / "optimization" / args.name
    original, legacy_output = selected_config(base, folder)
    if original.get("reranker_path"):
        parser.error("The current vLLM adapter requires reranker_path=null")
    if args.stage == "stop":
        if not sys.platform.startswith("linux"):
            parser.error("--stage stop requires Linux /proc")
        stop_existing(args.config, args.name)
        # Wait for the original writer to finish its safe shutdown.
    with (folder / ".run.lock").open("a") as lock:
        deadline = time.monotonic() + 120
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if time.monotonic() > deadline:
                    raise RuntimeError("Another score-v2 job is running; stop it before changing engines") from error
                time.sleep(.5)
        if args.stage == "stop":
            return
        manifest = read_manifest(Path(base["work_dir"]).parent / "vllm/server.json")
        cfg = {**original, "load_in_4bit": False, "inference_backend": "vllm_http_bf16",
               "vllm_deployment": fingerprint(manifest)}
        output = legacy_output.with_name("test.vllm.jsonl")
        reference = reference_pilot(original, folder, args.pilot_size)
        report_path = folder / "vllm-validation" / f"concurrency-{args.concurrency}.json"
        if args.stage in ("validate", "all"):
            validate(cfg, original, folder, manifest, reference, args.concurrency, args.pilot_size,
                     report_path, legacy_output, args.min_speedup)
        if args.stage in ("predict", "all"):
            report = read_json(report_path)
            if (not report["passed"] or report["speedup"] < args.min_speedup
                    or report["identity"] != proof(cfg, manifest, reference, args.concurrency)):
                raise ValueError("vLLM pilot did not pass, or deployment changed; inspect report before prediction")
            save_json(output.with_name("vllm-config.json"), cfg)
            predict(cfg, folder, output, manifest, args.concurrency)


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Interrupted; completed BF16 predictions retained")

    signal.signal(signal.SIGTERM, interrupt)
    main()
