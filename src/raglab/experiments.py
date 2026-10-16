from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from raglab.config import PipelineConfig
from raglab.data import file_sha256, iter_examples, iter_records, write_record
from raglab.pipeline import build_pipeline
from raglab.prompts import PROMPT_VERSION


def git_revision() -> dict:
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def source_sha256() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def recover_trace_tail(path: Path) -> None:
    """Only discard a final unterminated line; malformed complete lines still raise."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
        end = stream.tell()
        position = end
        last_newline = -1
        while position > 0:
            start = max(0, position - 8192)
            stream.seek(start)
            block = stream.read(position - start)
            found = block.rfind(b"\n")
            if found != -1:
                last_newline = start + found
                break
            position = start
        stream.truncate(last_newline + 1)
    print(
        "Recovered interrupted final trace line; the affected question will run again",
        file=sys.stderr,
    )


def run_inference(
    input_path: Path,
    config: PipelineConfig,
    run_dir: Path,
    limit: int | None = None,
    resume: bool = False,
) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    identity = {
        "input_sha256": file_sha256(input_path),
        "config": config.to_dict(),
        "limit": limit,
        "prompt_version": PROMPT_VERSION,
        "source_sha256": source_sha256(),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    trace_path, meta_path = run_dir / "traces.jsonl", run_dir / "run.json"
    if resume:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata["fingerprint"] != fingerprint:
            raise ValueError("Resume refused: input, config, limit, or prompt version changed")
        if metadata.get("git") != git_revision():
            raise ValueError(
                "Resume refused: code revision or dirty status changed; start a new run"
            )
        recover_trace_tail(trace_path)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            **identity,
            "fingerprint": fingerprint,
            "git": git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "timing": "Per-question retrieval + model calls; model loading excluded. HTTP cold start unknown.",
            "status": "running",
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["status"] = "running"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    done = {}
    if trace_path.exists():
        for record in iter_records(trace_path):
            key = str(record["id"])
            if key in done:
                raise ValueError(f"Duplicate completed ID: {key}")
            done[key] = record["question"]
    pipeline, seen, count = None, set(), 0
    with trace_path.open("a", encoding="utf-8") as stream:
        for example in itertools.islice(iter_examples(input_path), limit):
            count += 1
            key = str(example.id)
            seen.add(key)
            if key in done:
                if done[key] != example.question:
                    raise ValueError("Trace question mismatch")
                continue
            if pipeline is None:
                pipeline = build_pipeline(config)
            trace = pipeline.run(example.id, example.question, example.contexts)
            write_record(stream, trace.to_dict())
            stream.flush()
            os.fsync(stream.fileno())
            print(f"completed {count}: {key} ({trace.stop_reason})", file=sys.stderr)
    if set(done) - seen:
        raise ValueError("Existing traces include IDs outside this input selection")
    if not count:
        raise ValueError("Input selection was empty")
    metadata["status"], metadata["n_predictions"] = "complete", count
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def comparison(run_dirs: list[Path]) -> dict:
    if len(run_dirs) < 2:
        raise ValueError("Compare at least two runs")
    baseline, reference_ids, rows = None, None, []
    for directory in run_dirs:
        meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if meta["status"] != "complete" or metrics["coverage"] != 1:
            raise ValueError("Comparison requires complete runs with full evaluation coverage")
        config = meta["config"]
        signature = {
            "input_sha256": meta["input_sha256"],
            "prompt_version": meta["prompt_version"],
            "source_sha256": meta["source_sha256"],
            "limit": meta["limit"],
            "generator": config["generator"],
            "retriever": config["retriever"],
            "top_k": config["top_k"],
            "max_revisions": config["max_revisions"],
            "format_retries": config["format_retries"],
        }
        ids = [
            (str(t["id"]), t["question"], t["documents"])
            for t in iter_records(directory / "traces.jsonl")
        ]
        if baseline is None:
            baseline, reference_ids = signature, ids
        elif signature != baseline or ids != reference_ids:
            raise ValueError(
                "Runs differ in input, retrieval, generator, budgets, or document order"
            )
        rows.append(
            {
                "run": str(directory),
                "mode": config["mode"],
                "metrics": metrics,
                "critic": config["critic"],
                "environment": meta["platform"],
            }
        )
    return {
        "notice": "Check matching GPU/server/batch conditions before comparing timing",
        "runs": rows,
    }
