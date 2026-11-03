import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

from .data import fingerprint, gold_answers, normalize, records, validation_group
from .generation import DryRun, HuggingFace, clean_answer
from .metrics import scores
from .retrieval import retrieve

DEFAULTS = {
    "backend": "hf", "model": "Qwen/Qwen2.5-7B-Instruct", "seed": 42,
    "top_k": 10, "batch_size": 1, "max_input_tokens": 6144,
    "max_new_tokens": 64, "dtype": "bfloat16", "load_in_4bit": False,
    "retrieval_mode": "input",
}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_traces(path):
    result = {}
    if Path(path).exists():
        for row in records(path):
            if row["id"] in result:
                raise ValueError("Duplicate IDs in traces")
            result[row["id"]] = row
    return result


def summarize(traces):
    rows = list(traces.values())
    labeled = [r for r in rows if r["metrics"] is not None]
    times = [r["timing"]["pipeline_seconds_amortized"] for r in rows]
    return {
        "samples": len(rows), "labeled_samples": len(labeled),
        "backend": rows[0]["backend"] if rows else None,
        "diagnostic_only": True,
        "metric_note": "Local EM/F1 and substring_accuracy (FlashRAG-style acc); Kaggle matching rules are unverified. Dry-run scores are meaningless.",
        **{k: statistics.mean(r["metrics"][k] for r in labeled) if labeled else None
           for k in ("raw_em", "normalized_em", "token_f1", "substring_accuracy")},
        "average_revision_count": 0,
        "average_pipeline_seconds_amortized": statistics.mean(times) if times else None,
        "timing_note": "Excludes model loading; generation time divided by actual batch size. Use batch_size=1 for latency comparison.",
    }


def run(args):
    config = dict(DEFAULTS)
    if args.config:
        custom = json.loads(Path(args.config).read_text())
        if set(custom) - set(config):
            raise ValueError(f"Unknown config keys: {set(custom) - set(config)}")
        config.update(custom)
    if args.dry_run:
        config["backend"] = "dry-run"
    if config["backend"] not in ("hf", "dry-run"):
        raise ValueError("backend must be hf or dry-run")
    if config["retrieval_mode"] not in ("bm25", "input"):
        raise ValueError("retrieval_mode must be bm25 or input")
    for k in ("top_k", "batch_size", "max_input_tokens", "max_new_tokens"):
        if not isinstance(config[k], int) or config[k] <= 0:
            raise ValueError(f"{k} must be a positive integer")
    if config["dtype"] not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported dtype")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    out = Path(args.output)
    implementation = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        implementation.update(source.name.encode())
        implementation.update(source.read_bytes())
    identity = {"config": config, "input_sha256": fingerprint(args.input), "limit": args.limit,
                "implementation_sha256": implementation.hexdigest()}
    manifest_path = out / "manifest.json"
    if args.resume:
        saved = json.loads(manifest_path.read_text())
        if saved["identity"] != identity:
            raise ValueError("Resume refused: input/config/limit changed; use a new output directory")
    else:
        out.mkdir(parents=True, exist_ok=False)
        versions = {}
        for package in ("torch", "transformers", "accelerate", "bitsandbytes"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        dump(manifest_path, {"identity": identity, "python": platform.python_version(), "packages": versions})
    trace_path = out / "traces.jsonl"
    done = read_traces(trace_path)
    generator = None
    pending = []
    visited = set()
    started = time.perf_counter()
    with trace_path.open("a", encoding="utf-8") as stream:
        def flush():
            if not pending:
                return
            before = time.perf_counter()
            raw = generator.generate([p[1] for p in pending])
            elapsed = time.perf_counter() - before
            if len(raw) != len(pending):
                raise ValueError("Backend returned wrong number of answers")
            for item, answer in zip(pending, raw):
                row, prompt, docs, count, prep_seconds, gold = item
                final = clean_answer(answer)
                if not final:
                    raise ValueError(f"Empty model response for {row['id']}; no submission generated")
                trace = {
                    "id": row["id"], "question": row["question"], "backend": config["backend"],
                    "retrieved_documents": docs, "messages": prompt, "input_tokens": count,
                    "raw_output": answer, "initial_answer": final, "final_answer": final,
                    "revision_count": 0, "stop_reason": "single_pass", "critic_calls": 0,
                    "timing": {"retrieval_and_prompt_seconds": prep_seconds,
                               "generation_seconds_amortized": elapsed / len(pending),
                               "pipeline_seconds_amortized": prep_seconds + elapsed / len(pending),
                               "batch_size": len(pending)},
                    "metrics": scores(final, gold) if gold else None,
                }
                stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                done[row["id"]] = trace
            print(f"Completed {len(done)} samples", flush=True)
            pending.clear()

        for i, original in enumerate(records(args.input)):
            if args.limit is not None and i >= args.limit:
                break
            row = normalize(original, i)
            if row["id"] in visited:
                raise ValueError(f"Duplicate input id: {row['id']}")
            visited.add(row["id"])
            if row["id"] in done:
                continue
            gold = gold_answers(original)
            if generator is None:
                generator = DryRun() if config["backend"] == "dry-run" else HuggingFace(config)
            before = time.perf_counter()
            docs = retrieve(row["question"], row["ctxs"], config["top_k"], config["retrieval_mode"])
            prompt, packed, count = generator.pack(row["question"], docs, config["max_input_tokens"])
            pending.append((row, prompt, packed, count, time.perf_counter() - before, gold))
            if len(pending) >= config["batch_size"]:
                flush()
        flush()
    if not visited:
        raise ValueError("Input has no samples")
    if set(done) != visited:
        raise ValueError("Trace IDs differ from input IDs")
    summary = summarize(done)
    summary["this_invocation_wall_seconds_including_loading"] = time.perf_counter() - started
    dump(out / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def split(args):
    if not 0 < args.fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    counts = {"train": 0, "validation": 0}
    seen_ids = set()
    with (out / "train.jsonl").open("w", encoding="utf-8") as train, \
         (out / "validation.jsonl").open("w", encoding="utf-8") as val:
        for i, row in enumerate(records(args.input)):
            sample = normalize(row, i)
            if sample["id"] in seen_ids:
                raise ValueError(f"Duplicate id: {sample['id']}")
            seen_ids.add(sample["id"])
            answers = gold_answers(row)
            if not answers:
                raise ValueError(f"Missing labels in training row {i}")
            partition = "validation" if validation_group(sample["question"], args.seed, args.fraction) else "train"
            sample["answers"] = answers
            target = val if partition == "validation" else train
            target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[partition] += 1
    dump(out / "split.json", {"seed": args.seed, "fraction": args.fraction,
                              "input_sha256": fingerprint(args.input), "counts": counts,
                              "grouping": "lowercase whitespace-normalized question SHA256"})
    print(counts)


def submission(args):
    traces = read_traces(args.traces)
    if not traces or any(r["backend"] != "hf" for r in traces.values()):
        raise ValueError("Submission requires real HF model traces; dry-run export is forbidden")
    with open(args.template, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "answer"]:
            raise ValueError("Template columns must be exactly id,answer")
        ids = [row["id"] for row in reader]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Template IDs are empty or duplicated")
    if set(ids) != set(traces):
        raise ValueError(f"IDs mismatch: missing={len(set(ids)-set(traces))}, extra={len(set(traces)-set(ids))}")
    if any(not traces[rid]["final_answer"].strip() for rid in ids):
        raise ValueError("Empty prediction")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        writer.writerows((rid, traces[rid]["final_answer"]) for rid in ids)
    print(f"Wrote {len(ids)} predictions: {output}")


def main():
    parser = argparse.ArgumentParser(description="Qwen single-pass naive RAG baseline")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("run")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=run)
    p = sub.add_parser("split")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fraction", type=float, default=0.1)
    p.set_defaults(func=split)
    p = sub.add_parser("submission")
    p.add_argument("--traces", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=submission)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
