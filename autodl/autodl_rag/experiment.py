from __future__ import annotations

import csv
import itertools
import os
import random
import time
from collections import Counter
from pathlib import Path

from .core import (
    InvalidOutput,
    digest_file,
    exact,
    public,
    read_json,
    records,
    save_json,
    token_f1,
    write_line,
)
from .data import prepared
from .models import model_identity
from .pipeline import Pipeline, source_identity


def adapter_identity(adapter):
    if not adapter:
        return None
    adapter = Path(adapter).resolve()
    files = sorted(adapter.glob("adapter*"))
    if not (adapter / "adapter_config.json").exists() or not any(
        p.suffix in (".safetensors", ".bin") for p in files
    ):
        raise FileNotFoundError(f"Invalid adapter directory: {adapter}")
    return {
        "path": str(adapter),
        "sha256": {p.name: digest_file(p) for p in files if p.is_file()},
    }


def repair_incomplete_tail(path):
    # Only a torn final JSONL line is discardable; complete invalid lines must error.
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return
    with path.open("r+b") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        size = f.tell()
        pos = size
        while pos:
            start = max(0, pos - 65536)
            f.seek(start)
            block = f.read(pos - start)
            found = block.rfind(b"\n")
            if found >= 0:
                cut = start + found + 1
                break
            pos = start
        else:
            cut = 0
        f.seek(cut)
        tail = f.read()
        backup = path.with_suffix(path.suffix + ".torn-tail")
        if backup.exists():
            raise FileExistsError(
                f"Preserve/remove previous tail backup before resuming: {backup}"
            )
        backup.write_bytes(tail)
        f.truncate(cut)


def run_dataset(
    cfg,
    split,
    mode,
    output,
    adapter=None,
    limit=0,
    resume=False,
    pipeline=None,
    restart_on_code_change=False,
):
    if mode == "rl" and not adapter:
        raise ValueError("--mode rl requires --adapter")
    if mode in ("rag", "critic") and adapter:
        raise ValueError(
            "Use mode rl for a trained adapter; critic denotes the original untrained LLaMA"
        )
    if split not in ("tune", "confirmation", "test") or limit < 0:
        raise ValueError("Invalid split/limit")
    source = prepared(cfg) / f"{split}.jsonl"
    output = Path(output).resolve()
    meta = output.with_suffix(".manifest.json")
    signature = {
        "config": cfg,
        "source_sha256": digest_file(source),
        "implementation": source_identity(),
        "mode": mode,
        "split": split,
        "limit": limit,
        "adapter": adapter_identity(adapter),
        "qwen": model_identity(cfg["qwen_path"]),
        "reranker": model_identity(cfg["reranker_path"]) if cfg.get("reranker_path") else None,
        "critic": model_identity(cfg["llama_path"]) if mode != "rag" else None,
    }
    done = []
    if output.exists() or meta.exists():
        if not resume:
            raise FileExistsError(f"{output}: use --resume or a new output path")
        old_signature = read_json(meta).get("signature") if meta.exists() else None
        if old_signature != signature:
            source_only = old_signature is not None and {
                k: v for k, v in old_signature.items() if k != "implementation"
            } == {k: v for k, v in signature.items() if k != "implementation"}
            if restart_on_code_change and source_only:
                archive = output.parent / ".archive" / f"{output.stem}-{time.time_ns()}"
                archive.mkdir(parents=True)
                save_json(
                    archive / "restart.json",
                    {
                        "reason": "explicit restart after implementation change; no predictions reused",
                        "previous_signature": old_signature,
                        "new_signature": signature,
                    },
                )
                for path in (
                    output,
                    meta,
                    output.with_suffix(".failure.json"),
                    output.with_suffix(".csv"),
                ):
                    if path.exists():
                        path.rename(archive / path.name)
                save_json(meta, {"signature": signature, "complete": False})
                print(
                    f"Archived previous-code predictions to {archive}; re-evaluating all rows.",
                    flush=True,
                )
            else:
                raise ValueError(
                    "Resume prediction config/data/code/model/adapter signature mismatch. "
                    "For implementation-only changes use --restart-on-code-change to archive and re-evaluate."
                )
        repair_incomplete_tail(output)
        done = list(records(output)) if output.exists() else []
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        save_json(meta, {"signature": signature, "complete": False})
    iterator = iter(records(source))
    if limit:
        iterator = itertools.islice(iterator, limit)
    for old in done:
        row = next(iterator, None)
        if row is None or old["id"] != row["id"]:
            raise ValueError("Existing predictions are not an exact dataset prefix")
    if len({r["id"] for r in done}) != len(done):
        raise ValueError("Duplicate IDs in predictions")
    first = next(iterator, None)
    if first is not None:
        pipeline = pipeline or Pipeline(cfg, mode=mode, adapter=adapter)
        with output.open("a", encoding="utf-8") as f:
            for row in itertools.chain([first], iterator):
                try:
                    trace = pipeline.run(public(row))
                except InvalidOutput as e:
                    failure = output.with_suffix(".failure.json")
                    save_json(
                        failure,
                        {
                            "id": row["id"],
                            "question": row["question"],
                            "error": str(e),
                            "completed_rows": len(done),
                            "note": "No placeholder prediction written. Fix and resume this row.",
                        },
                    )
                    raise InvalidOutput(
                        f"{e}. Failed sample id={row['id']}; details: {failure}"
                    ) from e
                trace.pop("_samples", None)
                write_line(f, trace)
                os.fsync(f.fileno())
                done.append({"id": row["id"]})
                print(
                    f"{mode}/{split}: {len(done)} id={row['id']} answer={trace['final']['answer']!r} stop={trace['stop_reason']}",
                    flush=True,
                )
                if trace["stop_reason"] == "change_rejected":
                    judges = trace.get("turns", [{}])[-1].get("verification", [])
                    if judges:
                        last = judges[-1]
                        print(
                            f"  rejection={last.get('rejection', 'unspecified')} "
                            f"reason={last.get('reason', last.get('invalid', ''))}",
                            flush=True,
                        )
    if not output.exists():
        raise ValueError("Dataset is empty")
    result = {
        "signature": signature,
        "complete": True,
        "rows": len(done),
        "predictions_sha256": digest_file(output),
    }
    save_json(meta, result)
    return result


def evaluate(cfg, predictions, split):
    if split not in ("tune", "confirmation"):
        raise ValueError("No labels for test; do not generate test accuracy")
    labels = {r["id"]: r["answers"] for r in records(prepared(cfg) / f"{split}.jsonl")}
    rows = list(records(predictions))
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty/duplicate predictions")
    correct = strict = fixes = breaks = 0
    f1 = revisions = seconds = wall = calls = 0.0
    types = {"boolean": [0, 0], "entity": [0, 0]}
    stops = Counter()
    for r in rows:
        if r["id"] not in labels:
            raise ValueError(f"Prediction {r['id']} is not in {split}")
        gold = labels[r["id"]]
        a, b = exact(r["initial"]["answer"], gold), exact(r["final"]["answer"], gold)
        correct += b
        strict += int(any(r["final"]["answer"].strip() == g.strip() for g in gold))
        fixes += int(not a and b)
        breaks += int(a and not b)
        f1 += token_f1(r["final"]["answer"], gold)
        revisions += r["revisions"]
        seconds += r["logical_seconds"]
        wall += r["wall_seconds"]
        calls += r["logical_model_calls"]
        stops[r["stop_reason"]] += 1
        kind = (
            "boolean"
            if all(g.strip().lower() in ("yes", "no") for g in gold)
            else "entity"
        )
        types[kind][0] += b
        types[kind][1] += 1
    n = len(rows)
    result = {
        "split": split,
        "n": n,
        "correct": correct,
        "normalized_em": correct / n,
        "strict_em": strict / n,
        "token_f1": f1 / n,
        "fixes": fixes,
        "breaks": breaks,
        "avg_revisions": revisions / n,
        "avg_logical_seconds": seconds / n,
        "avg_wall_seconds_this_run": wall / n,
        "avg_logical_model_calls": calls / n,
        "stop_reasons": dict(stops),
        "by_type": {
            k: {
                "correct": v[0],
                "n": v[1],
                "normalized_em": v[0] / v[1] if v[1] else None,
            }
            for k, v in types.items()
        },
        "timing_note": "logical_seconds includes originally measured shared initial generation plus actual additional work; wall_seconds may reuse cached initials",
        "metric_note": "Local metrics; not a reproduction of Kaggle's undisclosed custom scorer",
    }
    save_json(Path(predictions).with_suffix(".metrics.json"), result)
    return result


def compare(cfg, paths, split, output):
    all_rows = {
        name: {r["id"]: r for r in records(path)} for name, path in paths.items()
    }
    base = all_rows["rag"]
    for name, rs in all_rows.items():
        if set(rs) != set(base):
            raise ValueError(f"{name} and rag have different evaluation ID sets")
        if any(
            rs[k]["initial_cache_key"] != base[k]["initial_cache_key"]
            or rs[k]["initial"] != base[k]["initial"]
            for k in base
        ):
            raise ValueError(
                "Comparisons must share the exact same initial evidence/answers"
            )
    metrics = {name: evaluate(cfg, path, split) for name, path in paths.items()}
    labels = {r["id"]: r["answers"] for r in records(prepared(cfg) / f"{split}.jsonl")}
    paired = {}
    for a, b in (("rag", "critic"), ("rag", "rl"), ("critic", "rl")):
        diffs = [
            exact(all_rows[b][k]["final"]["answer"], labels[k])
            - exact(all_rows[a][k]["final"]["answer"], labels[k])
            for k in base
        ]
        rng = random.Random(cfg["seed"])
        samples = sorted(
            sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(2000)
        )
        paired[b + "_minus_" + a] = {
            "delta_pp": 100 * sum(diffs) / len(diffs),
            "bootstrap_95_interval_pp": [100 * samples[49], 100 * samples[1949]],
            "resamples": 2000,
            "note": "Paired questions, fixed models, one training seed; not training-seed uncertainty",
        }
    result = {
        "same_questions_and_initial_answers": True,
        "methods": metrics,
        "paired_comparisons": paired,
    }
    save_json(output, result)
    return result


def select(cfg, candidates, output):
    candidates = Path(candidates)
    rows = []
    for p in sorted(candidates.glob("group-*.jsonl")):
        meta = read_json(p.with_suffix(".manifest.json"))
        if not meta.get("complete") or meta["signature"]["split"] != "tune":
            raise ValueError(
                "Checkpoint selection only accepts complete tune predictions"
            )
        metric = evaluate(cfg, p, "tune")
        group = int(p.stem.split("-")[-1])
        rows.append(
            {
                "group": group,
                "adapter": meta["signature"]["adapter"]["path"],
                "predictions": str(p),
                "metrics": metric,
            }
        )
    if not rows:
        raise ValueError(
            "No evaluated checkpoints; run tune for the saved group-* adapters"
        )
    signatures = [
        read_json(Path(r["predictions"]).with_suffix(".manifest.json"))["signature"]
        for r in rows
    ]
    for sig in signatures[1:]:
        if {k: v for k, v in sig.items() if k != "adapter"} != {
            k: v for k, v in signatures[0].items() if k != "adapter"
        }:
            raise ValueError("Candidate evaluations differ in data/config/code/limit")
    reference = {
        r["id"]: r["initial_cache_key"] for r in records(rows[0]["predictions"])
    }
    for r in rows[1:]:
        if {
            x["id"]: x["initial_cache_key"] for x in records(r["predictions"])
        } != reference:
            raise ValueError(
                "Candidate selection needs identical question IDs and shared initials"
            )
    rows.sort(
        key=lambda r: (
            -r["metrics"]["normalized_em"],
            r["metrics"]["breaks"],
            r["metrics"]["avg_revisions"],
            r["group"],
        )
    )
    result = {
        "selected": rows[0],
        "ranking": rows,
        "selection_data": "tune only; confirmation/test labels never used",
    }
    save_json(output, result)
    Path(output).with_suffix(".adapter.txt").write_text(
        rows[0]["adapter"] + "\n", encoding="utf-8"
    )
    return result


def export_csv(cfg, predictions, output):
    rows = list(records(predictions))
    answers = {}
    for r in rows:
        sid, answer = r["id"], r["final"]["answer"]
        if (
            sid in answers
            or not isinstance(answer, str)
            or not answer.strip()
            or any(c in answer for c in "\r\n")
        ):
            raise ValueError(f"Duplicate ID or invalid answer at {sid}")
        answers[sid] = answer
    with (prepared(cfg) / "sample_submission.csv").open(
        newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)
        ids = [r["id"] for r in reader]
    if set(answers) != set(ids) or len(answers) != len(ids):
        raise ValueError(
            "Predictions do not exactly cover the official template; partial runs cannot be exported"
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".building")
    with staging.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        writer.writerows((sid, answers[sid]) for sid in ids)
    staging.replace(output)
    result = {
        "rows": len(ids),
        "sha256": digest_file(output),
        "template_sha256": digest_file(prepared(cfg) / "sample_submission.csv"),
        "predictions_sha256": digest_file(predictions),
        "official_accuracy": None,
    }
    save_json(output.with_suffix(".validation.json"), result)
    return result
