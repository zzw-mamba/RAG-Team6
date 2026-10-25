from __future__ import annotations

import csv
import heapq
import shutil
from collections import Counter
from pathlib import Path

from .core import (
    digest_file,
    fingerprint,
    normalize_record,
    public,
    question_key,
    read_json,
    records,
    save_json,
    write_line,
)


def prepared(cfg):
    p = Path(cfg["work_dir"]) / "prepared"
    if not (p / "manifest.json").exists():
        raise RuntimeError("Run prepare first; incomplete datasets are not usable")
    return p


def smallest_sample(source, destination, size, seed):
    # Hash-ranked unique questions: bounded memory, deterministic and prefix-stable.
    heap, seen, serial = [], set(), 0
    for row in records(source):
        key = question_key(row["question"])
        if key in seen:
            continue
        seen.add(key)
        rank = int(fingerprint([seed, key]), 16)
        item = (-rank, serial, row)
        serial += 1
        if len(heap) < size:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    with Path(destination).open("w", encoding="utf-8") as f:
        for _, _, row in sorted(heap, key=lambda t: (-t[0], t[1])):
            write_line(f, row)
    return len(heap)


def prepare(cfg):
    source, work = Path(cfg["data_dir"]), Path(cfg["work_dir"])
    for name in ("train.json", "test.json", "sample_submission.csv"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    signature = {
        "seed": cfg["seed"],
        "tune_size": cfg["tune_size"],
        "confirmation_size": cfg["confirmation_size"],
        "train_pool_size": cfg["train_pool_size"],
        "files": {
            n: digest_file(source / n)
            for n in ("train.json", "test.json", "sample_submission.csv")
        },
    }
    dest = work / "prepared"
    if (dest / "manifest.json").exists():
        old = read_json(dest / "manifest.json")
        if old["signature"] != signature:
            raise ValueError("Prepared data/config changed; choose a NEW work_dir")
        return old
    staging = work / "prepared.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    test_keys, test_ids = set(), []
    with (staging / "test.jsonl").open("w", encoding="utf-8") as out:
        for i, raw in enumerate(records(source / "test.json")):
            row = normalize_record(raw, i, require_id=True)
            test_keys.add(question_key(row["question"]))
            test_ids.append(row["id"])
            write_line(out, public(row))
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Duplicate test IDs")
    with (source / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "answer"]:
            raise ValueError("Template columns must be id,answer")
        template_ids = [r["id"] for r in reader]
    if len(set(template_ids)) != len(template_ids) or set(template_ids) != set(
        test_ids
    ):
        raise ValueError("Test and template ID sets differ")
    counts, seen_ids = Counter(), set()
    files = {
        s: (staging / f"{s}.jsonl").open("w", encoding="utf-8")
        for s in ("train", "tune_all", "confirmation_all")
    }
    try:
        for i, raw in enumerate(records(source / "train.json")):
            row = normalize_record(raw, i)
            if not row["answers"]:
                raise ValueError(f"Training row {i} has no answers")
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate training id {row['id']}")
            seen_ids.add(row["id"])
            key = question_key(row["question"])
            if key in test_keys:
                counts["excluded_test_question_overlap"] += 1
                continue
            h = int(fingerprint([cfg["seed"], "split", key])[:16], 16) % 100
            split = "tune_all" if h < 5 else "confirmation_all" if h < 10 else "train"
            write_line(files[split], row)
            counts[split] += 1
            if (i + 1) % 10000 == 0:
                print(f"Prepared {i + 1:,} source rows", flush=True)
    finally:
        for f in files.values():
            f.close()
    for src, name, size in (
        ("train", "train_pool", cfg["train_pool_size"]),
        ("tune_all", "tune", cfg["tune_size"]),
        ("confirmation_all", "confirmation", cfg["confirmation_size"]),
    ):
        counts[name] = smallest_sample(
            staging / f"{src}.jsonl", staging / f"{name}.jsonl", size, cfg["seed"]
        )
        if not counts[name]:
            raise ValueError(f"Empty {name} split; check source data")
    shutil.copyfile(source / "sample_submission.csv", staging / "sample_submission.csv")
    counts["test"] = len(test_ids)
    manifest = {
        "signature": signature,
        "counts": dict(counts),
        "split_policy": "question-grouped SHA256: train90%, tune5%, confirmation5%; test-overlap excluded",
        "demos": "not used; rationale/qa_pairs never copied",
        "complete": True,
    }
    save_json(staging / "manifest.json", manifest)
    if dest.exists():
        raise ValueError(f"Unexpected destination {dest}")
    staging.rename(dest)
    return manifest
