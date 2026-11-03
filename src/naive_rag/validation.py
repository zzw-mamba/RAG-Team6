"""Held-out evaluation of labeled train data; all reference answers stay outside inference."""

import csv
import hashlib
import heapq
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from .data import fingerprint, gold_answers, normalize, records
from .metrics import METRIC_VERSION, normalize_answer, phrase_present, scores


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def question_key(question):
    return " ".join(question.lower().split())


def partition(question, seed):
    key = question_key(question)
    digest = hashlib.sha256(f"{seed}:partition:{key}".encode()).hexdigest()
    value = int(digest[:16], 16) / 2**64
    return "dev" if value < 0.1 else "holdout" if value < 0.2 else "train"


def prepare_eval(args):
    """Select fixed-size samples with a hash priority, keeping memory bounded by sample size."""
    if args.dev_size <= 0 or args.holdout_size <= 0:
        raise ValueError("dev-size and holdout-size must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new evaluation-data directory")
    counts = {"train": 0, "dev": 0, "holdout": 0}
    heaps = {"dev": [], "holdout": []}
    sizes = {"dev": args.dev_size, "holdout": args.holdout_size}
    seen_ids, seen_questions = set(), set()
    duplicates = 0
    for i, raw in enumerate(records(args.input)):
        sample = normalize(raw, i)
        if sample["id"] in seen_ids:
            raise ValueError(f"Duplicate input id: {sample['id']}")
        seen_ids.add(sample["id"])
        answers = gold_answers(raw)
        if not answers or not all(a.strip() for a in answers):
            raise ValueError(f"Missing/empty labels for {sample['id']}")
        key = question_key(sample["question"])
        if key in seen_questions:
            duplicates += 1
            continue
        seen_questions.add(key)
        group = partition(sample["question"], args.seed)
        counts[group] += 1
        if group == "train":
            continue
        sample["answers"] = answers
        if isinstance(raw.get("type"), str):
            sample["type"] = raw["type"]
        rank = int(hashlib.sha256(f"{args.seed}:sample:{key}".encode()).hexdigest(), 16)
        # The maximum rank is the first heap element, so retain the smallest ranks.
        item = (-rank, key, sample)
        heap = heaps[group]
        if len(heap) < sizes[group]:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    for group in sizes:
        if len(heaps[group]) < sizes[group]:
            raise ValueError(f"Only {len(heaps[group])} unique {group} questions; requested {sizes[group]}. Reduce sample size.")
    output.mkdir(parents=True)
    manifest = {"seed": args.seed, "source_sha256": fingerprint(args.input),
                "source_rows": len(seen_ids), "unique_question_counts": counts,
                "duplicate_question_rows_skipped": duplicates,
                "partition_rule": "question SHA256: train 80%, dev 10%, holdout 10%",
                "sampling_rule": "smallest seeded question hashes within each partition",
                "duplicate_policy": "one row per normalized question, first occurrence retained",
                "warning": "No entity-level or paraphrase isolation. Keep the holdout partition out of later training.",
                "files": {}}
    selected = {g: [x[2] for x in sorted(heap, key=lambda x: (-x[0], x[1]))]
                for g, heap in heaps.items()}
    selected["pilot"] = selected["dev"][:min(20, len(selected["dev"]))]
    for group, samples in selected.items():
        path = output / f"{group}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        manifest["files"][group] = {"samples": len(samples), "sha256": fingerprint(path)}
    write_json(output / "evaluation_split.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def contains_answer(answers, documents):
    text = " ".join(d.get("title", "") + " " + d["text"] for d in documents)
    return any(phrase_present(a, text) for a in answers)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def wilson_interval(successes, total):
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def report(args):
    from .cli import read_traces

    directory = Path(args.run_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    identity = manifest["identity"]
    if identity["input_sha256"] != fingerprint(args.input):
        raise ValueError("Report labels do not match the inference input fingerprint")
    if identity["limit"] is not None:
        raise ValueError("Report requires a complete input run; prepare a pilot file for small runs")
    traces = read_traces(directory / "traces.jsonl")
    if not traces:
        raise ValueError("No prediction traces")
    backend = identity["config"]["backend"]
    if any(t["backend"] != backend for t in traces.values()):
        raise ValueError("Trace backend differs from the run manifest")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Report output already exists")
    rows, examples, seen = [], {}, set()
    for i, raw in enumerate(records(args.input)):
        sample = normalize(raw, i)
        rid = sample["id"]
        if rid in seen or rid not in traces:
            raise ValueError(f"Duplicate input or missing prediction: {rid}")
        seen.add(rid)
        answers = gold_answers(raw)
        if not answers or not all(a.strip() for a in answers):
            raise ValueError(f"Missing/empty labels: {rid}")
        trace = traces[rid]
        metric = scores(trace["final_answer"], answers)
        selected_ids = {d["source_index"] for d in trace["retrieved_documents"]}
        selected = [d for j, d in enumerate(sample["ctxs"]) if j in selected_ids]
        candidate_hit = bool(contains_answer(answers, sample["ctxs"]))
        selected_hit = bool(contains_answer(answers, selected))
        packed_hit = bool(contains_answer(answers, trace["retrieved_documents"]))
        flags = []
        if metric["normalized_em"] == 0:
            if not candidate_hit:
                flags.append("gold_string_absent_from_candidates")
            elif not selected_hit:
                flags.append("gold_string_lost_after_selection")
            elif not packed_hit:
                flags.append("gold_string_lost_after_truncation")
            else:
                flags.append("answer_mismatch_with_gold_string_present")
        kind = raw.get("type") if isinstance(raw.get("type"), str) else "unspecified"
        row = {"id": rid, "question": sample["question"], "answers": answers,
               "prediction": trace["final_answer"], "type": kind, **metric,
               "answer_kind": "yes_no" if any(normalize_answer(a) in {"yes", "no"} for a in answers) else "other",
               "input_tokens": trace.get("input_tokens"),
               "pipeline_seconds_amortized": trace["timing"]["pipeline_seconds_amortized"],
               "unknown": normalize_answer(trace["final_answer"]) in {"unknown", "noanswer"},
               "candidate_gold_string_present": candidate_hit,
               "selected_gold_string_present": selected_hit,
               "prompt_gold_string_present": packed_hit,
               "truncated_documents": sum(d.get("truncated", False) for d in trace["retrieved_documents"]),
               "diagnostic_flags": flags}
        rows.append(row)
        category = flags[0] if flags else "correct"
        if len(examples.setdefault(category, [])) < 5:
            examples[category].append({**row, "documents_sent_to_model": trace["retrieved_documents"]})
    if seen != set(traces):
        raise ValueError("Extra predictions outside evaluation input")
    count = len(rows)
    times = [r["pipeline_seconds_amortized"] for r in rows]
    answered = [r for r in rows if not r["unknown"]]
    input_lengths = [r["input_tokens"] for r in rows if r["input_tokens"] is not None]
    summary = {"samples": count, "backend": backend, "dry_run": backend == "dry-run",
               "metric_version": METRIC_VERSION,
               "input_sha256": identity["input_sha256"], "config": identity["config"],
               "traces_sha256": fingerprint(directory / "traces.jsonl"),
               "metric_note": "Local metrics, not verified Kaggle Accuracy; dry-run scores/timing have no research meaning.",
               **{key: statistics.mean(r[key] for r in rows)
                  for key in ("raw_em", "normalized_em", "token_f1", "substring_accuracy", "legacy_substring_accuracy")},
               "legacy_substring_false_positive_count": sum(r["legacy_substring_accuracy"] > r["substring_accuracy"] for r in rows),
               "normalized_em_wilson_95_interval": wilson_interval(sum(r["normalized_em"] for r in rows), count),
               "unknown_rate": statistics.mean(r["unknown"] for r in rows),
               "unknown_count": sum(r["unknown"] for r in rows),
               "answered_count": len(answered),
               "answered_normalized_em": statistics.mean(r["normalized_em"] for r in answered) if answered else None,
               "unknown_with_gold_phrase_in_prompt_count": sum(r["unknown"] and r["prompt_gold_string_present"] for r in rows),
               "candidate_gold_phrase_rate": statistics.mean(r["candidate_gold_string_present"] for r in rows),
               "selected_gold_phrase_rate": statistics.mean(r["selected_gold_string_present"] for r in rows),
               "prompt_gold_phrase_rate": statistics.mean(r["prompt_gold_string_present"] for r in rows),
               "truncated_sample_count": sum(r["truncated_documents"] > 0 for r in rows),
               "mean_input_tokens": statistics.mean(input_lengths) if input_lengths else None,
               "max_input_tokens": max(input_lengths) if input_lengths else None,
               "diagnostic_counts": dict(Counter(flag for r in rows for flag in r["diagnostic_flags"])),
               "mean_pipeline_seconds_amortized": statistics.mean(times),
               "p50_pipeline_seconds_amortized": percentile(times, 0.5),
               "p95_pipeline_seconds_amortized": percentile(times, 0.95),
               "summed_pipeline_seconds_amortized": sum(times),
               "average_revision_count": 0,
               "timing_note": "Excludes model loading and file IO; batch_size>1 values are amortized, not request latency. No warm-up excluded.",
               "diagnostic_note": "Gold-string presence is not evidence-chain completeness or causal error attribution, especially for yes/no.",
               "by_type": {}, "by_answer_kind": {}}
    for kind in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == kind]
        summary["by_type"][kind] = {"samples": len(group),
                                    "normalized_em": statistics.mean(r["normalized_em"] for r in group),
                                    "token_f1": statistics.mean(r["token_f1"] for r in group)}
    for kind in sorted({r["answer_kind"] for r in rows}):
        group = [r for r in rows if r["answer_kind"] == kind]
        summary["by_answer_kind"][kind] = {"samples": len(group),
                                           "normalized_em": statistics.mean(r["normalized_em"] for r in group),
                                           "unknown_rate": statistics.mean(r["unknown"] for r in group)}
    output.mkdir(parents=True)
    write_json(output / "summary.json", summary)
    write_json(output / "cases.json", examples)
    with (output / "per_question.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "answers": json.dumps(row["answers"], ensure_ascii=False),
                             "diagnostic_flags": json.dumps(row["diagnostic_flags"])})
    banner = "DRY-RUN：仅检查流程，以下分数与耗时不代表模型效果。\n\n" if summary["dry_run"] else ""
    ci = summary["normalized_em_wilson_95_interval"]
    text = ("# RAG 本地验证报告\n\n" + banner
            + f"已完成样本数：{count}\n\n"
            + "| 指标 | 数值 |\n|---|---:|\n"
            + f"| Normalized EM | {summary['normalized_em']:.2%} |\n"
            + f"| Token F1 | {summary['token_f1']:.2%} |\n"
            + f"| Raw EM | {summary['raw_em']:.2%} |\n"
            + f"| 答案短语命中率（边界与类别检查） | {summary['substring_accuracy']:.2%} |\n"
            + f"| 旧版子串值（可能虚高，仅用于追溯） | {summary['legacy_substring_accuracy']:.2%} |\n"
            + f"| 旧版子串误命中题数 | {summary['legacy_substring_false_positive_count']} |\n"
            + f"| EM Wilson 95% 区间 | {ci[0]:.2%}–{ci[1]:.2%} |\n"
            + f"| unknown 比例 | {summary['unknown_rate']:.2%} |\n"
            + f"| unknown 题数 | {summary['unknown_count']} |\n"
            + f"| 非拒答题数 | {summary['answered_count']} |\n"
            + f"| 候选文档含答案短语的比例 | {summary['candidate_gold_phrase_rate']:.2%} |\n"
            + f"| 模型输入含答案短语的比例 | {summary['prompt_gold_phrase_rate']:.2%} |\n"
            + f"| unknown 且输入含答案短语的题数 | {summary['unknown_with_gold_phrase_in_prompt_count']} |\n"
            + f"| 有文档被截断的题数 | {summary['truncated_sample_count']} |\n"
            + f"| 平均耗时（秒/题） | {statistics.mean(times):.3f} |\n"
            + f"| P95 耗时（秒/题） | {percentile(times, .95):.3f} |\n\n"
            + "以上为本地指标，尚未确认与 Kaggle Accuracy 的匹配公式一致。\n\n"
            + "计时不含权重加载和日志写入，未排除首次推理预热；batch 大于 1 时为分摊耗时。\n\n"
            + "EM 区间仅描述抽样不确定性，不覆盖数据关联、模型记忆或调参选择偏差。\n\n"
            + "答案字符串出现不代表多跳证据完整，诊断标签需人工复核，尤其是 yes/no 题。\n\n"
            + "逐题结果见 [per_question.csv](per_question.csv)，案例和实际输入文档见 [cases.json](cases.json)。\n")
    (output / "report.md").write_text(text, encoding="utf-8")
    print(text)


def evaluate(args):
    from argparse import Namespace

    from .cli import run

    seen = set()
    for i, row in enumerate(records(args.input)):
        sample = normalize(row, i)
        labels = gold_answers(row)
        if not labels or not all(a.strip() for a in labels):
            raise ValueError(f"Evaluation requires nonempty labels: {sample['id']}")
        if sample["id"] in seen:
            raise ValueError(f"Duplicate input id: {sample['id']}")
        seen.add(sample["id"])
    if not seen:
        raise ValueError("Evaluation input is empty")
    run(Namespace(input=args.input, output=args.output, config=args.config,
                  limit=None, resume=args.resume, dry_run=args.dry_run))
    report_dir = Path(args.output) / "report"
    if args.resume and (report_dir / "report.md").exists():
        saved = json.loads((report_dir / "summary.json").read_text())
        if saved["traces_sha256"] != fingerprint(Path(args.output) / "traces.jsonl"):
            raise ValueError("Predictions changed since the report; use report with a new output directory")
        print((report_dir / "report.md").read_text())
        return
    report(Namespace(input=args.input, run_dir=args.output, output=str(report_dir)))
