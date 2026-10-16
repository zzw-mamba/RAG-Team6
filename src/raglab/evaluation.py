from __future__ import annotations

import re
import statistics
import string
from collections import Counter
from collections.abc import Iterable

from raglab.schema import Example


def normalize_answer(text: str) -> str:
    lowered = text.casefold()
    no_punctuation = "".join(c for c in lowered if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", no_punctuation).split())


def exact_match(prediction: str, answers: Iterable[str]) -> float:
    return float(any(normalize_answer(prediction) == normalize_answer(a) for a in answers))


def token_f1(prediction: str, answers: Iterable[str]) -> float:
    pred = normalize_answer(prediction).split()
    best = 0.0
    for answer in answers:
        gold = normalize_answer(answer).split()
        if not pred or not gold:
            best = max(best, float(pred == gold))
            continue
        if pred in (["yes"], ["no"], ["unknown"]) or gold in (["yes"], ["no"], ["unknown"]):
            best = max(best, float(pred == gold))
            continue
        common = sum((Counter(pred) & Counter(gold)).values())
        best = max(best, 2 * common / (len(pred) + len(gold)))
    return best


def evaluate(
    examples: Iterable[Example], traces: Iterable[dict], allow_partial: bool = False
) -> dict:
    indexed = {}
    for trace in traces:
        key = str(trace["id"])
        if key in indexed:
            raise ValueError(f"Duplicate trace ID: {key}")
        indexed[key] = trace
    if not indexed:
        raise ValueError("Cannot evaluate empty predictions")
    scores, f1s, times, revisions, backends = [], [], [], [], set()
    fixes = breaks = initially_wrong = initially_right = labeled = 0
    stop_reasons, missing, format_errors = Counter(), [], 0
    matched = set()
    for example in examples:
        if not example.answers:
            raise ValueError("Evaluation requires gold answers; never evaluate hidden test labels")
        labeled += 1
        key = str(example.id)
        trace = indexed.get(key)
        if trace is None:
            missing.append(key)
            continue
        if trace["question"] != example.question:
            raise ValueError(f"Question mismatch for ID {key}")
        matched.add(key)
        initial = exact_match(trace["initial_answer"], example.answers)
        final = exact_match(trace["final_answer"], example.answers)
        initially_right += int(initial)
        initially_wrong += int(not initial)
        fixes += int(not initial and final)
        breaks += int(initial and not final)
        scores.append(final)
        f1s.append(token_f1(trace["final_answer"], example.answers))
        times.append(trace["elapsed_seconds"])
        revisions.append(trace["revisions"])
        stop_reasons[trace["stop_reason"]] += 1
        format_errors += trace.get("format_errors", 0)
        backends.add(trace["backend"])
    extra = set(indexed) - matched
    if extra:
        raise ValueError(f"Predictions contain unknown IDs: {sorted(extra)[:5]}")
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} predictions; use --allow-partial explicitly")
    sorted_times = sorted(times)
    return {
        "metric_notice": "Local normalized EM / Token F1; Kaggle Accuracy matching is unverified",
        "synthetic_mock": any("mock" in b.split("+") for b in backends),
        "n_evaluated": len(scores),
        "n_labeled": labeled,
        "coverage": len(scores) / labeled,
        "normalized_em": statistics.mean(scores),
        "token_f1": statistics.mean(f1s),
        "mean_revisions": statistics.mean(revisions),
        "mean_seconds": statistics.mean(times),
        "median_seconds": statistics.median(times),
        "p95_seconds": sorted_times[max(0, (95 * len(times) + 99) // 100 - 1)],
        "correction_rate": fixes / initially_wrong if initially_wrong else None,
        "regression_rate": breaks / initially_right if initially_right else None,
        "n_initially_wrong": initially_wrong,
        "n_initially_right": initially_right,
        "n_corrected": fixes,
        "n_regressed": breaks,
        "stop_reasons": dict(stop_reasons),
        "format_errors": format_errors,
    }
