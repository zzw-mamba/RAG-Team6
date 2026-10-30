"""Tune evidence/answer policies with real Qwen; keep confirmation separate."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .core import digest_file, fingerprint, load_config, normalized, read_json, records, save_json
from .data import prepared
from .experiment import evaluate, export_csv, run_dataset
from .pipeline import Pipeline, source_identity


def profiles(base):
    coverage = {
        **base,
        "retrieval_policy": "coverage",
        "retrieval_plan_rounds": 0,
        "retrieval_plan_trigger": "unknown",
        "max_input_tokens": 16384,
        "max_documents": 32,
        "document_chars": 4000,
        "global_top_k": 8,
        "answer_tokens": 384,
        "allow_model_knowledge": False,
    }
    return {
        "baseline": {**base, "retrieval_policy": "legacy", "retrieval_plan_rounds": 0,
                     "allow_model_knowledge": False},
        "coverage": coverage,
        "knowledge": {**coverage, "allow_model_knowledge": True},
        "planned": {**coverage, "allow_model_knowledge": True,
                    "retrieval_plan_rounds": 1, "retrieval_plan_trigger": "always"},
    }


def diagnose(cfg, output, split):
    """Labels stay in offline diagnostics, never retrieval/planner/model prompts."""
    labels = {r["id"]: r["answers"] for r in records(prepared(cfg) / f"{split}.jsonl")}
    stats = Counter()
    for row in records(output):
        stats["rows"] += 1
        stats["unknown"] += normalized(row["final"]["answer"]) == "unknown"
        docs = row["initial_documents"]
        haystack = " " + normalized(" ".join(d["title"] + " " + d["text"] for d in docs)) + " "
        gold = labels[row["id"]]
        entities = [g for g in gold if normalized(g) not in ("yes", "no", "unknown")]
        if entities:
            stats["entity_rows"] += 1
            stats["gold_string_visible"] += any(
                " " + normalized(g) + " " in haystack for g in entities
            )
        stats["visible_documents"] += len(docs)
        stats["selected_documents"] += row.get("selected_document_count", len(docs))
        stats["input_contexts"] += row.get("input_context_count", 0)
        stats["planner_calls"] += len(row.get("retrieval_plans", []))
    return {
        **dict(stats),
        "unknown_rate": stats["unknown"] / max(1, stats["rows"]),
        "note": "Gold string presence is a diagnostic proxy, not proof of complete evidence. Labels are used only here after inference.",
    }


def choose(results):
    # Predeclared tune metric; ties favor the cheaper observed pipeline.
    return max(results, key=lambda name: (
        results[name]["normalized_em"], results[name]["token_f1"],
        -results[name]["avg_logical_seconds"],
    ))


def verify_selection(base, folder):
    selection = read_json(folder / "selection.json")
    if selection["base_signature"] != fingerprint(base) or selection["source"] != source_identity():
        raise ValueError("Config/code changed since tuning; use a new --name and rerun tune")
    for name, expected in selection["tune_manifests"].items():
        if digest_file(folder / name / "tune.manifest.json") != expected:
            raise ValueError("Tune outputs changed since selection")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("tune", "confirmation", "predict"), required=True)
    parser.add_argument("--name", default="score-v2")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a single directory name")
    base = load_config(args.config)
    folder = Path(base["work_dir"]) / "optimization" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (folder / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Another optimization stage is using this --name") from e
        run_stage(args, base, folder)


def run_stage(args, base, folder):
    configs = profiles(base)
    selection = None if args.stage == "tune" else verify_selection(base, folder)
    split = {"tune": "tune", "confirmation": "confirmation", "predict": "test"}[args.stage]
    if args.stage == "tune":
        names = list(configs)
    elif args.stage == "confirmation":
        names = list(dict.fromkeys(["baseline", selection["selected"]]))
    else:
        confirmation = read_json(folder / "confirmation.json")
        if confirmation["selection_sha256"] != digest_file(folder / "selection.json"):
            raise ValueError("Confirmation belongs to another selection")
        if not confirmation["passed"]:
            raise ValueError("Selected policy regressed on confirmation; inspect metrics before full prediction")
        names = [selection["selected"]]

    from .models import LocalModel

    qwen = None
    results = {}
    for name in names:
        cfg = configs[name]
        save_json(folder / name / "config.json", cfg)
        if qwen is None:
            qwen = LocalModel(cfg, "qwen")
        qwen.cfg = cfg
        pipeline = Pipeline(cfg, mode="rag", qwen=qwen)
        output = folder / name / f"{split}.jsonl"
        try:
            result = run_dataset(cfg, split, "rag", output, resume=True, pipeline=pipeline)
        finally:
            pipeline.retriever.db.close()
        if split == "test":
            if not output.with_suffix(".csv").exists():
                export_csv(cfg, output, output.with_suffix(".csv"))
            print(f"Submission: {output.with_suffix('.csv')}", flush=True)
        else:
            metrics = evaluate(cfg, output, split)
            metrics["diagnostics"] = diagnose(cfg, output, split)
            results[name] = metrics
            save_json(folder / name / f"{split}.metrics.json", metrics)
            print(f"{name}: EM={metrics['normalized_em']:.4f} F1={metrics['token_f1']:.4f} "
                  f"unknown={metrics['diagnostics']['unknown_rate']:.1%}", flush=True)
        if not result["complete"]:
            raise RuntimeError("Incomplete evaluation")

    if args.stage == "tune":
        save_json(folder / "selection.json", {
            "selected": choose(results), "metric": "tune normalized_em; F1 then time break ties",
            "source": source_identity(), "base_signature": fingerprint(base),
            "tune_manifests": {n: digest_file(folder / n / "tune.manifest.json") for n in names},
            "results": results,
        })
        print(f"Tune selected: {choose(results)}; report: {folder / 'selection.json'}", flush=True)
    elif args.stage == "confirmation":
        selected = selection["selected"]
        save_json(folder / "confirmation.json", {
            "selection_sha256": digest_file(folder / "selection.json"),
            "passed": results[selected]["normalized_em"] >= results["baseline"]["normalized_em"],
            "results": results,
            "note": "Policy is frozen by tune. Confirmation is a regression check, not another model selection step. Local EM is not Kaggle accuracy.",
        })
        print(f"Confirmation report: {folder / 'confirmation.json'}", flush=True)


if __name__ == "__main__":
    main()
