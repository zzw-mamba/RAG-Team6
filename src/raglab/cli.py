from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from raglab.config import PipelineConfig, load_config
from raglab.data import file_sha256, iter_examples, iter_records, prepare
from raglab.evaluation import evaluate
from raglab.experiments import comparison, run_inference
from raglab.submission import export_submission


def dump(value, path: Path | None = None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def save_json(value, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_run(input_path: Path, run_dir: Path, allow_partial: bool = False) -> dict:
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if file_sha256(input_path) != metadata["input_sha256"]:
        raise ValueError("Evaluation input differs from the inference input")
    examples = itertools.islice(iter_examples(input_path), metadata["limit"])
    metrics = evaluate(examples, iter_records(run_dir / "traces.jsonl"), allow_partial)
    metrics["selection_limit"] = metadata["limit"]
    return metrics


def smoke(fixtures: Path, output: Path) -> dict:
    if not fixtures.is_dir():
        raise ValueError("Run from the project root or pass --fixtures path/to/fixtures")
    output.mkdir(parents=True, exist_ok=False)
    data_path = fixtures / "questions.json"
    summaries = {}
    for mode in ("vanilla", "critique", "rl"):
        run_dir = output / mode
        config = PipelineConfig(mode=mode)
        run_inference(data_path, config, run_dir)
        metrics = evaluate_run(data_path, run_dir)
        save_json(metrics, run_dir / "metrics.json")
        summaries[mode] = metrics
    count = export_submission(
        output / "critique" / "traces.jsonl",
        fixtures / "sample_submission.csv",
        output / "synthetic_submission.csv",
        expected_rows=3,
        allow_mock=True,
    )
    compared = comparison([output / mode for mode in ("vanilla", "critique", "rl")])
    save_json(compared, output / "comparison.json")
    return {
        "notice": "Synthetic mock plumbing only; no models were trained",
        "csv_rows": count,
        "output": str(output),
        "modes": list(summaries),
    }


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(prog="raglab", description="RAG course research pipeline")
    commands = app.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser(
        "inspect", help="Validate and summarize data without calling a model"
    )
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--limit", type=int, default=5)

    prepare_cmd = commands.add_parser(
        "prepare", help="Stream and split labeled data by question hash"
    )
    prepare_cmd.add_argument("--input", type=Path, required=True)
    prepare_cmd.add_argument("--output", type=Path, required=True)
    prepare_cmd.add_argument("--seed", type=int, default=42)
    prepare_cmd.add_argument("--validation-fraction", type=float, default=0.1)

    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)

    infer = commands.add_parser(
        "infer", help="Generate public-only traces; no gold labels in prompts"
    )
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--config", type=Path, required=True)
    infer.add_argument("--run-dir", type=Path, required=True)
    infer.add_argument("--limit", type=int)
    infer.add_argument("--resume", action="store_true")

    evaluation = commands.add_parser(
        "evaluate", help="Local EM/F1 and process metrics; not Kaggle metric"
    )
    evaluation.add_argument("--input", type=Path, required=True)
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument("--allow-partial", action="store_true")

    compare = commands.add_parser("compare")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)

    submission = commands.add_parser(
        "export", help="Validate and write a submission; never uploads"
    )
    submission.add_argument("--traces", type=Path, required=True)
    submission.add_argument("--template", type=Path, required=True)
    submission.add_argument("--output", type=Path, required=True)
    submission.add_argument("--expected-rows", type=int, default=12576)

    demo = commands.add_parser("demo", help="Local trace replay interface")
    demo.add_argument("--run-dir", type=Path, required=True)
    demo.add_argument("--port", type=int, default=8765)

    smoke_cmd = commands.add_parser("smoke", help="Offline synthetic end-to-end workflow")
    smoke_cmd.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    smoke_cmd.add_argument("--output", type=Path, default=Path("outputs/smoke"))

    collect = commands.add_parser(
        "collect-annotations", help="Create pending critic examples for review"
    )
    collect.add_argument("--input", type=Path, required=True)
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--split", choices=["train", "validation"], required=True)
    collect.add_argument("--limit", type=int, default=100)

    for name in ("train-sft", "train-grpo"):
        training = commands.add_parser(name)
        training.add_argument("--config", type=Path, required=True)
        training.add_argument("--annotations", type=Path, required=True)
        training.add_argument("--check-only", action="store_true")
        if name == "train-sft":
            training.add_argument("--validation", type=Path)
    return app


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        match args.command:
            case "inspect":
                if args.limit < 1:
                    raise ValueError("limit must be positive")
                values = [
                    {
                        "id": e.id,
                        "question": e.question,
                        "n_contexts": len(e.contexts),
                        "n_answers": len(e.answers),
                        "explicit_id": e.explicit_id,
                    }
                    for e in itertools.islice(iter_examples(args.input), args.limit)
                ]
                dump({"preview_only": True, "samples": values})
            case "prepare":
                dump(prepare(args.input, args.output, args.seed, args.validation_fraction))
            case "validate-config":
                dump(load_config(args.config).to_dict())
            case "infer":
                dump(
                    run_inference(
                        args.input, load_config(args.config), args.run_dir, args.limit, args.resume
                    )
                )
            case "evaluate":
                dump(
                    evaluate_run(args.input, args.run_dir, args.allow_partial),
                    args.run_dir / "metrics.json",
                )
            case "compare":
                dump(comparison(args.runs), args.output)
            case "export":
                dump(
                    {
                        "rows": export_submission(
                            args.traces, args.template, args.output, args.expected_rows
                        ),
                        "output": str(args.output),
                    }
                )
            case "demo":
                from raglab.demo import serve

                serve(args.run_dir, args.port)
            case "smoke":
                dump(smoke(args.fixtures, args.output))
            case "collect-annotations":
                from raglab.training.annotations import collect_annotations

                dump(
                    {
                        "pending": collect_annotations(
                            args.input,
                            args.output,
                            load_config(args.config),
                            args.split,
                            args.limit,
                        )
                    }
                )
            case "train-sft":
                from raglab.training.sft import train_sft

                dump(train_sft(args.config, args.annotations, args.validation, args.check_only))
            case "train-grpo":
                from raglab.training.grpo import train_grpo

                dump(train_grpo(args.config, args.annotations, args.check_only))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"raglab: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
