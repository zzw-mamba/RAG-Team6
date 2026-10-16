from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from raglab.config import PipelineConfig
from raglab.data import file_sha256, iter_examples, iter_records, question_key, write_record
from raglab.models import build_backend
from raglab.prompts import answer_messages, critic_messages
from raglab.retrieval import retrieve
from raglab.schema import Answer, Document, parse_answer, parse_critique


def collect_annotations(
    input_path: Path,
    output: Path,
    config: PipelineConfig,
    split: str,
    limit: int,
) -> int:
    """Collect initial responses for review. Gold fields are never sent to the generator."""
    if split not in ("train", "validation") or limit < 1:
        raise ValueError("Use a labeled train/validation split and a positive limit")
    source_sha = file_sha256(input_path)
    backend = None
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        import itertools

        for example in itertools.islice(iter_examples(input_path), limit):
            if not example.answers:
                raise ValueError("Critic annotations require labeled data")
            if backend is None:
                backend = build_backend(config.generator, "generator")
            docs = retrieve(example.question, example.contexts, config.top_k, config.retriever)
            current = parse_answer(backend.complete(answer_messages(example.question, docs)), docs)
            write_record(
                stream,
                {
                    "id": example.id,
                    "split": split,
                    "question": example.question,
                    "documents": [asdict(d) for d in docs],
                    "current_answer": asdict(current),
                    "gold_answers": list(example.answers),
                    "target_critique": None,
                    "reviewed": False,
                    "reviewer": "",
                    "notes": "",
                    "source_sha256": source_sha,
                    "generator": asdict(config.generator),
                    "synthetic": config.generator.backend == "mock",
                },
            )
            count += 1
    if not count:
        raise ValueError("No annotation candidates collected")
    return count


def annotation_state(row: dict) -> tuple[tuple[Document, ...], Answer]:
    docs = tuple(Document(**{**d, "score": d.get("score")}) for d in row["documents"])
    if not docs or len({d.id for d in docs}) != len(docs):
        raise ValueError("Annotation documents must have unique IDs")
    # Reuse the same inference output contract for initial answers and reviewed targets.
    current = parse_answer(json.dumps(row["current_answer"]), docs)
    return docs, current


def read_annotations(path: Path, split: str) -> list[dict]:
    rows, ids = [], set()
    for row in iter_records(path):
        key = str(row["id"])
        if key in ids:
            raise ValueError(f"Duplicate annotation ID: {key}")
        ids.add(key)
        if row.get("split") != split:
            raise ValueError(f"Annotation {key}: expected split={split}")
        if row.get("reviewed") is not True or not row.get("reviewer", "").strip():
            raise ValueError(f"Annotation {key}: human/approved-teacher review is required")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Annotation {key}: missing question")
        answers = row.get("gold_answers")
        if (
            not isinstance(answers, list)
            or not answers
            or any(not isinstance(a, str) or not a.strip() for a in answers)
        ):
            raise ValueError(f"Annotation {key}: missing gold_answers")
        docs, _ = annotation_state(row)
        parse_critique(json.dumps(row.get("target_critique")), docs)
        rows.append(row)
    if not rows:
        raise ValueError("No reviewed annotation rows")
    return rows


def sft_pair(row: dict) -> tuple[list[dict], str]:
    docs, current = annotation_state(row)
    target = json.dumps(row["target_critique"], ensure_ascii=False)
    return critic_messages(row["question"], docs, current), target


def check_disjoint(train: list[dict], validation: list[dict]) -> None:
    shared = {question_key(r["question"]) for r in train} & {
        question_key(r["question"]) for r in validation
    }
    if shared:
        raise ValueError(f"Train/validation overlap: {len(shared)} duplicate questions")
