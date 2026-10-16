from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from raglab.schema import Document, Example


def iter_records(path: str | Path) -> Iterator[dict]:
    """JSONL streams natively. Large JSON arrays require the optional ijson extra."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix == ".jsonl":
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSONL") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{number}: expected object")
                yield item
            return
        if path.suffix != ".json":
            raise ValueError("Input must be a .json array or a .jsonl file")
        first = stream.read(1)
        while first and first.isspace():
            first = stream.read(1)
        if first != "[":
            raise ValueError("JSON input must contain a top-level array")
        stream.seek(0)
        try:
            import ijson
        except ImportError:
            if path.stat().st_size > 32 * 1024 * 1024:
                raise RuntimeError("Large JSON: install pip install -e '.[data]' for streaming")
            yield from json.load(stream)
        else:
            yield from ijson.items(stream, "item", use_float=True)


def normalize(raw: dict, index: int = 0) -> Example:
    if not isinstance(raw, dict):
        raise ValueError(f"Sample {index}: expected an object")
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Sample {index}: missing question")
    explicit = raw.get("explicit_id", "id" in raw or "_id" in raw)
    sample_id = raw.get("id", raw.get("_id", f"row:{index}"))
    if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)):
        raise ValueError(f"Sample {index}: id must be a string or integer")
    if not str(sample_id).strip():
        raise ValueError(f"Sample {index}: empty id")
    contexts = raw.get("ctxs", raw.get("contexts", raw.get("context")))
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"Sample {sample_id}: missing non-empty ctxs/context/contexts")
    documents = []
    for n, item in enumerate(contexts):
        doc_id, title, text, score = f"doc_{n}", "", None, None
        if isinstance(item, dict):
            doc_id = str(item.get("id", doc_id))
            title = item.get("title", "")
            text = item.get("text", item.get("contents", item.get("passage")))
            if text is None and isinstance(item.get("sentences"), list):
                text = " ".join(item["sentences"])
            score = item.get("score")
        elif isinstance(item, list) and len(item) == 2:
            title, sentences = item
            text = " ".join(sentences) if isinstance(sentences, list) else sentences
        elif isinstance(item, str):
            text = item
        if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Sample {sample_id}, context {n}: unsupported document schema; "
                "extend normalize() explicitly after inspecting the actual data"
            )
        if not doc_id.strip() or any(d.id == doc_id for d in documents):
            raise ValueError(f"Sample {sample_id}: empty or duplicate document ID {doc_id!r}")
        documents.append(Document(doc_id, title, text, float(score) if score is not None else None))
    answers = raw.get("answers", raw.get("answer", []))
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list) or any(
        not isinstance(a, str) or not a.strip() for a in answers
    ):
        raise ValueError(f"Sample {sample_id}: answers must be strings")
    rationale = raw.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        raise ValueError(f"Sample {sample_id}: rationale must be a string or null")
    return Example(
        sample_id, question.strip(), tuple(documents), tuple(answers), rationale, explicit
    )


def iter_examples(path: str | Path) -> Iterator[Example]:
    seen = set()
    for index, raw in enumerate(iter_records(path)):
        item = normalize(raw, index)
        key = str(item.id)
        if key in seen:
            raise ValueError(f"Duplicate sample ID: {key}")
        seen.add(key)
        yield item


def example_record(example: Example) -> dict:
    return {
        "id": example.id,
        "explicit_id": example.explicit_id,
        "question": example.question,
        "ctxs": [asdict(d) for d in example.contexts],
        "answers": list(example.answers),
        "rationale": example.rationale,
    }


def write_record(stream, record: dict) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def question_key(question: str) -> str:
    return " ".join(re.findall(r"\w+", question.casefold()))


def is_validation(question: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{question_key(question)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(input_path: Path, output_dir: Path, seed: int = 42, fraction: float = 0.1) -> dict:
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    output_dir.mkdir(parents=True, exist_ok=False)
    counts = {"train": 0, "validation": 0}
    with (
        (output_dir / "train.jsonl").open("w", encoding="utf-8") as train,
        (output_dir / "validation.jsonl").open("w", encoding="utf-8") as validation,
    ):
        for example in iter_examples(input_path):
            if not example.answers:
                raise ValueError("prepare requires labeled training data, not the test set")
            split = "validation" if is_validation(example.question, seed, fraction) else "train"
            write_record(validation if split == "validation" else train, example_record(example))
            counts[split] += 1
    manifest = {
        "source_sha256": file_sha256(input_path),
        "seed": seed,
        "validation_fraction": fraction,
        "grouping": "casefolded question tokens; duplicate questions stay in the same split",
        "counts": counts,
        "status": "complete",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
