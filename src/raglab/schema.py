from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

SampleID = str | int


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    score: float | None = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Document ID must be a non-empty string")
        if (
            not isinstance(self.title, str)
            or not isinstance(self.text, str)
            or not self.text.strip()
        ):
            raise ValueError("Document requires a string title and non-empty text")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("Document score must be finite")


@dataclass(frozen=True)
class Example:
    id: SampleID
    question: str
    contexts: tuple[Document, ...]
    answers: tuple[str, ...] = ()
    rationale: str | None = None
    explicit_id: bool = True


@dataclass(frozen=True)
class Answer:
    answer: str
    explanation: str = ""
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Critique:
    decision: Literal["stop", "revise"]
    feedback: str
    evidence_ids: tuple[str, ...] = ()


@dataclass
class Step:
    index: int
    kind: str
    content: dict
    seconds: float


@dataclass
class Trace:
    id: SampleID
    question: str
    mode: str
    backend: str
    documents: list[dict]
    initial_answer: str
    final_answer: str
    revisions: int
    stop_reason: str
    elapsed_seconds: float
    steps: list[Step] = field(default_factory=list)
    format_errors: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ModelOutputError(ValueError):
    """A model response failed the public JSON output contract."""


def parse_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ModelOutputError(
            "Model must return one JSON object, without surrounding prose"
        ) from exc
    if not isinstance(value, dict):
        raise ModelOutputError("Model response is not an object")
    return value


def _evidence(value: dict, documents: tuple[Document, ...]) -> tuple[str, ...]:
    ids = value.get("evidence_ids", [])
    if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
        raise ModelOutputError("evidence_ids must be a list of strings")
    if len(set(ids)) != len(ids):
        raise ModelOutputError("Duplicate evidence IDs")
    unknown = set(ids) - {d.id for d in documents}
    if unknown:
        raise ModelOutputError(f"Unknown evidence IDs: {sorted(unknown)}")
    return tuple(ids)


def parse_answer(raw: str, documents: tuple[Document, ...]) -> Answer:
    value = parse_object(raw)
    answer = value.get("answer")
    explanation = value.get("explanation", "")
    if not isinstance(answer, str) or not answer.strip():
        raise ModelOutputError("answer must be a non-empty string")
    if not isinstance(explanation, str):
        raise ModelOutputError("explanation must be a string")
    return Answer(answer.strip(), explanation, _evidence(value, documents))


def parse_critique(raw: str, documents: tuple[Document, ...]) -> Critique:
    value = parse_object(raw)
    decision = value.get("decision")
    feedback = value.get("feedback")
    if decision not in ("stop", "revise"):
        raise ModelOutputError("decision must be stop or revise")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ModelOutputError("feedback must be non-empty, including a reason when stopping")
    return Critique(decision, feedback.strip(), _evidence(value, documents))
