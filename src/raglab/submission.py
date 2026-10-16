from __future__ import annotations

import csv
from pathlib import Path

from raglab.data import iter_records


def export_submission(
    traces_path: Path,
    template_path: Path,
    output_path: Path,
    expected_rows: int = 12576,
    allow_mock: bool = False,
) -> int:
    if expected_rows < 1:
        raise ValueError("expected_rows must be positive")
    if allow_mock and expected_rows == 12576:
        raise ValueError("Mock predictions cannot be exported as a full competition submission")
    with template_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "answer"]:
            raise ValueError("Template header must be exactly id,answer")
        ids = [row["id"] for row in reader]
    if len(ids) != expected_rows or len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError(f"Template must have {expected_rows} unique, non-empty IDs")
    predictions = {}
    for trace in iter_records(traces_path):
        if "mock" in trace["backend"].split("+") and not allow_mock:
            raise ValueError("Mock predictions are for smoke tests only; use real model outputs")
        key = str(trace["id"])
        if key in predictions:
            raise ValueError(f"Duplicate prediction ID: {key}")
        answer = trace["final_answer"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Empty answer for {key}")
        if trace["stop_reason"] in ("invalid_critique", "invalid_revision"):
            raise ValueError(f"Resolve model format failures before exporting ID {key}")
        predictions[key] = answer.strip()
    if set(predictions) != set(ids):
        raise ValueError("Prediction IDs do not match the complete template")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "answer"])
        for key in ids:
            writer.writerow([key, predictions[key]])
    return len(ids)
