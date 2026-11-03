import hashlib
import json
from pathlib import Path


def records(path):
    """Stream JSONL or a top-level JSON array without loading the 1.65 GB train file."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as f:
        if path.suffix.lower() == ".jsonl":
            for line in f:
                if line.strip():
                    yield json.loads(line)
            return
        decoder = json.JSONDecoder()
        buf, eof = "", False

        def refill():
            nonlocal buf, eof
            chunk = f.read(65536)
            buf += chunk
            eof = not chunk

        def whitespace():
            nonlocal buf
            while True:
                buf = buf.lstrip()
                if buf or eof:
                    return
                refill()

        whitespace()
        if not buf.startswith("["):
            raise ValueError("Expected a top-level JSON array or a .jsonl file")
        buf = buf[1:]
        whitespace()
        if buf.startswith("]"):
            buf = buf[1:]
        else:
            while True:
                while True:
                    try:
                        item, end = decoder.raw_decode(buf)
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise ValueError(f"Malformed/incomplete JSON in {path}")
                        refill()
                if not isinstance(item, dict):
                    raise ValueError("Every sample must be a JSON object")
                yield item
                buf = buf[end:]
                whitespace()
                if buf.startswith("]"):
                    buf = buf[1:]
                    break
                if not buf.startswith(","):
                    raise ValueError("Expected comma between samples")
                buf = buf[1:]
                whitespace()
        whitespace()
        if buf:
            raise ValueError("Unexpected data after JSON array")


def normalize(row, index):
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Row {index}: missing question")
    docs = row.get("ctxs")
    if docs is None and "context" in row:
        docs = [{"title": t, "text": " ".join(ss)} for t, ss in row["context"]]
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"Row {index}: ctxs must be a nonempty list")
    clean, seen = [], set()
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
            raise ValueError(f"Row {index}, document {i}: expected ctxs[].text string")
        title, text = str(doc.get("title") or ""), doc["text"].strip()
        if not text or (title, text) in seen:
            continue
        seen.add((title, text))
        clean.append({"id": str(doc.get("id", i)), "title": title, "text": text})
    if not clean:
        raise ValueError(f"Row {index}: no usable documents")
    # Deliberately discard rationale, qa_pairs, answers, and supporting_facts here.
    rid = row.get("id", row.get("_id", f"row-{index}"))
    if rid is None or isinstance(rid, (dict, list, bool)):
        raise ValueError(f"Row {index}: invalid id")
    return {"id": str(rid), "question": question.strip(), "ctxs": clean}


def gold_answers(row):
    value = row.get("answers", row.get("answer"))
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    raise ValueError("answers must be a string or a list of strings")


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validation_group(question, seed, fraction):
    key = " ".join(question.lower().split())
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:16], 16) / 2**64 < fraction
