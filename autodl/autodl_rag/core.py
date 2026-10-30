from __future__ import annotations

import hashlib
import json
import os
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

PROMPT_VERSION = "autodl-v3-bounded-json"


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_line(f, value):
    f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    f.flush()


def records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    with path.open("rb") as f:
        offset = 3 if f.read(3) == b"\xef\xbb\xbf" else 0
        f.seek(offset)
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if first != b"[":
            raise ValueError(f"{path}: expected a top-level JSON array or .jsonl")
        f.seek(offset)
        try:
            import ijson
        except ImportError:
            if path.stat().st_size > 16 * 1024 * 1024:
                raise RuntimeError("Large JSON requires ijson: pip install -e .")
            yield from json.load(f)
        else:
            yield from ijson.items(f, "item", use_float=True)


def question_key(q):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", q).casefold()))


def normalized(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def exact(answer, gold):
    return int(any(normalized(answer) == normalized(g) for g in gold))


def token_f1(answer, gold):
    def one(g):
        a, b = normalized(answer).split(), normalized(g).split()
        if not a or not b:
            return float(a == b)
        n = sum((Counter(a) & Counter(b)).values())
        return 2 * n / (len(a) + len(b))

    return max((one(g) for g in gold), default=0.0)


def normalize_record(raw, index, require_id=False):
    q = raw.get("question")
    if not isinstance(q, str) or not q.strip():
        raise ValueError(f"Row {index}: missing question")
    if require_id and "id" not in raw and "_id" not in raw:
        raise ValueError(f"Test row {index}: explicit id is required")
    sid = raw.get("id", raw.get("_id", f"train:{index}"))
    if isinstance(sid, bool) or not isinstance(sid, (str, int)) or not str(sid).strip():
        raise ValueError(f"Row {index}: invalid id")
    ctxs = raw.get("ctxs", raw.get("context", raw.get("contexts")))
    if not isinstance(ctxs, list) or not ctxs:
        raise ValueError(f"Row {index}: nonempty ctxs required")
    docs, seen = [], set()
    for j, d in enumerate(ctxs):
        title = ""
        if isinstance(d, str):
            text = d
        elif isinstance(d, dict):
            title = d.get("title", "")
            text = d.get("text", d.get("contents", d.get("passage")))
            if text is None and isinstance(d.get("sentences"), list):
                text = " ".join(d["sentences"])
        elif isinstance(d, (list, tuple)) and len(d) == 2:
            title, text = d
            if isinstance(text, list):
                text = " ".join(text)
        else:
            raise ValueError(f"Row {index}, context {j}: unknown document schema")
        if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Row {index}, context {j}: invalid title/text")
        key = fingerprint([title.strip(), text.strip()])
        if key not in seen:
            docs.append(
                {"id": "d_" + key[:20], "title": title.strip(), "text": text.strip()}
            )
            seen.add(key)
    answers = raw.get("answers", raw.get("answer", []))
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list) or any(
        not isinstance(a, str) or not a.strip() for a in answers
    ):
        raise ValueError(f"Row {index}: invalid answers")
    # No rationale, qa_pairs, supporting_facts, or labels on any document.
    return {"id": str(sid), "question": q.strip(), "ctxs": docs, "answers": answers}


def public(row):
    return {k: row[k] for k in ("id", "question", "ctxs")}


class InvalidOutput(ValueError):
    pass


def json_object(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise InvalidOutput(f"Invalid JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise InvalidOutput("Expected one JSON object")
    return obj


def citations(obj, allowed):
    refs = obj.get("evidence_ids")
    if not isinstance(refs, list) or any(not isinstance(r, str) or r not in allowed for r in refs):
        raise InvalidOutput(
            f"Invalid evidence_ids: expected a JSON list containing only visible document IDs "
            f"{sorted(allowed)!r}; got {refs!r}"
        )
    return list(dict.fromkeys(refs))


def parse_answer(raw, allowed):
    obj = json_object(raw)
    answer = obj.get("answer")
    if (
        not isinstance(answer, str)
        or not answer.strip()
        or any(c in answer for c in "\r\n")
    ):
        raise InvalidOutput("answer must be a nonempty single-line string")
    if len(answer) > 500:
        raise InvalidOutput("answer is too long")
    why = obj.get("explanation")
    if not isinstance(why, str) or not why.strip():
        raise InvalidOutput("Missing explanation")
    return {
        "answer": answer.strip(),
        "explanation": why.strip(),
        "evidence_ids": citations(obj, allowed),
    }


def parse_critique(raw, allowed):
    obj = json_object(raw)
    if obj.get("action") not in ("stop", "revise"):
        raise InvalidOutput("action must be stop or revise")
    if not isinstance(obj.get("feedback"), str) or not obj["feedback"].strip():
        raise InvalidOutput("feedback required")
    refs = citations(obj, allowed)
    if obj["action"] == "revise" and not refs:
        raise InvalidOutput("revise needs actual evidence_ids")
    queries = obj.get("search_queries", [])
    if not isinstance(queries, list) or any(not isinstance(q, str) for q in queries):
        raise InvalidOutput("search_queries must be strings")
    suggestion = obj.get("suggested_answer", "")
    if not isinstance(suggestion, str):
        raise InvalidOutput("suggested_answer must be a string")
    return {
        "action": obj["action"],
        "feedback": obj["feedback"],
        "evidence_ids": refs,
        "suggested_answer": suggestion,
        "search_queries": [q[:240] for q in queries[:2]],
    }


def reward(trace, gold, cfg):
    a = exact(trace["initial"]["answer"], gold)
    b = exact(trace["final"]["answer"], gold)
    base = {(0, 1): 2.0, (1, 1): 1.0, (1, 0): -3.0, (0, 0): -0.3}[(a, b)]
    # A rejected unsafe candidate still costs generation, but is not a final break.
    unsafe = sum(
        bool(a and t.get("candidate") and not exact(t["candidate"]["answer"], gold))
        for t in trace["turns"]
    )
    invalid = sum(
        t.get("status") in ("invalid_critique", "invalid_revision")
        for t in trace["turns"]
    )
    result = base - cfg["reward_revision_cost"] * trace["revisions"]
    result -= cfg["reward_invalid_cost"] * invalid + cfg["reward_unsafe_cost"] * unsafe
    return {
        "total": result,
        "initial_em": a,
        "final_em": b,
        "unsafe": unsafe,
        "invalid": invalid,
    }


def load_config(path):
    cfg = read_json(path)
    required = ("data_dir", "work_dir", "qwen_path", "llama_path")
    for k in required:
        if not cfg.get(k):
            raise ValueError(f"Missing config: {k}")
    defaults = read_json(Path(__file__).parent / "defaults.json")
    cfg = {**defaults, **cfg}
    if cfg.get("retrieval_policy", "legacy") not in ("legacy", "coverage"):
        raise ValueError("retrieval_policy must be legacy or coverage")
    rounds = cfg.get("retrieval_plan_rounds", 0)
    if type(rounds) is not int or not 0 <= rounds <= 2:
        raise ValueError("retrieval_plan_rounds must be an integer from 0 to 2")
    if cfg.get("retrieval_plan_trigger", "unknown") not in ("unknown", "always"):
        raise ValueError("retrieval_plan_trigger must be unknown or always")
    for k in (
        "max_input_tokens",
        "max_rounds",
        "group_size",
        "local_top_k",
        "max_documents",
    ):
        if not isinstance(cfg[k], int) or cfg[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if (
        cfg["group_size"] < 2
        or cfg["temperature"] <= 0
        or cfg["advantage_std_floor"] <= 0
    ):
        raise ValueError("Invalid GRPO group/temperature/advantage settings")
    return cfg
