import re
import string
from collections import Counter

METRIC_VERSION = "answer-metrics-v2"
CATEGORICAL_ANSWERS = {"yes", "no", "noanswer", "unknown"}


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def phrase_present(answer, text):
    """Normalized whole-token phrase presence; 'no' must not match 'unknown'."""
    answer, text = normalize_answer(answer), normalize_answer(text)
    return bool(answer) and f" {answer} " in f" {text} "


def scores(prediction, answers):
    p = normalize_answer(prediction)
    def f1(gold):
        g = normalize_answer(gold)
        # FlashRAG/Hotpot-style handling: categorical answers require exact agreement.
        if (p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}) and p != g:
            return 0.0
        a, b = p.split(), g.split()
        if not a or not b:
            return float(a == b)
        overlap = sum((Counter(a) & Counter(b)).values())
        return 2 * overlap / (len(a) + len(b))
    def containment(gold):
        g = normalize_answer(gold)
        if not p or not g or p in CATEGORICAL_ANSWERS or g in CATEGORICAL_ANSWERS:
            return float(p == g)
        return float(phrase_present(gold, prediction))

    return {"normalized_em": max(float(p == normalize_answer(g)) for g in answers),
            "token_f1": max(f1(g) for g in answers),
            "raw_em": max(float(prediction == g) for g in answers),
            # Keep the field name for reports, but enforce token boundaries and exact
            # categorical answers. The old raw substring value is explicitly legacy.
            "substring_accuracy": max(containment(g) for g in answers),
            "legacy_substring_accuracy": max(float(normalize_answer(g) in p) if normalize_answer(g)
                                             else float(not p) for g in answers)}
