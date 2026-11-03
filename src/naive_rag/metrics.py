import re
import string
from collections import Counter


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


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
    return {"normalized_em": max(float(p == normalize_answer(g)) for g in answers),
            "token_f1": max(f1(g) for g in answers),
            "raw_em": max(float(prediction == g) for g in answers),
            # A separate lenient diagnostic; never label this as Kaggle Accuracy.
            "substring_accuracy": max(float(normalize_answer(g) in p) if normalize_answer(g)
                                      else float(not p) for g in answers)}
