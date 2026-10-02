"""LoCoMo paper scoring, kept local so this evaluation is self-contained."""

import re
import string
from collections import Counter

from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()


def _normalize(value: str) -> str:
    value = str(value).replace(",", "")
    value = re.sub(r"\b(a|an|the|and)\b", " ", value, flags=re.IGNORECASE)
    value = "".join(char for char in value if char not in string.punctuation)
    return " ".join(value.lower().split())


def _token_f1(prediction: str, reference: str) -> float:
    predicted = [_STEMMER.stem(word) for word in _normalize(prediction).split()]
    expected = [_STEMMER.stem(word) for word in _normalize(reference).split()]
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap or not predicted or not expected:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, reference: str, category: int) -> float:
    prediction, reference = str(prediction), str(reference)
    if category in (2, 3, 4):
        return _token_f1(prediction, reference)
    if category == 1:
        predicted = [part.strip() for part in prediction.split(",")]
        expected = [part.strip() for part in reference.split(",")]
        return sum(max(_token_f1(part, answer) for part in predicted)
                   for answer in expected) / len(expected)
    if category == 5:
        abstentions = ("no information", "not mentioned", "not provided", "don't know",
                       "do not know", "cannot find", "no record")
        return 1.0 if any(phrase in prediction.lower() for phrase in abstentions) else 0.0
    raise ValueError(f"Unknown LoCoMo category: {category}")
