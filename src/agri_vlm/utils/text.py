"""Text normalization and lightweight metrics."""

import re
from typing import Iterable, List, Sequence, Set


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_text(text: str) -> str:
    return normalize_whitespace(re.sub(r"[^a-z0-9\s:/_-]+", " ", (text or "").lower()))


def normalize_label(label: str) -> str:
    normalized = normalize_text(label).replace("/", " ").replace("_", " ")
    normalized = normalized.replace("___", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def token_set(text: str) -> Set[str]:
    return {token for token in normalize_text(text).split(" ") if token}


def overlap_ratio(reference_terms: Iterable[str], prediction: str) -> float:
    reference = {normalize_label(term) for term in reference_terms if normalize_label(term)}
    if not reference:
        return 0.0
    prediction_terms = token_set(prediction)
    hits = 0
    for term in reference:
        term_tokens = token_set(term)
        if term_tokens and term_tokens.issubset(prediction_terms):
            hits += 1
    return hits / float(len(reference))


def exact_match(reference: str, prediction: str) -> float:
    return 1.0 if normalize_text(reference) == normalize_text(prediction) else 0.0


def best_exact_match(references: Sequence[str], prediction: str) -> float:
    return max([exact_match(reference, prediction) for reference in references] + [0.0])


def word_count(text: str) -> int:
    return len([token for token in normalize_whitespace(text).split(" ") if token])


def contains_any(text: str, phrases: Sequence[str]) -> bool:
    normalized = normalize_text(text)
    normalized_phrases = [normalize_text(phrase) for phrase in phrases if normalize_text(phrase)]
    return any(phrase in normalized for phrase in normalized_phrases)


def section_headers_present(text: str, headers: Sequence[str]) -> List[str]:
    normalized = normalize_text(text)
    return [header for header in headers if normalize_text(header + ":") in normalized]
