"""Parsing helpers for deterministic RL rewards."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

from agri_vlm.utils.text import normalize_label as _normalize_label
from agri_vlm.utils.text import normalize_text, normalize_whitespace, word_count


ANSWER_RE = re.compile(r"(?im)^\s*Answer\s*:\s*(?P<value>.+?)\s*$")
DECISION_RE = re.compile(r"(?im)^\s*Decision\s*:\s*(?P<value>clarify|respond)\b")
SECTION_RE = re.compile(r"(?im)^\s*(?P<header>[A-Za-z][A-Za-z -]{0,40})\s*:\s*(?P<body>.*)$")
REPEATED_TOKEN_RE = re.compile(r"\b(\w+)(?:\s+\1\b){4,}", re.IGNORECASE)


def extract_answer_field(text: str) -> str:
    """Return the first line-start Answer field, falling back to stripped text."""
    match = ANSWER_RE.search(text or "")
    if match:
        return normalize_whitespace(match.group("value"))
    return normalize_whitespace(text or "")


def extract_decision_field(text: str) -> str:
    """Return an explicit line-start Decision value if present."""
    match = DECISION_RE.search(text or "")
    return match.group("value").lower() if match else ""


def extract_structured_sections(text: str) -> Dict[str, str]:
    """Parse line-start section headers and their bodies.

    Only true line-start headers are accepted. Continuation lines are attached
    to the current section until the next section header.
    """
    sections: Dict[str, List[str]] = {}
    current_header = ""
    for raw_line in (text or "").splitlines():
        match = SECTION_RE.match(raw_line)
        if match:
            current_header = normalize_ag_label(match.group("header"))
            sections.setdefault(current_header, [])
            body = normalize_whitespace(match.group("body"))
            if body:
                sections[current_header].append(body)
            continue
        if current_header and raw_line.strip():
            sections[current_header].append(normalize_whitespace(raw_line))
    return {header: normalize_whitespace(" ".join(parts)) for header, parts in sections.items()}


def normalize_ag_label(text: str) -> str:
    """Normalize agricultural labels across dataset naming conventions."""
    normalized = _normalize_label(text)
    normalized = normalized.replace("healthy healthy", "healthy")
    return normalize_whitespace(normalized)


def _is_short_option(answer: str) -> bool:
    normalized = normalize_text(answer)
    return normalized in {"yes", "no", "true", "false", "unknown", "unclear"}


def safe_contains_answer(prediction: str, accepted_answers: Sequence[str]) -> bool:
    """Check an extracted Answer field against accepted answers.

    Short options require exact equality. Longer answers accept exact equality
    or conservative containment in either direction to support brief VQA
    variants without rewarding unrelated full-completion text.
    """
    candidate = normalize_text(extract_answer_field(prediction))
    if not candidate:
        return False
    for answer in accepted_answers:
        reference = normalize_text(str(answer))
        if not reference:
            continue
        if candidate == reference:
            return True
        if _is_short_option(reference):
            continue
        if word_count(reference) <= 8 and (reference in candidate or candidate in reference):
            return True
    return False


def max_length_or_repetition_penalty(text: str, task_type: str) -> float:
    """Return a small deterministic penalty for runaway completions."""
    token_count = word_count(text or "")
    limit_by_task = {
        "classification": 80,
        "vqa": 80,
        "clarify_or_respond": 120,
        "consultation": 260,
    }
    limit = limit_by_task.get(task_type, 160)
    penalty = 0.0
    if token_count > limit:
        penalty -= min(1.0, (token_count - limit) / float(limit))
    if REPEATED_TOKEN_RE.search(text or ""):
        penalty -= 0.5
    return penalty


def any_normalized_phrase(text: str, phrases: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    for phrase in phrases:
        normalized_phrase = normalize_text(str(phrase))
        if not normalized_phrase:
            continue
        if " " not in normalized_phrase:
            if re.search(r"\b%s\b" % re.escape(normalized_phrase), normalized):
                return True
            continue
        if normalized_phrase in normalized:
            return True
    return False
