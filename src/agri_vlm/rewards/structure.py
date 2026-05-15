"""Structured output rewards."""

import re
from collections import Counter
from typing import Dict, List, Tuple

from agri_vlm.rewards.parsing import extract_structured_sections, normalize_ag_label
from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import normalize_text, word_count


SECTION_LINE_RE = re.compile(r"(?im)^\s*(?P<header>[A-Za-z][A-Za-z -]{0,40})\s*:\s*(?P<body>.*)$")


MIN_SECTION_WORDS = {
    "diagnosis": 2,
    "evidence": 4,
    "uncertainty": 4,
    "management": 4,
    "follow up": 4,
}

LOW_INFORMATION_BODIES = {
    "",
    "n a",
    "na",
    "none",
    "unknown",
    "uncertain",
    "not sure",
    "tbd",
    "todo",
    "same as above",
}


def _required_key(section: str) -> str:
    return normalize_ag_label(section).replace("-", " ")


def _section_heading_stats(text: str, required_sections: List[str]) -> Tuple[Counter[str], Counter[str]]:
    required_keys = {_required_key(section) for section in required_sections}
    heading_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    for match in SECTION_LINE_RE.finditer(text or ""):
        key = _required_key(match.group("header"))
        if key not in required_keys:
            continue
        heading_counts[key] += 1
        if not normalize_text(match.group("body")):
            empty_counts[key] += 1
    return heading_counts, empty_counts


def _is_meaningful_section(section_key: str, content: str) -> bool:
    normalized = normalize_text(content)
    if normalized in LOW_INFORMATION_BODIES:
        return False
    minimum_words = MIN_SECTION_WORDS.get(section_key, 3)
    if word_count(content) < minimum_words:
        return False
    alphabetic_tokens = [token for token in normalized.split() if any(ch.isalpha() for ch in token)]
    return len(alphabetic_tokens) >= minimum_words


def structured_format_reward(reward_input: RewardInput) -> float:
    required_sections = reward_input.required_sections
    if not required_sections:
        return 0.0
    parsed = extract_structured_sections(reward_input.prediction)
    heading_counts, empty_counts = _section_heading_stats(reward_input.prediction, required_sections)
    meaningful = []
    for section in required_sections:
        section_key = _required_key(section)
        content = parsed.get(section_key) or parsed.get(normalize_ag_label(section)) or ""
        if _is_meaningful_section(section_key, content):
            meaningful.append(section_key)
    base_score = len(meaningful) / float(len(required_sections))
    duplicate_count = sum(max(0, heading_counts[_required_key(section)] - 1) for section in required_sections)
    empty_count = sum(empty_counts[_required_key(section)] for section in required_sections)
    penalty = min(0.6, duplicate_count * 0.15 + empty_count * 0.20)
    return max(0.0, min(1.0, base_score - penalty))
