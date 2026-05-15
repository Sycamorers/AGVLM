"""Management advice coverage reward."""

import re
from typing import List

from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import normalize_label, normalize_text, word_count


ACTION_CONTEXT_MARKERS = [
    "remove",
    "prune",
    "monitor",
    "avoid",
    "improve",
    "inspect",
    "isolate",
    "rotate",
    "water",
    "irrigation",
    "mulch",
    "sanitize",
    "dispose",
    "extension",
    "confirm",
    "scout",
    "apply",
    "use",
    "treat",
]


def _unique_keywords(keywords: List[str]) -> List[str]:
    normalized = []
    seen = set()
    for keyword in keywords:
        value = normalize_label(keyword)
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"[\n.;!?]+", text or "") if part.strip()]


def _contains_keyword(sentence: str, keyword: str) -> bool:
    normalized_sentence = normalize_text(sentence)
    if not keyword or not normalized_sentence:
        return False
    if keyword in normalized_sentence:
        return True
    keyword_tokens = {token for token in keyword.split() if token}
    sentence_tokens = set(normalized_sentence.split())
    return bool(keyword_tokens) and keyword_tokens.issubset(sentence_tokens)


def _is_meaningful_context(sentence: str, keyword: str) -> bool:
    normalized = normalize_text(sentence)
    tokens = normalized.split()
    if len(tokens) < max(5, len(keyword.split()) + 2):
        return False
    unique_ratio = len(set(tokens)) / float(len(tokens)) if tokens else 0.0
    if unique_ratio < 0.45:
        return False
    if "," in sentence and len(tokens) <= 8:
        return False
    return any(marker in normalized for marker in ACTION_CONTEXT_MARKERS)


def _keyword_repetition_penalty(prediction: str, keywords: List[str]) -> float:
    normalized = normalize_text(prediction)
    total_hits = 0
    repeated_hits = 0
    for keyword in keywords:
        count = normalized.count(keyword)
        total_hits += count
        if count > 2:
            repeated_hits += count - 2
    penalty = min(0.25, repeated_hits * 0.05)
    token_count = word_count(prediction)
    if token_count > 280:
        penalty += min(0.25, (token_count - 280) / 560.0)
    tokens = normalized.split()
    if tokens and len(set(tokens)) / float(len(tokens)) < 0.35:
        penalty += 0.2
    if total_hits > max(6, len(keywords) * 3):
        penalty += 0.15
    return min(0.5, penalty)


def management_coverage_reward(reward_input: RewardInput) -> float:
    keywords = _unique_keywords(reward_input.management_keywords)
    if not keywords:
        return 0.0
    matched = set()
    for keyword in keywords:
        for sentence in _sentences(reward_input.prediction):
            if _contains_keyword(sentence, keyword) and _is_meaningful_context(sentence, keyword):
                matched.add(keyword)
                break
    base_score = min(0.5, 0.5 * (len(matched) / float(len(keywords))))
    return max(-0.25, min(0.5, base_score - _keyword_repetition_penalty(reward_input.prediction, keywords)))
