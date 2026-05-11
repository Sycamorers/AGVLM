"""Parsing helpers for inference-only agriculture VLM benchmarks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any


SECTION_HEADERS = ["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"]
PLACEHOLDER_PATH_MARKERS = {"", "todo", "tbd", "change_me", "changeme", "none", "null", "/path/to/checkpoint"}


def normalize_text(text: str | None) -> str:
    """Normalize answer text while preserving enough words for label matching."""
    value = (text or "").strip().lower()
    value = value.replace("\u2019", "'")
    value = re.sub(r"[^a-z0-9.+%:/_-]+", " ", value)
    value = value.replace("/", " ").replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_label(text: str | None) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(r"^\d+\s+", "", normalized)
    return normalized.strip()


def extract_answer_field(raw_output: str | None) -> tuple[str, str]:
    """Return explicit Answer field text and parse status."""
    text = raw_output or ""
    pattern = re.compile(
        r"(?im)^\s*(?:final\s+)?answer\s*:\s*(?P<answer>.*?)(?=\n\s*(?:decision|diagnosis|evidence|uncertainty|management|follow-up|explanation|confidence)\s*:|\Z)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return text.strip(), "raw"
    answer = match.group("answer").strip()
    return answer, "exact" if answer else "failed"


def extract_decision_field(raw_output: str | None) -> tuple[str, str]:
    """Parse clarify/respond decision with explicit field preferred."""
    text = raw_output or ""
    explicit = re.search(r"(?im)^\s*decision\s*:\s*(clarify|respond)\b", text)
    if explicit:
        return explicit.group(1).lower(), "exact"

    answer_text, answer_status = extract_answer_field(text)
    normalized = normalize_text(answer_text)
    first_token = normalized.split()[:1]
    if first_token and first_token[0] in {"clarify", "respond"}:
        return first_token[0], "exact" if answer_status == "exact" else "inferred"

    clarify_markers = [
        "need more information",
        "more information is needed",
        "additional information",
        "cannot determine",
        "can't determine",
        "not enough information",
        "image is insufficient",
        "please provide",
        "could you provide",
        "can you provide",
        "need a closer image",
        "need another image",
    ]
    if any(marker in normalized for marker in clarify_markers) or normalized.endswith("?"):
        return "clarify", "inferred"
    if normalized:
        return "respond", "inferred"
    return "", "failed"


def extract_structured_sections(raw_output: str | None) -> dict[str, str]:
    """Extract line-start consultation sections only.

    Loose substrings are intentionally ignored. A header must begin a line and
    match one of the known benchmark section names.
    """
    text = raw_output or ""
    header_re = re.compile(r"(?im)^\s*(diagnosis|evidence|uncertainty|management|follow-up)\s*:\s*")
    matches = list(header_re.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


def _label_regex(label_key: str) -> re.Pattern[str]:
    tokens = [re.escape(token) for token in label_key.split() if token]
    if not tokens:
        return re.compile(r"a^")
    return re.compile(r"(?<![a-z0-9])" + r"\s+".join(tokens) + r"(?![a-z0-9])")


def detect_ambiguous_label_mentions(text: str | None, label_space: list[str]) -> dict[str, Any]:
    normalized = normalize_label(text)
    matches: list[str] = []
    by_key: dict[str, str] = {}
    for label in label_space:
        key = normalize_label(label)
        if key:
            by_key.setdefault(key, label)
    for key, label in by_key.items():
        if _label_regex(key).search(normalized):
            matches.append(label)
    unique = sorted(set(matches), key=lambda value: normalize_label(value))
    return {
        "matched_labels": unique,
        "ambiguous": len(unique) > 1,
        "match_count": len(unique),
    }


def extract_label_from_answer(raw_output: str | None, label_space: list[str]) -> dict[str, Any]:
    """Extract a classification label and classify extraction quality."""
    answer, answer_status = extract_answer_field(raw_output)
    answer_norm = normalize_label(answer)
    if not answer_norm:
        return {
            "parsed_prediction": "",
            "normalized_prediction": "",
            "parse_status": "failed",
            "invalid_prediction": True,
            "label_mentions": [],
        }

    key_to_label: dict[str, str] = {}
    for label in label_space:
        key = normalize_label(label)
        if key:
            key_to_label.setdefault(key, label)

    if answer_norm in key_to_label:
        label = key_to_label[answer_norm]
        return {
            "parsed_prediction": label,
            "normalized_prediction": normalize_label(label),
            "parse_status": "exact",
            "invalid_prediction": False,
            "label_mentions": [label],
        }

    mentions = detect_ambiguous_label_mentions(answer, label_space)
    matched = mentions["matched_labels"]
    if len(matched) == 1:
        label = matched[0]
        return {
            "parsed_prediction": label,
            "normalized_prediction": normalize_label(label),
            "parse_status": "inferred" if answer_status != "exact" else "exact",
            "invalid_prediction": False,
            "label_mentions": matched,
        }
    if len(matched) > 1:
        return {
            "parsed_prediction": answer.strip(),
            "normalized_prediction": answer_norm,
            "parse_status": "ambiguous",
            "invalid_prediction": True,
            "label_mentions": matched,
        }
    return {
        "parsed_prediction": answer.strip(),
        "normalized_prediction": answer_norm,
        "parse_status": "failed" if answer_status == "failed" else "inferred",
        "invalid_prediction": True,
        "label_mentions": [],
    }


def parse_numeric_answer(text: str | None) -> float | None:
    answer, _ = extract_answer_field(text)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", answer.replace(",", ""))
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def normalize_yes_no(text: str | None) -> tuple[str, str]:
    answer, answer_status = extract_answer_field(text)
    normalized = normalize_text(answer)
    yes = bool(re.search(r"(?<![a-z0-9])yes(?![a-z0-9])", normalized))
    no = bool(re.search(r"(?<![a-z0-9])no(?![a-z0-9])", normalized))
    if yes and no:
        return "", "ambiguous"
    if yes:
        return "yes", "exact" if normalized in {"yes", "yes."} and answer_status == "exact" else "inferred"
    if no:
        return "no", "exact" if normalized in {"no", "no."} and answer_status == "exact" else "inferred"
    return "", "failed"


def detect_forbidden_claims(text: str | None, forbidden_claims: list[str] | None = None) -> list[str]:
    normalized = normalize_text(text)
    claims = list(forbidden_claims or [])
    default_claims = [
        "guaranteed cure",
        "no follow up needed",
        "no follow-up needed",
        "100 percent certain",
        "100% certain",
        "definitely safe",
        "always safe",
        "never harmful",
    ]
    for claim in default_claims:
        if claim not in claims:
            claims.append(claim)
    matched = []
    for claim in claims:
        if normalize_text(claim) and normalize_text(claim) in normalized:
            matched.append(claim)
    return matched


def detect_overconfidence(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    markers = [
        "definitely",
        "certainly",
        "100 percent",
        "100%",
        "without doubt",
        "guaranteed",
        "always",
        "never",
        "no need to verify",
        "no need for follow up",
        "no need for follow-up",
    ]
    return [marker for marker in markers if normalize_text(marker) in normalized]


def repetition_stats(text: str | None) -> dict[str, float | int]:
    tokens = normalize_text(text).split()
    if not tokens:
        return {
            "token_count": 0,
            "unique_token_count": 0,
            "repetition_rate": 0.0,
            "max_trigram_count": 0,
        }
    counts = Counter(tokens)
    trigrams = Counter(tuple(tokens[index : index + 3]) for index in range(max(len(tokens) - 2, 0)))
    repeated_tokens = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "token_count": len(tokens),
        "unique_token_count": len(counts),
        "repetition_rate": repeated_tokens / float(len(tokens)),
        "max_trigram_count": max(trigrams.values()) if trigrams else 0,
    }


@dataclass(frozen=True)
class ParsedOutput:
    parsed_prediction: str
    normalized_prediction: str
    parse_status: str
    invalid_prediction: bool
    extra: dict[str, Any]


def parse_prediction_output(
    *,
    raw_output: str | None,
    task_type: str,
    verifier_mode: str,
    label_space: list[str] | None = None,
) -> ParsedOutput:
    label_space = label_space or []
    if verifier_mode == "label":
        result = extract_label_from_answer(raw_output, label_space)
        extra = {"label_mentions": result.get("label_mentions", [])}
        return ParsedOutput(
            parsed_prediction=str(result["parsed_prediction"]),
            normalized_prediction=str(result["normalized_prediction"]),
            parse_status=str(result["parse_status"]),
            invalid_prediction=bool(result["invalid_prediction"]),
            extra=extra,
        )

    if verifier_mode == "clarify" or task_type == "clarify_or_respond":
        decision, status = extract_decision_field(raw_output)
        return ParsedOutput(
            parsed_prediction=decision,
            normalized_prediction=decision,
            parse_status=status,
            invalid_prediction=decision not in {"clarify", "respond"},
            extra={},
        )

    if task_type == "consultation" or verifier_mode == "structured":
        sections = extract_structured_sections(raw_output)
        normalized = normalize_text(raw_output)
        return ParsedOutput(
            parsed_prediction=(raw_output or "").strip(),
            normalized_prediction=normalized,
            parse_status="exact" if sections else ("failed" if not normalized else "raw"),
            invalid_prediction=not bool(normalized),
            extra={"sections": sections},
        )

    answer, status = extract_answer_field(raw_output)
    normalized = normalize_text(answer)
    return ParsedOutput(
        parsed_prediction=answer.strip(),
        normalized_prediction=normalized,
        parse_status=status if normalized else "failed",
        invalid_prediction=not bool(normalized),
        extra={},
    )
