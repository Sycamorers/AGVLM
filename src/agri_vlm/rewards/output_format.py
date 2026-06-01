"""Output-format penalties for malformed agricultural completions."""

from agri_vlm.rewards.clarify_decision import infer_decision
from agri_vlm.rewards.parsing import extract_answer_field, extract_structured_sections, max_length_or_repetition_penalty
from agri_vlm.schemas.reward_schema import RewardInput
from agri_vlm.utils.text import normalize_label, normalize_text, word_count


MALFORMED_SHORT_ANSWERS = {
    "",
    ":",
    "::",
    "answer",
    "answer:",
    "final answer",
    "final answer:",
}
GENERIC_AGRICULTURE_ANSWERS = {
    "plant",
    "plant disease",
    "plant.",
    "crop",
    "crop disease",
    "disease",
    "leaf",
    "image",
    "unknown plant disease",
}
LOW_INFORMATION_SECTION_BODIES = {
    "",
    "n a",
    "na",
    "none",
    "unknown",
    "uncertain",
    "not sure",
    "tbd",
    "todo",
}


def _answer_text(prediction: str) -> str:
    answer = extract_answer_field(prediction)
    normalized = normalize_text(answer)
    if normalized in MALFORMED_SHORT_ANSWERS:
        return ""
    return answer


def _is_generic_answer(answer: str, reward_input: RewardInput) -> bool:
    normalized = normalize_label(answer)
    if normalized in GENERIC_AGRICULTURE_ANSWERS:
        return True
    if reward_input.crop and normalized == normalize_label(reward_input.crop):
        return True
    if reward_input.disease and normalized == normalize_label(reward_input.disease):
        return True
    return False


def _section_key(section: str) -> str:
    return normalize_label(section).replace("-", " ")


def _meaningful_section_count(sections: dict[str, str], required_sections: list[str]) -> int:
    count = 0
    normalized_sections = {_section_key(key): value for key, value in sections.items()}
    for section in required_sections:
        value = normalized_sections.get(_section_key(section), "")
        normalized_body = normalize_text(value)
        if normalized_body in LOW_INFORMATION_SECTION_BODIES:
            continue
        if word_count(value) < 3:
            continue
        count += 1
    return count


def output_format_penalty(reward_input: RewardInput) -> float:
    """Return a non-positive penalty for malformed or incomplete outputs."""
    prediction = reward_input.prediction or ""
    penalty = max_length_or_repetition_penalty(prediction, reward_input.task_type)
    task_type = reward_input.task_type

    if task_type in {"classification", "vqa", "label_diagnosis"}:
        answer = _answer_text(prediction)
        if not normalize_text(answer):
            penalty -= 1.0
        elif _is_generic_answer(answer, reward_input):
            penalty -= 0.75

    if task_type == "clarify_or_respond":
        answer = _answer_text(prediction)
        decision = infer_decision(prediction)
        if not normalize_text(answer) or decision == "none":
            penalty -= 1.0

    if task_type == "consultation" and reward_input.required_sections:
        sections = extract_structured_sections(prediction)
        meaningful_count = _meaningful_section_count(sections, reward_input.required_sections)
        missing = max(0, len(reward_input.required_sections) - meaningful_count)
        if missing:
            penalty -= min(1.0, missing / float(len(reward_input.required_sections)))

    return min(0.0, max(-2.0, penalty))
