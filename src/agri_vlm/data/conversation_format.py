"""Convert normalized samples into model-facing chat representations."""

import copy
import json
from typing import Any, Dict, List

from agri_vlm.schemas.dataset_schema import Message, UnifiedSample


PLAIN_FORMAT = "plain"
INSTRUCTIONAL_FORMAT = "instructional"
CLASSIFICATION_LABEL_ONLY_FORMAT = "classification_label_only"
MANIFEST_PROMPT_FORMAT = "manifest"


def _plain_target_to_text(sample: UnifiedSample) -> str:
    target = sample.target
    if target.answer_text:
        return target.answer_text
    if target.canonical_label:
        return target.canonical_label
    if target.canonical_labels:
        return ", ".join(target.canonical_labels)
    if target.decision and target.structured:
        payload = {"decision": target.decision, "content": target.structured}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if target.decision:
        return target.decision
    if target.structured:
        return json.dumps(target.structured, ensure_ascii=False, sort_keys=True)
    if target.acceptable_answers:
        return target.acceptable_answers[0]
    raise ValueError("Unable to render target text for sample %s" % sample.sample_id)


def _is_yes_no_target(sample: UnifiedSample) -> bool:
    values = []
    if sample.target.answer_text:
        values.append(sample.target.answer_text)
    values.extend(sample.target.acceptable_answers)
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    return bool(normalized) and normalized.issubset({"yes", "no"})


def _classification_label_space(sample: UnifiedSample) -> List[str]:
    metadata = sample.metadata or {}
    raw_labels = metadata.get("classification_label_space") or metadata.get("allowed_classification_labels") or []
    if not isinstance(raw_labels, list):
        return []
    labels: List[str] = []
    seen = set()
    for raw_label in raw_labels:
        label = str(raw_label or "").strip()
        key = " ".join(label.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def _classification_choice_options(sample: UnifiedSample) -> List[Dict[str, str]]:
    metadata = sample.metadata or {}
    raw_options = metadata.get("classification_choice_options") or []
    if not isinstance(raw_options, list):
        return []
    options: List[Dict[str, str]] = []
    seen_letters = set()
    seen_labels = set()
    for raw_option in raw_options:
        if not isinstance(raw_option, dict):
            return []
        letter = str(raw_option.get("letter") or "").strip().upper()
        label = str(raw_option.get("label") or "").strip()
        label_key = " ".join(label.lower().split())
        if not letter or len(letter) > 2 or not label or letter in seen_letters or label_key in seen_labels:
            return []
        seen_letters.add(letter)
        seen_labels.add(label_key)
        options.append({"letter": letter, "label": label})
    return options


def _classification_choice_answer(sample: UnifiedSample) -> Dict[str, str]:
    options = _classification_choice_options(sample)
    if not options:
        return {}
    target_label = str(sample.target.canonical_label or sample.target.answer_text or "").strip()
    target_key = " ".join(target_label.lower().split())
    for option in options:
        if " ".join(option["label"].lower().split()) == target_key:
            return option
    return {}


def _is_classification_sample(sample: UnifiedSample) -> bool:
    return sample.task_type == "classification" or sample.verifier.mode == "label"


def label_only_instruction_for_classification(sample: UnifiedSample) -> str:
    label_space = _classification_label_space(sample)
    label_instruction = ""
    if len(label_space) > 1:
        label_instruction = (
            "Choose exactly one label from this allowed label set:\n"
            f"Allowed labels: {'; '.join(label_space)}\n"
        )
    return (
        f"{label_instruction}"
        "Respond with only the selected label text.\n"
        "Do not include Answer:, Evidence:, option letters, punctuation, Markdown, JSON, or explanations."
    )


def output_instruction_for_sample(sample: UnifiedSample) -> str:
    """Return the preferred explicit output contract for SFT-style training."""
    if sample.task_type == "clarify_or_respond" or sample.verifier.mode == "clarify" or sample.target.decision:
        return (
            "Respond using exactly one of these formats:\n"
            "Decision: clarify\n"
            "Clarifying question: <one question needed before diagnosis or management>\n\n"
            "Decision: respond\n"
            "Answer: <concise agricultural answer>"
        )
    if sample.task_type == "consultation" or sample.verifier.mode == "structured" or sample.target.structured:
        return (
            "Respond using these line-start section headers exactly once:\n"
            "Diagnosis:\nEvidence:\nUncertainty:\nManagement:\nFollow-up:"
        )
    if sample.task_type == "classification" or sample.verifier.mode == "label" or sample.target.canonical_label:
        choice_options = _classification_choice_options(sample)
        if choice_options:
            rendered_options = "\n".join(
                "%s. %s" % (option["letter"], option["label"]) for option in choice_options
            )
            return (
                "Choose exactly one option from this list:\n"
                "%s\n"
                "Respond in this format:\n"
                "Choice: <option letter>\n"
                "Answer: <label text from the selected option>\n"
                "Evidence: <brief visible symptom evidence>\n"
                "Do not leave Choice or Answer blank, invent labels, or copy the placeholder text."
                % rendered_options
            )
        label_space = _classification_label_space(sample)
        label_instruction = ""
        answer_placeholder = "<canonical agricultural label>"
        if len(label_space) > 1:
            label_instruction = (
                "Choose exactly one label from this allowed label set:\n"
                f"Allowed labels: {'; '.join(label_space)}\n"
            )
            answer_placeholder = "<one allowed label>"
        return (
            f"{label_instruction}"
            "Respond in this format:\n"
            f"Answer: {answer_placeholder}\n"
            "Evidence: <brief visible symptom evidence>\n"
            "Do not leave Answer blank, invent labels, or copy the placeholder text."
        )
    if _is_yes_no_target(sample):
        return "Respond in this format:\nAnswer: <Yes or No>"
    if sample.task_type == "vqa" or sample.verifier.mode in {"exact_match", "synonym"}:
        return "Respond in this format:\nAnswer: <short answer>"
    return "Respond in a concise agriculture-focused format."


def _has_output_instruction(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    markers = [
        "respond in this format",
        "respond using exactly",
        "respond using these line start",
        "respond using these line-start",
        "answer:",
        "decision:",
        "choice:",
    ]
    return any(marker in normalized for marker in markers)


def _append_output_instruction(messages: List[Dict[str, Any]], instruction: str) -> List[Dict[str, Any]]:
    rendered = copy.deepcopy(messages)
    for message in rendered:
        if message.get("role") != "user":
            continue
        for content in message.get("content") or []:
            if content.get("type") != "text":
                continue
            text = str(content.get("text") or "").strip()
            if _has_output_instruction(text):
                return rendered
            content["text"] = "%s\n\n%s" % (text, instruction.strip()) if text else instruction.strip()
            return rendered
    raise ValueError("Sample %s is missing a user text prompt." % messages)


def sample_to_prompt_messages(
    sample: UnifiedSample,
    *,
    prompt_format: str = MANIFEST_PROMPT_FORMAT,
) -> List[Dict[str, Any]]:
    messages = [message.model_dump(mode="json", exclude_none=True) for message in sample.messages]
    if prompt_format == MANIFEST_PROMPT_FORMAT:
        return messages
    if prompt_format == INSTRUCTIONAL_FORMAT:
        return _append_output_instruction(messages, output_instruction_for_sample(sample))
    if prompt_format == CLASSIFICATION_LABEL_ONLY_FORMAT:
        instruction = (
            label_only_instruction_for_classification(sample)
            if _is_classification_sample(sample)
            else output_instruction_for_sample(sample)
        )
        return _append_output_instruction(messages, instruction)
    raise ValueError("Unsupported prompt_format=%r for sample %s." % (prompt_format, sample.sample_id))


def target_to_text(sample: UnifiedSample, *, target_format: str = PLAIN_FORMAT) -> str:
    if target_format == PLAIN_FORMAT:
        return _plain_target_to_text(sample)
    if target_format == CLASSIFICATION_LABEL_ONLY_FORMAT:
        if _is_classification_sample(sample):
            return sample.target.canonical_label or sample.target.answer_text or _plain_target_to_text(sample)
        target_format = INSTRUCTIONAL_FORMAT
    if target_format != INSTRUCTIONAL_FORMAT:
        raise ValueError("Unsupported target_format=%r for sample %s." % (target_format, sample.sample_id))

    target = sample.target
    if sample.task_type == "clarify_or_respond" or sample.verifier.mode == "clarify" or target.decision:
        decision = target.decision or "respond"
        answer = target.answer_text or (target.acceptable_answers[0] if target.acceptable_answers else "")
        if decision == "clarify":
            return "Decision: clarify\nClarifying question: %s" % answer
        return "Decision: respond\nAnswer: %s" % answer
    if sample.task_type == "consultation" or sample.verifier.mode == "structured" or target.structured:
        return _structured_target_to_text(sample)
    if sample.task_type == "classification" or sample.verifier.mode == "label" or target.canonical_label:
        label = target.canonical_label or target.answer_text or _plain_target_to_text(sample)
        choice_answer = _classification_choice_answer(sample)
        if choice_answer:
            return "Choice: %s\nAnswer: %s\nEvidence: %s" % (
                choice_answer["letter"],
                choice_answer["label"],
                _classification_evidence_to_text(sample),
            )
        return "Answer: %s\nEvidence: %s" % (label, _classification_evidence_to_text(sample))
    if sample.task_type == "vqa" or sample.verifier.mode in {"exact_match", "synonym"}:
        return "Answer: %s" % _plain_target_to_text(sample)
    return _plain_target_to_text(sample)


def _first_nonempty_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = _first_nonempty_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("text", "evidence", "description", "value"):
            text = _first_nonempty_text(value.get(key))
            if text:
                return text
        return ""
    text = str(value or "").strip()
    return text


def _classification_evidence_to_text(sample: UnifiedSample) -> str:
    metadata = sample.metadata or {}
    verifier = sample.verifier.model_dump(mode="json", exclude_none=True) if sample.verifier else {}
    for container in (metadata, verifier):
        for field_name in ("visual_evidence", "known_facts", "symptoms"):
            evidence = _first_nonempty_text(container.get(field_name))
            if evidence:
                return evidence
    crop = str(metadata.get("crop") or sample.verifier.crop or "").strip()
    disease = str(metadata.get("disease") or sample.verifier.disease or "").strip()
    if crop and disease:
        return "Visible %s symptoms support the %s label." % (crop, disease)
    return "Visible agricultural symptoms or pest features support this label."


def _answer_text_field(sample: UnifiedSample, field_name: str) -> str:
    prefix = "%s:" % field_name.lower()
    for line in (sample.target.answer_text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped.split(":", 1)[1].strip()
    return ""


def _structured_target_to_text(sample: UnifiedSample) -> str:
    target = sample.target
    structured = target.structured or {}
    diagnosis = (
        str(structured.get("diagnosis") or "").strip()
        or str(target.canonical_label or "").strip()
        or _answer_text_field(sample, "Diagnosis")
        or _plain_target_to_text(sample)
    )
    symptoms = _answer_text_field(sample, "Symptoms")
    management_value = structured.get("management_steps") or _answer_text_field(sample, "Management")
    if isinstance(management_value, list):
        management = " ".join(str(step).strip() for step in management_value if str(step).strip())
    else:
        management = str(management_value or "").strip()
    if not management:
        management = "No source-specific management step was provided; recommend local extension follow-up."
    evidence = symptoms or "Visible symptoms should be checked against the image and crop context."
    return "\n".join(
        [
            "Diagnosis: %s" % diagnosis,
            "Evidence: %s" % evidence,
            "Uncertainty: Image-only assessment; confirm with field context before treatment.",
            "Management: %s" % management,
            "Follow-up: Share close-up images and crop history if symptoms progress or the diagnosis is uncertain.",
        ]
    )


def sample_to_training_messages(
    sample: UnifiedSample,
    *,
    prompt_format: str = MANIFEST_PROMPT_FORMAT,
    target_format: str = PLAIN_FORMAT,
) -> List[Dict[str, Any]]:
    messages = sample_to_prompt_messages(sample, prompt_format=prompt_format)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_to_text(sample, target_format=target_format)}],
        }
    )
    return messages


def strip_assistant_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    return [message.model_dump(mode="json", exclude_none=True) for message in messages if message.role != "assistant"]
