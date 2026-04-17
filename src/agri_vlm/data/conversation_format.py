"""Convert normalized samples into model-facing chat representations."""

import json
from typing import Any, Dict, List

from agri_vlm.schemas.dataset_schema import Message, UnifiedSample


def target_to_text(sample: UnifiedSample) -> str:
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


def sample_to_prompt_messages(sample: UnifiedSample) -> List[Dict[str, Any]]:
    return [message.model_dump(mode="json", exclude_none=True) for message in sample.messages]


def sample_to_training_messages(sample: UnifiedSample) -> List[Dict[str, Any]]:
    messages = sample_to_prompt_messages(sample)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_to_text(sample)}],
        }
    )
    return messages


def strip_assistant_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    return [message.model_dump(mode="json", exclude_none=True) for message in messages if message.role != "assistant"]
