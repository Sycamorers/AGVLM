"""Data collators for multimodal SFT."""

from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.conversation_format import sample_to_prompt_messages, sample_to_training_messages
from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.image import open_image


def _assistant_suffix_start(full_input_ids: List[int], assistant_input_ids: List[int]) -> int:
    """Return the start index of the assistant suffix inside a tokenized chat sample."""
    if not assistant_input_ids:
        raise ValueError("Assistant token suffix cannot be empty.")
    if len(assistant_input_ids) > len(full_input_ids):
        raise ValueError("Assistant token suffix is longer than the tokenized sample.")
    start_index = len(full_input_ids) - len(assistant_input_ids)
    if full_input_ids[start_index:] != assistant_input_ids:
        raise ValueError("Assistant token suffix is not aligned with the tokenized sample.")
    return start_index


class QwenVLChatCollator:
    """Tokenize multimodal chat samples for causal LM training."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        samples = [UnifiedSample.model_validate(feature) for feature in features]
        texts = []
        image_batches = []
        assistant_suffixes = []
        for sample in samples:
            prompt_messages = sample_to_prompt_messages(sample)
            messages = sample_to_training_messages(sample)
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(rendered)
            image_batches.append([open_image(Path(path)) for path in sample.images])
            assistant_suffixes.append(
                self.processor.tokenizer.encode_message_with_chat_template(
                    messages[-1],
                    conversation_history=prompt_messages,
                )
            )

        batch = self.processor(
            text=texts,
            images=image_batches,
            padding=True,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        labels = batch["input_ids"].clone()
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            raise ValueError("Processor output is missing attention_mask required for assistant-only loss masking.")
        for index, assistant_suffix in enumerate(assistant_suffixes):
            sequence_length = int(attention_mask[index].sum().item())
            assistant_start = _assistant_suffix_start(
                labels[index, :sequence_length].tolist(),
                assistant_suffix,
            )
            labels[index, :assistant_start] = -100
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch["labels"] = labels
        return batch
