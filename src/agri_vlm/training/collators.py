"""Data collators for multimodal SFT."""

from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.conversation_format import sample_to_prompt_messages, sample_to_training_messages
from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.image import open_image


class VisionLanguageChatCollator:
    """Tokenize multimodal chat samples for causal LM training."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        samples = [UnifiedSample.model_validate(feature) for feature in features]
        prompt_texts = []
        texts = []
        image_batches = []
        for sample in samples:
            prompt = self.processor.apply_chat_template(
                sample_to_prompt_messages(sample),
                tokenize=False,
                add_generation_prompt=True,
            )
            messages = sample_to_training_messages(sample)
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_texts.append(prompt)
            texts.append(rendered)
            image_batches.append([open_image(Path(path)) for path in sample.images])

        batch = self.processor(
            text=texts,
            images=image_batches,
            padding=True,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=image_batches,
            padding=True,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        labels = batch["input_ids"].clone()
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        prompt_attention_mask = prompt_batch.get("attention_mask")
        attention_mask = batch.get("attention_mask")
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")
        for row_index in range(labels.shape[0]):
            if prompt_attention_mask is not None:
                prompt_length = int(prompt_attention_mask[row_index].sum().item())
            elif pad_token_id is not None:
                prompt_length = int(prompt_batch["input_ids"][row_index].ne(pad_token_id).sum().item())
            else:
                prompt_length = int(prompt_batch["input_ids"].shape[1])

            if attention_mask is not None:
                sequence_length = int(attention_mask[row_index].sum().item())
            elif pad_token_id is not None:
                sequence_length = int(batch["input_ids"][row_index].ne(pad_token_id).sum().item())
            else:
                sequence_length = int(batch["input_ids"].shape[1])

            sequence_start = labels.shape[1] - sequence_length if padding_side == "left" else 0
            prompt_end = min(sequence_start + prompt_length, labels.shape[1])
            labels[row_index, sequence_start:prompt_end] = -100
        batch["labels"] = labels
        return batch


def build_sft_data_collator(model_config: Any, processor: Any) -> Any:
    _ = model_config
    return VisionLanguageChatCollator(processor=processor)


QwenVLChatCollator = VisionLanguageChatCollator
