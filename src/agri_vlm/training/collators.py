"""Data collators for multimodal SFT."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.conversation_format import (
    MANIFEST_PROMPT_FORMAT,
    PLAIN_FORMAT,
    sample_to_prompt_messages,
    sample_to_training_messages,
    target_to_text,
)
from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.image import open_image

LOGGER = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


class VisionLanguageChatCollator:
    """Tokenize multimodal chat samples for causal LM training."""

    def __init__(
        self,
        processor: Any,
        *,
        prompt_format: str = MANIFEST_PROMPT_FORMAT,
        target_format: str = PLAIN_FORMAT,
    ) -> None:
        self.processor = processor
        self.prompt_format = prompt_format
        self.target_format = target_format
        self.log_batches = _env_flag("AGRI_VLM_LOG_COLLATOR_BATCHES")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        samples = [UnifiedSample.model_validate(feature) for feature in features]
        if self.log_batches:
            LOGGER.info(
                "SFT collator start rank=%s local_rank=%s sample_ids=%s images=%s",
                os.environ.get("RANK", "0"),
                os.environ.get("LOCAL_RANK", "0"),
                [sample.sample_id for sample in samples],
                [sample.images for sample in samples],
            )
        prompt_texts = []
        texts = []
        image_batches = []
        for sample in samples:
            prompt = self.processor.apply_chat_template(
                sample_to_prompt_messages(sample, prompt_format=self.prompt_format),
                tokenize=False,
                add_generation_prompt=True,
            )
            messages = sample_to_training_messages(
                sample,
                prompt_format=self.prompt_format,
                target_format=self.target_format,
            )
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_texts.append(prompt)
            texts.append(rendered)
            image_batches.append([open_image(Path(path)) for path in sample.images])

        if self.log_batches:
            LOGGER.info(
                "SFT collator images loaded rank=%s local_rank=%s sample_ids=%s",
                os.environ.get("RANK", "0"),
                os.environ.get("LOCAL_RANK", "0"),
                [sample.sample_id for sample in samples],
            )
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
        _mask_prompt_and_padding_tokens(
            batch=batch,
            prompt_batch=prompt_batch,
            processor=self.processor,
        )
        if self.log_batches:
            LOGGER.info(
                "SFT collator done rank=%s local_rank=%s sample_ids=%s input_shape=%s",
                os.environ.get("RANK", "0"),
                os.environ.get("LOCAL_RANK", "0"),
                [sample.sample_id for sample in samples],
                tuple(batch["input_ids"].shape),
            )
        return batch


def _attention_sequence_start(mask_row: Any, *, default_start: int) -> int:
    if mask_row is None:
        return default_start
    positions = mask_row.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return default_start
    return int(positions[0].item())


def _mask_prompt_and_padding_tokens(
    *,
    batch: Dict[str, Any],
    prompt_batch: Dict[str, Any],
    processor: Any,
) -> None:
    labels = batch["input_ids"].clone()
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    prompt_attention_mask = prompt_batch.get("attention_mask")
    attention_mask = batch.get("attention_mask")
    padding_side = getattr(processor.tokenizer, "padding_side", "right")
    for row_index in range(labels.shape[0]):
        if prompt_attention_mask is not None:
            prompt_length = int(prompt_attention_mask[row_index].sum().item())
        elif pad_token_id is not None:
            prompt_ids = prompt_batch["input_ids"][row_index]
            prompt_length = int(prompt_ids.ne(pad_token_id).sum().item())
        else:
            prompt_length = int(prompt_batch["input_ids"].shape[1])

        if attention_mask is not None:
            sequence_length = int(attention_mask[row_index].sum().item())
        elif pad_token_id is not None:
            sequence_length = int(batch["input_ids"][row_index].ne(pad_token_id).sum().item())
        else:
            sequence_length = int(batch["input_ids"].shape[1])

        default_start = labels.shape[1] - sequence_length if padding_side == "left" else 0
        mask_row = attention_mask[row_index] if attention_mask is not None else None
        sequence_start = _attention_sequence_start(mask_row, default_start=default_start)
        prompt_end = min(sequence_start + prompt_length, labels.shape[1])
        labels[row_index, sequence_start:prompt_end] = -100
    batch["labels"] = labels


def _render_phi4_multimodal_messages(
    sample: UnifiedSample,
    *,
    include_target: bool,
    prompt_format: str = MANIFEST_PROMPT_FORMAT,
    target_format: str = PLAIN_FORMAT,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    image_index = 1
    for message in sample_to_prompt_messages(sample, prompt_format=prompt_format):
        parts = []
        for content in message.get("content") or []:
            if content.get("type") == "image":
                parts.append("<|image_%s|>" % image_index)
                image_index += 1
            elif content.get("text"):
                parts.append(content["text"])
        messages.append({"role": message.get("role", "user"), "content": "".join(parts)})
    if include_target:
        messages.append({"role": "assistant", "content": target_to_text(sample, target_format=target_format)})
    return messages


def _render_phi4_reasoning_vision_messages(
    sample: UnifiedSample,
    *,
    include_target: bool,
    prompt_format: str = MANIFEST_PROMPT_FORMAT,
    target_format: str = PLAIN_FORMAT,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for message in sample_to_prompt_messages(sample, prompt_format=prompt_format):
        parts = []
        for content in message.get("content") or []:
            if content.get("type") == "image":
                parts.append("<image>")
            elif content.get("text"):
                parts.append(content["text"])
        messages.append({"role": message.get("role", "user"), "content": "".join(parts)})
    if include_target:
        messages.append({"role": "assistant", "content": target_to_text(sample, target_format=target_format)})
    return messages


class Phi4MultimodalVisionCollator(VisionLanguageChatCollator):
    """Tokenize Phi-4 multimodal vision batches for causal LM training."""

    def _render_messages(self, sample: UnifiedSample, *, include_target: bool) -> List[Dict[str, str]]:
        return _render_phi4_multimodal_messages(
            sample,
            include_target=include_target,
            prompt_format=self.prompt_format,
            target_format=self.target_format,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        samples = [UnifiedSample.model_validate(feature) for feature in features]
        prompt_texts = []
        texts = []
        images = []
        for sample in samples:
            prompt_texts.append(
                self.processor.tokenizer.apply_chat_template(
                    self._render_messages(sample, include_target=False),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            texts.append(
                self.processor.tokenizer.apply_chat_template(
                    self._render_messages(sample, include_target=True),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            images.extend(open_image(Path(path)) for path in sample.images)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        _mask_prompt_and_padding_tokens(
            batch=batch,
            prompt_batch=prompt_batch,
            processor=self.processor,
        )
        if self.log_batches:
            LOGGER.info(
                "SFT collator done rank=%s local_rank=%s sample_ids=%s input_shape=%s",
                os.environ.get("RANK", "0"),
                os.environ.get("LOCAL_RANK", "0"),
                [sample.sample_id for sample in samples],
                tuple(batch["input_ids"].shape),
            )
        return batch


class Phi4ReasoningVisionCollator(Phi4MultimodalVisionCollator):
    """Tokenize Phi-4 reasoning vision batches for causal LM training."""

    def _render_messages(self, sample: UnifiedSample, *, include_target: bool) -> List[Dict[str, str]]:
        return _render_phi4_reasoning_vision_messages(
            sample,
            include_target=include_target,
            prompt_format=self.prompt_format,
            target_format=self.target_format,
        )


def build_sft_data_collator(model_config: Any, processor: Any, train_config: Any = None) -> Any:
    model_name = "%s %s" % (
        getattr(model_config, "name", ""),
        getattr(model_config, "model_name_or_path", ""),
    )
    lower_name = model_name.lower()
    prompt_format = getattr(train_config, "sft_prompt_format", MANIFEST_PROMPT_FORMAT)
    target_format = getattr(train_config, "sft_target_format", PLAIN_FORMAT)
    if "phi-4-reasoning-vision" in lower_name or "phi4_reasoning_vision" in lower_name:
        return Phi4ReasoningVisionCollator(
            processor=processor,
            prompt_format=prompt_format,
            target_format=target_format,
        )
    if "phi-4-multimodal" in lower_name or "phi4_multimodal" in lower_name:
        return Phi4MultimodalVisionCollator(
            processor=processor,
            prompt_format=prompt_format,
            target_format=target_format,
        )
    return VisionLanguageChatCollator(
        processor=processor,
        prompt_format=prompt_format,
        target_format=target_format,
    )
