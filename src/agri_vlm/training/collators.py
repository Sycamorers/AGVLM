"""Data collators for multimodal SFT."""

from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.conversation_format import sample_to_training_messages
from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.image import open_image


class QwenVLChatCollator:
    """Tokenize multimodal chat samples for causal LM training."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        samples = [UnifiedSample.model_validate(feature) for feature in features]
        texts = []
        image_batches = []
        for sample in samples:
            messages = sample_to_training_messages(sample)
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(rendered)
            image_batches.append([open_image(Path(path)) for path in sample.images])

        batch = self.processor(
            text=texts,
            images=image_batches,
            padding=True,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        labels = batch["input_ids"].clone()
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch["labels"] = labels
        return batch
