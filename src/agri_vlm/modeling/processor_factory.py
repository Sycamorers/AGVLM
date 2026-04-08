"""Processor loading helpers."""

from typing import Any


def load_processor(model_config: Any) -> Any:
    """Load the processor from pretrained weights."""
    from transformers import AutoProcessor

    processor_name = model_config.processor_name_or_path or model_config.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=model_config.trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor
