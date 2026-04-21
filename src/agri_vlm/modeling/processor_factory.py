"""Processor loading helpers."""

from pathlib import Path
from typing import Any, Optional


def load_processor(model_config: Any, checkpoint_path: Optional[str] = None) -> Any:
    """Load the processor from pretrained weights."""
    from transformers import AutoProcessor

    processor_name = model_config.processor_name_or_path or model_config.model_name_or_path
    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path)
        processor_files = [
            "processor_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ]
        if any((checkpoint_dir / name).exists() for name in processor_files):
            processor_name = str(checkpoint_dir)
    processor_kwargs = {"trust_remote_code": model_config.trust_remote_code}
    if model_config.min_pixels is not None:
        processor_kwargs["min_pixels"] = model_config.min_pixels
    if model_config.max_pixels is not None:
        processor_kwargs["max_pixels"] = model_config.max_pixels
    processor = AutoProcessor.from_pretrained(processor_name, **processor_kwargs)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor
