"""Model-loading utilities for Qwen vision-language checkpoints."""

from typing import Any, Dict


def torch_dtype_from_name(dtype_name: str) -> Any:
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError("Unsupported torch dtype: %s" % dtype_name)
    return mapping[dtype_name]


def build_model_init_kwargs(model_config: Any) -> Dict[str, Any]:
    """Create a common kwargs payload for `from_pretrained`."""
    kwargs: Dict[str, Any] = {
        "trust_remote_code": model_config.trust_remote_code,
        "device_map": model_config.device_map,
    }
    if model_config.attn_implementation:
        kwargs["attn_implementation"] = model_config.attn_implementation
    dtype = torch_dtype_from_name(model_config.torch_dtype)
    kwargs["torch_dtype"] = dtype

    if model_config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch_dtype_from_name(model_config.bnb_4bit_compute_dtype),
        )
    return kwargs


def load_model(model_name_or_path: str, model_config: Any) -> Any:
    """Load the configured vision-language model."""
    kwargs = build_model_init_kwargs(model_config)
    try:
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    except ImportError:  # pragma: no cover - depends on transformers install
        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            model_cls = AutoModelForVision2Seq

    model = model_cls.from_pretrained(model_name_or_path, **kwargs)
    if model_config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model
