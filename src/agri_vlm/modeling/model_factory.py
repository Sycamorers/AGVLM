"""Model-loading utilities for Qwen vision-language checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from agri_vlm.utils.distributed import DistributedContext


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


def _resolve_device_map(
    model_config: Any,
    distributed_context: Optional[DistributedContext],
) -> Any:
    if distributed_context and distributed_context.is_distributed:
        if not model_config.load_in_4bit:
            return None
        if model_config.distributed_device_map == "local_process":
            return {"": distributed_context.local_rank}
        return None
    return model_config.device_map


def build_model_init_kwargs(
    model_config: Any,
    distributed_context: Optional[DistributedContext] = None,
) -> Dict[str, Any]:
    """Create a common kwargs payload for `from_pretrained`."""
    kwargs: Dict[str, Any] = {
        "trust_remote_code": model_config.trust_remote_code,
        "low_cpu_mem_usage": model_config.low_cpu_mem_usage,
    }
    device_map = _resolve_device_map(model_config, distributed_context=distributed_context)
    if device_map is not None:
        kwargs["device_map"] = device_map
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


def load_model(
    model_name_or_path: str,
    model_config: Any,
    distributed_context: Optional[DistributedContext] = None,
) -> Any:
    """Load the configured vision-language model."""
    kwargs = build_model_init_kwargs(model_config, distributed_context=distributed_context)
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
    if hasattr(model, "config"):
        model.config.use_cache = model_config.use_cache
    return model


def load_sft_checkpoint_model(
    model_config: Any,
    checkpoint_path: str,
    distributed_context: Optional[DistributedContext] = None,
) -> Any:
    """Load a trainable model from an SFT checkpoint path."""
    checkpoint_dir = Path(checkpoint_path)
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        return load_model(
            model_name_or_path=checkpoint_path,
            model_config=model_config,
            distributed_context=distributed_context,
        )

    from peft import PeftModel

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model_name_or_path = adapter_config.get("base_model_name_or_path") or model_config.model_name_or_path
    model = load_model(
        model_name_or_path=base_model_name_or_path,
        model_config=model_config,
        distributed_context=distributed_context,
    )
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
    if hasattr(model, "config"):
        model.config.use_cache = model_config.use_cache
    return model
