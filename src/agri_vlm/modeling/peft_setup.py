"""PEFT / LoRA setup helpers."""

from typing import Any


def build_lora_config(train_config: Any) -> Any:
    """Build a LoRA config from train settings."""
    from peft import LoraConfig, TaskType

    lora = train_config.lora
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        target_modules=lora.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )


def maybe_wrap_with_peft(model: Any, train_config: Any) -> Any:
    """Wrap a model with LoRA if requested."""
    if not train_config.use_peft:
        return model
    from peft import get_peft_model, prepare_model_for_kbit_training

    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_config.gradient_checkpointing,
        )

    return get_peft_model(model, build_lora_config(train_config))
