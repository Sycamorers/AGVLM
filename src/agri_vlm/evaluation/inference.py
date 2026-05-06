"""Model inference helpers for evaluation."""

from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

from agri_vlm.data.conversation_format import sample_to_prompt_messages, target_to_text
from agri_vlm.modeling.model_factory import load_inference_model
from agri_vlm.modeling.processor_factory import load_processor
from agri_vlm.utils.image import open_image


def _batched(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    for start in range(0, len(items), max(batch_size, 1)):
        yield items[start : start + max(batch_size, 1)]


def _resolve_generation_device(model: Any, fallback_device: Optional[Any] = None) -> Optional[Any]:
    if fallback_device is not None:
        return fallback_device
    device = getattr(model, "device", None)
    if device is not None:
        return device
    module = getattr(model, "module", None)
    if module is not None:
        device = getattr(module, "device", None)
        if device is not None:
            return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def generate_predictions_from_loaded_model(
    samples: Iterable[Any],
    model: Any,
    processor: Any,
    max_new_tokens: int,
    batch_size: int = 1,
    device: Optional[Any] = None,
    synced_gpus: bool = False,
) -> List[str]:
    """Run generation for a list of normalized samples with an already loaded model."""
    rows = list(samples)
    predictions = []
    generation_model = model if hasattr(model, "generate") else getattr(model, "module", model)
    generation_device = _resolve_generation_device(generation_model, fallback_device=device)
    was_training = bool(getattr(generation_model, "training", False))
    model_config = getattr(generation_model, "config", None)
    generation_config = getattr(generation_model, "generation_config", None)
    original_model_use_cache = getattr(model_config, "use_cache", None)
    original_generation_use_cache = getattr(generation_config, "use_cache", None)
    try:
        import torch

        generation_model.eval()
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = True
        if generation_config is not None and hasattr(generation_config, "use_cache"):
            generation_config.use_cache = True
        with torch.no_grad():
            for batch_rows in _batched(rows, batch_size=batch_size):
                prompts = [
                    processor.apply_chat_template(
                        sample_to_prompt_messages(sample),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for sample in batch_rows
                ]
                image_batch = [[open_image(Path(path)) for path in sample.images] for sample in batch_rows]
                batch = processor(text=prompts, images=image_batch, padding=True, return_tensors="pt")
                batch.pop("token_type_ids", None)
                if generation_device is not None:
                    batch = batch.to(generation_device)
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "use_cache": True,
                }
                if synced_gpus:
                    generation_kwargs["synced_gpus"] = True
                output_ids = generation_model.generate(**batch, **generation_kwargs)
                prompt_length = int(batch["input_ids"].shape[1])
                for row_index in range(len(batch_rows)):
                    decoded = processor.batch_decode(
                        [output_ids[row_index, prompt_length:]],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    predictions.append(decoded.strip())
    finally:
        if model_config is not None and original_model_use_cache is not None:
            model_config.use_cache = original_model_use_cache
        if generation_config is not None and original_generation_use_cache is not None:
            generation_config.use_cache = original_generation_use_cache
        if was_training:
            generation_model.train()
    return predictions


def generate_predictions(
    samples: Iterable[Any],
    model_config: Any,
    max_new_tokens: int,
    batch_size: int = 1,
    checkpoint_path: Optional[str] = None,
) -> List[str]:
    """Run local generation for a list of normalized samples."""
    processor = load_processor(model_config, checkpoint_path=checkpoint_path)
    model = load_inference_model(model_config=model_config, checkpoint_path=checkpoint_path)
    return generate_predictions_from_loaded_model(
        samples,
        model=model,
        processor=processor,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )


def oracle_predictions(samples: Iterable[Any]) -> List[str]:
    return [target_to_text(sample) for sample in samples]
