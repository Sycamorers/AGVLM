"""Model inference helpers for evaluation."""

from pathlib import Path
from typing import Any, Iterable, List

from agri_vlm.data.conversation_format import sample_to_prompt_messages, target_to_text
from agri_vlm.modeling.model_factory import load_model
from agri_vlm.modeling.processor_factory import load_processor
from agri_vlm.utils.image import open_image


def generate_predictions(samples: Iterable[Any], model_config: Any, max_new_tokens: int) -> List[str]:
    """Run local generation for a list of normalized samples."""
    processor = load_processor(model_config)
    model = load_model(model_config.model_name_or_path, model_config=model_config)
    predictions = []
    for sample in samples:
        prompt = processor.apply_chat_template(
            sample_to_prompt_messages(sample),
            tokenize=False,
            add_generation_prompt=True,
        )
        image_batch = [[open_image(Path(path)) for path in sample.images]]
        batch = processor(text=[prompt], images=image_batch, padding=True, return_tensors="pt")
        batch.pop("token_type_ids", None)
        batch = batch.to(model.device)
        output_ids = model.generate(**batch, max_new_tokens=max_new_tokens)
        prompt_length = batch["input_ids"].shape[1]
        decoded = processor.batch_decode(
            output_ids[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        predictions.append(decoded.strip())
    return predictions


def oracle_predictions(samples: Iterable[Any]) -> List[str]:
    return [target_to_text(sample) for sample in samples]
