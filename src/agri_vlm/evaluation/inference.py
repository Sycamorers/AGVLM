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


def generate_predictions(
    samples: Iterable[Any],
    model_config: Any,
    max_new_tokens: int,
    batch_size: int = 1,
    checkpoint_path: Optional[str] = None,
) -> List[str]:
    """Run local generation for a list of normalized samples."""
    rows = list(samples)
    processor = load_processor(model_config, checkpoint_path=checkpoint_path)
    model = load_inference_model(model_config=model_config, checkpoint_path=checkpoint_path)
    predictions = []
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
        batch = batch.to(model.device)
        output_ids = model.generate(**batch, max_new_tokens=max_new_tokens)
        prompt_lengths = batch["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            decoded = processor.batch_decode(
                [output_ids[row_index, int(prompt_length) :]],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            predictions.append(decoded.strip())
    return predictions


def oracle_predictions(samples: Iterable[Any]) -> List[str]:
    return [target_to_text(sample) for sample in samples]
