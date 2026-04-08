"""Module-freezing helpers."""

from typing import Any, Dict


VISION_KEYWORDS = ("vision", "visual", "image_tower", "vision_tower")
PROJECTOR_KEYWORDS = ("projector", "multi_modal_projector", "visual_projection", "mm_projector")


def apply_freezing(model: Any, freeze_config: Any) -> Dict[str, int]:
    """Freeze selected model modules by name."""
    total = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        total += 1
        should_freeze = False
        if freeze_config.freeze_vision_encoder and any(keyword in name for keyword in VISION_KEYWORDS):
            should_freeze = True
        if freeze_config.freeze_projector and any(keyword in name for keyword in PROJECTOR_KEYWORDS):
            should_freeze = True
        if should_freeze:
            parameter.requires_grad = False
            frozen += 1
    return {"total_parameters": total, "frozen_parameter_tensors": frozen}
