from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "vlm_baselines"))

from run_baselines import _generation_config, _resolved_quantization  # noqa: E402


def test_resolved_quantization_uses_model_config_when_cli_unset():
    args = SimpleNamespace(quantization=None)

    assert _resolved_quantization(args, {"quantization": "4bit"}) == "4bit"


def test_resolved_quantization_cli_override_wins():
    args = SimpleNamespace(quantization="none")

    assert _resolved_quantization(args, {"quantization": "4bit"}) == "none"


def test_resolved_quantization_defaults_to_none_without_config_value():
    args = SimpleNamespace(quantization=None)

    assert _resolved_quantization(args, {}) == "none"


def test_resolved_quantization_rejects_invalid_config_value():
    args = SimpleNamespace(quantization=None)

    with pytest.raises(ValueError, match="Unsupported quantization"):
        _resolved_quantization(args, {"quantization": "8bit"})


def test_generation_config_adds_constrained_classification_mode_only_when_requested():
    sample = SimpleNamespace(task_type="classification", verifier_mode="label")
    vqa_sample = SimpleNamespace(task_type="vqa", verifier_mode="exact_match")
    base_args = {
        "max_new_tokens": 0,
        "min_new_tokens": 0,
        "batch_size": 1,
        "seed": 42,
    }

    free_config = _generation_config(SimpleNamespace(**base_args, classification_decode_mode="free"), sample)
    constrained_config = _generation_config(
        SimpleNamespace(**base_args, classification_decode_mode="constrained"),
        sample,
    )

    assert "classification_decode_mode" not in free_config
    assert constrained_config["classification_decode_mode"] == "constrained"
    assert "classification_decode_mode" not in _generation_config(
        SimpleNamespace(**base_args, classification_decode_mode="constrained"),
        vqa_sample,
    )
