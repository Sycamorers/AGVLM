from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "vlm_baselines"))

from checkpoint_config import resolve_model_entry, validate_model_entry  # noqa: E402


def test_placeholder_checkpoint_path_fails_when_selected():
    entry = {
        "model_key": "agvlm_phi4_sft_completed",
        "model_name": "agvlm_phi4_sft_completed",
        "adapter_type": "agvlm_sft",
        "checkpoint_type": "sft",
        "base_model_name_or_path": "microsoft/Phi-4-reasoning-vision-15B",
        "checkpoint_path": "CHANGE_ME",
        "adapter_path": "",
    }
    result = validate_model_entry(entry, phase="sft", require_runnable=True)
    assert not result.ok
    assert "placeholder" in result.errors[0]


def test_completed_checkpoint_path_passes(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    entry = {
        "model_key": "agvlm_phi4_sft_completed",
        "model_name": "agvlm_phi4_sft_completed",
        "adapter_type": "merged_checkpoint",
        "checkpoint_type": "sft",
        "base_model_name_or_path": "microsoft/Phi-4-reasoning-vision-15B",
        "checkpoint_path": str(checkpoint_dir),
        "adapter_path": "",
    }
    result = validate_model_entry(entry, phase="sft", require_runnable=True)
    assert result.ok


def test_raw_base_model_is_not_accepted_as_rl_checkpoint(tmp_path):
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    entry = {
        "model_key": "agvlm_phi4_rl_completed",
        "model_name": "agvlm_phi4_rl_completed",
        "adapter_type": "agvlm_rl",
        "checkpoint_type": "rl",
        "base_model_name_or_path": "microsoft/Phi-4-reasoning-vision-15B",
        "checkpoint_path": "microsoft/Phi-4-reasoning-vision-15B",
        "adapter_path": "",
        "initialized_from_sft_checkpoint": str(sft_dir),
    }
    result = validate_model_entry(entry, phase="rl", require_runnable=True)
    assert not result.ok
    assert any("raw microsoft/Phi-4-reasoning-vision-15B" in error for error in result.errors)


def test_resolve_external_model_from_config(tmp_path):
    config = tmp_path / "baseline.yaml"
    config.write_text("models:\n  - name: HuggingFaceTB/SmolVLM2-2.2B-Instruct\n", encoding="utf-8")
    entry = resolve_model_entry(
        model_key=None,
        model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        model_config_path=config,
        checkpoint_config_path=None,
    )
    assert entry["checkpoint_type"] == "external_baseline"
    assert entry["base_model_name_or_path"] == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
