import pytest

from agri_vlm.utils.checkpointing import find_latest_checkpoint, resolve_resume_checkpoint, validate_peft_adapter_checkpoint


def _write_standard_checkpoint(path) -> None:
    path.mkdir()
    (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    (path / "training_args.bin").write_bytes(b"args")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "scheduler.pt").write_bytes(b"scheduler")


def _write_deepspeed_checkpoint(path, tag: str) -> None:
    path.mkdir()
    (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    (path / "training_args.bin").write_bytes(b"args")
    (path / "latest").write_text(tag, encoding="utf-8")
    tag_dir = path / tag
    tag_dir.mkdir()
    (tag_dir / "zero_pp_rank_0_mp_rank_00_model_states.pt").write_bytes(b"model")
    (tag_dir / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"optimizer")


def test_find_latest_checkpoint_uses_numeric_step_order(tmp_path) -> None:
    _write_standard_checkpoint(tmp_path / "checkpoint-900")
    _write_standard_checkpoint(tmp_path / "checkpoint-1000")
    _write_standard_checkpoint(tmp_path / "checkpoint-1200")

    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-1200"


def test_find_latest_checkpoint_ignores_non_numeric_checkpoint_dirs(tmp_path) -> None:
    (tmp_path / "checkpoint-final").mkdir()
    _write_standard_checkpoint(tmp_path / "checkpoint-42")

    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-42"


def test_find_latest_checkpoint_skips_incomplete_higher_step(tmp_path) -> None:
    _write_standard_checkpoint(tmp_path / "checkpoint-1200")
    (tmp_path / "checkpoint-1300").mkdir()

    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-1200"


def test_find_latest_checkpoint_accepts_deepspeed_checkpoint(tmp_path) -> None:
    _write_deepspeed_checkpoint(tmp_path / "checkpoint-1300", "global_step1300")

    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-1300"


def test_resolve_resume_checkpoint_auto_uses_numeric_latest(tmp_path) -> None:
    _write_standard_checkpoint(tmp_path / "checkpoint-900")
    _write_standard_checkpoint(tmp_path / "checkpoint-1100")

    assert resolve_resume_checkpoint(tmp_path, "auto") == tmp_path / "checkpoint-1100"


def test_validate_peft_adapter_checkpoint_rejects_empty_tensor_file(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        validate_peft_adapter_checkpoint(adapter_dir)


def test_validate_peft_adapter_checkpoint_accepts_lora_tensors(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    safetensors_torch.save_file(
        {"base_model.model.layers.0.self_attn.qkv_proj.lora_A.weight": torch.zeros(2, 2)},
        str(adapter_dir / "adapter_model.safetensors"),
    )

    summary = validate_peft_adapter_checkpoint(adapter_dir)

    assert summary["num_tensors"] == 1
    assert summary["non_empty_tensors"] == 1
