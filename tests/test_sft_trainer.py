import sys
import types

import pytest

from agri_vlm.schemas.config_schema import TrainConfigSchema


def test_chunked_causal_lm_loss_matches_torch_cross_entropy() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from agri_vlm.training.sft_trainer import _chunked_causal_lm_loss

    logits = torch.randn(2, 7, 11, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor(
        [
            [1, 2, 3, -100, 5, 6, 7],
            [0, 4, -100, 8, 9, 10, 1],
        ]
    )

    actual = _chunked_causal_lm_loss(logits, labels, chunk_size=2)
    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )

    assert torch.allclose(actual, expected)


def test_chunked_causal_lm_loss_skips_all_ignored_chunks(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from agri_vlm.training.sft_trainer import _chunked_causal_lm_loss

    calls = []
    real_cross_entropy = F.cross_entropy

    def fake_cross_entropy(input, target, **kwargs):
        calls.append(target.detach().clone())
        return real_cross_entropy(input, target, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", fake_cross_entropy)
    logits = torch.randn(1, 6, 7, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, -100, -100, 4]])

    _chunked_causal_lm_loss(logits, labels, chunk_size=2)

    assert len(calls) == 1
    assert calls[0].tolist() == [4]


def test_train_config_accepts_loss_chunk_size() -> None:
    config = TrainConfigSchema.model_validate(
        {
            "manifest_path": "data/manifests/partial_10pct/sft_manifest.jsonl",
            "output_dir": "outputs/smoke/sft-qwen3-vl-4b",
            "loss_chunk_size": 1024,
        }
    )

    assert config.loss_chunk_size == 1024


def test_train_config_accepts_max_images_per_sample() -> None:
    config = TrainConfigSchema.model_validate(
        {
            "manifest_path": "data/manifests/partial_10pct/sft_manifest.jsonl",
            "output_dir": "outputs/smoke/sft-qwen3-vl-4b",
            "max_images_per_sample": 5,
        }
    )

    assert config.max_images_per_sample == 5


def test_train_config_accepts_deepspeed_and_max_steps() -> None:
    config = TrainConfigSchema.model_validate(
        {
            "manifest_path": "data/manifests/partial_10pct/sft_manifest.jsonl",
            "output_dir": "outputs/smoke/sft-qwen3-vl-4b",
            "deepspeed": "configs/deepspeed/zero3_qlora_turin_24g.json",
            "max_steps": 1,
        }
    )

    assert config.deepspeed == "configs/deepspeed/zero3_qlora_turin_24g.json"
    assert config.max_steps == 1


def test_train_config_accepts_save_strategy() -> None:
    config = TrainConfigSchema.model_validate(
        {
            "manifest_path": "data/manifests/partial_10pct/sft_manifest.jsonl",
            "output_dir": "outputs/smoke/sft-qwen3-vl-4b",
            "save_strategy": "no",
        }
    )

    assert config.save_strategy == "no"


def test_train_config_accepts_eval_loss_and_overlap_guard_options() -> None:
    config = TrainConfigSchema.model_validate(
        {
            "manifest_path": "data/manifests/partial_10pct/sft_manifest.jsonl",
            "output_dir": "outputs/smoke/sft-qwen3-vl-4b",
            "prediction_loss_only": True,
            "fail_on_train_eval_overlap": True,
        }
    )

    assert config.prediction_loss_only is True
    assert config.fail_on_train_eval_overlap is True


def test_train_eval_overlap_guard_rejects_group_overlap() -> None:
    from agri_vlm.schemas.dataset_schema import UnifiedSample
    from agri_vlm.training.sft_trainer import _assert_no_train_eval_overlap

    payload = {
        "sample_id": "sample-1",
        "source_dataset": "plantdoc",
        "task_type": "classification",
        "split": "train",
        "images": ["image-a.png"],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Agricultural RGB consultation only."}]},
            {"role": "user", "content": [{"type": "image", "image": "image-a.png"}]},
        ],
        "target": {"answer_text": "leaf spot"},
        "metadata": {"source_image_id": "same-source-image"},
    }
    train_row = UnifiedSample.model_validate(payload)
    eval_payload = dict(payload, sample_id="sample-2", split="holdout", images=["image-b.png"])
    eval_row = UnifiedSample.model_validate(eval_payload)

    with pytest.raises(ValueError, match="Train/eval manifest overlap"):
        _assert_no_train_eval_overlap([train_row], [eval_row])


def test_peft_adapter_save_passes_raw_lora_state_dict(monkeypatch, tmp_path) -> None:
    torch = pytest.importorskip("torch")

    from agri_vlm.training.sft_trainer import _save_peft_adapter_model

    captured = {}
    lora_name = "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"

    class FakePeftModel:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.ones(2, 2))

        def named_parameters(self):
            return [(lora_name, self.parameter)]

        def save_pretrained(self, output_dir, **kwargs):
            captured["output_dir"] = output_dir
            captured["kwargs"] = kwargs

    def fake_get_peft_model_state_dict(_model, *, state_dict, **_kwargs):
        return {
            key.replace(".default.", "."): value
            for key, value in state_dict.items()
            if ".default." in key
        }

    fake_peft = types.SimpleNamespace(get_peft_model_state_dict=fake_get_peft_model_state_dict)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    _save_peft_adapter_model(FakePeftModel(), tmp_path, should_save=True)

    saved_state_dict = captured["kwargs"]["state_dict"]
    assert list(saved_state_dict) == [lora_name]
    assert captured["kwargs"]["safe_serialization"] is True
    assert captured["kwargs"]["is_main_process"] is True
