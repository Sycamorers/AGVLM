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


def test_train_config_accepts_loss_chunk_size() -> None:
    config = TrainConfigSchema.model_validate(
        {
            "manifest_path": "data/manifests/partial_10pct/sft_manifest.jsonl",
            "output_dir": "outputs/smoke/sft-qwen3-vl-4b",
            "loss_chunk_size": 1024,
        }
    )

    assert config.loss_chunk_size == 1024
