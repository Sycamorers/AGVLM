import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agri_vlm.schemas.config_schema import ModelConfigSchema, RLTrainConfigSchema, load_config
from agri_vlm.training.rl_trainer import run_rl_grpo, validate_rl_sft_checkpoint_path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing else "src:%s" % existing
    return env


def _row(sample_id: str, image_path: Path, verifier_mode: str = "label") -> dict:
    verifier = {
        "mode": verifier_mode,
        "accepted_labels": ["leaf spot"],
        "accepted_answers": ["leaf spot"],
    }
    if verifier_mode == "clarify":
        verifier["expected_decision"] = "clarify"
    return {
        "sample_id": sample_id,
        "source_dataset": "unit",
        "task_type": "classification" if verifier_mode != "clarify" else "clarify_or_respond",
        "split": "train",
        "images": [str(image_path)],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an agricultural assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "Identify the disease."},
                ],
            },
        ],
        "target": {
            "answer_text": "leaf spot",
            "canonical_label": "leaf spot",
            "decision": "clarify" if verifier_mode == "clarify" else None,
            "acceptable_answers": ["leaf spot"],
        },
        "metadata": {"crop": "tomato"},
        "verifier": verifier,
        "reward_meta": {"weights": {"exact_match": 1.0, "normalized_label": 1.0}},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_rl_manifest_audit_reports_counts(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.png"
    image_path.write_bytes(b"not-a-real-image")
    manifest_path = tmp_path / "rl.jsonl"
    _write_jsonl(manifest_path, [_row("sample-1", image_path)])
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/data/audit_rl_manifest.py",
            "--manifest-path",
            str(manifest_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--fail-on-critical",
        ],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["total_samples"] == 1
    assert report["counts"]["by_verifier_mode"]["label"] == 1
    assert report["critical_issue_count"] == 0
    assert output_md.exists()


def test_rl_manifest_audit_fails_on_critical_issues(tmp_path: Path) -> None:
    missing_image = tmp_path / "missing.png"
    manifest_path = tmp_path / "rl_bad.jsonl"
    _write_jsonl(manifest_path, [_row("dup", missing_image), _row("dup", missing_image)])
    output_json = tmp_path / "audit_bad.json"
    output_md = tmp_path / "audit_bad.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/data/audit_rl_manifest.py",
            "--manifest-path",
            str(manifest_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--fail-on-critical",
        ],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["issues"]["duplicate_sample_ids"]["count"] == 1
    assert report["issues"]["image_paths_not_exist"]["count"] == 2


def test_reward_sanity_check_scores_target_above_empty(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.png"
    image_path.write_bytes(b"not-a-real-image")
    manifest_path = tmp_path / "rl.jsonl"
    _write_jsonl(manifest_path, [_row("sample-1", image_path)])
    output_json = tmp_path / "reward.json"
    output_md = tmp_path / "reward.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train/rl_reward_sanity_check.py",
            "--manifest-path",
            str(manifest_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--max-samples",
            "1",
        ],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    averages = report["average_reward_by_candidate"]
    assert averages["target_answer"] > averages["empty"]
    assert output_md.exists()


def test_rl_dataset_format_check_uses_expected_columns(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.png"
    image_path.write_bytes(b"not-a-real-image")
    manifest_path = tmp_path / "rl.jsonl"
    _write_jsonl(manifest_path, [_row("sample-1", image_path)])
    output_json = tmp_path / "format.json"
    output_md = tmp_path / "format.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train/check_rl_dataset_format.py",
            "--manifest-path",
            str(manifest_path),
            "--model-config",
            "configs/model/phi4_reasoning_vision_15b_turin_24g.yaml",
            "--max-samples",
            "1",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["reward_function_columns_ok"] is True
    assert report["dataset_columns"] == [
        "prompt",
        "image_paths",
        "task_type",
        "sample_id",
        "target_json",
        "verifier_json",
        "reward_meta_json",
        "metadata_json",
    ]
    assert report["issues"] == {}


def test_phi4_rl_configs_load() -> None:
    for config_path in [
        "configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_readiness.yaml",
        "configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_smoke_after_sft.yaml",
        "configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_full_after_sft.yaml",
    ]:
        config = load_config(REPO_ROOT / config_path, RLTrainConfigSchema)
        assert config.manifest_path == "data/manifests/full/rl_manifest.jsonl"
        assert config.lora.target_modules == ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        assert "qwen" not in config.sft_checkpoint_path.lower()


def test_rl_dry_run_allows_placeholder_sft_path(tmp_path: Path) -> None:
    image_path = tmp_path / "leaf.png"
    image_path.write_bytes(b"not-a-real-image")
    manifest_path = tmp_path / "rl.jsonl"
    _write_jsonl(manifest_path, [_row("sample-1", image_path)])
    model_config = load_config(
        REPO_ROOT / "configs/model/phi4_reasoning_vision_15b_turin_24g.yaml",
        ModelConfigSchema,
    )
    train_config = RLTrainConfigSchema(
        sft_checkpoint_path="/tmp/<FINAL_SFT_CHECKPOINT_OR_ADAPTER>",
        manifest_path=str(manifest_path),
        output_dir=str(tmp_path / "out"),
        checkpoint_output_dir=str(tmp_path / "ckpt"),
        dry_run=True,
        smoke_max_samples=1,
        reward_modules=["exact_match"],
        reward_weights={"exact_match": 1.0},
    )

    summary = run_rl_grpo(model_config=model_config, train_config=train_config)

    assert summary["train_rows"] == 1
    assert (tmp_path / "out" / "dry_run_summary.json").exists()


def test_non_dry_run_rejects_placeholder_sft_path() -> None:
    model_config = load_config(
        REPO_ROOT / "configs/model/phi4_reasoning_vision_15b_turin_24g.yaml",
        ModelConfigSchema,
    )
    train_config = load_config(
        REPO_ROOT / "configs/train/rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_smoke_after_sft.yaml",
        RLTrainConfigSchema,
    )
    with pytest.raises(ValueError, match="placeholder"):
        validate_rl_sft_checkpoint_path(model_config, train_config)


def test_non_dry_run_rejects_missing_sft_path(tmp_path: Path) -> None:
    model_config = load_config(
        REPO_ROOT / "configs/model/phi4_reasoning_vision_15b_turin_24g.yaml",
        ModelConfigSchema,
    )
    train_config = RLTrainConfigSchema(
        sft_checkpoint_path=str(tmp_path / "missing-checkpoint"),
        manifest_path="data/manifests/full/rl_manifest.jsonl",
        output_dir=str(tmp_path / "out"),
        dry_run=False,
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_rl_sft_checkpoint_path(model_config, train_config)


def test_non_dry_run_rejects_base_model_path(tmp_path: Path) -> None:
    model_config = load_config(
        REPO_ROOT / "configs/model/phi4_reasoning_vision_15b_turin_24g.yaml",
        ModelConfigSchema,
    )
    train_config = RLTrainConfigSchema(
        sft_checkpoint_path="microsoft/Phi-4-reasoning-vision-15B",
        manifest_path="data/manifests/full/rl_manifest.jsonl",
        output_dir=str(tmp_path / "out"),
        dry_run=False,
    )
    with pytest.raises(ValueError, match="raw/base model"):
        validate_rl_sft_checkpoint_path(model_config, train_config)


def test_slurm_wrapper_static_checks() -> None:
    slurm_text = (
        REPO_ROOT / "scripts/hpc/run_rl_grpo_b200_4gpu_phi4_reasoning_vision_15b.slurm"
    ).read_text(encoding="utf-8")
    assert "torchrun" in slurm_text
    assert "--nproc_per_node=4" in slurm_text
    assert "configs/model/phi4_reasoning_vision_15b_turin_24g.yaml" in slurm_text
    assert "TRAIN_CONFIG" in slurm_text
    assert "rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_smoke_after_sft.yaml" in slurm_text
    assert "rl_grpo_phi4_reasoning_vision_15b_b200_4gpu_full_after_sft.yaml" not in slurm_text
