from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "vlm_baselines"))

from build_phase_splits import RL_PHASE, SFT_PHASE, _phase_report, build_phase_splits  # noqa: E402
from dataset_adapter import semantic_prompt  # noqa: E402


def _row(sample_id: str, image: str, *, split: str = "holdout") -> dict:
    return {
        "sample_id": sample_id,
        "source_dataset": "synthetic",
        "task_type": "classification",
        "split": split,
        "images": [image],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Identify the issue."}]}],
        "target": {"canonical_label": "late blight"},
        "verifier": {"mode": "label", "accepted_labels": ["late blight"]},
        "metadata": {"source_image_id": image, "crop": "tomato", "disease": "late blight"},
    }


def test_phase_report_detects_duplicates_missing_images_and_overlap():
    train = [_row("a", "missing/a.jpg")]
    rows_by_split = {"val": [_row("a", "missing/a.jpg")], "test": [_row("a", "missing/a.jpg")]}
    report = _phase_report(phase=SFT_PHASE, rows_by_split=rows_by_split, train_rows=train)
    assert report["duplicate_sample_id_count"] == 1
    assert report["missing_image_sample_count"] == 2
    assert report["train_eval_overlap"]["exact_sample_id_count"] == 1
    assert report["train_eval_overlap"]["group_key_count"] == 1


def test_real_phase_split_manifests_are_phase_tagged(tmp_path):
    report = build_phase_splits(
        phase="both",
        output_dir=tmp_path,
        seed=123,
        max_samples=12,
        force=True,
        write_report=True,
    )
    assert SFT_PHASE in report["phases"]
    assert RL_PHASE in report["phases"]
    for path in [
        tmp_path / "sft_val_manifest.jsonl",
        tmp_path / "sft_test_manifest.jsonl",
        tmp_path / "rl_val_manifest.jsonl",
        tmp_path / "rl_test_manifest.jsonl",
    ]:
        assert path.exists()
    sft_lines = (tmp_path / "sft_test_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    sft_lines += (tmp_path / "sft_val_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    sample_line = sft_lines[0]
    assert '"phase": "sft_benchmark"' in sample_line


def test_semantic_prompt_does_not_duplicate_existing_output_contract():
    row = _row("a", "missing/a.jpg")
    row["messages"][0]["content"][0]["text"] = (
        "Identify the issue.\n\n"
        "Respond in this format:\n"
        "Answer: <canonical agricultural label>"
    )

    prompt = semantic_prompt(row)

    assert prompt.count("Respond in this format:") == 1
