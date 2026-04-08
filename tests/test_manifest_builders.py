from pathlib import Path

from agri_vlm.data.builders import build_eval_manifests, build_rl_manifest, build_sft_manifest
from agri_vlm.data.manifest_io import read_manifest, write_manifest


def sample_row(sample_id: str, dataset: str, task_type: str, split: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_dataset": dataset,
        "task_type": task_type,
        "split": split,
        "images": ["data/raw/_smoke/%s.png" % sample_id],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an agricultural assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data/raw/_smoke/%s.png" % sample_id},
                    {"type": "text", "text": "Answer the question."},
                ],
            },
        ],
        "target": {"answer_text": "leaf spot", "canonical_label": "leaf spot"},
        "metadata": {"source_image_id": "%s.png" % sample_id, "benchmark_track": "mmst"},
        "verifier": {
            "mode": "label" if task_type == "classification" else "exact_match",
            "accepted_labels": ["leaf spot"],
            "accepted_answers": ["leaf spot"],
        },
        "reward_meta": {"weights": {"exact_match": 1.0}},
    }


def test_manifest_builders_filter_and_merge(tmp_path: Path) -> None:
    plantdoc_path = tmp_path / "plantdoc.jsonl"
    mirage_path = tmp_path / "mirage.jsonl"
    agmmu_path = tmp_path / "agmmu.jsonl"
    ip102_path = tmp_path / "ip102.jsonl"
    vqa_path = tmp_path / "vqa.jsonl"

    write_manifest(
        plantdoc_path,
        [
            sample_row("p1", "plantdoc", "classification", "train"),
            sample_row("p2", "plantdoc", "classification", "test"),
        ],
    )
    write_manifest(mirage_path, [sample_row("m1", "mirage", "consultation", "validation")])
    write_manifest(agmmu_path, [sample_row("a1", "agmmu", "vqa", "test")])
    write_manifest(ip102_path, [sample_row("i1", "ip102", "classification", "train")])
    write_manifest(vqa_path, [sample_row("v1", "plantvillage_vqa", "vqa", "train")])

    sft_output = tmp_path / "sft.jsonl"
    rows = build_sft_manifest(
        source_paths=[plantdoc_path, mirage_path],
        output_path=sft_output,
        allowed_task_types=["classification", "consultation"],
        exclude_splits=["test"],
    )
    assert len(rows) == 2
    assert sft_output.exists()

    rl_output = tmp_path / "rl.jsonl"
    rl_rows = build_rl_manifest(
        source_paths=[plantdoc_path, vqa_path],
        output_path=rl_output,
        allowed_task_types=["classification", "vqa"],
        exclude_splits=["test"],
        allowed_verifier_modes=["label", "exact_match"],
        max_answer_words=10,
    )
    assert len(rl_rows) == 2

    summary = build_eval_manifests(
        source_paths={
            "agmmu": agmmu_path,
            "mirage": mirage_path,
            "plantdoc": plantdoc_path,
            "ip102": ip102_path,
            "plantvillage_vqa": vqa_path,
        },
        output_paths={
            "agmmu": tmp_path / "agmmu_eval.jsonl",
            "mirage_mmst": tmp_path / "mmst.jsonl",
            "mirage_mmmt": tmp_path / "mmmt.jsonl",
            "local_holdout": tmp_path / "holdout.jsonl",
        },
        holdout_ratio=0.1,
        holdout_datasets=["plantdoc", "ip102", "plantvillage_vqa"],
        salt="test-salt",
    )
    assert summary["agmmu"] == 1
    assert len(read_manifest(tmp_path / "holdout.jsonl")) >= 1
