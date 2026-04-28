from pathlib import Path

from agri_vlm.data.builders import (
    build_eval_manifests,
    build_rl_manifest,
    build_sft_manifest,
    build_sft_train_eval_manifests,
)
from agri_vlm.data.manifest_io import read_manifest, write_manifest


def sample_row(sample_id: str, dataset: str, task_type: str, split: str, image_count: int = 1) -> dict:
    images = ["data/raw/_smoke/%s_%s.png" % (sample_id, index) for index in range(image_count)]
    return {
        "sample_id": sample_id,
        "source_dataset": dataset,
        "task_type": task_type,
        "split": split,
        "images": images,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an agricultural assistant."}],
            },
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
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
    write_manifest(ip102_path, [sample_row("i1", "ip102", "classification", "train")])
    write_manifest(
        vqa_path,
        [
            sample_row("v1", "plantvillage_vqa", "vqa", "train"),
            sample_row("v2", "plantvillage_vqa", "vqa", "train", image_count=2),
        ],
    )

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
        max_images_per_sample=1,
    )
    assert len(rl_rows) == 2
    assert all(len(row["images"]) == 1 for row in rl_rows)

    summary = build_eval_manifests(
        source_paths={
            "mirage": mirage_path,
            "plantdoc": plantdoc_path,
            "ip102": ip102_path,
            "plantvillage_vqa": vqa_path,
        },
        output_paths={
            "mirage_mmst": tmp_path / "mmst.jsonl",
            "mirage_mmmt": tmp_path / "mmmt.jsonl",
            "local_holdout": tmp_path / "holdout.jsonl",
        },
        holdout_ratio=0.1,
        holdout_datasets=["plantdoc", "ip102", "plantvillage_vqa"],
        salt="test-salt",
    )
    assert summary["mirage_mmst"] == 1
    assert summary["mirage_mmmt"] == 0
    assert len(read_manifest(tmp_path / "holdout.jsonl")) >= 1


def test_build_sft_train_eval_manifests_removes_eval_overlap(tmp_path: Path) -> None:
    source_path = tmp_path / "sft_source.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    train_output = tmp_path / "train.jsonl"
    eval_output = tmp_path / "eval.jsonl"
    summary_output = tmp_path / "summary.json"

    write_manifest(
        source_path,
        [
            sample_row("train-keep", "plantdoc", "classification", "train"),
            sample_row("train-holdout", "plantdoc", "classification", "train"),
            sample_row("train-val-group", "ip102", "classification", "train"),
            sample_row("val", "ip102", "classification", "validation"),
            sample_row("test", "ip102", "classification", "test"),
            sample_row("multi", "ip102", "classification", "train", image_count=2),
        ],
    )
    rows = [row.model_dump(mode="json") for row in read_manifest(source_path)]
    rows[2]["metadata"]["source_image_id"] = "shared.png"
    rows[3]["metadata"]["source_image_id"] = "shared.png"
    write_manifest(source_path, rows)
    write_manifest(holdout_path, [sample_row("train-holdout", "plantdoc", "classification", "holdout")])

    summary = build_sft_train_eval_manifests(
        source_manifest_path=source_path,
        holdout_manifest_path=holdout_path,
        train_output_path=train_output,
        eval_output_path=eval_output,
        train_splits=["train"],
        eval_splits=["validation"],
        max_images_per_sample=1,
        eval_sample_size=8,
        min_eval_samples_per_stratum=1,
        salt="unit-test",
        summary_output_path=summary_output,
    )

    train_ids = {row.sample_id for row in read_manifest(train_output)}
    eval_ids = {row.sample_id for row in read_manifest(eval_output)}
    assert train_ids == {"train-keep"}
    assert eval_ids == {"train-holdout", "val"}
    assert summary["overlap"] == {"exact_sample_id": 0, "group_key": 0}
    assert summary_output.exists()
