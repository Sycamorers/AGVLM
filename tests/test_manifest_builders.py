from collections import Counter
from pathlib import Path

import pytest

from agri_vlm.data.builders import (
    build_balanced_sft_v2_manifest,
    build_classification_probe_manifests,
    build_closed_label_eval_manifest,
    build_closed_label_sft_manifest,
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


def test_build_rl_manifest_makes_duplicate_sample_ids_unique(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    duplicate = sample_row("dup", "mirage", "vqa", "train")
    second_duplicate = sample_row("dup", "mirage", "vqa", "train")
    second_duplicate["images"] = ["data/raw/_smoke/dup_alt.png"]
    write_manifest(source_path, [duplicate, second_duplicate])

    rl_output = tmp_path / "rl.jsonl"
    rl_rows = build_rl_manifest(
        source_paths=[source_path],
        output_path=rl_output,
        allowed_task_types=["vqa"],
        exclude_splits=["test"],
        allowed_verifier_modes=["exact_match"],
        max_answer_words=10,
        max_images_per_sample=1,
    )

    sample_ids = [row["sample_id"] for row in rl_rows]
    assert len(sample_ids) == len(set(sample_ids))
    assert sample_ids[0] == "dup"
    assert sample_ids[1].startswith("dup-rl-0002-")
    assert rl_rows[1]["metadata"]["original_sample_id"] == "dup"


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


def test_build_balanced_sft_v2_manifest_caps_and_repeats(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "balanced.jsonl"
    summary_path = tmp_path / "summary.json"
    rows = []
    for index in range(6):
        row = sample_row("class-%s" % index, "plantdoc", "classification", "train")
        row["target"]["canonical_label"] = "label-%s" % (index % 2)
        rows.append(row)
    for index in range(4):
        rows.append(sample_row("vqa-%s" % index, "plantvillage_vqa", "vqa", "train"))
    for index in range(2):
        row = sample_row("clarify-%s" % index, "mirage", "clarify_or_respond", "train")
        row["target"] = {"answer_text": "Which crop is affected?", "decision": "clarify"}
        row["verifier"]["mode"] = "clarify"
        rows.append(row)
    write_manifest(source_path, rows)

    summary = build_balanced_sft_v2_manifest(
        input_manifest_path=source_path,
        output_manifest_path=output_path,
        summary_output_path=summary_path,
        task_targets={"classification": 4, "vqa": 3, "clarify_or_respond": 5},
        stratify_fields_by_task={
            "classification": ["source_dataset", "target.canonical_label"],
            "vqa": ["source_dataset"],
            "clarify_or_respond": ["source_dataset", "target.decision"],
        },
        seed=11,
        shuffle=True,
    )

    output_rows = read_manifest(output_path)
    counts = Counter(row.task_type for row in output_rows)
    assert counts == {"classification": 4, "vqa": 3, "clarify_or_respond": 5}
    assert summary["task_plan"]["clarify_or_respond"]["unique_selected_rows"] == 2
    assert summary["task_plan"]["clarify_or_respond"]["repeated_rows_added"] == 3
    assert len({row.sample_id for row in output_rows if row.task_type == "clarify_or_respond"}) == 2
    assert summary_path.exists()


def test_build_closed_label_sft_manifest_balances_and_adds_label_space(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "closed.jsonl"
    summary_path = tmp_path / "summary.json"
    rows = []
    for index in range(3):
        row = sample_row("ip-a-%s" % index, "ip102", "classification", "train")
        row["target"]["canonical_label"] = "23 corn borer"
        row["target"]["answer_text"] = "23 corn borer"
        row["verifier"]["accepted_labels"] = ["23 corn borer"]
        rows.append(row)
    row = sample_row("ip-b", "ip102", "classification", "train")
    row["target"]["canonical_label"] = "6 rice gall midge"
    row["target"]["answer_text"] = "6 rice gall midge"
    row["verifier"]["accepted_labels"] = ["6 rice gall midge"]
    rows.append(row)
    row = sample_row("plantdoc-a", "plantdoc", "classification", "train")
    row["target"]["canonical_label"] = "tomato leaf mold"
    row["target"]["answer_text"] = "tomato leaf mold"
    row["verifier"]["accepted_labels"] = ["tomato leaf mold"]
    rows.append(row)
    for index in range(2):
        rows.append(sample_row("vqa-%s" % index, "plantvillage_vqa", "vqa", "train"))
    write_manifest(source_path, rows)

    summary = build_closed_label_sft_manifest(
        input_manifest_path=source_path,
        output_manifest_path=output_path,
        summary_output_path=summary_path,
        classification_per_label_target=2,
        task_targets={"vqa": 3},
        stratify_fields_by_task={"vqa": ["source_dataset"]},
        strip_leading_numeric_prefix_sources=["ip102"],
        seed=17,
    )

    output_rows = read_manifest(output_path)
    counts = Counter(row.task_type for row in output_rows)
    assert counts == {"classification": 6, "vqa": 3}
    labels = Counter(
        (row.source_dataset, row.target.canonical_label)
        for row in output_rows
        if row.task_type == "classification"
    )
    assert labels[("ip102", "corn borer")] == 2
    assert labels[("ip102", "rice gall midge")] == 2
    assert labels[("plantdoc", "tomato leaf mold")] == 2
    ip_row = next(row for row in output_rows if row.task_type == "classification" and row.source_dataset == "ip102")
    assert ip_row.metadata["classification_label_space"] == ["corn borer", "rice gall midge"]
    assert ip_row.metadata["classification_label_space_size"] == 2
    assert "23 corn borer" in ip_row.verifier.accepted_labels
    assert summary["classification_label_space_sizes_by_source"]["ip102"] == 2
    assert summary_path.exists()


def test_build_closed_label_eval_manifest_repairs_labels_and_adds_source_label_space(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    label_space_path = tmp_path / "label_space.jsonl"
    output_path = tmp_path / "eval_closed.jsonl"
    summary_path = tmp_path / "summary.json"

    eval_row = sample_row("eval-ip", "ip102", "classification", "validation")
    eval_row["target"]["canonical_label"] = "23 corn borer"
    eval_row["target"]["answer_text"] = "23 corn borer"
    eval_row["verifier"]["accepted_labels"] = ["23 corn borer"]
    vqa_row = sample_row("eval-vqa", "plantvillage_vqa", "vqa", "validation")
    write_manifest(eval_path, [eval_row, vqa_row])

    label_rows = []
    for sample_id, label in [("train-a", "23 corn borer"), ("train-b", "6 rice gall midge")]:
        row = sample_row(sample_id, "ip102", "classification", "train")
        row["target"]["canonical_label"] = label
        row["target"]["answer_text"] = label
        row["verifier"]["accepted_labels"] = [label]
        label_rows.append(row)
    write_manifest(label_space_path, label_rows)

    summary = build_closed_label_eval_manifest(
        input_manifest_path=eval_path,
        label_space_manifest_path=label_space_path,
        output_manifest_path=output_path,
        summary_output_path=summary_path,
        strip_leading_numeric_prefix_sources=["ip102"],
    )

    output_rows = read_manifest(output_path)
    repaired = next(row for row in output_rows if row.sample_id == "eval-ip")
    assert repaired.target.canonical_label == "corn borer"
    assert "23 corn borer" in repaired.verifier.accepted_labels
    assert repaired.metadata["classification_label_space"] == ["corn borer", "rice gall midge"]
    assert repaired.metadata["classification_label_space_size"] == 2
    assert summary["repaired_rows_by_source"]["ip102"] == 1
    assert summary_path.exists()


def test_build_closed_label_eval_manifest_repairs_gray_light_alias(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    label_space_path = tmp_path / "label_space.jsonl"
    output_path = tmp_path / "eval_closed.jsonl"

    eval_row = sample_row("eval-tea", "tea_sickness", "classification", "validation")
    eval_row["target"]["canonical_label"] = "gray light"
    eval_row["target"]["answer_text"] = "gray light"
    eval_row["verifier"]["accepted_labels"] = ["gray light"]
    label_row = sample_row("train-tea", "tea_sickness", "classification", "train")
    label_row["target"]["canonical_label"] = "gray light"
    label_row["target"]["answer_text"] = "gray light"
    label_row["verifier"]["accepted_labels"] = ["gray light"]
    write_manifest(eval_path, [eval_row])
    write_manifest(label_space_path, [label_row])

    build_closed_label_eval_manifest(
        input_manifest_path=eval_path,
        label_space_manifest_path=label_space_path,
        output_manifest_path=output_path,
    )

    repaired = read_manifest(output_path)[0]
    assert repaired.target.canonical_label == "gray blight"
    assert repaired.metadata["classification_label_space"] == ["gray blight"]
    assert "gray light" in repaired.verifier.accepted_labels


def test_build_closed_label_eval_manifest_fails_when_eval_label_missing_from_space(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    label_space_path = tmp_path / "label_space.jsonl"
    output_path = tmp_path / "eval_closed.jsonl"

    eval_row = sample_row("eval-a", "digigreen_crop_disease", "classification", "validation")
    eval_row["target"]["canonical_label"] = "coriander healthy"
    eval_row["target"]["answer_text"] = "coriander healthy"
    eval_row["verifier"]["accepted_labels"] = ["coriander healthy"]
    label_row = sample_row("train-a", "digigreen_crop_disease", "classification", "train")
    label_row["target"]["canonical_label"] = "maize healthy"
    label_row["target"]["answer_text"] = "maize healthy"
    label_row["verifier"]["accepted_labels"] = ["maize healthy"]
    write_manifest(eval_path, [eval_row])
    write_manifest(label_space_path, [label_row])

    with pytest.raises(ValueError, match="missing from their source label spaces"):
        build_closed_label_eval_manifest(
            input_manifest_path=eval_path,
            label_space_manifest_path=label_space_path,
            output_manifest_path=output_path,
        )


def test_build_classification_probe_manifests_selects_shared_labels(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_output_path = tmp_path / "probe_train.jsonl"
    eval_output_path = tmp_path / "probe_eval.jsonl"
    summary_path = tmp_path / "probe_summary.json"

    train_rows = []
    eval_rows = []
    for label in ["apple scab", "late blight"]:
        for index in range(2):
            row = sample_row("train-%s-%s" % (label.replace(" ", "-"), index), "plantdoc", "classification", "train")
            row["target"]["canonical_label"] = label
            row["target"]["answer_text"] = label
            row["verifier"]["accepted_labels"] = [label]
            train_rows.append(row)
        eval_row = sample_row("eval-%s" % label.replace(" ", "-"), "plantdoc", "classification", "validation")
        eval_row["target"]["canonical_label"] = label
        eval_row["target"]["answer_text"] = label
        eval_row["verifier"]["accepted_labels"] = [label]
        eval_rows.append(eval_row)
    missing_eval = sample_row("eval-only", "plantdoc", "classification", "validation")
    missing_eval["target"]["canonical_label"] = "powdery mildew"
    missing_eval["target"]["answer_text"] = "powdery mildew"
    missing_eval["verifier"]["accepted_labels"] = ["powdery mildew"]
    eval_rows.append(missing_eval)
    write_manifest(train_path, train_rows)
    write_manifest(eval_path, eval_rows)

    summary = build_classification_probe_manifests(
        train_source_manifest_path=train_path,
        eval_source_manifest_path=eval_path,
        train_output_path=train_output_path,
        eval_output_path=eval_output_path,
        summary_output_path=summary_path,
        train_per_label=3,
        eval_per_label=1,
        max_labels_per_source=2,
        seed=31,
        sources=["plantdoc"],
    )

    train_output_rows = read_manifest(train_output_path)
    eval_output_rows = read_manifest(eval_output_path)
    assert Counter(row.target.canonical_label for row in train_output_rows) == {
        "apple scab": 3,
        "late blight": 3,
    }
    assert Counter(row.target.canonical_label for row in eval_output_rows) == {
        "apple scab": 1,
        "late blight": 1,
    }
    for row in train_output_rows + eval_output_rows:
        assert row.metadata["classification_probe"] is True
        assert row.metadata["classification_label_space"] == ["apple scab", "late blight"]
    assert summary["eligible_label_count_by_source"]["plantdoc"] == 2
    assert summary_path.exists()


def test_build_classification_probe_manifests_can_emit_multiple_choice_options(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_output_path = tmp_path / "probe_mc_train.jsonl"
    eval_output_path = tmp_path / "probe_mc_eval.jsonl"

    train_rows = []
    eval_rows = []
    for label in ["apple scab", "late blight"]:
        for index in range(2):
            row = sample_row("train-%s-%s" % (label.replace(" ", "-"), index), "plantdoc", "classification", "train")
            row["target"]["canonical_label"] = label
            row["target"]["answer_text"] = label
            row["verifier"]["accepted_labels"] = [label]
            train_rows.append(row)
        eval_row = sample_row("eval-%s" % label.replace(" ", "-"), "plantdoc", "classification", "validation")
        eval_row["target"]["canonical_label"] = label
        eval_row["target"]["answer_text"] = label
        eval_row["verifier"]["accepted_labels"] = [label]
        eval_rows.append(eval_row)
    write_manifest(train_path, train_rows)
    write_manifest(eval_path, eval_rows)

    summary = build_classification_probe_manifests(
        train_source_manifest_path=train_path,
        eval_source_manifest_path=eval_path,
        train_output_path=train_output_path,
        eval_output_path=eval_output_path,
        train_per_label=1,
        eval_per_label=1,
        max_labels_per_source=2,
        seed=41,
        sources=["plantdoc"],
        choice_format="multiple_choice",
    )

    output_rows = read_manifest(train_output_path) + read_manifest(eval_output_path)
    assert summary["choice_format"] == "multiple_choice"
    for row in output_rows:
        assert row.metadata["classification_format"] == "multiple_choice"
        assert row.metadata["classification_choice_count"] == 2
        options = row.metadata["classification_choice_options"]
        assert {option["letter"] for option in options} == {"A", "B"}
        assert {option["label"] for option in options} == {"apple scab", "late blight"}
        answer = row.metadata["classification_choice_answer"]
        assert answer in options
        assert answer["label"] == row.target.canonical_label
