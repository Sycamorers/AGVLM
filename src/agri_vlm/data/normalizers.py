"""Dataset normalization functions."""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agri_vlm.constants import (
    TASK_TYPE_CLASSIFICATION,
    TASK_TYPE_CLARIFY,
    TASK_TYPE_CONSULTATION,
    TASK_TYPE_SYMPTOM_EXPLANATION,
    TASK_TYPE_VQA,
)
from agri_vlm.data.loaders import read_records, require_records_file
from agri_vlm.data.registry import read_download_info
from agri_vlm.data.split_utils import assign_hash_split
from agri_vlm.data.transforms import (
    build_structured_consultation_answer,
    build_user_message_text,
    default_system_prompt,
    detect_split_from_path,
    metadata_with_license,
    parse_ip102_label,
    parse_plant_label,
    relative_posix_path,
    normalize_split_name,
    slugify,
)
from agri_vlm.utils.image import collect_image_paths
from agri_vlm.utils.text import normalize_label


def _build_messages(question: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": default_system_prompt()}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": path} for path in image_paths]
            + [{"type": "text", "text": question}],
        },
    ]


def _base_sample(
    sample_id: str,
    source_dataset: str,
    task_type: str,
    split: str,
    image_paths: List[str],
    question: str,
    target: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    verifier: Optional[Dict[str, Any]] = None,
    reward_meta: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged_metadata = dict(provenance or {})
    merged_metadata.update(metadata or {})
    return {
        "sample_id": sample_id,
        "source_dataset": source_dataset,
        "task_type": task_type,
        "split": split,
        "images": image_paths,
        "messages": _build_messages(question=question, image_paths=image_paths),
        "target": target,
        "metadata": merged_metadata,
        "verifier": verifier or {"mode": "none"},
        "reward_meta": reward_meta or {"weights": {}},
    }


def load_provenance_metadata(raw_dir: Path) -> Dict[str, Any]:
    payload = read_download_info(raw_dir)
    if not payload:
        return {}
    keys = [
        "subset_tag",
        "download_mode",
        "sample_fraction",
        "source_type",
        "access",
        "source_repo_id",
    ]
    return {key: payload[key] for key in keys if payload.get(key) is not None}


def normalize_classification_directory_dataset(
    raw_dir: Path,
    repo_root: Path,
    dataset_name: str,
    salt: str,
    pest_mode: bool = False,
    license_name: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    image_paths = collect_image_paths(raw_dir)
    if not image_paths:
        raise FileNotFoundError("No image files found under %s" % raw_dir)

    rows: List[Dict[str, Any]] = []
    for image_path in image_paths:
        split = detect_split_from_path(image_path) or assign_hash_split(
            relative_posix_path(image_path, raw_dir), salt=salt
        )
        label_name = image_path.parent.name
        crop_name, disease_name = parse_plant_label(label_name)
        canonical_label = parse_ip102_label(label_name) if pest_mode else normalize_label(label_name)
        metadata = metadata_with_license(
            {
                "crop": crop_name,
                "disease": disease_name if not pest_mode else None,
                "pest": canonical_label if pest_mode else None,
                "original_label": label_name,
                "normalized_label": canonical_label,
                "template_origin": "templated_from_label",
                "source_image_id": relative_posix_path(image_path, raw_dir),
            },
            license_name=license_name,
        )
        question = build_user_message_text(TASK_TYPE_CLASSIFICATION)
        target = {"answer_text": canonical_label, "canonical_label": canonical_label}
        verifier = {"mode": "label", "accepted_labels": [canonical_label, normalize_label(label_name)]}
        reward_meta = {"weights": {"normalized_label": 1.0, "hallucination_penalty": 1.0}}
        rows.append(
            _base_sample(
                sample_id="%s-%s" % (dataset_name, slugify(relative_posix_path(image_path, raw_dir))),
                source_dataset=dataset_name,
                task_type=TASK_TYPE_CLASSIFICATION,
                split=split,
                image_paths=[relative_posix_path(image_path, repo_root)],
                question=question,
                target=target,
                metadata=metadata,
                verifier=verifier,
                reward_meta=reward_meta,
                provenance=provenance,
            )
        )
    return rows


def normalize_classification_records_dataset(
    raw_dir: Path,
    repo_root: Path,
    dataset_name: str,
    pest_mode: bool = False,
    license_name: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    records_path = require_records_file(raw_dir)
    records = read_records(records_path)
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(records):
        image_paths = _extract_image_paths(row, raw_dir=raw_dir, repo_root=repo_root)
        split = normalize_split_name(row.get("split")) or assign_hash_split(
            "%s-%s" % (dataset_name, index), salt=dataset_name
        )
        raw_labels = row.get("all_labels") or row.get("labels") or row.get("categories") or []
        if isinstance(raw_labels, str):
            raw_labels = [raw_labels]
        label_name = str(row.get("label") or row.get("category") or "").strip()
        if not label_name:
            if raw_labels:
                counts = Counter(str(item) for item in raw_labels if str(item).strip())
                label_name = counts.most_common(1)[0][0]
            else:
                raise ValueError("Classification record is missing a label in %s" % records_path)
        canonical_candidates = [str(item) for item in raw_labels if str(item).strip()] or [label_name]
        canonical_labels = [
            parse_ip102_label(item) if pest_mode else normalize_label(item) for item in canonical_candidates
        ]
        canonical_label = canonical_labels[0]
        crop_name, disease_name = parse_plant_label(label_name)
        metadata = metadata_with_license(
            {
                "crop": row.get("crop") or crop_name,
                "disease": row.get("disease") or (disease_name if not pest_mode else None),
                "pest": row.get("pest") or (canonical_label if pest_mode else None),
                "original_label": label_name,
                "original_labels": canonical_candidates,
                "normalized_label": canonical_label,
                "normalized_labels": canonical_labels,
                "source_image_id": row.get("image") or row.get("id") or str(index),
                "template_origin": row.get("template_origin", "source_authored"),
            },
            license_name=license_name,
        )
        rows.append(
            _base_sample(
                sample_id="%s-%s" % (dataset_name, row.get("id") or index),
                source_dataset=dataset_name,
                task_type=TASK_TYPE_CLASSIFICATION,
                split=split,
                image_paths=image_paths,
                question=build_user_message_text(TASK_TYPE_CLASSIFICATION, row.get("question")),
                target={"answer_text": canonical_label, "canonical_label": canonical_label},
                metadata=metadata,
                verifier={"mode": "label", "accepted_labels": canonical_labels},
                reward_meta={"weights": {"normalized_label": 1.0, "hallucination_penalty": 1.0}},
                provenance=provenance,
            )
        )
    if not rows:
        raise ValueError("No classification rows were parsed from %s" % records_path)
    return rows


def normalize_ip102_dataset(
    raw_dir: Path,
    repo_root: Path,
    license_name: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    split_files = {
        "train": raw_dir / "train.txt",
        "validation": raw_dir / "val.txt",
        "test": raw_dir / "test.txt",
    }
    if not split_files["train"].exists():
        return normalize_classification_directory_dataset(
            raw_dir=raw_dir,
            repo_root=repo_root,
            dataset_name="ip102",
            salt="ip102-fallback",
            pest_mode=True,
            license_name=license_name,
            provenance=provenance,
        )

    label_lookup = {}
    for candidate_name in ["classes.txt", "class_names.txt", "species.txt", "categories.txt"]:
        candidate_path = raw_dir / candidate_name
        if candidate_path.exists():
            labels = [
                line.strip()
                for line in candidate_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            label_lookup = {str(index): parse_ip102_label(label) for index, label in enumerate(labels)}
            break

    rows: List[Dict[str, Any]] = []
    for split_name, split_file in split_files.items():
        if not split_file.exists():
            continue
        for line in split_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            relative_path = parts[0]
            label_id = parts[1] if len(parts) > 1 else None
            image_path = raw_dir / relative_path
            if not image_path.exists():
                raise FileNotFoundError("IP102 image listed in split file does not exist: %s" % image_path)
            fallback_label = parse_ip102_label(image_path.parent.name)
            canonical_label = label_lookup.get(label_id, fallback_label)
            metadata = metadata_with_license(
                {
                    "pest": canonical_label,
                    "label_id": label_id,
                    "original_label": image_path.parent.name or label_id,
                    "normalized_label": canonical_label,
                    "source_image_id": relative_path,
                },
                license_name=license_name,
            )
            rows.append(
                _base_sample(
                    sample_id="ip102-%s" % slugify(relative_path),
                    source_dataset="ip102",
                    task_type=TASK_TYPE_CLASSIFICATION,
                    split=split_name,
                    image_paths=[relative_posix_path(image_path, repo_root)],
                    question="Identify the insect or pest shown in this agricultural image.",
                    target={"answer_text": canonical_label, "canonical_label": canonical_label},
                    metadata=metadata,
                    verifier={"mode": "label", "accepted_labels": [canonical_label]},
                    reward_meta={"weights": {"normalized_label": 1.0, "hallucination_penalty": 1.0}},
                    provenance=provenance,
                )
            )
    if not rows:
        raise ValueError("No IP102 rows were parsed from %s" % raw_dir)
    return rows


def _extract_image_paths(row: Dict[str, Any], raw_dir: Path, repo_root: Path) -> List[str]:
    image_value = row.get("images") or row.get("image_paths") or row.get("image") or row.get("img")
    if isinstance(image_value, list):
        candidates = image_value
    elif image_value:
        candidates = [image_value]
    else:
        raise ValueError("Row is missing image fields.")
    resolved = []
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = raw_dir / candidate_path
        if not candidate_path.exists():
            raise FileNotFoundError("Referenced image path does not exist: %s" % candidate_path)
        resolved.append(relative_posix_path(candidate_path, repo_root))
    return resolved


def normalize_vqa_like_dataset(
    raw_dir: Path,
    repo_root: Path,
    dataset_name: str,
    default_task_type: str = TASK_TYPE_VQA,
    license_name: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    records_path = require_records_file(raw_dir)
    records = read_records(records_path)
    rows = []
    for index, row in enumerate(records):
        image_paths = _extract_image_paths(row, raw_dir=raw_dir, repo_root=repo_root)
        split = normalize_split_name(row.get("split")) or assign_hash_split(
            "%s-%s" % (dataset_name, index), salt=dataset_name
        )
        question = row.get("question") or row.get("query") or row.get("instruction")
        if not question:
            raise ValueError("VQA-like row is missing a question field in %s" % records_path)
        answer = row.get("answer") or row.get("response") or row.get("label")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not answer:
            raise ValueError("VQA-like row is missing an answer field in %s" % records_path)
        task_type = row.get("task_type") or default_task_type
        verifier_mode = "exact_match"
        expected_decision = None
        if task_type == TASK_TYPE_CLARIFY:
            verifier_mode = "clarify"
            expected_decision = row.get("decision") or row.get("expected_decision")
        rows.append(
            _base_sample(
                sample_id="%s-%s" % (dataset_name, row.get("id") or index),
                source_dataset=dataset_name,
                task_type=task_type,
                split=split,
                image_paths=image_paths,
                question=question,
                target={
                    "answer_text": str(answer),
                    "decision": expected_decision,
                    "acceptable_answers": [str(answer)],
                },
                metadata=metadata_with_license(
                    {
                        "crop": row.get("crop"),
                    "disease": row.get("disease"),
                    "pest": row.get("pest"),
                    "source_image_id": row.get("image") or row.get("id") or str(index),
                    "template_origin": row.get("template_origin", "human_or_source_authored"),
                    "benchmark_track": row.get("benchmark_track"),
                    "options": row.get("options"),
                },
                license_name=license_name,
            ),
                verifier={
                    "mode": verifier_mode,
                    "accepted_answers": [str(answer)],
                    "expected_decision": expected_decision,
                    "management_keywords": row.get("management_keywords") or [],
                },
                reward_meta={"weights": row.get("reward_weights") or {}},
                provenance=provenance,
            )
        )
    if not rows:
        raise ValueError("No rows parsed from %s" % records_path)
    return rows


def normalize_consultation_dataset(
    raw_dir: Path,
    repo_root: Path,
    dataset_name: str,
    license_name: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    records_path = require_records_file(raw_dir)
    records = read_records(records_path)
    rows = []
    for index, row in enumerate(records):
        image_paths = _extract_image_paths(row, raw_dir=raw_dir, repo_root=repo_root)
        split = normalize_split_name(row.get("split")) or assign_hash_split(
            "%s-%s" % (dataset_name, index), salt=dataset_name
        )
        diagnosis = str(row.get("diagnosis") or row.get("answer") or row.get("label") or "").strip()
        if not diagnosis:
            raise ValueError("Consultation row is missing diagnosis/answer text in %s" % records_path)
        management_steps = row.get("management_steps") or row.get("management") or []
        if isinstance(management_steps, str):
            management_steps = [management_steps]
        question = row.get("question") or row.get("query") or build_user_message_text(TASK_TYPE_CONSULTATION)
        answer_text = row.get("answer_text")
        if not answer_text:
            answer_text = build_structured_consultation_answer(
                diagnosis=diagnosis,
                management_steps=management_steps,
                uncertainty=str(row.get("uncertainty") or "State uncertainty when evidence is incomplete."),
            )
        rows.append(
            _base_sample(
                sample_id="%s-%s" % (dataset_name, row.get("id") or index),
                source_dataset=dataset_name,
                task_type=row.get("task_type") or TASK_TYPE_CONSULTATION,
                split=split,
                image_paths=image_paths,
                question=question,
                target={
                    "answer_text": answer_text,
                    "canonical_label": normalize_label(diagnosis),
                    "structured": {
                        "diagnosis": diagnosis,
                        "management_steps": management_steps,
                    },
                },
                metadata=metadata_with_license(
                    {
                        "crop": row.get("crop"),
                        "disease": row.get("disease") or diagnosis,
                        "pest": row.get("pest"),
                        "source_image_id": row.get("image") or row.get("id") or str(index),
                        "template_origin": row.get("template_origin", "human_or_source_authored"),
                        "benchmark_track": row.get("benchmark_track"),
                    },
                    license_name=license_name,
                ),
                verifier={
                    "mode": "structured",
                    "accepted_labels": [normalize_label(diagnosis)],
                    "required_sections": row.get("required_sections")
                    or ["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
                    "management_keywords": management_steps,
                    "uncertainty_required": bool(row.get("uncertainty_required", True)),
                },
                reward_meta={
                    "weights": row.get("reward_weights")
                    or {
                        "normalized_label": 1.0,
                        "structured_format": 0.5,
                        "uncertainty_calibration": 0.5,
                        "management_coverage": 0.5,
                        "hallucination_penalty": 1.0,
                    },
                    "structured_output_required": True,
                },
                provenance=provenance,
            )
        )
    if not rows:
        raise ValueError("No rows parsed from %s" % records_path)
    return rows
