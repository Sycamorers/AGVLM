"""Hugging Face dataset acquisition helpers."""

from __future__ import annotations

from collections import Counter
import csv
import io
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple
import zipfile

from PIL import Image

from agri_vlm.data.paths import normalize_download_mode, normalize_sample_fraction
from agri_vlm.data.registry import DatasetRegistry, DatasetSpec, create_manual_slot, write_download_info
from agri_vlm.data.transforms import parse_plant_label
from agri_vlm.utils.io import ensure_dir


def _require_hf_datasets():
    try:
        from datasets import load_dataset, load_dataset_builder
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "The `datasets` package is required for automatic dataset downloads. "
            "Install the project environment first."
        ) from exc
    return load_dataset, load_dataset_builder


def _require_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "The `huggingface_hub` package is required for archive-backed dataset downloads. "
            "Install the project environment first."
        ) from exc
    return hf_hub_download


def _canonical_split(value: str) -> str:
    lowered = str(value).strip().lower()
    if lowered in {"val", "valid"}:
        return "validation"
    if lowered == "dev":
        return "dev"
    return lowered


def _split_target_count(total_examples: Optional[int], download_mode: str, sample_fraction: float) -> Optional[int]:
    normalized_mode = normalize_download_mode(download_mode)
    normalized_fraction = normalize_sample_fraction(sample_fraction)
    if normalized_mode == "full":
        return None
    if total_examples is None:
        return None
    return max(1, int(total_examples * normalized_fraction))


def _first_non_empty(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return value
    return None


def _listify(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _save_image_value(value: Any, path: Path) -> None:
    ensure_dir(path.parent)
    if path.exists() and path.stat().st_size > 0:
        return
    if hasattr(value, "save"):
        value.convert("RGB").save(path)
        return
    if isinstance(value, dict):
        if value.get("bytes"):
            with Image.open(io.BytesIO(value["bytes"])) as image:
                image.convert("RGB").save(path)
            return
        if value.get("path"):
            shutil.copyfile(value["path"], path)
            return
    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as image:
            image.convert("RGB").save(path)
        return
    raise ValueError("Unsupported image value type: %s" % type(value).__name__)


def _decode_label_value(feature: Any, value: Any) -> Any:
    if isinstance(value, list):
        inner_feature = getattr(feature, "feature", feature)
        return [_decode_label_value(inner_feature, item) for item in value]
    names = getattr(feature, "names", None)
    if names and isinstance(value, int):
        return names[value]
    return value


def _resolve_feature(features: Any, key_path: Tuple[str, ...]) -> Any:
    feature = features
    for key in key_path:
        if feature is None:
            return None
        if hasattr(feature, "feature") and not isinstance(feature, dict):
            feature = feature.feature
        if isinstance(feature, dict):
            feature = feature.get(key)
            continue
        try:
            feature = feature[key]
        except Exception:
            return None
    return feature


def _extract_image_pairs(row: Dict[str, Any]) -> List[Tuple[str, Any]]:
    pairs: List[Tuple[str, Any]] = []
    for key, value in row.items():
        lowered = key.lower()
        is_supported_key = lowered == "image" or lowered in {"png", "jpg", "jpeg"}
        if lowered.startswith("image_") and lowered.replace("image_", "").isdigit():
            is_supported_key = True
        if not is_supported_key or value is None:
            continue
        pairs.append((key, value))
    return pairs


def _load_builder_split_sizes(builder: Any, split_names: Iterable[str]) -> Dict[str, Optional[int]]:
    sizes: Dict[str, Optional[int]] = {}
    info_splits = getattr(builder.info, "splits", {}) or {}
    for split_name in split_names:
        split_info = info_splits.get(split_name)
        sizes[split_name] = getattr(split_info, "num_examples", None) if split_info else None
    return sizes


def _iter_dataset_rows(
    repo_id: str,
    config_name: Optional[str],
    split_name: str,
    target_count: Optional[int],
    token: Optional[str],
) -> Iterable[Dict[str, Any]]:
    load_dataset, _ = _require_hf_datasets()
    stream = load_dataset(repo_id, name=config_name, split=split_name, streaming=True, token=token)
    for index, row in enumerate(stream):
        if target_count is not None and index >= target_count:
            break
        yield dict(row)


def _write_jsonl_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _relative_posix_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _normalize_decision_value(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().lower().strip("<>")
    if not normalized:
        return None
    if "respond" in normalized:
        return "respond"
    if "clarify" in normalized or "more info" in normalized or "ask" in normalized:
        return "clarify"
    return normalized


def _download_plantvillage(
    spec: DatasetSpec,
    raw_dir: Path,
    download_mode: str,
    sample_fraction: float,
    token: Optional[str],
) -> Dict[str, Any]:
    _, load_dataset_builder = _require_hf_datasets()
    builder = load_dataset_builder(spec.hf_repo_id, name=spec.hf_config_names[0], token=token)
    split_names = spec.hf_split_names or tuple(builder.info.splits.keys())
    split_sizes = _load_builder_split_sizes(builder, split_names)
    label_feature = _resolve_feature(builder.info.features, ("label",))
    rows: List[Dict[str, Any]] = []
    saved = 0

    for split_name in split_names:
        target_count = _split_target_count(split_sizes.get(split_name), download_mode, sample_fraction)
        for index, row in enumerate(
            _iter_dataset_rows(spec.hf_repo_id, spec.hf_config_names[0], split_name, target_count, token)
        ):
            label_name = str(_decode_label_value(label_feature, row["label"]))
            image_path = raw_dir / "images" / split_name / ("%06d.png" % index)
            _save_image_value(row["image"], image_path)
            crop_name, disease_name = parse_plant_label(label_name)
            rows.append(
                {
                    "id": "%s-%06d" % (split_name, index),
                    "image": str(image_path.relative_to(raw_dir)).replace("\\", "/"),
                    "label": label_name,
                    "split": _canonical_split(split_name),
                    "crop": crop_name,
                    "disease": disease_name,
                    "template_origin": "source_authored",
                }
            )
            saved += 1
    _write_jsonl_rows(raw_dir / "records.jsonl", rows)
    return {"saved_rows": saved, "split_sizes": split_sizes}


def _download_plantdoc(
    spec: DatasetSpec,
    raw_dir: Path,
    download_mode: str,
    sample_fraction: float,
    token: Optional[str],
) -> Dict[str, Any]:
    _, load_dataset_builder = _require_hf_datasets()
    builder = load_dataset_builder(spec.hf_repo_id, token=token)
    split_names = spec.hf_split_names or tuple(builder.info.splits.keys())
    split_sizes = _load_builder_split_sizes(builder, split_names)
    category_feature = _resolve_feature(builder.info.features, ("objects", "category"))
    rows: List[Dict[str, Any]] = []
    saved = 0

    for split_name in split_names:
        target_count = _split_target_count(split_sizes.get(split_name), download_mode, sample_fraction)
        for index, row in enumerate(_iter_dataset_rows(spec.hf_repo_id, None, split_name, target_count, token)):
            categories = _decode_label_value(category_feature, row.get("objects", {}).get("category") or [])
            labels = [str(item) for item in _listify(categories) if str(item).strip()]
            if not labels:
                raise ValueError("PlantDoc row is missing object categories for split %s index %s" % (split_name, index))
            primary_label = Counter(labels).most_common(1)[0][0]
            image_path = raw_dir / "images" / split_name / ("%06d.png" % index)
            _save_image_value(row["image"], image_path)
            rows.append(
                {
                    "id": str(row.get("image_id") or "%s-%06d" % (split_name, index)),
                    "image": str(image_path.relative_to(raw_dir)).replace("\\", "/"),
                    "label": primary_label,
                    "all_labels": labels,
                    "split": _canonical_split(split_name),
                    "template_origin": "source_authored",
                }
            )
            saved += 1
    _write_jsonl_rows(raw_dir / "records.jsonl", rows)
    return {"saved_rows": saved, "split_sizes": split_sizes}


def _download_generic_vqa_records(
    spec: DatasetSpec,
    raw_dir: Path,
    download_mode: str,
    sample_fraction: float,
    token: Optional[str],
) -> Dict[str, Any]:
    _, load_dataset_builder = _require_hf_datasets()
    config_name = spec.hf_config_names[0] if spec.hf_config_names else None
    builder = load_dataset_builder(spec.hf_repo_id, name=config_name, token=token)
    split_names = spec.hf_split_names or tuple(builder.info.splits.keys())
    split_sizes = _load_builder_split_sizes(builder, split_names)
    rows: List[Dict[str, Any]] = []
    saved = 0

    for split_name in split_names:
        target_count = _split_target_count(split_sizes.get(split_name), download_mode, sample_fraction)
        for index, row in enumerate(_iter_dataset_rows(spec.hf_repo_id, config_name, split_name, target_count, token)):
            image_pairs = _extract_image_pairs(row)
            if not image_pairs:
                raise ValueError("%s row is missing image content in split %s index %s" % (spec.name, split_name, index))
            image_paths = []
            for image_index, (_, value) in enumerate(image_pairs, start=1):
                file_name = "%s-%06d-%02d.png" % (split_name, index, image_index)
                image_path = raw_dir / "images" / split_name / file_name
                _save_image_value(value, image_path)
                image_paths.append(str(image_path.relative_to(raw_dir)).replace("\\", "/"))
            row_split = _canonical_split(str(row.get("split") or split_name))
            record = {
                "id": str(row.get("id") or row.get("question_id") or "%s-%06d" % (split_name, index)),
                "images": image_paths,
                "image": image_paths[0],
                "question": str(
                    _first_non_empty(row, ["question", "query", "prompt", "instruction", "dialog_context", "user", "text"])
                    or ""
                ).strip(),
                "answer": _first_non_empty(
                    row, ["answer", "response", "utterance", "assistant_response", "expert_response", "label", "output"]
                ),
                "split": row_split,
                "task_type": spec.default_task_type or row.get("task_type"),
                "template_origin": "source_authored",
            }
            if not record["question"]:
                raise ValueError("%s row is missing a usable question field at split %s index %s" % (spec.name, split_name, index))
            if record["answer"] is None or (isinstance(record["answer"], str) and not record["answer"].strip()):
                raise ValueError("%s row is missing a usable answer field at split %s index %s" % (spec.name, split_name, index))
            record["answer"] = str(record["answer"]).strip()
            for key in ["crop", "disease", "pest", "question_type", "options"]:
                if row.get(key) is not None:
                    record[key] = row.get(key)
            rows.append(record)
            saved += 1
    _write_jsonl_rows(raw_dir / "records.jsonl", rows)
    return {"saved_rows": saved, "split_sizes": split_sizes}


def _download_plantvillage_vqa_archive(
    spec: DatasetSpec,
    raw_dir: Path,
    download_mode: str,
    sample_fraction: float,
    token: Optional[str],
) -> Dict[str, Any]:
    hf_hub_download = _require_hf_hub_download()
    archive_path = hf_hub_download(
        repo_id=spec.hf_repo_id,
        repo_type="dataset",
        filename="PlantVillageVQA.zip",
        token=token,
    )

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        csv_name = next((name for name in names if name.lower().endswith(".csv")), None)
        if csv_name is None:
            raise ValueError("PlantVillageVQA archive is missing a CSV annotations file.")

        image_members = {
            Path(name).name.lower(): name
            for name in names
            if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        if not image_members:
            raise ValueError("PlantVillageVQA archive is missing image files.")

        with archive.open(csv_name) as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
            source_rows = [dict(row) for row in reader]

        if not source_rows:
            raise ValueError("PlantVillageVQA CSV did not contain any rows.")

        split_sizes = Counter(_canonical_split(row.get("split") or "train") for row in source_rows)
        split_targets = {
            split_name: _split_target_count(split_sizes[split_name], download_mode, sample_fraction)
            for split_name in split_sizes
        }

        selected_rows: List[Dict[str, Any]] = []
        selected_counts: Counter[str] = Counter()
        for row in source_rows:
            split_name = _canonical_split(row.get("split") or "train")
            target_count = split_targets[split_name]
            if target_count is not None and selected_counts[split_name] >= target_count:
                continue
            selected_rows.append(row)
            selected_counts[split_name] += 1

        extracted_images: Dict[Tuple[str, str], str] = {}
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(selected_rows):
            split_name = _canonical_split(row.get("split") or "train")
            image_name = Path(row.get("image_path") or row.get("image_id") or "").name
            if not image_name:
                raise ValueError("PlantVillageVQA row %s is missing an image path." % index)
            image_key = (split_name, image_name.lower())
            member_name = image_members.get(image_name.lower())
            if member_name is None:
                raise FileNotFoundError("PlantVillageVQA image not found in archive: %s" % image_name)

            if image_key not in extracted_images:
                image_path = raw_dir / "images" / split_name / image_name
                ensure_dir(image_path.parent)
                with archive.open(member_name) as source, image_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                extracted_images[image_key] = _relative_posix_path(image_path, raw_dir)

            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not question or not answer:
                raise ValueError("PlantVillageVQA row %s is missing question/answer text." % index)

            rows.append(
                {
                    "id": "%s-%06d" % (row.get("image_id") or "plantvillage_vqa", index),
                    "images": [extracted_images[image_key]],
                    "image": extracted_images[image_key],
                    "question": question,
                    "answer": answer,
                    "split": split_name,
                    "task_type": spec.default_task_type or "vqa",
                    "question_type": row.get("question_type"),
                    "template_origin": "source_authored",
                }
            )

    _write_jsonl_rows(raw_dir / "records.jsonl", rows)
    return {"saved_rows": len(rows), "split_sizes": dict(split_sizes)}


def _download_mirage(
    spec: DatasetSpec,
    raw_dir: Path,
    download_mode: str,
    sample_fraction: float,
    token: Optional[str],
) -> Dict[str, Any]:
    _, load_dataset_builder = _require_hf_datasets()
    rows: List[Dict[str, Any]] = []
    split_sizes: Dict[str, Optional[int]] = {}
    saved = 0

    for config_name in spec.hf_config_names:
        builder = load_dataset_builder(spec.hf_repo_id, name=config_name, token=token)
        available_splits = tuple((builder.info.splits or {}).keys())
        requested_splits = spec.hf_split_names or available_splits
        config_splits = [split_name for split_name in requested_splits if split_name in available_splits]
        if not config_splits:
            raise ValueError("No supported MIRAGE splits found for %s" % config_name)
        config_sizes = _load_builder_split_sizes(builder, config_splits)
        split_sizes.update({"%s:%s" % (config_name, key): value for key, value in config_sizes.items()})
        for split_name in config_splits:
            target_count = _split_target_count(config_sizes.get(split_name), download_mode, sample_fraction)
            for index, row in enumerate(_iter_dataset_rows(spec.hf_repo_id, config_name, split_name, target_count, token)):
                image_pairs = _extract_image_pairs(row)
                if not image_pairs:
                    raise ValueError("MIRAGE row is missing images for %s %s index %s" % (config_name, split_name, index))
                image_paths = []
                for image_index, (_, value) in enumerate(image_pairs, start=1):
                    file_name = "%s-%s-%06d-%02d.png" % (config_name, split_name, index, image_index)
                    image_path = raw_dir / "images" / config_name / split_name / file_name
                    _save_image_value(value, image_path)
                    image_paths.append(str(image_path.relative_to(raw_dir)).replace("\\", "/"))

                question = _first_non_empty(row, ["dialog_context", "question", "query", "prompt", "user"])
                answer = _first_non_empty(row, ["utterance", "answer", "response", "assistant_response", "expert_response"])
                if question is None or answer is None:
                    raise ValueError("MIRAGE row is missing question/answer for %s %s index %s" % (config_name, split_name, index))
                decision = _normalize_decision_value(row.get("decision"))
                task_type = "clarify_or_respond" if decision else "consultation"
                benchmark_track = "mmmt" if config_name.lower().startswith("mmmt") else "mmst"
                rows.append(
                    {
                        "id": str(row.get("id") or "%s-%06d" % (config_name, index)),
                        "images": image_paths,
                        "image": image_paths[0],
                        "question": str(question).strip(),
                        "answer": str(answer).strip(),
                        "split": _canonical_split(split_name),
                        "task_type": task_type,
                        "decision": decision,
                        "management_keywords": _listify(
                            _first_non_empty(row, ["known_goal", "goals", "management_keywords"])
                        ),
                        "benchmark_track": benchmark_track,
                        "template_origin": "source_authored",
                    }
                )
                saved += 1
    _write_jsonl_rows(raw_dir / "records.jsonl", rows)
    return {"saved_rows": saved, "split_sizes": split_sizes}


def _materialize_spec(
    spec: DatasetSpec,
    raw_dir: Path,
    download_mode: str,
    sample_fraction: float,
    token: Optional[str],
) -> Dict[str, Any]:
    materializer = spec.materializer or spec.name
    if materializer == "plantvillage":
        return _download_plantvillage(spec, raw_dir, download_mode, sample_fraction, token)
    if materializer == "plantdoc":
        return _download_plantdoc(spec, raw_dir, download_mode, sample_fraction, token)
    if materializer == "plantvillage_vqa":
        return _download_plantvillage_vqa_archive(spec, raw_dir, download_mode, sample_fraction, token)
    if materializer == "mirage":
        return _download_mirage(spec, raw_dir, download_mode, sample_fraction, token)
    raise ValueError("Unsupported materializer for %s: %s" % (spec.name, materializer))


def download_supported_datasets(
    registry: DatasetRegistry,
    repo_root: Path,
    subset_tag: str,
    download_mode: str,
    sample_fraction: float,
    data_root: Optional[str] = None,
    dataset_names: Optional[Iterable[str]] = None,
    token: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Dict[str, Any]]:
    selected_names = list(dataset_names or registry.specs.keys())
    summary: Dict[str, Dict[str, Any]] = {}

    for dataset_name in selected_names:
        spec = registry.specs[dataset_name]
        raw_dir = spec.raw_dir(
            repo_root=repo_root,
            defaults=registry.defaults,
            subset_tag=subset_tag,
            data_root=data_root,
            download_mode=download_mode,
            sample_fraction=sample_fraction,
        )

        if spec.source_type != "hf_dataset":
            create_manual_slot(
                spec=spec,
                repo_root=repo_root,
                defaults=registry.defaults,
                subset_tag=subset_tag,
                data_root=data_root,
                download_mode=download_mode,
                sample_fraction=sample_fraction,
                reason="No verified partial-download adapter is configured for this dataset.",
            )
            summary[dataset_name] = {"status": "manual_required", "raw_dir": str(raw_dir)}
            continue

        if dry_run:
            summary[dataset_name] = {"status": "dry_run", "raw_dir": str(raw_dir), "source_repo_id": spec.hf_repo_id}
            continue

        try:
            ensure_dir(raw_dir)
            materialized = _materialize_spec(spec, raw_dir, download_mode, sample_fraction, token)
            write_download_info(
                raw_dir,
                {
                    "dataset_name": dataset_name,
                    "subset_tag": subset_tag,
                    "download_mode": download_mode,
                    "sample_fraction": sample_fraction,
                    "source_type": spec.source_type,
                    "access": spec.access,
                    "source_repo_id": spec.hf_repo_id,
                    "materialized": True,
                    "manual_required": False,
                    "saved_rows": materialized["saved_rows"],
                    "split_sizes": materialized["split_sizes"],
                },
            )
            summary[dataset_name] = {"status": "downloaded", "raw_dir": str(raw_dir), **materialized}
        except Exception as exc:
            reason = "Automatic download failed: %s" % exc
            create_manual_slot(
                spec=spec,
                repo_root=repo_root,
                defaults=registry.defaults,
                subset_tag=subset_tag,
                data_root=data_root,
                download_mode=download_mode,
                sample_fraction=sample_fraction,
                reason=reason,
            )
            summary[dataset_name] = {
                "status": "manual_required",
                "raw_dir": str(raw_dir),
                "reason": reason,
            }
    return summary
