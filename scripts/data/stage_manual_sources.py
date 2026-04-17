#!/usr/bin/env python3
"""Stage manual or authenticated dataset sources into subset-tagged raw directories."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any, Dict, Iterable, List, Optional

from agri_vlm.data.paths import normalize_download_mode, normalize_sample_fraction
from agri_vlm.data.pipeline import resolve_runtime_settings
from agri_vlm.data.registry import create_manual_slot, load_dataset_registry, write_download_info
from agri_vlm.utils.io import ensure_dir


IP102_DRIVE_URL = "https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing"
AGMMU_REPO_ID = "AgMMU/AgMMU_v1"
AGRILLAVA_REPO_ID = "Agri-LLaVA-Anonymous/Agri_LLaVA_VQA_Bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/datasets.yaml")
    parser.add_argument("--download-mode", choices=["partial", "full"], default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--subset-tag", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--datasets", nargs="*", default=["ip102", "agbase", "agrillava"])
    return parser.parse_args()


def _split_target_count(total_examples: int, download_mode: str, sample_fraction: float) -> Optional[int]:
    normalized_mode = normalize_download_mode(download_mode)
    normalized_fraction = normalize_sample_fraction(sample_fraction)
    if normalized_mode == "full":
        return None
    return max(1, int(total_examples * normalized_fraction))


def _write_jsonl_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _normalize_relative_path(value: str) -> str:
    return str(Path(str(value).lstrip("./"))).replace("\\", "/")


def _resolve_existing_relative_path(raw_dir: Path, relative_path: str) -> Optional[str]:
    normalized = _normalize_relative_path(relative_path)
    candidates = [normalized]
    if normalized.startswith("images_ft/"):
        candidates.append(normalized.replace("images_ft/", "", 1))
    for candidate in candidates:
        if (raw_dir / candidate).exists():
            return candidate
    return None


def _find_first_by_name(root: Path, filename: str) -> Optional[Path]:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def _looks_like_ip102_classification_split(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        return ".jpg" in candidate.lower()
    return False


def _require_hf_hub() -> tuple[Any, Any]:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "The `huggingface_hub` package is required for staging Hub-backed manual datasets."
        ) from exc
    return hf_hub_download, list_repo_files


def _require_gdown() -> Any:
    try:
        import gdown
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("The `gdown` package is required for staging IP102 from Google Drive.") from exc
    return gdown


def _require_rarfile() -> Any:
    try:
        import rarfile
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("The `rarfile` package is required for staging Agri-LLaVA images.") from exc
    return rarfile


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON list at %s" % path)
    return [dict(item) for item in payload]


def _extract_tar_if_needed(archive_path: Path, output_dir: Path, marker_name: str) -> None:
    marker_path = output_dir / marker_name
    if marker_path.exists():
        return
    ensure_dir(output_dir)
    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(output_dir)
    marker_path.write_text("ok\n", encoding="utf-8")


def _extract_rar_if_needed(archive_path: Path, output_dir: Path, marker_name: str) -> None:
    marker_path = output_dir / marker_name
    if marker_path.exists():
        return
    ensure_dir(output_dir)
    rarfile = _require_rarfile()
    with rarfile.RarFile(archive_path) as archive:
        archive.extractall(output_dir)
    marker_path.write_text("ok\n", encoding="utf-8")


class _JoinedPartReader(io.RawIOBase):
    def __init__(self, part_paths: List[Path]) -> None:
        self._part_paths = list(part_paths)
        self._index = 0
        self._handle = None

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()

    def _advance(self) -> bool:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._index >= len(self._part_paths):
            return False
        self._handle = self._part_paths[self._index].open("rb")
        self._index += 1
        return True

    def readinto(self, buffer: bytearray) -> int:
        total = 0
        view = memoryview(buffer)
        while total < len(buffer):
            if self._handle is None and not self._advance():
                break
            chunk = self._handle.read(len(buffer) - total)
            if not chunk:
                if not self._advance():
                    break
                continue
            view[total : total + len(chunk)] = chunk
            total += len(chunk)
        return total


def _extract_tar_parts_if_needed(part_paths: List[Path], output_dir: Path, marker_name: str) -> None:
    marker_path = output_dir / marker_name
    if marker_path.exists():
        return
    ensure_dir(output_dir)
    reader = _JoinedPartReader(part_paths)
    try:
        with tarfile.open(
            fileobj=io.BufferedReader(reader, buffer_size=1024 * 1024),
            mode="r|gz",
        ) as archive:
            archive.extractall(output_dir)
    finally:
        reader.close()
    marker_path.write_text("ok\n", encoding="utf-8")


def _write_stage_info(
    raw_dir: Path,
    dataset_name: str,
    source_type: str,
    access: str,
    source_repo_id: Optional[str],
    download_mode: str,
    sample_fraction: float,
    subset_tag: str,
    extra: Dict[str, Any],
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "subset_tag": subset_tag,
        "download_mode": download_mode,
        "sample_fraction": sample_fraction,
        "source_type": source_type,
        "access": access,
        "source_repo_id": source_repo_id,
        "materialized": True,
        "manual_required": False,
    }
    payload.update(extra)
    write_download_info(raw_dir, payload)


def _stage_agbase(
    raw_dir: Path,
    download_mode: str,
    sample_fraction: float,
    subset_tag: str,
    token: Optional[str],
) -> Dict[str, Any]:
    hf_hub_download, list_repo_files = _require_hf_hub()
    json_path = Path(
        hf_hub_download(
            repo_id=AGMMU_REPO_ID,
            filename="agmmu_ft_hf1.json",
            repo_type="dataset",
            token=token,
        )
    )
    part_names = sorted(
        name
        for name in list_repo_files(AGMMU_REPO_ID, repo_type="dataset", token=token)
        if name.startswith("images_ft.tar.gz.part-")
    )
    if not part_names:
        raise FileNotFoundError("AgBase image archive parts were not found in %s" % AGMMU_REPO_ID)
    part_paths = [
        Path(hf_hub_download(repo_id=AGMMU_REPO_ID, filename=name, repo_type="dataset", token=token))
        for name in part_names
    ]
    _extract_tar_parts_if_needed(part_paths, raw_dir, ".agbase_extract_complete")

    source_rows = _load_json_list(json_path)
    target_count = _split_target_count(len(source_rows), download_mode, sample_fraction)
    selected_rows = source_rows[:target_count] if target_count is not None else source_rows

    records: List[Dict[str, Any]] = []
    skipped_no_images = 0
    for index, row in enumerate(selected_rows):
        image_paths = []
        for image_path in row.get("images") or []:
            resolved_path = _resolve_existing_relative_path(raw_dir, str(image_path))
            if resolved_path is not None:
                image_paths.append(resolved_path)
        if not image_paths:
            skipped_no_images += 1
            continue

        qa_payload = row.get("finetuning qa") or {}
        species = str((qa_payload.get("species") or {}).get("a") or "").strip()
        issue = str((qa_payload.get("disease/issue identification") or {}).get("a") or "").strip()
        symptom = str((qa_payload.get("symptom description") or {}).get("a") or "").strip()
        management = str((qa_payload.get("management instructions") or {}).get("a") or "").strip()
        diagnosis = issue or species or symptom or management or "See source-authored answer."

        answer_lines = []
        if species:
            answer_lines.append("Species: %s" % species)
        if issue:
            answer_lines.append("Diagnosis: %s" % issue)
        if symptom:
            answer_lines.append("Symptoms: %s" % symptom)
        if management:
            answer_lines.append("Management: %s" % management)
        answer_text = "\n".join(answer_lines) if answer_lines else diagnosis

        records.append(
            {
                "id": "agbase-%s" % (row.get("faq-id") or index),
                "images": image_paths,
                "image": image_paths[0],
                "question": "Provide an expert agricultural diagnosis and management plan for the issue shown.",
                "diagnosis": diagnosis,
                "management_steps": [management] if management else [],
                "answer_text": answer_text,
                "split": "train",
                "crop": species or None,
                "template_origin": "source_authored",
            }
        )

    _write_jsonl_rows(raw_dir / "records.jsonl", records)
    _write_stage_info(
        raw_dir=raw_dir,
        dataset_name="agbase",
        source_type="manual",
        access="manual",
        source_repo_id=AGMMU_REPO_ID,
        download_mode=download_mode,
        sample_fraction=sample_fraction,
        subset_tag=subset_tag,
        extra={
            "saved_rows": len(records),
            "input_rows": len(selected_rows),
            "skipped_no_image_rows": skipped_no_images,
        },
    )
    return {
        "status": "downloaded",
        "raw_dir": str(raw_dir),
        "saved_rows": len(records),
        "input_rows": len(selected_rows),
        "skipped_no_image_rows": skipped_no_images,
    }


def _stage_agrillava(
    raw_dir: Path,
    download_mode: str,
    sample_fraction: float,
    subset_tag: str,
    token: Optional[str],
) -> Dict[str, Any]:
    hf_hub_download, _ = _require_hf_hub()
    json_path = Path(
        hf_hub_download(
            repo_id=AGRILLAVA_REPO_ID,
            filename="agri_llava_vqa_train.json",
            repo_type="dataset",
            token=token,
        )
    )
    archive_path = Path(
        hf_hub_download(
            repo_id=AGRILLAVA_REPO_ID,
            filename="Img.rar",
            repo_type="dataset",
            token=token,
        )
    )
    _extract_rar_if_needed(archive_path, raw_dir, ".agrillava_extract_complete")

    source_rows = _load_json_list(json_path)
    target_count = _split_target_count(len(source_rows), download_mode, sample_fraction)
    selected_rows = source_rows[:target_count] if target_count is not None else source_rows

    image_root = raw_dir / "Img"
    if not image_root.exists():
        raise FileNotFoundError("Agri-LLaVA image directory was not extracted under %s" % image_root)

    records: List[Dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        image_name = Path(str(row.get("image") or "")).name
        if not image_name:
            raise ValueError("Agri-LLaVA row %s is missing an image filename." % index)
        image_path = image_root / image_name
        if not image_path.exists():
            raise FileNotFoundError("Agri-LLaVA image does not exist: %s" % image_path)
        conversations = list(row.get("conversations") or [])
        if len(conversations) < 2:
            raise ValueError("Agri-LLaVA row %s is missing a conversation pair." % index)
        question = str(conversations[0].get("value") or "").strip()
        answer = str(conversations[1].get("value") or "").strip()
        if not question or not answer:
            raise ValueError("Agri-LLaVA row %s is missing question or answer text." % index)
        records.append(
            {
                "id": "agrillava-%06d" % index,
                "image": "Img/%s" % image_name,
                "question": question,
                "diagnosis": answer,
                "answer_text": answer,
                "management_steps": [],
                "split": "train",
                "template_origin": "source_authored",
            }
        )

    _write_jsonl_rows(raw_dir / "records.jsonl", records)
    _write_stage_info(
        raw_dir=raw_dir,
        dataset_name="agrillava",
        source_type="manual",
        access="manual",
        source_repo_id=AGRILLAVA_REPO_ID,
        download_mode=download_mode,
        sample_fraction=sample_fraction,
        subset_tag=subset_tag,
        extra={"saved_rows": len(records), "input_rows": len(selected_rows)},
    )
    return {
        "status": "downloaded",
        "raw_dir": str(raw_dir),
        "saved_rows": len(records),
        "input_rows": len(selected_rows),
    }


def _stage_ip102(
    raw_dir: Path,
    download_mode: str,
    sample_fraction: float,
    subset_tag: str,
) -> Dict[str, Any]:
    gdown = _require_gdown()
    downloads_dir = raw_dir / "_downloads" / "ip102_drive"
    tar_path = downloads_dir / "Classification" / "ip102_v1.1.tar"
    if not tar_path.exists():
        ensure_dir(downloads_dir)
        gdown.download_folder(
            url=IP102_DRIVE_URL,
            output=str(downloads_dir),
            quiet=False,
            use_cookies=False,
        )
    if not tar_path.exists():
        raise FileNotFoundError("IP102 classification archive was not downloaded into %s" % downloads_dir)

    _extract_tar_if_needed(tar_path, raw_dir, ".ip102_extract_complete")

    classes_txt = _find_first_by_name(downloads_dir, "classes.txt")
    if classes_txt and not (raw_dir / "classes.txt").exists():
        shutil.copyfile(classes_txt, raw_dir / "classes.txt")

    extracted_root = raw_dir / "ip102_v1.1"
    for name in ["train.txt", "val.txt", "test.txt"]:
        preferred_path = extracted_root / name
        source_path = preferred_path if _looks_like_ip102_classification_split(preferred_path) else None
        if source_path is None:
            for candidate in sorted(raw_dir.rglob(name)):
                if candidate.parent == raw_dir:
                    continue
                if _looks_like_ip102_classification_split(candidate):
                    source_path = candidate
                    break
        if source_path is None:
            continue
        rewritten_lines = []
        for line in source_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            image_path = parts[0]
            if "/" not in image_path:
                image_path = "ip102_v1.1/images/%s" % image_path
            rewritten_line = image_path
            if len(parts) > 1:
                rewritten_line = "%s %s" % (image_path, parts[1])
            rewritten_lines.append(rewritten_line)
        (raw_dir / name).write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")

    copied_split_files = [name for name in ["train.txt", "val.txt", "test.txt"] if (raw_dir / name).exists()]
    _write_stage_info(
        raw_dir=raw_dir,
        dataset_name="ip102",
        source_type="manual",
        access="manual",
        source_repo_id=None,
        download_mode=download_mode,
        sample_fraction=sample_fraction,
        subset_tag=subset_tag,
        extra={"copied_split_files": copied_split_files, "classes_file_present": (raw_dir / "classes.txt").exists()},
    )
    return {
        "status": "downloaded",
        "raw_dir": str(raw_dir),
        "copied_split_files": copied_split_files,
        "classes_file_present": (raw_dir / "classes.txt").exists(),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_dataset_registry(repo_root / args.config)
    runtime = resolve_runtime_settings(
        registry=registry,
        repo_root=repo_root,
        subset_tag=args.subset_tag,
        download_mode=args.download_mode,
        sample_fraction=args.fraction,
        data_root=args.data_root,
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    summary: Dict[str, Dict[str, Any]] = {}

    for dataset_name in args.datasets:
        spec = registry.specs[dataset_name]
        raw_dir = spec.raw_dir(
            repo_root=repo_root,
            defaults=registry.defaults,
            subset_tag=runtime["subset_tag"],
            data_root=str(runtime["data_root"]),
            download_mode=runtime["download_mode"],
            sample_fraction=runtime["sample_fraction"],
        )
        ensure_dir(raw_dir)
        try:
            if dataset_name == "agbase":
                summary[dataset_name] = _stage_agbase(
                    raw_dir=raw_dir,
                    download_mode=runtime["download_mode"],
                    sample_fraction=runtime["sample_fraction"],
                    subset_tag=runtime["subset_tag"],
                    token=token,
                )
            elif dataset_name == "agrillava":
                summary[dataset_name] = _stage_agrillava(
                    raw_dir=raw_dir,
                    download_mode=runtime["download_mode"],
                    sample_fraction=runtime["sample_fraction"],
                    subset_tag=runtime["subset_tag"],
                    token=token,
                )
            elif dataset_name == "ip102":
                summary[dataset_name] = _stage_ip102(
                    raw_dir=raw_dir,
                    download_mode=runtime["download_mode"],
                    sample_fraction=runtime["sample_fraction"],
                    subset_tag=runtime["subset_tag"],
                )
            else:
                raise ValueError("Unsupported manual staging target: %s" % dataset_name)
        except Exception as exc:
            reason = "Manual staging failed: %s" % exc
            create_manual_slot(
                spec=spec,
                repo_root=repo_root,
                defaults=registry.defaults,
                subset_tag=runtime["subset_tag"],
                data_root=str(runtime["data_root"]),
                download_mode=runtime["download_mode"],
                sample_fraction=runtime["sample_fraction"],
                reason=reason,
            )
            summary[dataset_name] = {"status": "manual_required", "raw_dir": str(raw_dir), "reason": reason}

    print(
        json.dumps(
            {
                "subset_tag": runtime["subset_tag"],
                "download_mode": runtime["download_mode"],
                "sample_fraction": runtime["sample_fraction"],
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
