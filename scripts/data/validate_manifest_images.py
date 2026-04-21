#!/usr/bin/env python3
"""Validate image payloads referenced by JSONL manifests."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

from PIL import Image

from agri_vlm.utils.io import ensure_dir, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest to validate.")
    parser.add_argument("--repo-root", default=None, help="Repository root for relative image paths.")
    parser.add_argument(
        "--valid-output",
        default=None,
        help="Optional JSONL path where rows with all valid images are written.",
    )
    parser.add_argument(
        "--invalid-output",
        default=None,
        help="Optional JSONL path where invalid row diagnostics are written.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON path for aggregate validation counts.",
    )
    parser.add_argument(
        "--mode",
        choices=["decode", "header"],
        default="decode",
        help="Use PIL decode validation or faster file-signature validation.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress to stderr after this many rows or image paths. Disabled by default.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use for validating unique image paths.",
    )
    parser.add_argument(
        "--allow-invalid-with-report",
        action="store_true",
        help="Return success when invalid rows are found only if an invalid-output report is written.",
    )
    return parser.parse_args()


def _resolve_image_path(image_path: str, repo_root: Path) -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _validate_image_header(path: Path) -> Optional[str]:
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError as exc:
        return "%s: %s" % (type(exc).__name__, exc)

    if header.startswith(b"\xff\xd8\xff"):
        return None
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    if header.startswith(b"BM"):
        return None
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return None
    if header.startswith(b"%PDF"):
        return "unsupported image signature: PDF document"
    return "unsupported image signature: %s" % header[:8].hex()


def _validate_image(path: Path, mode: str) -> Optional[str]:
    if not path.exists():
        return "missing file"
    if not path.is_file():
        return "not a file"
    if mode == "header":
        return _validate_image_header(path)
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    return None


def _validate_image_reference(payload: Tuple[str, str, str]) -> Tuple[str, Optional[str], str]:
    image_path, repo_root, mode = payload
    resolved_path = _resolve_image_path(image_path, Path(repo_root))
    return image_path, _validate_image(resolved_path, mode), str(resolved_path)


def _collect_unique_images(manifest_path: Path, progress_every: int) -> List[str]:
    seen = set()
    unique_images: List[str] = []
    for row_count, row in enumerate(read_jsonl(manifest_path), start=1):
        for image_path in row.get("images") or []:
            normalized = str(image_path)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_images.append(normalized)
        if progress_every and row_count % progress_every == 0:
            print(
                "indexed_rows=%s unique_images=%s" % (row_count, len(unique_images)),
                file=sys.stderr,
                flush=True,
            )
    return unique_images


def _validate_unique_images(
    image_paths: List[str],
    repo_root: Path,
    mode: str,
    workers: int,
    progress_every: int,
) -> Dict[str, Dict[str, str]]:
    errors: Dict[str, Dict[str, str]] = {}
    payloads = [(image_path, str(repo_root), mode) for image_path in image_paths]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_validate_image_reference, payloads, chunksize=256)
            for index, (image_path, error, resolved_path) in enumerate(iterator, start=1):
                if error:
                    errors[image_path] = {"error": error, "resolved_path": resolved_path}
                if progress_every and index % progress_every == 0:
                    print(
                        "validated_images=%s invalid_unique_images=%s" % (index, len(errors)),
                        file=sys.stderr,
                        flush=True,
                    )
    else:
        for index, payload in enumerate(payloads, start=1):
            image_path, error, resolved_path = _validate_image_reference(payload)
            if error:
                errors[image_path] = {"error": error, "resolved_path": resolved_path}
            if progress_every and index % progress_every == 0:
                print(
                    "validated_images=%s invalid_unique_images=%s" % (index, len(errors)),
                    file=sys.stderr,
                    flush=True,
                )
    return errors


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest)
    valid_output = Path(args.valid_output) if args.valid_output else None
    invalid_output = Path(args.invalid_output) if args.invalid_output else None
    summary_output = Path(args.summary_output) if args.summary_output else None

    dataset_counts: Dict[str, Dict[str, int]] = {}
    total_rows = 0
    valid_row_count = 0
    invalid_row_count = 0
    total_images = 0
    invalid_image_count = 0
    unique_images = _collect_unique_images(manifest_path, args.progress_every)
    image_error_lookup = _validate_unique_images(
        image_paths=unique_images,
        repo_root=repo_root,
        mode=args.mode,
        workers=max(1, args.workers),
        progress_every=args.progress_every,
    )

    if valid_output:
        ensure_dir(valid_output.parent)
    if invalid_output:
        ensure_dir(invalid_output.parent)
    valid_handle = valid_output.open("w", encoding="utf-8") if valid_output else None
    invalid_handle = invalid_output.open("w", encoding="utf-8") if invalid_output else None

    try:
        for line_number, row in enumerate(read_jsonl(manifest_path), start=1):
            total_rows += 1
            dataset = str(row.get("source_dataset") or "unknown")
            dataset_counts.setdefault(dataset, {"rows": 0, "invalid_rows": 0, "images": 0, "invalid_images": 0})
            dataset_counts[dataset]["rows"] += 1

            image_errors = []
            for image_path in row.get("images") or []:
                total_images += 1
                dataset_counts[dataset]["images"] += 1
                error_payload = image_error_lookup.get(str(image_path))
                if error_payload:
                    invalid_image_count += 1
                    dataset_counts[dataset]["invalid_images"] += 1
                    image_errors.append(
                        {
                            "image": image_path,
                            "resolved_path": error_payload["resolved_path"],
                            "error": error_payload["error"],
                        }
                    )
            if not row.get("images"):
                image_errors.append(
                    {
                        "image": None,
                        "resolved_path": None,
                        "error": "row is missing images",
                    }
                )

            if image_errors:
                invalid_row_count += 1
                dataset_counts[dataset]["invalid_rows"] += 1
                invalid_payload = {
                    "line_number": line_number,
                    "sample_id": row.get("sample_id"),
                    "source_dataset": row.get("source_dataset"),
                    "split": row.get("split"),
                    "task_type": row.get("task_type"),
                    "image_errors": image_errors,
                    "row": row,
                }
                if invalid_handle:
                    invalid_handle.write(json.dumps(invalid_payload, ensure_ascii=False, sort_keys=True))
                    invalid_handle.write("\n")
            else:
                valid_row_count += 1
                if valid_handle:
                    valid_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    valid_handle.write("\n")

            if args.progress_every and total_rows % args.progress_every == 0:
                print(
                    "validated_rows=%s invalid_rows=%s invalid_images=%s"
                    % (total_rows, invalid_row_count, invalid_image_count),
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if valid_handle:
            valid_handle.close()
        if invalid_handle:
            invalid_handle.close()

    summary = {
        "manifest": str(manifest_path),
        "repo_root": str(repo_root),
        "mode": args.mode,
        "unique_images": len(unique_images),
        "invalid_unique_images": len(image_error_lookup),
        "total_rows": total_rows,
        "valid_rows": valid_row_count,
        "invalid_rows": invalid_row_count,
        "total_images": total_images,
        "invalid_images": invalid_image_count,
        "by_dataset": dataset_counts,
    }

    if summary_output:
        write_json(summary_output, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if invalid_row_count and not (args.allow_invalid_with_report and invalid_output):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
