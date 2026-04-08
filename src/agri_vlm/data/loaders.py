"""Dataset loading helpers."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.image import open_image
from agri_vlm.utils.io import read_jsonl


def load_manifest(path: Path) -> List[UnifiedSample]:
    return [UnifiedSample.model_validate(row) for row in read_jsonl(path)]


def load_manifest_dicts(path: Path) -> List[Dict[str, Any]]:
    return [row for row in read_jsonl(path)]


def open_images(paths: Iterable[str]) -> List[Any]:
    return [open_image(Path(path)) for path in paths]


def read_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [row for row in read_jsonl(path)]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                return [row for row in payload["data"] if isinstance(row, dict)]
            raise ValueError("JSON file must be a list or contain a `data` list: %s" % path)
        raise ValueError("Unsupported JSON payload: %s" % path)
    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [dict(row) for row in reader]
    raise ValueError("Unsupported record file format: %s" % path)


def find_first_records_file(root: Path) -> Optional[Path]:
    for candidate in sorted(root.rglob("*")):
        if candidate.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv"}:
            return candidate
    return None


def require_records_file(root: Path) -> Path:
    candidate = find_first_records_file(root)
    if candidate is None:
        raise FileNotFoundError("No JSON/JSONL/CSV/TSV records file found under %s" % root)
    return candidate
