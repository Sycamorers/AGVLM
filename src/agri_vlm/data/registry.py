"""Dataset registry and manual-slot helpers."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict

from agri_vlm.constants import MANUAL_DATASET_PLACEHOLDER_PREFIX
from agri_vlm.utils.io import ensure_dir, load_yaml


@dataclass(frozen=True)
class DatasetSpec:
    """Describes one dataset slot in the repository."""

    name: str
    task_family: str
    raw_dir: str
    interim_path: str
    download_mode: str
    manual_url: str
    notes: str


def load_dataset_registry(config_path: Path) -> Dict[str, DatasetSpec]:
    payload = load_yaml(config_path)
    specs = {}
    for row in payload.get("datasets", []):
        spec = DatasetSpec(**row)
        specs[spec.name] = spec
    return specs


def create_manual_slot(spec: DatasetSpec, repo_root: Path) -> Path:
    raw_dir = ensure_dir(repo_root / spec.raw_dir)
    readme_path = raw_dir / "README.manual.md"
    manifest_path = raw_dir / "MANIFEST.stub.json"

    readme_lines = [
        "# %s" % spec.name,
        "",
        "Task family: `%s`" % spec.task_family,
        "Download mode: `%s`" % spec.download_mode,
        "",
        "Manual source URL:",
        spec.manual_url,
        "",
        "Notes:",
        spec.notes,
        "",
        "Expected next step:",
        "Place the raw dataset contents inside this directory, then run the matching normalization script.",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    stub_payload = {
        "name": spec.name,
        "download_mode": spec.download_mode,
        "manual_url": spec.manual_url,
        "placeholder_url": spec.manual_url.startswith(MANUAL_DATASET_PLACEHOLDER_PREFIX),
        "raw_dir": spec.raw_dir,
        "interim_path": spec.interim_path,
    }
    manifest_path.write_text(json.dumps(stub_payload, indent=2) + "\n", encoding="utf-8")
    return raw_dir
