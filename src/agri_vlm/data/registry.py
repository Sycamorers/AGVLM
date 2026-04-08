"""Dataset registry and dataset-slot helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from agri_vlm.data.paths import (
    build_template_context,
    compute_subset_tag,
    get_data_root,
    normalize_download_mode,
    normalize_sample_fraction,
    render_path_template,
)
from agri_vlm.utils.io import ensure_dir, load_yaml, read_json, write_json


DOWNLOAD_INFO_NAME = "DOWNLOAD_INFO.json"
MANUAL_STUB_NAME = "MANIFEST.stub.json"


@dataclass(frozen=True)
class DatasetRegistryDefaults:
    """Global dataset prep defaults."""

    download_mode: str = "partial"
    sample_fraction: float = 0.1
    raw_dir_template: str = "{data_root}/raw/{dataset_name}/{subset_tag}"
    interim_path_template: str = "{data_root}/interim/{subset_tag}/{dataset_name}.jsonl"
    processed_dir_template: str = "{data_root}/processed/{dataset_name}/{subset_tag}"
    manifest_dir_template: str = "{data_root}/manifests/{subset_tag}"

    def normalized(self) -> "DatasetRegistryDefaults":
        return DatasetRegistryDefaults(
            download_mode=normalize_download_mode(self.download_mode),
            sample_fraction=normalize_sample_fraction(self.sample_fraction),
            raw_dir_template=self.raw_dir_template,
            interim_path_template=self.interim_path_template,
            processed_dir_template=self.processed_dir_template,
            manifest_dir_template=self.manifest_dir_template,
        )


@dataclass(frozen=True)
class DatasetSpec:
    """Describes one dataset entry."""

    name: str
    task_family: str
    source_type: str
    access: str
    normalizer: str
    manual_url: str = ""
    notes: str = ""
    materializer: str = ""
    hf_repo_id: str = ""
    hf_config_names: tuple[str, ...] = ()
    hf_split_names: tuple[str, ...] = ()
    default_task_type: str = ""
    license_name: str = ""
    eval_only: bool = False
    include_in_sft: bool = True
    include_in_rl: bool = True
    include_in_eval: bool = True
    raw_dir_template: Optional[str] = None
    interim_path_template: Optional[str] = None

    def raw_dir(
        self,
        repo_root: Path,
        defaults: DatasetRegistryDefaults,
        subset_tag: str,
        data_root: Optional[str] = None,
        download_mode: Optional[str] = None,
        sample_fraction: Optional[float] = None,
    ) -> Path:
        actual_data_root = get_data_root(repo_root, data_root)
        context = build_template_context(
            repo_root=repo_root,
            data_root=actual_data_root,
            subset_tag=subset_tag,
            dataset_name=self.name,
            download_mode=download_mode or defaults.download_mode,
            sample_fraction=sample_fraction or defaults.sample_fraction,
        )
        return render_path_template(self.raw_dir_template or defaults.raw_dir_template, context)

    def interim_path(
        self,
        repo_root: Path,
        defaults: DatasetRegistryDefaults,
        subset_tag: str,
        data_root: Optional[str] = None,
        download_mode: Optional[str] = None,
        sample_fraction: Optional[float] = None,
    ) -> Path:
        actual_data_root = get_data_root(repo_root, data_root)
        context = build_template_context(
            repo_root=repo_root,
            data_root=actual_data_root,
            subset_tag=subset_tag,
            dataset_name=self.name,
            download_mode=download_mode or defaults.download_mode,
            sample_fraction=sample_fraction or defaults.sample_fraction,
        )
        return render_path_template(self.interim_path_template or defaults.interim_path_template, context)


@dataclass(frozen=True)
class DatasetRegistry:
    """Resolved dataset registry."""

    defaults: DatasetRegistryDefaults
    specs: Dict[str, DatasetSpec]

    def subset_tag(
        self,
        download_mode: Optional[str] = None,
        sample_fraction: Optional[float] = None,
    ) -> str:
        return compute_subset_tag(
            download_mode or self.defaults.download_mode,
            sample_fraction or self.defaults.sample_fraction,
        )


def _tupled(value: Optional[Iterable[str]]) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()))


def load_dataset_registry(config_path: Path) -> DatasetRegistry:
    payload = load_yaml(config_path)
    defaults_payload = payload.get("defaults", {})
    defaults = DatasetRegistryDefaults(**defaults_payload).normalized()

    specs = {}
    for row in payload.get("datasets", []):
        materializer = row.get("materializer") or row.get("name") or ""
        spec = DatasetSpec(
            name=row["name"],
            task_family=row["task_family"],
            source_type=row.get("source_type", "manual"),
            access=row.get("access", "manual"),
            normalizer=row["normalizer"],
            manual_url=row.get("manual_url", ""),
            notes=row.get("notes", ""),
            materializer=materializer,
            hf_repo_id=row.get("hf_repo_id", ""),
            hf_config_names=_tupled(row.get("hf_config_names")),
            hf_split_names=_tupled(row.get("hf_split_names")),
            default_task_type=row.get("default_task_type", ""),
            license_name=row.get("license_name", ""),
            eval_only=bool(row.get("eval_only", False)),
            include_in_sft=bool(row.get("include_in_sft", True)),
            include_in_rl=bool(row.get("include_in_rl", True)),
            include_in_eval=bool(row.get("include_in_eval", True)),
            raw_dir_template=row.get("raw_dir_template"),
            interim_path_template=row.get("interim_path_template"),
        )
        specs[spec.name] = spec
    return DatasetRegistry(defaults=defaults, specs=specs)


def create_manual_slot(
    spec: DatasetSpec,
    repo_root: Path,
    defaults: DatasetRegistryDefaults,
    subset_tag: str,
    data_root: Optional[str] = None,
    download_mode: Optional[str] = None,
    sample_fraction: Optional[float] = None,
    reason: str = "Manual dataset preparation is required.",
) -> Path:
    raw_dir = ensure_dir(
        spec.raw_dir(
            repo_root=repo_root,
            defaults=defaults,
            subset_tag=subset_tag,
            data_root=data_root,
            download_mode=download_mode,
            sample_fraction=sample_fraction,
        )
    )
    download_info = {
        "dataset_name": spec.name,
        "subset_tag": subset_tag,
        "download_mode": download_mode or defaults.download_mode,
        "sample_fraction": sample_fraction or defaults.sample_fraction,
        "source_type": spec.source_type,
        "access": spec.access,
        "source_repo_id": spec.hf_repo_id or None,
        "materialized": False,
        "manual_required": True,
        "manual_reason": reason,
    }
    write_download_info(raw_dir, download_info)

    readme_path = raw_dir / "README.manual.md"
    stub_path = raw_dir / MANUAL_STUB_NAME
    interim_path = spec.interim_path(
        repo_root=repo_root,
        defaults=defaults,
        subset_tag=subset_tag,
        data_root=data_root,
        download_mode=download_mode,
        sample_fraction=sample_fraction,
    )
    readme_lines = [
        "# %s" % spec.name,
        "",
        "Task family: `%s`" % spec.task_family,
        "Source type: `%s`" % spec.source_type,
        "Access mode: `%s`" % spec.access,
        "Subset tag: `%s`" % subset_tag,
        "Download mode: `%s`" % (download_mode or defaults.download_mode),
        "Sample fraction: `%s`" % (sample_fraction or defaults.sample_fraction),
        "",
        "Manual source URL:",
        spec.manual_url or "Not configured.",
        "",
        "Reason:",
        reason,
        "",
        "Notes:",
        spec.notes or "No additional notes.",
        "",
        "Expected next step:",
        "Place the raw dataset contents inside this directory, then rerun the matching normalization step.",
        "",
        "Expected normalized output:",
        str(interim_path),
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    stub_payload = {
        "dataset_name": spec.name,
        "subset_tag": subset_tag,
        "download_mode": download_mode or defaults.download_mode,
        "sample_fraction": sample_fraction or defaults.sample_fraction,
        "source_type": spec.source_type,
        "access": spec.access,
        "manual_url": spec.manual_url,
        "source_repo_id": spec.hf_repo_id or None,
        "raw_dir": str(raw_dir),
        "expected_interim_path": str(interim_path),
    }
    stub_path.write_text(json.dumps(stub_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return raw_dir


def write_download_info(raw_dir: Path, payload: Dict[str, object]) -> Path:
    raw_dir = ensure_dir(raw_dir)
    path = raw_dir / DOWNLOAD_INFO_NAME
    write_json(path, payload)
    return path


def read_download_info(raw_dir: Path) -> Dict[str, object]:
    path = Path(raw_dir) / DOWNLOAD_INFO_NAME
    if not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Download info payload must be a mapping: %s" % path)
    return payload
