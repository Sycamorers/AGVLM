"""Dataset reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from agri_vlm.data.manifest_io import read_manifest, summarize_manifest
from agri_vlm.data.registry import DatasetRegistry, read_download_info
from agri_vlm.utils.io import write_json


def build_dataset_report(
    registry: DatasetRegistry,
    repo_root: Path,
    subset_tag: str,
    data_root: Optional[str] = None,
    download_mode: Optional[str] = None,
    sample_fraction: Optional[float] = None,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "subset_tag": subset_tag,
        "download_mode": download_mode or registry.defaults.download_mode,
        "sample_fraction": sample_fraction or registry.defaults.sample_fraction,
        "datasets": {},
    }
    for dataset_name, spec in registry.specs.items():
        raw_dir = spec.raw_dir(
            repo_root=repo_root,
            defaults=registry.defaults,
            subset_tag=subset_tag,
            data_root=data_root,
            download_mode=download_mode,
            sample_fraction=sample_fraction,
        )
        interim_path = spec.interim_path(
            repo_root=repo_root,
            defaults=registry.defaults,
            subset_tag=subset_tag,
            data_root=data_root,
            download_mode=download_mode,
            sample_fraction=sample_fraction,
        )
        dataset_entry = {
            "source_type": spec.source_type,
            "access": spec.access,
            "task_family": spec.task_family,
            "eval_only": spec.eval_only,
            "raw_dir": str(raw_dir),
            "interim_path": str(interim_path),
            "manual_url": spec.manual_url,
            "notes": spec.notes,
            "download_info": read_download_info(raw_dir),
            "normalized_rows": 0,
            "summary": {},
            "status": "missing",
        }
        if interim_path.exists():
            rows = read_manifest(interim_path)
            dataset_entry["normalized_rows"] = len(rows)
            dataset_entry["summary"] = summarize_manifest(rows)
            dataset_entry["status"] = "normalized"
        elif dataset_entry["download_info"]:
            dataset_entry["status"] = (
                "manual_required" if dataset_entry["download_info"].get("manual_required") else "downloaded_only"
            )
        report["datasets"][dataset_name] = dataset_entry
    return report


def write_dataset_report(report: Dict[str, object], json_path: Path, markdown_path: Path) -> None:
    write_json(json_path, report)
    lines = [
        "# Dataset Report",
        "",
        "Subset tag: `%s`" % report["subset_tag"],
        "Download mode: `%s`" % report["download_mode"],
        "Sample fraction: `%s`" % report["sample_fraction"],
        "",
        "| Dataset | Status | Source | Access | Rows |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for dataset_name, payload in sorted(report["datasets"].items()):
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                dataset_name,
                payload["status"],
                payload["source_type"],
                payload["access"],
                payload["normalized_rows"],
            )
        )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
