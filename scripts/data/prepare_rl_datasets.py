#!/usr/bin/env python3
"""Prepare RL source manifests without touching SFT training inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image

from agri_vlm.data.hf_download import download_supported_datasets
from agri_vlm.data.pipeline import has_materialized_raw_data, normalize_dataset_spec, resolve_runtime_settings
from agri_vlm.data.registry import create_manual_slot, load_dataset_registry
from agri_vlm.utils.io import ensure_dir, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/data/datasets.yaml")
    parser.add_argument("--rl-config", default="configs/data/rl_build.yaml")
    parser.add_argument("--subset", choices=["full", "debug"], default="debug")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--skip-bad-rows", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def _runtime_from_subset(subset: str) -> Dict[str, Any]:
    if subset == "full":
        return {"download_mode": "full", "sample_fraction": 1.0, "subset_tag": "full"}
    return {"download_mode": "partial", "sample_fraction": 0.001, "subset_tag": "debug"}


def _rl_dataset_names(repo_root: Path, rl_config_path: Path) -> List[str]:
    from agri_vlm.utils.io import load_yaml

    payload = load_yaml(repo_root / rl_config_path)
    return [str(item) for item in payload.get("datasets") or []]


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in read_jsonl(path))


def _verify_manifest_images(rows: Iterable[Dict[str, Any]], repo_root: Path) -> Dict[str, int]:
    stats = Counter()
    for row in rows:
        images = row.get("images") or []
        if not images:
            stats["missing_image_field"] += 1
            continue
        for image_path in images:
            resolved = repo_root / str(image_path)
            if not resolved.exists():
                stats["missing_image_file"] += 1
                continue
            try:
                with Image.open(resolved) as image:
                    image.convert("RGB")
                stats["verified_images"] += 1
            except Exception:
                stats["invalid_image_file"] += 1
    return dict(stats)


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# RL Data Preparation Report",
        "",
        "- Subset: `%s`" % report["subset"],
        "- Subset tag: `%s`" % report["subset_tag"],
        "- Download mode: `%s`" % report["download_mode"],
        "- Sample fraction: `%s`" % report["sample_fraction"],
        "- Data root: `%s`" % report["data_root"],
        "",
        "## Datasets",
        "",
        "| Dataset | Status | Rows | Raw Dir | Interim Path | Notes |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for name, payload in report["datasets"].items():
        lines.append(
            "| %s | %s | %s | `%s` | `%s` | %s |"
            % (
                name,
                payload.get("status"),
                payload.get("rows", 0),
                payload.get("raw_dir", ""),
                payload.get("interim_path", ""),
                str(payload.get("reason") or payload.get("notes") or "").replace("|", "\\|"),
            )
        )
    lines.extend(["", "## Image Verification", ""])
    for key, value in sorted(report.get("image_verification", {}).items()):
        lines.append("- `%s`: `%s`" % (key, value))
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_dataset_registry(repo_root / args.dataset_config)
    subset_runtime = _runtime_from_subset(args.subset)
    runtime = resolve_runtime_settings(
        registry=registry,
        repo_root=repo_root,
        subset_tag=subset_runtime["subset_tag"],
        download_mode=subset_runtime["download_mode"],
        sample_fraction=subset_runtime["sample_fraction"],
        data_root=args.data_root,
    )
    dataset_names = _rl_dataset_names(repo_root, Path(args.rl_config))
    report: Dict[str, Any] = {
        "subset": args.subset,
        "subset_tag": runtime["subset_tag"],
        "download_mode": runtime["download_mode"],
        "sample_fraction": runtime["sample_fraction"],
        "data_root": str(runtime["data_root"]),
        "datasets": {},
        "image_verification": {},
    }

    public_to_download = []
    for dataset_name in dataset_names:
        spec = registry.specs[dataset_name]
        raw_dir = spec.raw_dir(
            repo_root=repo_root,
            defaults=registry.defaults,
            subset_tag=runtime["subset_tag"],
            data_root=str(runtime["data_root"]),
            download_mode=runtime["download_mode"],
            sample_fraction=runtime["sample_fraction"],
        )
        interim_path = spec.interim_path(
            repo_root=repo_root,
            defaults=registry.defaults,
            subset_tag=runtime["subset_tag"],
            data_root=str(runtime["data_root"]),
            download_mode=runtime["download_mode"],
            sample_fraction=runtime["sample_fraction"],
        )
        report["datasets"][dataset_name] = {
            "status": "pending",
            "raw_dir": str(raw_dir),
            "interim_path": str(interim_path),
            "rows": _count_jsonl(interim_path),
        }
        if spec.source_type == "hf_dataset" and (args.force or not has_materialized_raw_data(raw_dir)):
            public_to_download.append(dataset_name)
        elif spec.source_type != "hf_dataset" and not has_materialized_raw_data(raw_dir):
            create_manual_slot(
                spec=spec,
                repo_root=repo_root,
                defaults=registry.defaults,
                subset_tag=runtime["subset_tag"],
                data_root=str(runtime["data_root"]),
                download_mode=runtime["download_mode"],
                sample_fraction=runtime["sample_fraction"],
                reason="Manual or licensed staging is required before RL preparation can include this source.",
            )
            report["datasets"][dataset_name]["status"] = "manual_required"

    if public_to_download:
        download_summary = download_supported_datasets(
            registry=registry,
            repo_root=repo_root,
            subset_tag=runtime["subset_tag"],
            download_mode=runtime["download_mode"],
            sample_fraction=runtime["sample_fraction"],
            data_root=str(runtime["data_root"]),
            dataset_names=public_to_download,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
            dry_run=False,
        )
        for dataset_name, payload in download_summary.items():
            report["datasets"][dataset_name].update(payload)

    unavailable = []
    all_image_stats = Counter()
    for dataset_name in dataset_names:
        spec = registry.specs[dataset_name]
        raw_dir = spec.raw_dir(
            repo_root=repo_root,
            defaults=registry.defaults,
            subset_tag=runtime["subset_tag"],
            data_root=str(runtime["data_root"]),
            download_mode=runtime["download_mode"],
            sample_fraction=runtime["sample_fraction"],
        )
        if not has_materialized_raw_data(raw_dir):
            unavailable.append(dataset_name)
            if report["datasets"][dataset_name].get("status") == "pending":
                report["datasets"][dataset_name]["status"] = "missing_raw"
            continue
        try:
            rows = normalize_dataset_spec(
                spec=spec,
                registry=registry,
                repo_root=repo_root,
                subset_tag=runtime["subset_tag"],
                data_root=str(runtime["data_root"]),
                download_mode=runtime["download_mode"],
                sample_fraction=runtime["sample_fraction"],
            )
        except Exception as exc:
            if not args.skip_bad_rows:
                raise
            unavailable.append(dataset_name)
            report["datasets"][dataset_name].update({"status": "normalization_error", "reason": str(exc)})
            continue
        report["datasets"][dataset_name].update({"status": "normalized", "rows": len(rows)})
        if args.verify_images:
            all_image_stats.update(_verify_manifest_images(rows, repo_root=repo_root))

    report["unavailable_datasets"] = unavailable
    report["image_verification"] = dict(all_image_stats)
    if args.write_report:
        write_json(repo_root / "reports" / "rl_data_prep_report.json", report)
        _write_markdown(report, repo_root / "reports" / "rl_data_prep_report.md")
    if unavailable and not args.skip_unavailable:
        print("unavailable_rl_datasets=%s" % ",".join(unavailable))
        return 2
    print("prepared_rl_datasets=%s unavailable=%s" % (len(dataset_names), len(unavailable)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
