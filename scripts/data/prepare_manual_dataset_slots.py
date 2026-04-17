#!/usr/bin/env python3
"""Create subset-tagged dataset slots and optional synthetic raw smoke data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from agri_vlm.data.pipeline import resolve_runtime_settings
from agri_vlm.data.registry import create_manual_slot, load_dataset_registry, write_download_info
from agri_vlm.utils.image import save_solid_image
from agri_vlm.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/datasets.yaml")
    parser.add_argument("--download-mode", choices=["partial", "full"], default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--subset-tag", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--with-smoke-data", action="store_true")
    return parser.parse_args()


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _rel(path: Path, base_dir: Path) -> str:
    return str(path.relative_to(base_dir)).replace("\\", "/")


def _seed_public_slot(raw_dir: Path, dataset_name: str, subset_tag: str, download_mode: str, sample_fraction: float) -> None:
    write_download_info(
        raw_dir,
        {
            "dataset_name": dataset_name,
            "subset_tag": subset_tag,
            "download_mode": download_mode,
            "sample_fraction": sample_fraction,
            "source_type": "hf_dataset",
            "materialized": False,
            "manual_required": False,
        },
    )


def _build_smoke_raw_data(raw_dir: Path, dataset_name: str, subset_tag: str, download_mode: str, sample_fraction: float) -> None:
    images_dir = raw_dir / "images"
    ensure_dir(images_dir)

    def image(name: str, color: List[int]) -> str:
        path = images_dir / name
        save_solid_image(path, color)
        return _rel(path, raw_dir)

    if dataset_name == "plantvillage":
        _write_jsonl(
            raw_dir / "records.jsonl",
            [
                {
                    "id": "plantvillage-smoke-1",
                    "image": image("plantvillage_smoke_1.png", [120, 40, 40]),
                    "label": "Tomato___Early_blight",
                    "split": "train",
                    "crop": "tomato",
                    "disease": "early blight",
                }
            ],
        )
    elif dataset_name == "plantdoc":
        _write_jsonl(
            raw_dir / "records.jsonl",
            [
                {
                    "id": "plantdoc-smoke-1",
                    "image": image("plantdoc_smoke_1.png", [80, 120, 40]),
                    "label": "Tomato Early Blight",
                    "all_labels": ["Tomato Early Blight", "Tomato Early Blight"],
                    "split": "train",
                }
            ],
        )
    elif dataset_name == "ip102":
        pest_dir = raw_dir / "images" / "rice_leaf_roller"
        pest_path = pest_dir / "ip102_smoke_1.png"
        save_solid_image(pest_path, [40, 120, 40])
        ensure_dir(raw_dir)
        (raw_dir / "classes.txt").write_text("rice leaf roller\n", encoding="utf-8")
        (raw_dir / "train.txt").write_text(
            "%s 0\n" % _rel(pest_path, raw_dir),
            encoding="utf-8",
        )
    elif dataset_name == "plantvillage_vqa":
        _write_jsonl(
            raw_dir / "records.jsonl",
            [
                {
                    "id": "plantvillage-vqa-smoke-1",
                    "image": image("plantvillage_vqa_smoke_1.png", [40, 40, 120]),
                    "question": "Is the leaf healthy? Answer yes or no.",
                    "answer": "no",
                    "split": "train",
                    "crop": "tomato",
                }
            ],
        )
    elif dataset_name == "agbase":
        _write_jsonl(
            raw_dir / "records.jsonl",
            [
                {
                    "id": "agbase-smoke-1",
                    "image": image("agbase_smoke_1.png", [120, 70, 120]),
                    "question": "Provide an expert diagnosis and management plan.",
                    "diagnosis": "tomato early blight",
                    "management_steps": ["remove affected leaves", "improve airflow"],
                    "uncertainty": "moderate because there is only one image",
                    "split": "train",
                    "crop": "tomato",
                }
            ],
        )
    elif dataset_name == "mirage":
        _write_jsonl(
            raw_dir / "records.jsonl",
            [
                {
                    "id": "mirage-smoke-mmmt-1",
                    "images": [
                        image("mirage_smoke_1.png", [120, 120, 40]),
                        image("mirage_smoke_2.png", [120, 100, 50]),
                    ],
                    "question": "What should I spray right now?",
                    "answer": "Please share a sharper close-up image and the crop before spraying anything.",
                    "decision": "clarify",
                    "task_type": "clarify_or_respond",
                    "benchmark_track": "mmmt",
                    "split": "dev",
                },
                {
                    "id": "mirage-smoke-mmst-1",
                    "image": image("mirage_smoke_3.png", [150, 100, 40]),
                    "question": "Provide a concise diagnosis and management plan.",
                    "answer": "Diagnosis: probable leaf spot\nEvidence: visible spotting on the leaf\nUncertainty: confirm with a closer image and field spread\nManagement: isolate affected leaves; monitor spread\nFollow-up: capture both sides of the leaf",
                    "task_type": "consultation",
                    "management_keywords": ["isolate affected leaves", "monitor spread"],
                    "benchmark_track": "mmst",
                    "split": "dev",
                },
            ],
        )
    elif dataset_name == "agrillava":
        _write_jsonl(
            raw_dir / "records.jsonl",
            [
                {
                    "id": "agrillava-smoke-1",
                    "image": image("agrillava_smoke_1.png", [70, 120, 120]),
                    "question": "What is the most likely issue and what should the grower do next?",
                    "diagnosis": "possible nutrient deficiency",
                    "management_steps": ["check recent fertilization", "avoid immediate pesticide use"],
                    "uncertainty": "high because cultivar and fertilization history are unknown",
                    "split": "train",
                }
            ],
        )
    else:
        raise ValueError("Unsupported smoke dataset: %s" % dataset_name)

    write_download_info(
        raw_dir,
        {
            "dataset_name": dataset_name,
            "subset_tag": subset_tag,
            "download_mode": download_mode,
            "sample_fraction": sample_fraction,
            "source_type": "synthetic_smoke",
            "materialized": True,
            "manual_required": False,
            "saved_rows": 1,
        },
    )


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

    for spec in registry.specs.values():
        raw_dir = spec.raw_dir(
            repo_root=repo_root,
            defaults=registry.defaults,
            subset_tag=runtime["subset_tag"],
            data_root=str(runtime["data_root"]),
            download_mode=runtime["download_mode"],
            sample_fraction=runtime["sample_fraction"],
        )
        ensure_dir(raw_dir)
        if spec.source_type != "hf_dataset" or spec.access == "gated":
            create_manual_slot(
                spec=spec,
                repo_root=repo_root,
                defaults=registry.defaults,
                subset_tag=runtime["subset_tag"],
                data_root=str(runtime["data_root"]),
                download_mode=runtime["download_mode"],
                sample_fraction=runtime["sample_fraction"],
                reason="Manual or authenticated dataset staging is still required for this dataset.",
            )
        else:
            _seed_public_slot(
                raw_dir=raw_dir,
                dataset_name=spec.name,
                subset_tag=runtime["subset_tag"],
                download_mode=runtime["download_mode"],
                sample_fraction=runtime["sample_fraction"],
            )
        print("prepared_slot=%s" % raw_dir)

    if args.with_smoke_data:
        smoke_datasets = [
            "plantvillage",
            "plantdoc",
            "ip102",
            "plantvillage_vqa",
            "agbase",
            "mirage",
            "agrillava",
        ]
        for dataset_name in smoke_datasets:
            spec = registry.specs[dataset_name]
            raw_dir = spec.raw_dir(
                repo_root=repo_root,
                defaults=registry.defaults,
                subset_tag=runtime["subset_tag"],
                data_root=str(runtime["data_root"]),
                download_mode=runtime["download_mode"],
                sample_fraction=runtime["sample_fraction"],
            )
            _build_smoke_raw_data(
                raw_dir=raw_dir,
                dataset_name=dataset_name,
                subset_tag=runtime["subset_tag"],
                download_mode=runtime["download_mode"],
                sample_fraction=runtime["sample_fraction"],
            )
            print("wrote_smoke_raw=%s" % raw_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
