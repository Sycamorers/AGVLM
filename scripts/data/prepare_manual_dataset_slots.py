#!/usr/bin/env python3
"""Create manual dataset slots and optional synthetic smoke data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from agri_vlm.data.registry import create_manual_slot, load_dataset_registry
from agri_vlm.data.manifest_io import write_manifest
from agri_vlm.utils.image import save_solid_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data/datasets.yaml",
        help="Dataset registry configuration file.",
    )
    parser.add_argument(
        "--with-smoke-data",
        action="store_true",
        help="Create tiny synthetic interim manifests and images for smoke testing.",
    )
    return parser.parse_args()


def _sample(
    sample_id: str,
    dataset: str,
    task_type: str,
    split: str,
    image_path: str,
    prompt: str,
    answer: str,
    verifier: Dict[str, object],
    metadata: Dict[str, object],
    target_extra: Dict[str, object] = None,
    reward_meta: Dict[str, object] = None,
) -> Dict[str, object]:
    target = {"answer_text": answer}
    target.update(target_extra or {})
    return {
        "sample_id": sample_id,
        "source_dataset": dataset,
        "task_type": task_type,
        "split": split,
        "images": [image_path],
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an agricultural assistant focused on crop disease and pest analysis.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "target": target,
        "metadata": metadata,
        "verifier": verifier,
        "reward_meta": reward_meta or {"weights": {}},
    }


def build_smoke_rows(repo_root: Path) -> Dict[str, List[Dict[str, object]]]:
    smoke_root = repo_root / "data" / "raw" / "_smoke"
    image_specs = {
        "plantdoc_leaf.png": (120, 40, 40),
        "ip102_insect.png": (40, 120, 40),
        "vqa_leaf.png": (40, 40, 120),
        "mirage_leaf.png": (120, 120, 40),
        "agbase_leaf.png": (120, 70, 120),
        "agrillava_leaf.png": (70, 120, 120),
        "agmmu_leaf.png": (90, 90, 90),
    }
    for file_name, color in image_specs.items():
        save_solid_image(smoke_root / file_name, color)

    def rel(name: str) -> str:
        return "data/raw/_smoke/%s" % name

    return {
        "plantdoc": [
            _sample(
                sample_id="plantdoc-smoke-1",
                dataset="plantdoc",
                task_type="classification",
                split="train",
                image_path=rel("plantdoc_leaf.png"),
                prompt="Identify the disease in this field image.",
                answer="tomato early blight",
                verifier={"mode": "label", "accepted_labels": ["tomato early blight"]},
                metadata={"crop": "tomato", "disease": "early blight", "source_image_id": "plantdoc_leaf.png"},
                target_extra={"canonical_label": "tomato early blight"},
                reward_meta={"weights": {"normalized_label": 1.0}},
            )
        ],
        "ip102": [
            _sample(
                sample_id="ip102-smoke-1",
                dataset="ip102",
                task_type="classification",
                split="train",
                image_path=rel("ip102_insect.png"),
                prompt="Identify the insect or pest in this image.",
                answer="rice leaf roller",
                verifier={"mode": "label", "accepted_labels": ["rice leaf roller"]},
                metadata={"pest": "rice leaf roller", "source_image_id": "ip102_insect.png"},
                target_extra={"canonical_label": "rice leaf roller"},
                reward_meta={"weights": {"normalized_label": 1.0}},
            )
        ],
        "plantvillage_vqa": [
            _sample(
                sample_id="plantvillage-vqa-smoke-1",
                dataset="plantvillage_vqa",
                task_type="vqa",
                split="train",
                image_path=rel("vqa_leaf.png"),
                prompt="Is the leaf healthy? Answer yes or no.",
                answer="no",
                verifier={"mode": "exact_match", "accepted_answers": ["no"]},
                metadata={"crop": "tomato", "source_image_id": "vqa_leaf.png"},
                reward_meta={"weights": {"exact_match": 1.0}},
            )
        ],
        "mirage": [
            _sample(
                sample_id="mirage-smoke-1",
                dataset="mirage",
                task_type="clarify_or_respond",
                split="validation",
                image_path=rel("mirage_leaf.png"),
                prompt="What should I spray right now?",
                answer="Please upload a sharper close-up image and share the crop and affected organ before spraying anything.",
                verifier={"mode": "clarify", "expected_decision": "clarify", "accepted_answers": ["please upload a sharper close-up image and share the crop and affected organ before spraying anything"]},
                metadata={"benchmark_track": "mmmt", "source_image_id": "mirage_leaf.png"},
                target_extra={"decision": "clarify"},
                reward_meta={"weights": {"clarify_vs_respond": 1.0}},
            ),
            _sample(
                sample_id="mirage-smoke-2",
                dataset="mirage",
                task_type="consultation",
                split="validation",
                image_path=rel("mirage_leaf.png"),
                prompt="Provide a concise diagnosis and management plan.",
                answer="Diagnosis: probable leaf spot\nEvidence: visible spotting on the leaf\nUncertainty: confirm with a closer image and field spread\nManagement: isolate affected leaves; monitor spread; avoid unnecessary broad-spectrum spraying\nFollow-up: capture both sides of the leaf and nearby plants",
                verifier={
                    "mode": "structured",
                    "accepted_labels": ["probable leaf spot"],
                    "required_sections": ["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
                    "management_keywords": ["isolate affected leaves", "monitor spread"],
                    "uncertainty_required": True,
                },
                metadata={"benchmark_track": "mmst", "source_image_id": "mirage_leaf.png"},
                target_extra={"canonical_label": "probable leaf spot"},
                reward_meta={"weights": {"structured_format": 1.0, "management_coverage": 1.0}},
            ),
        ],
        "agbase": [
            _sample(
                sample_id="agbase-smoke-1",
                dataset="agbase",
                task_type="consultation",
                split="train",
                image_path=rel("agbase_leaf.png"),
                prompt="Provide an expert-style diagnosis and management recommendation.",
                answer="Diagnosis: tomato early blight\nEvidence: dark lesions consistent with early blight\nUncertainty: moderate because only one image is available\nManagement: remove heavily affected leaves; improve airflow; avoid overhead irrigation\nFollow-up: verify stem symptoms and disease spread in neighboring plants",
                verifier={
                    "mode": "structured",
                    "accepted_labels": ["tomato early blight"],
                    "required_sections": ["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
                    "management_keywords": ["remove heavily affected leaves", "improve airflow"],
                    "uncertainty_required": True,
                },
                metadata={"crop": "tomato", "disease": "early blight", "source_image_id": "agbase_leaf.png"},
                target_extra={"canonical_label": "tomato early blight"},
                reward_meta={"weights": {"structured_format": 1.0, "management_coverage": 1.0}},
            )
        ],
        "agrillava": [
            _sample(
                sample_id="agrillava-smoke-1",
                dataset="agrillava",
                task_type="consultation",
                split="train",
                image_path=rel("agrillava_leaf.png"),
                prompt="What is the most likely issue and what should the grower do next?",
                answer="Diagnosis: possible nutrient deficiency\nEvidence: diffuse discoloration rather than distinct lesions\nUncertainty: high because cultivar and recent fertilization history are unknown\nManagement: check recent fertilization; inspect newer and older leaves separately; avoid immediate pesticide use\nFollow-up: share soil, irrigation, and fertilization context",
                verifier={
                    "mode": "structured",
                    "required_sections": ["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
                    "management_keywords": ["check recent fertilization", "avoid immediate pesticide use"],
                    "uncertainty_required": True,
                },
                metadata={"source_image_id": "agrillava_leaf.png"},
                reward_meta={"weights": {"structured_format": 1.0, "uncertainty_calibration": 1.0}},
            )
        ],
        "agmmu": [
            _sample(
                sample_id="agmmu-smoke-1",
                dataset="agmmu",
                task_type="vqa",
                split="test",
                image_path=rel("agmmu_leaf.png"),
                prompt="Question: Which option best matches the symptom? Options: A) healthy B) leaf spot C) pest-free D) nutrient burn",
                answer="B) leaf spot",
                verifier={"mode": "exact_match", "accepted_answers": ["B) leaf spot", "leaf spot"]},
                metadata={"benchmark_track": "agmmu", "source_image_id": "agmmu_leaf.png"},
                reward_meta={"weights": {"exact_match": 1.0}},
            )
        ],
        "plantvillage": [
            _sample(
                sample_id="plantvillage-smoke-1",
                dataset="plantvillage",
                task_type="classification",
                split="train",
                image_path=rel("plantdoc_leaf.png"),
                prompt="Identify the crop issue in this image.",
                answer="tomato early blight",
                verifier={"mode": "label", "accepted_labels": ["tomato early blight"]},
                metadata={"crop": "tomato", "disease": "early blight", "source_image_id": "plantdoc_leaf.png"},
                target_extra={"canonical_label": "tomato early blight"},
                reward_meta={"weights": {"normalized_label": 1.0}},
            )
        ],
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_dataset_registry(repo_root / args.config)
    for spec in registry.values():
        create_manual_slot(spec, repo_root=repo_root)
        print("prepared_slot=%s" % spec.raw_dir)

    if args.with_smoke_data:
        smoke_rows = build_smoke_rows(repo_root)
        for dataset_name, rows in smoke_rows.items():
            output_path = repo_root / "data" / "interim" / ("%s.jsonl" % dataset_name)
            write_manifest(output_path, rows)
            print("wrote_smoke_manifest=%s rows=%s" % (output_path, len(rows)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
