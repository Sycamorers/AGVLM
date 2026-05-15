#!/usr/bin/env python3
"""Prepare pairwise expert preference rows from an RL manifest.

This script does not train a reward model. It only validates and exports
chosen/rejected pairs for a future learned reward or preference model stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def _preference_pair(row: Dict[str, Any]) -> Dict[str, Any] | None:
    preference = row.get("preference") or {}
    chosen = str(preference.get("chosen_response") or "").strip()
    rejected = str(preference.get("rejected_response") or "").strip()
    if not chosen and not rejected:
        return None
    if not chosen or not rejected:
        raise ValueError("Preference row %s must include both chosen_response and rejected_response." % row.get("sample_id"))
    if chosen == rejected:
        raise ValueError("Preference row %s has identical chosen and rejected responses." % row.get("sample_id"))
    return {
        "sample_id": row.get("sample_id"),
        "source_dataset": row.get("source_dataset"),
        "task_type": row.get("task_type"),
        "images": row.get("images") or [],
        "messages": row.get("messages") or [],
        "target": row.get("target") or {},
        "verifier": row.get("verifier") or {},
        "metadata": row.get("metadata") or {},
        "chosen": chosen,
        "rejected": rejected,
        "preference_score": preference.get("preference_score"),
        "preference_rationale": preference.get("preference_rationale"),
        "expert_quality_score": preference.get("expert_quality_score"),
        "agronomic_correctness_score": preference.get("agronomic_correctness_score"),
        "management_usefulness_score": preference.get("management_usefulness_score"),
        "uncertainty_calibration_score": preference.get("uncertainty_calibration_score"),
        "safety_score": preference.get("safety_score"),
    }


def prepare_pairwise_preference_data(
    *,
    manifest_path: Path,
    output_path: Path,
    summary_output_path: Path,
    allow_empty: bool,
) -> Dict[str, Any]:
    output_rows: List[Dict[str, Any]] = []
    input_rows = 0
    for row in read_jsonl(manifest_path):
        input_rows += 1
        UnifiedSample.model_validate(row)
        pair = _preference_pair(row)
        if pair:
            output_rows.append(pair)
    if not output_rows and not allow_empty:
        raise ValueError(
            "No pairwise preference rows found. Add top-level preference.chosen_response and "
            "preference.rejected_response fields, or pass --allow-empty for schema-only checks."
        )
    write_jsonl(output_path, output_rows)
    summary = {
        "manifest": str(manifest_path),
        "output": str(output_path),
        "input_rows": input_rows,
        "pairwise_rows": len(output_rows),
        "format": "messages/images plus chosen/rejected expert preference responses",
        "trains_reward_model": False,
    }
    write_json(summary_output_path, summary)
    return summary


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    summary_output = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary = prepare_pairwise_preference_data(
        manifest_path=Path(args.manifest),
        output_path=output_path,
        summary_output_path=summary_output,
        allow_empty=args.allow_empty,
    )
    print("prepared_pairwise_preference_data=%s rows=%s" % (args.output, summary["pairwise_rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
