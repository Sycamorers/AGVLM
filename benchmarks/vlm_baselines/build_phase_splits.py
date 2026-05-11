#!/usr/bin/env python3
"""Build explicit SFT and RL benchmark split manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
from typing import Any

from dataset_adapter import (
    ACTIVE_EVAL_MANIFEST,
    ACTIVE_SPLIT_SUMMARY,
    ACTIVE_TRAIN_MANIFEST,
    FALLBACK_SOURCE_MANIFESTS,
    accepted_references,
    distribution_report,
    group_key,
    image_status,
    stratum_key,
    user_prompt,
)
from utils import BENCHMARK_ROOT, REPO_ROOT, ensure_dir, load_yaml, read_jsonl, stable_hash, write_json, write_jsonl
from prediction_parsing import normalize_text


SFT_PHASE = "sft_benchmark"
RL_PHASE = "rl_benchmark"
DEFAULT_RL_TRAIN_MANIFEST = Path("data/manifests/full/rl_manifest.jsonl")
DEFAULT_RL_EVAL_MANIFEST = Path("data/manifests/full/rl_local_holdout_eval.jsonl")
DEFAULT_RL_CONFIG = Path("configs/data/rl_build.yaml")
LAST_RL_FILTER_INFO: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["sft", "rl", "both"], default="both")
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT / "splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-fallback-split", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _clone(row: dict[str, Any], *, phase: str, split: str, policy: str) -> dict[str, Any]:
    cloned = copy.deepcopy(row)
    cloned["phase"] = phase
    cloned["benchmark_phase"] = phase
    cloned["benchmark_split"] = split
    cloned["split"] = row.get("split")
    cloned["benchmark_split_policy"] = policy
    return cloned


def _deterministic_limit(rows: list[dict[str, Any]], *, max_samples: int, seed: int) -> list[dict[str, Any]]:
    if not max_samples or len(rows) <= max_samples:
        return rows
    return sorted(rows, key=lambda row: stable_hash(str(row.get("sample_id") or ""), seed=seed))[:max_samples]


def _split_groups(rows: list[dict[str, Any]], *, val_ratio: float, seed: int) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    groups_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group, group_rows in groups.items():
        stratum = Counter(stratum_key(row) for row in group_rows).most_common(1)[0][0]
        groups_by_stratum[stratum].append(group)
    assignment: dict[str, str] = {}
    for stratum, group_ids in sorted(groups_by_stratum.items()):
        ordered = sorted(group_ids, key=lambda value: stable_hash("%s|%s" % (stratum, value), seed=seed))
        total = sum(len(groups[group]) for group in ordered)
        target_val = max(1, int(round(total * val_ratio))) if total > 1 else 0
        running = 0
        for group in ordered:
            if running < target_val:
                assignment[group] = "val"
                running += len(groups[group])
            else:
                assignment[group] = "test"
    return assignment


def _fallback_source_manifest() -> Path:
    for relative in FALLBACK_SOURCE_MANIFESTS:
        candidate = REPO_ROOT / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No fallback SFT source manifest found under data/manifests.")


def build_sft_phase(
    *,
    output_dir: Path,
    seed: int,
    max_samples: int = 0,
    allow_fallback_split: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    eval_path = REPO_ROOT / ACTIVE_EVAL_MANIFEST
    rows_by_split: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    if eval_path.exists():
        rows = read_jsonl(eval_path)
        rows = _deterministic_limit(rows, max_samples=max_samples, seed=seed)
        for row in rows:
            split = "val" if row.get("split") == "validation" else "test"
            rows_by_split[split].append(
                _clone(row, phase=SFT_PHASE, split=split, policy="active_sft_heldout_disjoint")
            )
        if not rows_by_split["val"]:
            assignment = _split_groups(rows, val_ratio=0.20, seed=seed)
            rows_by_split = {"val": [], "test": []}
            for row in rows:
                split = assignment[group_key(row)]
                rows_by_split[split].append(
                    _clone(row, phase=SFT_PHASE, split=split, policy="active_sft_heldout_group_val")
                )
        return rows_by_split

    if not allow_fallback_split:
        raise FileNotFoundError(
            "Missing primary SFT benchmark manifest %s. Re-run with --allow-fallback-split only for an explicit deterministic fallback."
            % ACTIVE_EVAL_MANIFEST
        )
    source_path = _fallback_source_manifest()
    rows = read_jsonl(source_path)
    rows = _deterministic_limit(rows, max_samples=max_samples, seed=seed)
    assignment = _split_groups(rows, val_ratio=0.20, seed=seed)
    for row in rows:
        split = assignment[group_key(row)]
        rows_by_split[split].append(_clone(row, phase=SFT_PHASE, split=split, policy="fallback_group_stratified"))
    return rows_by_split


def _rl_paths_from_config() -> tuple[Path, Path]:
    config_path = REPO_ROOT / DEFAULT_RL_CONFIG
    if not config_path.exists():
        return REPO_ROOT / DEFAULT_RL_TRAIN_MANIFEST, REPO_ROOT / DEFAULT_RL_EVAL_MANIFEST
    payload = load_yaml(config_path)
    context = {"data_root": str(REPO_ROOT / "data"), "subset_tag": "full"}
    train = Path(str(payload.get("output_path") or DEFAULT_RL_TRAIN_MANIFEST).format(**context))
    holdout = Path(str(payload.get("holdout_output_path") or DEFAULT_RL_EVAL_MANIFEST).format(**context))
    if not train.is_absolute():
        train = REPO_ROOT / train
    if not holdout.is_absolute():
        holdout = REPO_ROOT / holdout
    return train, holdout


def build_rl_phase(
    *,
    output_dir: Path,
    seed: int,
    max_samples: int = 0,
    allow_fallback_split: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    global LAST_RL_FILTER_INFO
    LAST_RL_FILTER_INFO = {}
    train_path, holdout_path = _rl_paths_from_config()
    rows_by_split: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    if holdout_path.exists():
        rows = read_jsonl(holdout_path)
        train_rows = read_jsonl(train_path) if train_path.exists() else []
        train_ids = _sample_ids(train_rows)
        train_groups = _group_keys(train_rows)
        original_count = len(rows)
        rows = [
            row
            for row in rows
            if str(row.get("sample_id") or "") not in train_ids and group_key(row) not in train_groups
        ]
        LAST_RL_FILTER_INFO = {
            "source_holdout_rows": original_count,
            "filtered_train_overlap_rows": original_count - len(rows),
            "filter_policy": "drop exact sample_id or group_key overlap with RL train manifest before benchmark split",
        }
        rows = _deterministic_limit(rows, max_samples=max_samples, seed=seed)
        assignment = _split_groups(rows, val_ratio=0.20, seed=seed)
        for row in rows:
            split = assignment[group_key(row)]
            rows_by_split[split].append(_clone(row, phase=RL_PHASE, split=split, policy="rl_local_holdout_group_split"))
        return rows_by_split

    if not allow_fallback_split:
        raise FileNotFoundError(
            "Missing RL holdout benchmark manifest %s. Re-run with --allow-fallback-split to derive a local holdout from %s."
            % (holdout_path, train_path)
        )
    if not train_path.exists():
        raise FileNotFoundError("Cannot derive RL benchmark split because RL train manifest is missing: %s" % train_path)
    rows = read_jsonl(train_path)
    rows = [row for row in rows if row.get("split") != "test"]
    rows = _deterministic_limit(rows, max_samples=max_samples, seed=seed)
    assignment = _split_groups(rows, val_ratio=0.20, seed=seed)
    for row in rows:
        split = assignment[group_key(row)]
        rows_by_split[split].append(_clone(row, phase=RL_PHASE, split=split, policy="rl_train_fallback_group_holdout"))
    return rows_by_split


def _sample_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("sample_id") or "") for row in rows if row.get("sample_id")}


def _group_keys(rows: list[dict[str, Any]]) -> set[str]:
    return {group_key(row) for row in rows}


def _missing_image_report(rows: list[dict[str, Any]]) -> tuple[int, int, list[dict[str, Any]]]:
    samples = 0
    files = 0
    examples: list[dict[str, Any]] = []
    for row in rows:
        count, missing = image_status(row)
        if count:
            samples += 1
            files += count
            if len(examples) < 20:
                examples.append({"sample_id": row.get("sample_id"), "missing_images": missing})
    return samples, files, examples


def _prompt_leakage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_hits = []
    target_hits = []
    for row in rows:
        prompt = normalize_text(user_prompt(row))
        refs = accepted_references(row)
        for ref in refs:
            normalized_ref = normalize_text(ref)
            if normalized_ref and len(normalized_ref) >= 4 and normalized_ref in prompt:
                prompt_hits.append({"sample_id": row.get("sample_id"), "reference": ref, "task_type": row.get("task_type")})
                break
        assistant_text = " ".join(
            str(block.get("text") or "")
            for message in row.get("messages") or []
            if message.get("role") == "assistant"
            for block in message.get("content") or []
        )
        normalized_assistant = normalize_text(assistant_text)
        if normalized_assistant:
            for ref in refs:
                normalized_ref = normalize_text(ref)
                if normalized_ref and normalized_ref in normalized_assistant:
                    target_hits.append({"sample_id": row.get("sample_id"), "reference": ref, "task_type": row.get("task_type")})
                    break
    return {
        "prompt_leakage_count": len(prompt_hits),
        "prompt_leakage_examples": prompt_hits[:20],
        "ground_truth_leakage_count": len(target_hits),
        "ground_truth_leakage_examples": target_hits[:20],
        "notes": "Prompt hits are heuristic. Yes/no questions may contain candidate labels without leaking the yes/no answer.",
    }


def _public_test_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flagged = []
    for row in rows:
        metadata = row.get("metadata") or {}
        split = normalize_text(str(row.get("split") or metadata.get("split") or ""))
        benchmark_track = normalize_text(str(metadata.get("benchmark_track") or ""))
        if split == "test" or "public test" in benchmark_track or benchmark_track == "test":
            flagged.append({"sample_id": row.get("sample_id"), "source_dataset": row.get("source_dataset"), "split": row.get("split")})
    return flagged


def _examples_by_phase_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    examples: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        phase = str(row.get("phase") or "missing")
        task = str(row.get("task_type") or "missing")
        if len(examples[phase][task]) >= 3:
            continue
        examples[phase][task].append(
            {
                "sample_id": row.get("sample_id"),
                "source_dataset": row.get("source_dataset"),
                "benchmark_split": row.get("benchmark_split"),
                "images": row.get("images"),
                "target_preview": str(row.get("target") or "")[:200],
            }
        )
    return {phase: dict(tasks) for phase, tasks in examples.items()}


def _counts_by_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    crop = Counter()
    disease = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        crop[str(metadata.get("crop") or "missing")] += 1
        disease[str(metadata.get("disease") or "missing")] += 1
    return {"crop": dict(crop.most_common(100)), "disease": dict(disease.most_common(100))}


def _phase_report(
    *,
    phase: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
    train_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    sample_ids = [str(row.get("sample_id") or "") for row in all_rows]
    duplicates = [sample_id for sample_id, count in Counter(sample_ids).items() if sample_id and count > 1]
    missing_samples, missing_files, missing_examples = _missing_image_report(all_rows)
    train_ids = _sample_ids(train_rows)
    eval_ids = _sample_ids(all_rows)
    train_groups = _group_keys(train_rows)
    eval_groups = _group_keys(all_rows)
    public_test = _public_test_rows(all_rows)
    leakage = _prompt_leakage(all_rows)
    return {
        "phase": phase,
        "rows_by_split": {split: len(rows) for split, rows in rows_by_split.items()},
        "rows_by_source_dataset": dict(Counter(str(row.get("source_dataset") or "missing") for row in all_rows)),
        "rows_by_task_type": dict(Counter(str(row.get("task_type") or "missing") for row in all_rows)),
        "rows_by_crop_disease": _counts_by_metadata(all_rows),
        "unique_image_groups": len(_group_keys(all_rows)),
        "missing_image_sample_count": missing_samples,
        "missing_image_file_count": missing_files,
        "missing_image_examples": missing_examples,
        "duplicate_sample_id_count": len(duplicates),
        "duplicate_sample_id_examples": duplicates[:20],
        "train_eval_overlap": {
            "exact_sample_id_count": len(train_ids & eval_ids),
            "group_key_count": len(train_groups & eval_groups),
            "exact_sample_id_examples": sorted(train_ids & eval_ids)[:20],
            "group_key_examples": sorted(train_groups & eval_groups)[:20],
        },
        "prompt_leakage": leakage,
        "contains_public_test_data": bool(public_test),
        "public_test_examples": public_test[:20],
        "multi_image_distribution": dict(Counter(str(len(row.get("images") or [])) for row in all_rows)),
        "examples_by_task_type": _examples_by_phase_task(all_rows).get(phase, {}),
    }


def build_phase_splits(
    *,
    phase: str,
    output_dir: Path,
    seed: int = 42,
    force: bool = False,
    max_samples: int = 0,
    allow_fallback_split: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    requested = ["sft", "rl"] if phase == "both" else [phase]
    expected_files = []
    for requested_phase in requested:
        prefix = "sft" if requested_phase == "sft" else "rl"
        expected_files.extend([output_dir / ("%s_val_manifest.jsonl" % prefix), output_dir / ("%s_test_manifest.jsonl" % prefix)])
    if not force and not max_samples and all(path.exists() for path in expected_files) and (output_dir / "benchmark_split_report.json").exists():
        return json.loads((output_dir / "benchmark_split_report.json").read_text(encoding="utf-8"))

    rows_by_phase_split: dict[str, dict[str, list[dict[str, Any]]]] = {}
    reports: dict[str, Any] = {}
    if "sft" in requested:
        rows_by_split = build_sft_phase(
            output_dir=output_dir,
            seed=seed,
            max_samples=max_samples,
            allow_fallback_split=allow_fallback_split,
        )
        rows_by_phase_split[SFT_PHASE] = rows_by_split
        train_rows = read_jsonl(REPO_ROOT / ACTIVE_TRAIN_MANIFEST) if (REPO_ROOT / ACTIVE_TRAIN_MANIFEST).exists() else []
        reports[SFT_PHASE] = _phase_report(phase=SFT_PHASE, rows_by_split=rows_by_split, train_rows=train_rows)
        write_jsonl(output_dir / "sft_val_manifest.jsonl", rows_by_split["val"])
        write_jsonl(output_dir / "sft_test_manifest.jsonl", rows_by_split["test"])

    if "rl" in requested:
        rows_by_split = build_rl_phase(
            output_dir=output_dir,
            seed=seed,
            max_samples=max_samples,
            allow_fallback_split=allow_fallback_split,
        )
        rows_by_phase_split[RL_PHASE] = rows_by_split
        rl_train_path, _ = _rl_paths_from_config()
        train_rows = read_jsonl(rl_train_path) if rl_train_path.exists() else []
        reports[RL_PHASE] = _phase_report(phase=RL_PHASE, rows_by_split=rows_by_split, train_rows=train_rows)
        reports[RL_PHASE]["rl_holdout_filter"] = dict(LAST_RL_FILTER_INFO)
        write_jsonl(output_dir / "rl_val_manifest.jsonl", rows_by_split["val"])
        write_jsonl(output_dir / "rl_test_manifest.jsonl", rows_by_split["test"])

    legacy_rows_by_split = {}
    if SFT_PHASE in rows_by_phase_split:
        legacy_rows_by_split = rows_by_phase_split[SFT_PHASE]
        write_jsonl(output_dir / "val_manifest.jsonl", legacy_rows_by_split["val"])
        write_jsonl(output_dir / "test_manifest.jsonl", legacy_rows_by_split["test"])

    all_rows = [
        row
        for phase_rows in rows_by_phase_split.values()
        for split_rows in phase_rows.values()
        for row in split_rows
    ]
    report = {
        "seed": seed,
        "max_samples": max_samples,
        "allow_fallback_split": allow_fallback_split,
        "output_dir": str(output_dir),
        "sft_sources": {
            "train_manifest": str(ACTIVE_TRAIN_MANIFEST),
            "eval_manifest": str(ACTIVE_EVAL_MANIFEST),
            "summary": _read_json_if_exists(REPO_ROOT / ACTIVE_SPLIT_SUMMARY),
        },
        "rl_sources": {
            "train_manifest": str(_rl_paths_from_config()[0].relative_to(REPO_ROOT))
            if _rl_paths_from_config()[0].is_relative_to(REPO_ROOT)
            else str(_rl_paths_from_config()[0]),
            "holdout_manifest": str(_rl_paths_from_config()[1].relative_to(REPO_ROOT))
            if _rl_paths_from_config()[1].is_relative_to(REPO_ROOT)
            else str(_rl_paths_from_config()[1]),
        },
        "phases": reports,
        "combined": {
            "rows_by_phase_split": {
                phase_name: {split: len(rows) for split, rows in splits.items()}
                for phase_name, splits in rows_by_phase_split.items()
            },
            "rows_by_source_dataset": dict(Counter(str(row.get("source_dataset") or "missing") for row in all_rows)),
            "rows_by_task_type": dict(Counter(str(row.get("task_type") or "missing") for row in all_rows)),
            "examples_by_phase_task_type": _examples_by_phase_task(all_rows),
        },
    }
    if write_report:
        write_json(output_dir / "benchmark_split_report.json", report)
        (output_dir / "benchmark_split_report.md").write_text(report_markdown(report), encoding="utf-8")
        if legacy_rows_by_split:
            legacy_report = distribution_report(legacy_rows_by_split, extra={"phase_report": reports.get(SFT_PHASE, {})})
            write_json(output_dir / "distribution_report.json", legacy_report)
    return report


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Split Report",
        "",
        "- seed: `%s`" % report.get("seed"),
        "- output directory: `%s`" % report.get("output_dir"),
        "- fallback enabled: `%s`" % report.get("allow_fallback_split"),
        "",
        "## Phase Summary",
        "",
        "| Phase | Val rows | Test rows | Duplicate IDs | Missing images | Sample-ID overlap | Group overlap | Public test rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase, payload in sorted((report.get("phases") or {}).items()):
        rows_by_split = payload.get("rows_by_split") or {}
        overlap = payload.get("train_eval_overlap") or {}
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                phase,
                rows_by_split.get("val", 0),
                rows_by_split.get("test", 0),
                payload.get("duplicate_sample_id_count", 0),
                payload.get("missing_image_sample_count", 0),
                overlap.get("exact_sample_id_count", 0),
                overlap.get("group_key_count", 0),
                len(payload.get("public_test_examples") or []),
            )
        )
    for phase, payload in sorted((report.get("phases") or {}).items()):
        lines.extend(
            [
                "",
                "## %s" % phase,
                "",
                "- rows by split: `%s`" % payload.get("rows_by_split", {}),
                "- rows by source dataset: `%s`" % payload.get("rows_by_source_dataset", {}),
                "- rows by task type: `%s`" % payload.get("rows_by_task_type", {}),
                "- multi-image distribution: `%s`" % payload.get("multi_image_distribution", {}),
                "- prompt leakage count: `%s`" % (payload.get("prompt_leakage") or {}).get("prompt_leakage_count", 0),
                "- ground-truth leakage count: `%s`"
                % (payload.get("prompt_leakage") or {}).get("ground_truth_leakage_count", 0),
                "",
                "### Example Rows",
                "",
            ]
        )
        for task_type, examples in sorted((payload.get("examples_by_task_type") or {}).items()):
            lines.append("- `%s`: `%s`" % (task_type, examples))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report = build_phase_splits(
        phase=args.phase,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        force=args.force,
        max_samples=args.max_samples,
        allow_fallback_split=args.allow_fallback_split,
        write_report=args.write_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
