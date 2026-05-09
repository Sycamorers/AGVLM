#!/usr/bin/env python3
"""Audit an RL manifest before GRPO training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from agri_vlm.utils.io import ensure_dir, read_jsonl, write_json
from agri_vlm.utils.text import normalize_label, word_count


SUPPORTED_VERIFIER_MODES = {"label", "exact_match", "synonym", "structured", "clarify"}
CRITICAL_ISSUES = {
    "no_samples",
    "duplicate_sample_ids",
    "missing_image_paths",
    "image_paths_not_exist",
    "unsupported_verifier_mode",
    "clarify_without_expected_decision",
    "no_applicable_reward_module",
}
FIELD_COVERAGE_NAMES = [
    "target.decision",
    "verifier.expected_decision",
    "management_keywords",
    "uncertainty_required",
    "forbidden_claims",
    "required_sections",
    "accepted_labels",
    "accepted_answers",
    "reward_meta.weights",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-examples-per-issue", type=int, default=20)
    parser.add_argument("--fail-on-critical", action="store_true")
    return parser.parse_args()


def _percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p95": _percentile(values, 0.95),
        "max": float(max(values)),
    }


def _target_text(row: Dict[str, Any]) -> str:
    target = row.get("target") or {}
    if target.get("answer_text"):
        return str(target["answer_text"])
    if target.get("canonical_label"):
        return str(target["canonical_label"])
    if target.get("canonical_labels"):
        return " ".join(str(item) for item in target["canonical_labels"])
    if target.get("acceptable_answers"):
        return str(target["acceptable_answers"][0])
    if target.get("decision"):
        return str(target["decision"])
    if target.get("structured"):
        return json.dumps(target["structured"], ensure_ascii=False, sort_keys=True)
    return ""


def _list_value(payload: Dict[str, Any], key: str) -> List[Any]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _dict_value(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def applicable_reward_modules(row: Dict[str, Any]) -> List[str]:
    target = _dict_value(row, "target")
    verifier = _dict_value(row, "verifier")
    modules = []
    if target.get("answer_text") or target.get("acceptable_answers") or verifier.get("accepted_answers"):
        modules.append("exact_match")
    if target.get("canonical_label") or target.get("canonical_labels") or verifier.get("accepted_labels"):
        modules.append("normalized_label")
    if verifier.get("synonyms") or target.get("canonical_label"):
        modules.append("synonym_match")
    if verifier.get("required_sections") or target.get("structured"):
        modules.append("structured_format")
    if verifier.get("uncertainty_required"):
        modules.append("uncertainty_calibration")
    if target.get("decision") or verifier.get("expected_decision"):
        modules.append("clarify_vs_respond")
    if verifier.get("management_keywords"):
        modules.append("management_coverage")
    if verifier.get("forbidden_claims") or verifier.get("uncertainty_required"):
        modules.append("hallucination_penalty")
    return modules


class IssueCollector:
    def __init__(self, max_examples: int) -> None:
        self.max_examples = max_examples
        self.counts: Counter[str] = Counter()
        self.examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add(self, name: str, row: Dict[str, Any], line_number: int, reason: str) -> None:
        self.counts[name] += 1
        if len(self.examples[name]) >= self.max_examples:
            return
        self.examples[name].append(
            {
                "line": line_number,
                "sample_id": row.get("sample_id"),
                "reason": reason,
            }
        )

    def report(self) -> Dict[str, Dict[str, Any]]:
        names = sorted(set(self.counts) | set(CRITICAL_ISSUES))
        return {
            name: {"count": int(self.counts.get(name, 0)), "examples": self.examples.get(name, [])}
            for name in names
        }


def _coverage(count: int, total: int) -> Dict[str, float]:
    percentage = (count / float(total) * 100.0) if total else 0.0
    return {"count": int(count), "percentage": percentage}


def _message_has_prompt(row: Dict[str, Any]) -> bool:
    messages = row.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        for content in message.get("content") or []:
            if content.get("type") == "text" and str(content.get("text") or "").strip():
                return True
    return False


def _is_management_related(row: Dict[str, Any]) -> bool:
    target = _dict_value(row, "target")
    verifier = _dict_value(row, "verifier")
    reward_meta = _dict_value(row, "reward_meta")
    structured = _dict_value(target, "structured")
    required_sections = " ".join(str(item).lower() for item in _list_value(verifier, "required_sections"))
    return (
        row.get("task_type") == "consultation"
        or bool(reward_meta.get("structured_output_required"))
        or "management" in required_sections
        or bool(structured.get("management_steps"))
    )


def _is_uncertainty_related(row: Dict[str, Any]) -> bool:
    target_text = _target_text(row).lower()
    verifier = _dict_value(row, "verifier")
    required_sections = " ".join(str(item).lower() for item in _list_value(verifier, "required_sections"))
    return "uncertainty" in required_sections or "uncertain" in target_text or "not enough evidence" in target_text


def _target_verifier_mismatch(row: Dict[str, Any]) -> str:
    target = _dict_value(row, "target")
    verifier = _dict_value(row, "verifier")
    target_decision = target.get("decision")
    expected_decision = verifier.get("expected_decision")
    if target_decision and expected_decision and target_decision != expected_decision:
        return "target.decision=%s differs from verifier.expected_decision=%s" % (
            target_decision,
            expected_decision,
        )
    canonical_labels = set()
    if target.get("canonical_label"):
        canonical_labels.add(normalize_label(str(target["canonical_label"])))
    canonical_labels.update(normalize_label(str(item)) for item in target.get("canonical_labels") or [])
    accepted_labels = {normalize_label(str(item)) for item in verifier.get("accepted_labels") or []}
    if canonical_labels and accepted_labels and not canonical_labels.intersection(accepted_labels):
        return "target canonical label does not overlap verifier.accepted_labels"
    return ""


def audit_manifest(
    manifest_path: Path,
    *,
    repo_root: Path,
    max_examples_per_issue: int = 20,
) -> Dict[str, Any]:
    rows = list(read_jsonl(manifest_path))
    total = len(rows)
    issues = IssueCollector(max_examples=max_examples_per_issue)
    if total == 0:
        issues.add("no_samples", {}, 0, "manifest has no rows")

    by_dataset: Counter[str] = Counter()
    by_task_type: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_verifier_mode: Counter[str] = Counter()
    by_image_count: Counter[str] = Counter()
    answer_lengths: List[int] = []
    field_counts: Counter[str] = Counter()
    module_counts: Counter[str] = Counter()
    seen_ids: Dict[str, int] = {}

    for line_number, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id") or "")
        by_dataset[str(row.get("source_dataset") or "")] += 1
        by_task_type[str(row.get("task_type") or "")] += 1
        by_split[str(row.get("split") or "")] += 1
        target = _dict_value(row, "target")
        verifier = _dict_value(row, "verifier")
        reward_meta = _dict_value(row, "reward_meta")
        mode = str(verifier.get("mode") or "")
        by_verifier_mode[mode] += 1
        images = row.get("images") or []
        image_count = len(images) if isinstance(images, list) else 0
        by_image_count[str(image_count)] += 1
        answer_lengths.append(word_count(_target_text(row)))

        if sample_id in seen_ids:
            issues.add("duplicate_sample_ids", row, line_number, "first seen on line %s" % seen_ids[sample_id])
        else:
            seen_ids[sample_id] = line_number

        if not isinstance(images, list) or not images or any(not str(path).strip() for path in images):
            issues.add("missing_image_paths", row, line_number, "row has no image path or an empty image path")
        else:
            for image_path in images:
                if not (repo_root / str(image_path)).exists():
                    issues.add("image_paths_not_exist", row, line_number, str(image_path))
                    break
        if image_count > 1:
            issues.add("multi_image_in_single_image_rl", row, line_number, "image_count=%s" % image_count)

        if not _message_has_prompt(row):
            issues.add("missing_prompt_messages", row, line_number, "row has no nonempty text prompt")
        if not _target_text(row):
            issues.add("missing_target", row, line_number, "target cannot be rendered to text")
        if row.get("split") == "test":
            issues.add("test_split_in_rl_manifest", row, line_number, "test split must be excluded")
        if mode not in SUPPORTED_VERIFIER_MODES:
            issues.add("unsupported_verifier_mode", row, line_number, "mode=%s" % mode)
        if mode == "clarify" and not verifier.get("expected_decision"):
            issues.add("clarify_without_expected_decision", row, line_number, "missing verifier.expected_decision")
        if row.get("task_type") == "clarify_or_respond" and not (
            target.get("decision") or verifier.get("expected_decision")
        ):
            issues.add("clarify_task_without_decision", row, line_number, "missing target/verifier decision")
        if row.get("task_type") == "consultation" and not target.get("structured"):
            issues.add("consultation_without_structured_target", row, line_number, "missing target.structured")
        if row.get("task_type") == "consultation" and not verifier.get("required_sections"):
            issues.add("consultation_without_required_sections", row, line_number, "missing required_sections")
        if _is_management_related(row) and not verifier.get("management_keywords"):
            issues.add("management_without_keywords", row, line_number, "management-related row has no keywords")
        if _is_uncertainty_related(row) and not verifier.get("uncertainty_required"):
            issues.add("uncertainty_without_flag", row, line_number, "uncertainty-related row has no flag")
        if answer_lengths[-1] > 80:
            issues.add("extremely_long_target_answers", row, line_number, "answer_words=%s" % answer_lengths[-1])
        mismatch = _target_verifier_mismatch(row)
        if mismatch:
            issues.add("target_verifier_mismatch", row, line_number, mismatch)

        modules = applicable_reward_modules(row)
        if not modules:
            issues.add("no_applicable_reward_module", row, line_number, "no reward module applies")
        for module_name in modules:
            module_counts[module_name] += 1

        if target.get("decision"):
            field_counts["target.decision"] += 1
        if verifier.get("expected_decision"):
            field_counts["verifier.expected_decision"] += 1
        if verifier.get("management_keywords"):
            field_counts["management_keywords"] += 1
        if verifier.get("uncertainty_required"):
            field_counts["uncertainty_required"] += 1
        if verifier.get("forbidden_claims"):
            field_counts["forbidden_claims"] += 1
        if verifier.get("required_sections"):
            field_counts["required_sections"] += 1
        if verifier.get("accepted_labels"):
            field_counts["accepted_labels"] += 1
        if verifier.get("accepted_answers"):
            field_counts["accepted_answers"] += 1
        if reward_meta.get("weights"):
            field_counts["reward_meta.weights"] += 1

    module_names = [
        "exact_match",
        "normalized_label",
        "synonym_match",
        "structured_format",
        "uncertainty_calibration",
        "clarify_vs_respond",
        "management_coverage",
        "hallucination_penalty",
    ]
    issue_report = issues.report()
    critical_count = sum(issue_report[name]["count"] for name in CRITICAL_ISSUES)
    return {
        "manifest_path": str(manifest_path),
        "total_samples": total,
        "counts": {
            "by_source_dataset": dict(sorted(by_dataset.items())),
            "by_task_type": dict(sorted(by_task_type.items())),
            "by_split": dict(sorted(by_split.items())),
            "by_verifier_mode": dict(sorted(by_verifier_mode.items())),
            "by_image_count": dict(sorted(by_image_count.items(), key=lambda item: int(item[0]) if item[0].isdigit() else -1)),
        },
        "answer_length_words": _distribution(answer_lengths),
        "field_coverage": {name: _coverage(field_counts.get(name, 0), total) for name in FIELD_COVERAGE_NAMES},
        "reward_module_applicability": {
            name: _coverage(module_counts.get(name, 0), total) for name in module_names
        },
        "issues": issue_report,
        "critical_issue_count": int(critical_count),
        "critical_issue_names": sorted(name for name in CRITICAL_ISSUES if issue_report[name]["count"] > 0),
    }


def _markdown_table(mapping: Dict[str, Any], key_header: str, value_header: str = "Count") -> List[str]:
    lines = ["| %s | %s |" % (key_header, value_header), "| --- | ---: |"]
    for key, value in mapping.items():
        lines.append("| %s | %s |" % (key, value))
    return lines


def write_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    lines: List[str] = [
        "# RL Manifest Audit",
        "",
        "- Manifest: `%s`" % report["manifest_path"],
        "- Total samples: `%s`" % report["total_samples"],
        "- Critical issue count: `%s`" % report["critical_issue_count"],
        "",
        "## Counts",
        "",
    ]
    for name, mapping in report["counts"].items():
        lines.extend(["### %s" % name.replace("_", " ").title(), ""])
        lines.extend(_markdown_table(mapping, "Value"))
        lines.append("")

    lines.extend(["## Answer Length Words", ""])
    lines.extend(_markdown_table(report["answer_length_words"], "Statistic", "Words"))
    lines.extend(["", "## Field Coverage", ""])
    lines.extend(["| Field | Count | Percent |", "| --- | ---: | ---: |"])
    for name, payload in report["field_coverage"].items():
        lines.append("| %s | %s | %.2f |" % (name, payload["count"], payload["percentage"]))

    lines.extend(["", "## Reward Module Applicability", ""])
    lines.extend(["| Module | Count | Percent |", "| --- | ---: | ---: |"])
    for name, payload in report["reward_module_applicability"].items():
        lines.append("| %s | %s | %.2f |" % (name, payload["count"], payload["percentage"]))

    lines.extend(["", "## Issues", ""])
    for name, payload in report["issues"].items():
        lines.append("### %s" % name)
        lines.append("")
        lines.append("- Count: `%s`" % payload["count"])
        if payload["examples"]:
            lines.append("")
            lines.append("| Line | Sample ID | Reason |")
            lines.append("| ---: | --- | --- |")
            for example in payload["examples"]:
                lines.append(
                    "| %s | %s | %s |"
                    % (example.get("line"), example.get("sample_id"), str(example.get("reason")).replace("|", "\\|"))
                )
        lines.append("")

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    report = audit_manifest(
        Path(args.manifest_path),
        repo_root=repo_root,
        max_examples_per_issue=args.max_examples_per_issue,
    )
    write_json(Path(args.output_json), report)
    write_markdown_report(report, Path(args.output_md))
    if args.fail_on_critical and report["critical_issue_count"] > 0:
        print("critical_rl_manifest_issues=%s" % ",".join(report["critical_issue_names"]))
        return 2
    print("rl_manifest_audit=%s samples=%s critical_issues=%s" % (
        args.output_json,
        report["total_samples"],
        report["critical_issue_count"],
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
