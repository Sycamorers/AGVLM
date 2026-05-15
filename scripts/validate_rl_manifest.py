#!/usr/bin/env python3
"""Validate RL manifests before reward scoring or GRPO smoke tests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import ValidationError

from agri_vlm.schemas.dataset_schema import UnifiedSample
from agri_vlm.utils.io import write_json
from agri_vlm.utils.text import normalize_label, normalize_text


ALLOWED_EXPECTED_DECISIONS = {"clarify", "respond"}
INVALID_LABELS = {
    "",
    "none",
    "null",
    "n a",
    "na",
    "unknown",
    "placeholder",
    "todo",
    "tbd",
    "label",
    "disease",
}


class IssueCollector:
    def __init__(self, max_examples: int) -> None:
        self.max_examples = max_examples
        self.counts: Counter[str] = Counter()
        self.examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add(self, name: str, line: int, sample_id: str, reason: str) -> None:
        self.counts[name] += 1
        if len(self.examples[name]) >= self.max_examples:
            return
        self.examples[name].append({"line": line, "sample_id": sample_id, "reason": reason})

    def report(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {"count": int(self.counts[name]), "examples": self.examples.get(name, [])}
            for name in sorted(self.counts)
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--split-manifests", nargs="*", default=[])
    parser.add_argument("--check-image-open", action="store_true")
    parser.add_argument("--max-examples-per-issue", type=int, default=20)
    return parser.parse_args()


def _read_jsonl(path: Path, issues: IssueCollector) -> List[Tuple[int, Dict[str, Any]]]:
    rows: List[Tuple[int, Dict[str, Any]]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        issues.add("manifest_file_error", 0, "", str(exc))
        return rows
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.add("invalid_jsonl", line_number, "", str(exc))
                continue
            if not isinstance(payload, dict):
                issues.add("jsonl_row_not_object", line_number, "", "row root is not an object")
                continue
            rows.append((line_number, payload))
    return rows


def _json_object_field(row: Dict[str, Any], key: str, line: int, issues: IssueCollector) -> Optional[Dict[str, Any]]:
    if key not in row:
        return None
    value = row.get(key)
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        issues.add("%s_invalid" % key, line, str(row.get("sample_id") or ""), str(exc))
        return None
    if not isinstance(payload, dict):
        issues.add("%s_invalid" % key, line, str(row.get("sample_id") or ""), "not a JSON object")
        return None
    return payload


def _list_of_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def _nonempty_list_of_strings(value: Any) -> bool:
    return _list_of_strings(value) and len(value) > 0


def _is_invalid_label(value: Any) -> bool:
    if value is None:
        return True
    normalized = normalize_label(str(value))
    return normalized in INVALID_LABELS or normalized.startswith("placeholder")


def _resolve_image_path(path_value: str, *, manifest_path: Path, repo_root: Path) -> Optional[Path]:
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [repo_root / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _image_can_open(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _user_prompt(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for message in row.get("messages") or []:
        if message.get("role") != "user":
            continue
        for content in message.get("content") or []:
            if content.get("type") == "text":
                parts.append(str(content.get("text") or ""))
    return "\n".join(parts)


def _duplicate_key(row: Dict[str, Any]) -> str:
    target = row.get("target") or {}
    label = target.get("canonical_label") or target.get("answer_text") or ""
    return "%s::%s::%s" % (
        normalize_text(",".join(str(path) for path in row.get("images") or [])),
        normalize_text(_user_prompt(row)),
        normalize_label(str(label)),
    )


def _reward_component_enabled(row: Dict[str, Any], component: str) -> bool:
    weights = (row.get("reward_meta") or {}).get("weights") or {}
    return float(weights.get(component, 0.0) or 0.0) > 0.0


def _validate_task_requirements(row: Dict[str, Any], line: int, issues: IssueCollector) -> None:
    sample_id = str(row.get("sample_id") or "")
    task_type = row.get("task_type")
    target = row.get("target") or {}
    verifier = row.get("verifier") or {}
    mode = verifier.get("mode")
    if task_type == "classification" or mode == "label":
        labels = []
        if target.get("canonical_label") is not None:
            labels.append(target.get("canonical_label"))
        labels.extend(target.get("canonical_labels") or [])
        labels.extend(verifier.get("accepted_labels") or [])
        if not labels or any(_is_invalid_label(label) for label in labels):
            issues.add("empty_or_invalid_classification_label", line, sample_id, "classification labels are missing or invalid")
    if task_type == "vqa" or mode == "exact_match":
        answers = [target.get("answer_text"), *(target.get("acceptable_answers") or []), *(verifier.get("accepted_answers") or [])]
        if not any(isinstance(answer, str) and answer.strip() for answer in answers):
            issues.add("missing_vqa_answer", line, sample_id, "no answer_text, acceptable_answers, or accepted_answers")
    if task_type == "consultation" or mode == "structured":
        if not _nonempty_list_of_strings(verifier.get("required_sections") or []):
            issues.add("invalid_required_sections", line, sample_id, "consultation rows require nonempty required_sections strings")
    if task_type == "clarify_or_respond" or mode == "clarify":
        decision = target.get("decision") or verifier.get("expected_decision")
        if decision not in ALLOWED_EXPECTED_DECISIONS:
            issues.add("invalid_expected_decision", line, sample_id, "expected_decision must be clarify or respond")


def _validate_reward_fields(row: Dict[str, Any], line: int, issues: IssueCollector) -> None:
    sample_id = str(row.get("sample_id") or "")
    target = row.get("target") or {}
    verifier = row.get("verifier") or {}
    if target.get("acceptable_answers") is not None and not _list_of_strings(target.get("acceptable_answers")):
        issues.add("invalid_acceptable_answers", line, sample_id, "acceptable_answers must be nonempty strings")
    if verifier.get("accepted_labels") is not None and not _list_of_strings(verifier.get("accepted_labels")):
        issues.add("invalid_accepted_labels", line, sample_id, "accepted_labels must be nonempty strings")
    for key in ["required_sections", "management_keywords", "forbidden_claims", "known_facts", "allowed_claims", "visual_evidence", "unsafe_recommendations"]:
        if verifier.get(key) is not None and not _list_of_strings(verifier.get(key)):
            issues.add("invalid_%s" % key, line, sample_id, "%s must be nonempty strings" % key)
    if verifier.get("expected_decision") is not None and verifier.get("expected_decision") not in ALLOWED_EXPECTED_DECISIONS:
        issues.add("invalid_expected_decision", line, sample_id, "verifier.expected_decision is invalid")
    if _reward_component_enabled(row, "management_coverage") and not _nonempty_list_of_strings(verifier.get("management_keywords") or []):
        issues.add("management_keywords_missing_when_enabled", line, sample_id, "management_coverage has positive weight")


def _validate_images(
    row: Dict[str, Any],
    *,
    line: int,
    manifest_path: Path,
    repo_root: Path,
    check_image_open: bool,
    issues: IssueCollector,
) -> None:
    sample_id = str(row.get("sample_id") or "")
    images = row.get("images")
    if not isinstance(images, list) or not images:
        issues.add("missing_image_references", line, sample_id, "images must be a nonempty list")
        return
    for image_path in images:
        if not isinstance(image_path, str) or not image_path.strip():
            issues.add("missing_image_references", line, sample_id, "empty image path")
            continue
        resolved = _resolve_image_path(image_path, manifest_path=manifest_path, repo_root=repo_root)
        if resolved is None:
            issues.add("image_path_missing", line, sample_id, image_path)
            continue
        if check_image_open and not _image_can_open(resolved):
            issues.add("image_path_broken", line, sample_id, image_path)


def _validate_schema(row: Dict[str, Any], line: int, issues: IssueCollector) -> None:
    sample_id = str(row.get("sample_id") or "")
    try:
        UnifiedSample.model_validate(row)
    except ValidationError as exc:
        issues.add("schema_validation_error", line, sample_id, str(exc.errors()[0]))


def _scan_split_files(paths: Iterable[Path], issues: IssueCollector) -> None:
    seen: Dict[str, Tuple[str, int]] = {}
    seen_keys: Dict[str, Tuple[str, int]] = {}
    for path in paths:
        split_rows = _read_jsonl(path, issues)
        for line, row in split_rows:
            split = str(row.get("split") or path.stem)
            sample_id = str(row.get("sample_id") or "")
            if sample_id:
                previous = seen.get(sample_id)
                if previous and previous[0] != split:
                    issues.add("duplicate_sample_across_splits", line, sample_id, "%s also appears in %s" % (sample_id, previous[0]))
                seen[sample_id] = (split, line)
            key = _duplicate_key(row)
            previous_key = seen_keys.get(key)
            if previous_key and previous_key[0] != split:
                issues.add("duplicate_content_across_splits", line, sample_id, "content duplicate also appears in %s" % previous_key[0])
            seen_keys[key] = (split, line)


def validate_manifest(
    *,
    manifest_path: Path,
    repo_root: Path,
    split_manifests: List[Path],
    check_image_open: bool,
    max_examples_per_issue: int,
) -> Dict[str, Any]:
    issues = IssueCollector(max_examples=max_examples_per_issue)
    rows = _read_jsonl(manifest_path, issues)
    seen_ids: Dict[str, Tuple[str, int]] = {}
    seen_keys: Dict[str, Tuple[str, int]] = {}
    by_task_type: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    for line, row in rows:
        sample_id = str(row.get("sample_id") or "")
        for json_key in ["target_json", "verifier_json", "reward_meta_json"]:
            _json_object_field(row, json_key, line, issues)
        _validate_schema(row, line, issues)
        _validate_task_requirements(row, line, issues)
        _validate_reward_fields(row, line, issues)
        _validate_images(
            row,
            line=line,
            manifest_path=manifest_path,
            repo_root=repo_root,
            check_image_open=check_image_open,
            issues=issues,
        )
        split = str(row.get("split") or "")
        by_task_type[str(row.get("task_type") or "")] += 1
        by_split[split] += 1
        if sample_id:
            previous = seen_ids.get(sample_id)
            if previous:
                issues.add("duplicate_sample_id", line, sample_id, "first seen on line %s" % previous[1])
                if previous[0] != split:
                    issues.add("duplicate_sample_across_splits", line, sample_id, "also appears in split %s" % previous[0])
            seen_ids[sample_id] = (split, line)
        key = _duplicate_key(row)
        previous_key = seen_keys.get(key)
        if previous_key and previous_key[0] != split:
            issues.add("duplicate_content_across_splits", line, sample_id, "also appears in split %s" % previous_key[0])
        seen_keys[key] = (split, line)
    if not rows:
        issues.add("no_rows", 0, "", "manifest is empty or all rows were invalid JSON")
    if split_manifests:
        _scan_split_files(split_manifests, issues)
    issue_report = issues.report()
    return {
        "manifest": str(manifest_path),
        "row_count": len(rows),
        "counts": {
            "by_task_type": dict(sorted(by_task_type.items())),
            "by_split": dict(sorted(by_split.items())),
        },
        "issues": issue_report,
        "issue_count": int(sum(payload["count"] for payload in issue_report.values())),
    }


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[1]
    report = validate_manifest(
        manifest_path=manifest_path,
        repo_root=repo_root,
        split_manifests=[Path(path) for path in args.split_manifests],
        check_image_open=bool(args.check_image_open),
        max_examples_per_issue=args.max_examples_per_issue,
    )
    if args.output_json:
        write_json(Path(args.output_json), report)
    print("validated_rl_manifest=%s rows=%s issues=%s" % (manifest_path, report["row_count"], report["issue_count"]))
    return 0 if report["issue_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
