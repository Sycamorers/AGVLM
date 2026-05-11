"""Dataset inspection and prompt adaptation for VLM baselines.

This module deliberately avoids importing the repository training package. It
reads normalized JSONL manifests as data and never mutates source datasets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from utils import REPO_ROOT, normalize_text, read_jsonl, resolve_repo_path


ACTIVE_EVAL_MANIFEST = Path("data/manifests/full/sft_eval_phi4_max3_stratified512.jsonl")
ACTIVE_TRAIN_MANIFEST = Path("data/manifests/full/sft_train_phi4_max3_no_eval_overlap.jsonl")
ACTIVE_SPLIT_SUMMARY = Path("data/manifests/full/sft_train_eval_phi4_max3_summary.json")
FALLBACK_SOURCE_MANIFESTS = [
    Path("data/manifests/full/sft_manifest.decode_aspect_valid_images.jsonl"),
    Path("data/manifests/full/sft_manifest.valid_images.jsonl"),
    Path("data/manifests/full/sft_manifest.jsonl"),
    Path("data/manifests/partial_10pct/sft_manifest.jsonl"),
]


@dataclass(frozen=True)
class BenchmarkSample:
    row: dict[str, Any]
    benchmark_split: str
    prompt: str
    system_prompt: str
    expected_answer: str
    references: list[str]
    label_space: list[str]

    @property
    def sample_id(self) -> str:
        return str(self.row.get("sample_id") or "")

    @property
    def task_type(self) -> str:
        return str(self.row.get("task_type") or "")

    @property
    def verifier_mode(self) -> str:
        return str((self.row.get("verifier") or {}).get("mode") or "")

    @property
    def image_paths(self) -> list[str]:
        return [str(path) for path in (self.row.get("images") or []) if path]


def _target(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("target") or {}


def _verifier(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("verifier") or {}


def expected_answer(row: dict[str, Any]) -> str:
    target = _target(row)
    if target.get("canonical_label"):
        return str(target["canonical_label"])
    if target.get("answer_text"):
        return str(target["answer_text"])
    if target.get("decision"):
        return str(target["decision"])
    canonical_labels = target.get("canonical_labels") or []
    if canonical_labels:
        return str(canonical_labels[0])
    acceptable_answers = target.get("acceptable_answers") or []
    if acceptable_answers:
        return str(acceptable_answers[0])
    structured = target.get("structured") or {}
    if structured:
        import json

        return json.dumps(structured, ensure_ascii=False, sort_keys=True)
    return ""


def accepted_references(row: dict[str, Any]) -> list[str]:
    target = _target(row)
    verifier = _verifier(row)
    refs: list[str] = []
    for value in verifier.get("accepted_labels") or []:
        if value:
            refs.append(str(value))
    for value in verifier.get("accepted_answers") or []:
        if value:
            refs.append(str(value))
    for value in target.get("acceptable_answers") or []:
        if value:
            refs.append(str(value))
    for key in ("canonical_label", "answer_text", "decision"):
        value = target.get(key)
        if value:
            refs.append(str(value))
    for value in target.get("canonical_labels") or []:
        if value:
            refs.append(str(value))
    deduped: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        key = normalize_text(ref)
        if key and key not in seen:
            seen.add(key)
            deduped.append(ref)
    return deduped


def output_instruction(row: dict[str, Any]) -> str:
    task_type = str(row.get("task_type") or "")
    mode = str(_verifier(row).get("mode") or "")
    reward_meta = row.get("reward_meta") or {}
    target = _target(row)
    refs = accepted_references(row)
    yes_no_refs = {normalize_text(ref) for ref in refs if ref}

    if reward_meta.get("structured_output_required") or mode == "structured" or target.get("structured"):
        return (
            "Respond using these line-start section headers exactly once:\n"
            "Diagnosis:\nEvidence:\nUncertainty:\nManagement:\nFollow-up:"
        )
    if mode == "clarify" or task_type == "clarify_or_respond":
        return (
            "Decide whether the case needs more information.\n"
            "Respond in this format:\nDecision: <clarify or respond>\nAnswer: <short answer or clarifying question>"
        )
    if mode == "label" or target.get("canonical_label"):
        return "Respond in this format:\nAnswer: <most specific crop issue, disease, pest, or label>"
    if yes_no_refs and yes_no_refs.issubset({"yes", "no"}):
        return "Respond in this format:\nAnswer: <Yes or No>"
    if task_type == "vqa" or mode == "exact_match":
        return "Respond in this format:\nAnswer: <short direct answer>"
    if task_type == "consultation":
        return (
            "Respond using these line-start section headers exactly once:\n"
            "Diagnosis:\nEvidence:\nUncertainty:\nManagement:\nFollow-up:"
        )
    return "Answer concisely."


def _message_texts(row: dict[str, Any], role: str | None = None) -> list[str]:
    texts: list[str] = []
    for message in row.get("messages") or []:
        if role and message.get("role") != role:
            continue
        for block in message.get("content") or []:
            if block.get("type") == "text" and block.get("text"):
                texts.append(str(block["text"]))
    return texts


def system_prompt(row: dict[str, Any]) -> str:
    texts = _message_texts(row, role="system")
    if texts:
        return "\n".join(texts)
    return (
        "You are an agricultural vision-language assistant focused on ground-level RGB crop "
        "disease, pest, symptom, and consultation tasks."
    )


def user_prompt(row: dict[str, Any]) -> str:
    texts = _message_texts(row, role="user")
    return "\n".join(texts).strip()


def semantic_prompt(row: dict[str, Any]) -> str:
    base = user_prompt(row)
    instruction = output_instruction(row)
    if base:
        return "%s\n\n%s" % (base, instruction)
    return instruction


def build_chat_messages(
    row: dict[str, Any],
    *,
    image_paths: list[str],
    include_image_paths: bool = False,
    include_system: bool = True,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if include_system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt(row)}]})
    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        if include_image_paths:
            content.append({"type": "image", "image": str(resolve_repo_path(image_path).as_uri())})
        else:
            content.append({"type": "image"})
    content.append({"type": "text", "text": semantic_prompt(row)})
    messages.append({"role": "user", "content": content})
    return messages


def build_plain_prompt(row: dict[str, Any], image_count: int = 1) -> str:
    image_tokens = "".join("<|image_%s|>" % (index + 1) for index in range(image_count))
    return "<|user|>%s%s<|end|><|assistant|>" % (image_tokens, semantic_prompt(row))


def label_space(rows: Iterable[dict[str, Any]]) -> list[str]:
    labels: dict[str, str] = {}
    for row in rows:
        if (row.get("verifier") or {}).get("mode") != "label" and not _target(row).get("canonical_label"):
            continue
        for ref in accepted_references(row):
            key = normalize_text(ref)
            if key:
                labels.setdefault(key, ref)
    return [labels[key] for key in sorted(labels)]


def load_benchmark_samples(manifest_path: Path, benchmark_split: str) -> list[BenchmarkSample]:
    rows = read_jsonl(manifest_path)
    labels = label_space(rows)
    samples = []
    for row in rows:
        samples.append(
            BenchmarkSample(
                row=row,
                benchmark_split=benchmark_split,
                prompt=semantic_prompt(row),
                system_prompt=system_prompt(row),
                expected_answer=expected_answer(row),
                references=accepted_references(row),
                label_space=labels,
            )
        )
    return samples


def _label_for_distribution(row: dict[str, Any]) -> str:
    target = _target(row)
    return (
        str(target.get("canonical_label") or "")
        or str(target.get("answer_text") or "")
        or str(target.get("decision") or "")
        or "missing"
    )


def image_status(row: dict[str, Any], repo_root: Path = REPO_ROOT) -> tuple[int, list[str]]:
    missing: list[str] = []
    for image_path in row.get("images") or []:
        resolved = resolve_repo_path(image_path, repo_root=repo_root)
        if not resolved.exists():
            missing.append(str(image_path))
    return len(missing), missing


def prompt_missing(row: dict[str, Any]) -> bool:
    return not bool(user_prompt(row).strip())


def distribution_report(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    repo_root: Path = REPO_ROOT,
    skipped: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    unique_ids = {str(row.get("sample_id") or "") for row in all_rows}
    samples_per_split = {split: len(rows) for split, rows in rows_by_split.items()}
    per_split_task: dict[str, dict[str, int]] = {}
    per_split_dataset: dict[str, dict[str, int]] = {}
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    original_split_counts: Counter[str] = Counter()
    missing_image_samples = 0
    missing_image_files = 0
    missing_text_samples = 0
    missing_examples: list[dict[str, Any]] = []

    for split, rows in rows_by_split.items():
        per_split_task[split] = dict(Counter(str(row.get("task_type") or "missing") for row in rows))
        per_split_dataset[split] = dict(Counter(str(row.get("source_dataset") or "missing") for row in rows))
        for row in rows:
            task_counts[str(row.get("task_type") or "missing")] += 1
            dataset_counts[str(row.get("source_dataset") or "missing")] += 1
            original_split_counts[str(row.get("split") or "missing")] += 1
            label_counts[_label_for_distribution(row)] += 1
            count, missing = image_status(row, repo_root=repo_root)
            if count:
                missing_image_samples += 1
                missing_image_files += count
                if len(missing_examples) < 20:
                    missing_examples.append({"sample_id": row.get("sample_id"), "missing_images": missing})
            if prompt_missing(row):
                missing_text_samples += 1

    return {
        "total_samples": len(all_rows),
        "unique_sample_count": len(unique_ids),
        "duplicate_sample_id_count_across_split_manifests": max(len(all_rows) - len(unique_ids), 0),
        "samples_per_split": samples_per_split,
        "samples_per_task_type": dict(task_counts),
        "samples_per_source_dataset": dict(dataset_counts),
        "samples_per_original_split": dict(original_split_counts),
        "samples_per_label_or_answer_top100": dict(label_counts.most_common(100)),
        "per_split_task_type": per_split_task,
        "per_split_source_dataset": per_split_dataset,
        "missing_image_sample_count": missing_image_samples,
        "missing_image_file_count": missing_image_files,
        "missing_text_prompt_count": missing_text_samples,
        "missing_image_examples": missing_examples,
        "skipped_samples": skipped or {},
        "extra": extra or {},
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VLM Baseline Dataset Distribution",
        "",
        "- total samples: %s" % report.get("total_samples", 0),
        "- unique sample ids: %s" % report.get("unique_sample_count", 0),
        "- duplicate sample ids across split manifests: %s"
        % report.get("duplicate_sample_id_count_across_split_manifests", 0),
        "- samples per split: `%s`" % report.get("samples_per_split", {}),
        "- samples per task type: `%s`" % report.get("samples_per_task_type", {}),
        "- samples per source dataset: `%s`" % report.get("samples_per_source_dataset", {}),
        "- original source splits: `%s`" % report.get("samples_per_original_split", {}),
        "- missing image samples: %s" % report.get("missing_image_sample_count", 0),
        "- missing image files: %s" % report.get("missing_image_file_count", 0),
        "- missing text prompts: %s" % report.get("missing_text_prompt_count", 0),
        "- skipped samples: `%s`" % report.get("skipped_samples", {}),
        "",
        "## Per Split Task Types",
        "",
    ]
    for split, counts in sorted((report.get("per_split_task_type") or {}).items()):
        lines.append("- %s: `%s`" % (split, counts))
    lines.extend(["", "## Per Split Source Datasets", ""])
    for split, counts in sorted((report.get("per_split_source_dataset") or {}).items()):
        lines.append("- %s: `%s`" % (split, counts))
    lines.extend(["", "## Top Labels Or Answers", ""])
    for label, count in list((report.get("samples_per_label_or_answer_top100") or {}).items())[:30]:
        lines.append("- `%s`: %s" % (label, count))
    lines.append("")
    return "\n".join(lines)


LEAKAGE_KEYS = [
    "scene_id",
    "dialogue_id",
    "conversation_id",
    "image_id",
    "video_id",
    "subject_id",
    "participant_id",
    "source_image_id",
]


def group_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    source_dataset = str(row.get("source_dataset") or metadata.get("source_dataset") or "missing_dataset")
    for key in LEAKAGE_KEYS:
        value = row.get(key) or metadata.get(key)
        if value:
            return "%s:%s:%s" % (source_dataset, key, value)
    images = row.get("images") or []
    if images:
        return "%s:source_file_stem:%s" % (source_dataset, Path(str(images[0])).stem)
    return "%s:sample_id:%s" % (source_dataset, row.get("sample_id", ""))


def stratum_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("source_dataset") or "missing_dataset"),
            str(row.get("task_type") or "missing_task"),
            str((_verifier(row)).get("mode") or "missing_mode"),
            normalize_text(_label_for_distribution(row))[:96],
        ]
    )
