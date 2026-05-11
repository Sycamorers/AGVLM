"""Manifest builders for SFT, RL, and evaluation."""

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from agri_vlm.data.manifest_io import (
    filter_rewardable_manifest,
    merge_manifests,
    read_manifest,
    write_manifest,
)
from agri_vlm.data.split_utils import assign_holdout
from agri_vlm.utils.io import read_jsonl, write_json
from agri_vlm.utils.text import normalize_label, normalize_text


RL_CONSULTATION_SECTIONS = ["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"]
RL_DEFAULT_FORBIDDEN_CLAIMS = [
    "guaranteed cure",
    "no follow-up needed",
    "100% certain diagnosis from image alone",
]


def build_sft_manifest(
    source_paths: Sequence[Path],
    output_path: Path,
    allowed_task_types: Sequence[str],
    exclude_splits: Sequence[str],
    max_samples_per_source: int = None,
) -> List[dict]:
    rows = merge_manifests(
        source_paths=source_paths,
        allowed_task_types=allowed_task_types,
        exclude_splits=exclude_splits,
        max_samples_per_source=max_samples_per_source,
    )
    return [sample.model_dump(mode="json") for sample in write_manifest(output_path, [row.model_dump(mode="json") for row in rows])]


def _manifest_group_key(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    images = row.get("images") or []
    source_image_id = metadata.get("source_image_id") or (images[0] if images else "")
    return "%s::%s" % (row.get("source_dataset"), source_image_id)


def _stable_hex(value: str, salt: str) -> str:
    return hashlib.sha256(("%s::%s" % (salt, value)).encode("utf-8")).hexdigest()


def _stratum_key(row: Dict[str, Any], fields: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def _counter_dict(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(Counter(str(row.get(field, "")) for row in rows))


def _with_unique_sample_ids(rows: Sequence[Any], salt: str) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    payloads: List[Dict[str, Any]] = []
    for row in rows:
        payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else dict(row)
        sample_id = str(payload["sample_id"])
        counts[sample_id] += 1
        if counts[sample_id] > 1:
            metadata = dict(payload.get("metadata") or {})
            metadata.setdefault("original_sample_id", sample_id)
            payload["metadata"] = metadata
            duplicate_key = jsonable_duplicate_key(payload)
            payload["sample_id"] = "%s-rl-%04d-%s" % (
                sample_id,
                counts[sample_id],
                _stable_hex(duplicate_key, salt)[:8],
            )
        payloads.append(payload)
    return payloads


def jsonable_duplicate_key(row: Dict[str, Any]) -> str:
    return "%s::%s::%s" % (
        row.get("sample_id"),
        ",".join(str(path) for path in row.get("images") or []),
        str((row.get("target") or {}).get("answer_text") or ""),
    )


def _iter_user_text_blocks(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for message in row.get("messages") or []:
        if message.get("role") != "user":
            continue
        for content in message.get("content") or []:
            if content.get("type") == "text":
                yield content


def _first_user_prompt(row: Dict[str, Any]) -> str:
    for content in _iter_user_text_blocks(row):
        text = str(content.get("text") or "").strip()
        if text:
            return text
    return ""


def _set_user_prompt(row: Dict[str, Any], prompt: str) -> None:
    for content in _iter_user_text_blocks(row):
        content["text"] = prompt
        return
    raise ValueError("RL row is missing a user text prompt: %s" % row.get("sample_id"))


def _append_instruction(prompt: str, instruction: str) -> str:
    prompt = str(prompt or "").strip()
    return "%s\n\n%s" % (prompt, instruction.strip()) if prompt else instruction.strip()


def _ensure_list(payload: Dict[str, Any], key: str) -> List[Any]:
    value = payload.get(key)
    if isinstance(value, list):
        return value
    if value is None:
        payload[key] = []
        return payload[key]
    payload[key] = [value]
    return payload[key]


def _apply_rl_output_contract(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(row)
    payload["messages"] = [dict(message) for message in row.get("messages") or []]
    for message in payload["messages"]:
        message["content"] = [dict(content) for content in message.get("content") or []]
    payload["target"] = dict(row.get("target") or {})
    payload["verifier"] = dict(row.get("verifier") or {})
    payload["reward_meta"] = dict(row.get("reward_meta") or {})
    payload["reward_meta"]["weights"] = dict(payload["reward_meta"].get("weights") or {})

    verifier = payload["verifier"]
    forbidden_claims = _ensure_list(verifier, "forbidden_claims")
    for claim in RL_DEFAULT_FORBIDDEN_CLAIMS:
        if claim not in forbidden_claims:
            forbidden_claims.append(claim)

    prompt = _first_user_prompt(payload)
    task_type = payload.get("task_type")
    if task_type == "classification":
        instruction = (
            "Respond in this format:\n"
            "Answer: <canonical agricultural label>\n"
            "Evidence: <brief visible symptom evidence>"
        )
    elif task_type == "vqa":
        instruction = "Respond in this format:\nAnswer: <short answer>"
    elif task_type == "consultation":
        if not verifier.get("required_sections"):
            verifier["required_sections"] = list(RL_CONSULTATION_SECTIONS)
        payload["reward_meta"]["structured_output_required"] = True
        payload["reward_meta"]["weights"].setdefault("structured_format", 0.5)
        payload["reward_meta"]["weights"].setdefault("hallucination_penalty", 1.0)
        if verifier.get("management_keywords"):
            payload["reward_meta"]["weights"].setdefault("management_coverage", 0.5)
        instruction = (
            "Respond using these line-start section headers:\n"
            "Diagnosis:\n"
            "Evidence:\n"
            "Uncertainty:\n"
            "Management:\n"
            "Follow-up:"
        )
    elif task_type == "clarify_or_respond":
        verifier["mode"] = "clarify"
        payload["reward_meta"]["allow_clarification"] = True
        instruction = (
            "Respond using exactly one of these formats:\n"
            "Decision: clarify\n"
            "Clarifying question: <one question needed before diagnosis or management>\n\n"
            "Decision: respond\n"
            "Answer: <concise agricultural answer>"
        )
    else:
        instruction = "Respond in a concise agriculture-focused format."

    _set_user_prompt(payload, _append_instruction(prompt, instruction))
    return payload


def _dedupe_target_text(row: Dict[str, Any]) -> str:
    target = row.get("target") or {}
    if target.get("canonical_label"):
        return str(target["canonical_label"])
    if target.get("answer_text"):
        return str(target["answer_text"])
    if target.get("acceptable_answers"):
        return str((target.get("acceptable_answers") or [""])[0])
    if target.get("decision"):
        return str(target["decision"])
    if target.get("structured"):
        return str(target["structured"])
    return ""


def _rl_dedupe_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    metadata = row.get("metadata") or {}
    images = row.get("images") or []
    image_identity = str(metadata.get("source_image_id") or (images[0] if images else ""))
    return (
        normalize_text(image_identity),
        normalize_text(_first_user_prompt(row)),
        normalize_label(_dedupe_target_text(row)),
    )


def _dedupe_rl_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    duplicates = 0
    for row in rows:
        key = _rl_dedupe_key(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, duplicates


def _with_split(rows: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        payload = dict(row)
        payload["split"] = split
        output.append(payload)
    return output


def _split_rl_train_holdout(
    rows: Sequence[Dict[str, Any]],
    *,
    holdout_ratio: float,
    max_holdout_samples: int,
    min_holdout_per_stratum: int,
    salt: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if holdout_ratio <= 0.0 or not rows:
        return _with_split(rows, "train"), []
    target_size = max(1, int(len(rows) * holdout_ratio))
    if max_holdout_samples > 0:
        target_size = min(target_size, max_holdout_samples)
    holdout_rows = _sample_stratified(
        rows,
        target_size=target_size,
        stratum_fields=["source_dataset", "task_type"],
        min_per_stratum=min_holdout_per_stratum,
        salt=salt,
    )
    holdout_ids = {str(row.get("sample_id")) for row in holdout_rows}
    train_rows = [row for row in rows if str(row.get("sample_id")) not in holdout_ids]
    if not train_rows:
        raise ValueError("No RL training rows remain after holdout split construction.")
    return _with_split(train_rows, "train"), _with_split(holdout_rows, "holdout")


def _nested_counter(rows: Sequence[Dict[str, Any]], *fields: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        values = []
        for field in fields:
            if field.startswith("metadata."):
                values.append(str((row.get("metadata") or {}).get(field.split(".", 1)[1]) or ""))
            else:
                values.append(str(row.get(field) or ""))
        counts["::".join(values)] += 1
    return dict(sorted(counts.items()))


def _sample_stratified(
    rows: Sequence[Dict[str, Any]],
    *,
    target_size: int,
    stratum_fields: Sequence[str],
    min_per_stratum: int,
    salt: str,
) -> List[Dict[str, Any]]:
    if target_size <= 0 or len(rows) <= target_size:
        return sorted(rows, key=lambda row: _stable_hex(str(row.get("sample_id")), salt))

    strata: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[_stratum_key(row, stratum_fields)].append(row)
    for key, stratum_rows in strata.items():
        strata[key] = sorted(stratum_rows, key=lambda row: _stable_hex(str(row.get("sample_id")), "%s::%s" % (salt, key)))

    allocations = {key: 0 for key in strata}
    remaining = target_size
    for key in sorted(strata, key=lambda item: len(strata[item]), reverse=True):
        if remaining <= 0:
            break
        take = min(min_per_stratum, len(strata[key]), remaining)
        allocations[key] = take
        remaining -= take

    if remaining > 0:
        capacities = {key: len(rows_for_key) - allocations[key] for key, rows_for_key in strata.items()}
        total_capacity = sum(max(value, 0) for value in capacities.values())
        fractional = []
        for key, capacity in capacities.items():
            if capacity <= 0 or total_capacity <= 0:
                continue
            raw = remaining * capacity / total_capacity
            take = min(capacity, int(raw))
            allocations[key] += take
            fractional.append((raw - take, capacity - take, key))
        remaining = target_size - sum(allocations.values())
        for _fraction, capacity, key in sorted(fractional, reverse=True):
            if remaining <= 0:
                break
            if capacity <= 0:
                continue
            allocations[key] += 1
            remaining -= 1

    sampled = []
    for key in sorted(strata):
        sampled.extend(strata[key][: allocations[key]])
    return sorted(sampled, key=lambda row: _stable_hex(str(row.get("sample_id")), salt))


def build_sft_train_eval_manifests(
    *,
    source_manifest_path: Path,
    holdout_manifest_path: Path,
    train_output_path: Path,
    eval_output_path: Path,
    train_splits: Sequence[str],
    eval_splits: Sequence[str],
    max_images_per_sample: int,
    eval_sample_size: int,
    min_eval_samples_per_stratum: int,
    salt: str,
    summary_output_path: Path = None,
) -> Dict[str, Any]:
    """Build non-overlapping SFT train and step-time validation manifests."""
    holdout_rows = [
        row
        for row in read_jsonl(holdout_manifest_path)
        if len(row.get("images") or []) <= max_images_per_sample
    ]
    eval_rows_by_id = {row["sample_id"]: row for row in holdout_rows}

    source_rows = list(read_jsonl(source_manifest_path))
    for row in source_rows:
        if len(row.get("images") or []) > max_images_per_sample:
            continue
        if row.get("split") in eval_splits:
            eval_rows_by_id.setdefault(row["sample_id"], row)

    eval_pool_rows = list(eval_rows_by_id.values())
    eval_ids = {row["sample_id"] for row in eval_pool_rows}
    eval_group_keys = {_manifest_group_key(row) for row in eval_pool_rows}

    train_rows = []
    excluded = Counter()
    for row in source_rows:
        if len(row.get("images") or []) > max_images_per_sample:
            excluded["max_images"] += 1
            continue
        if row.get("split") not in train_splits:
            excluded["split"] += 1
            continue
        if row["sample_id"] in eval_ids or _manifest_group_key(row) in eval_group_keys:
            excluded["eval_overlap"] += 1
            continue
        train_rows.append(row)

    if not train_rows:
        raise ValueError("No SFT training rows remain after train/eval split construction.")
    if not eval_pool_rows:
        raise ValueError("No SFT evaluation rows were selected for validation.")

    eval_rows = _sample_stratified(
        eval_pool_rows,
        target_size=eval_sample_size,
        stratum_fields=["source_dataset", "task_type", "split"],
        min_per_stratum=min_eval_samples_per_stratum,
        salt=salt,
    )

    train_ids = {row["sample_id"] for row in train_rows}
    train_group_keys = {_manifest_group_key(row) for row in train_rows}
    exact_overlap = sorted(train_ids.intersection(row["sample_id"] for row in eval_pool_rows))
    group_overlap = sorted(train_group_keys.intersection(_manifest_group_key(row) for row in eval_pool_rows))
    if exact_overlap or group_overlap:
        raise ValueError(
            "Train/eval overlap remains after split construction: exact=%s group=%s"
            % (len(exact_overlap), len(group_overlap))
        )

    write_manifest(train_output_path, train_rows)
    write_manifest(eval_output_path, eval_rows)
    summary = {
        "source_manifest_path": str(source_manifest_path),
        "holdout_manifest_path": str(holdout_manifest_path),
        "train_output_path": str(train_output_path),
        "eval_output_path": str(eval_output_path),
        "train_rows": len(train_rows),
        "eval_pool_rows": len(eval_pool_rows),
        "eval_rows": len(eval_rows),
        "train_splits": list(train_splits),
        "eval_splits": list(eval_splits),
        "max_images_per_sample": max_images_per_sample,
        "eval_sample_size": eval_sample_size,
        "min_eval_samples_per_stratum": min_eval_samples_per_stratum,
        "excluded": dict(excluded),
        "train_by_dataset": _counter_dict(train_rows, "source_dataset"),
        "train_by_task_type": _counter_dict(train_rows, "task_type"),
        "eval_by_dataset": _counter_dict(eval_rows, "source_dataset"),
        "eval_by_task_type": _counter_dict(eval_rows, "task_type"),
        "eval_by_split": _counter_dict(eval_rows, "split"),
        "overlap": {"exact_sample_id": 0, "group_key": 0},
    }
    if summary_output_path:
        write_json(summary_output_path, summary)
    return summary


def build_rl_manifest(
    source_paths: Sequence[Path],
    output_path: Path,
    allowed_task_types: Sequence[str],
    exclude_splits: Sequence[str],
    allowed_verifier_modes: Sequence[str],
    max_answer_words: int,
    max_images_per_sample: int = None,
    holdout_output_path: Path = None,
    holdout_ratio: float = 0.0,
    max_holdout_samples: int = 0,
    min_holdout_per_stratum: int = 1,
    summary_output_path: Path = None,
) -> List[dict]:
    merged_rows = merge_manifests(
        source_paths=source_paths,
        allowed_task_types=allowed_task_types,
        exclude_splits=exclude_splits,
    )
    rewardable_rows = filter_rewardable_manifest(
        merged_rows,
        allowed_verifier_modes=allowed_verifier_modes,
        max_answer_words=max_answer_words,
    )
    if max_images_per_sample is not None:
        rewardable_rows = [row for row in rewardable_rows if len(row.images) <= max_images_per_sample]
    rl_contract_rows = [
        _apply_rl_output_contract(row.model_dump(mode="json") if hasattr(row, "model_dump") else dict(row))
        for row in rewardable_rows
    ]
    deduped_rows, duplicate_count = _dedupe_rl_rows(rl_contract_rows)
    unique_rows = _with_unique_sample_ids(deduped_rows, salt="rl-manifest")
    train_rows, holdout_rows = _split_rl_train_holdout(
        unique_rows,
        holdout_ratio=holdout_ratio,
        max_holdout_samples=max_holdout_samples,
        min_holdout_per_stratum=min_holdout_per_stratum,
        salt="rl-local-holdout",
    )
    if holdout_output_path and holdout_rows:
        write_manifest(holdout_output_path, holdout_rows)
    if summary_output_path:
        write_json(
            summary_output_path,
            {
                "source_paths": [str(path) for path in source_paths],
                "train_output_path": str(output_path),
                "holdout_output_path": str(holdout_output_path) if holdout_output_path else None,
                "merged_rows": len(merged_rows),
                "rewardable_rows": len(rewardable_rows),
                "deduped_rows": len(deduped_rows),
                "duplicate_rows_removed": duplicate_count,
                "train_rows": len(train_rows),
                "holdout_rows": len(holdout_rows),
                "counts": {
                    "train_by_source_dataset": _nested_counter(train_rows, "source_dataset"),
                    "train_by_task_type": _nested_counter(train_rows, "task_type"),
                    "train_by_source_task": _nested_counter(train_rows, "source_dataset", "task_type"),
                    "train_by_crop": _nested_counter(train_rows, "metadata.crop"),
                    "train_by_disease": _nested_counter(train_rows, "metadata.disease"),
                    "holdout_by_source_dataset": _nested_counter(holdout_rows, "source_dataset"),
                    "holdout_by_task_type": _nested_counter(holdout_rows, "task_type"),
                    "holdout_by_source_task": _nested_counter(holdout_rows, "source_dataset", "task_type"),
                },
            },
        )
    return [
        sample.model_dump(mode="json")
        for sample in write_manifest(output_path, train_rows)
    ]


def build_eval_manifests(
    source_paths: Dict[str, Path],
    output_paths: Dict[str, Path],
    holdout_ratio: float,
    holdout_datasets: Sequence[str],
    salt: str,
) -> Dict[str, int]:
    summary = {}

    mirage_rows = read_manifest(source_paths["mirage"]) if source_paths.get("mirage") else []
    mmst_rows = []
    mmmt_rows = []
    for row in mirage_rows:
        track = str(row.metadata.get("benchmark_track") or "").lower()
        if "mmmt" in track or row.task_type == "clarify_or_respond":
            mmmt_rows.append(row)
        else:
            mmst_rows.append(row)
    write_manifest(output_paths["mirage_mmst"], [row.model_dump(mode="json") for row in mmst_rows])
    write_manifest(output_paths["mirage_mmmt"], [row.model_dump(mode="json") for row in mmmt_rows])
    summary["mirage_mmst"] = len(mmst_rows)
    summary["mirage_mmmt"] = len(mmmt_rows)

    holdout_rows = []
    fallback_rows = []
    for dataset_name in holdout_datasets:
        source_path = source_paths.get(dataset_name)
        if not source_path:
            continue
        for row in read_manifest(source_path):
            if row.split == "test":
                continue
            group_key = str(row.metadata.get("source_image_id") or row.images[0])
            payload = row.model_dump(mode="json")
            payload["split"] = "holdout"
            fallback_rows.append(payload)
            if not assign_holdout("%s::%s" % (dataset_name, group_key), salt=salt, holdout_ratio=holdout_ratio):
                continue
            holdout_rows.append(payload)
    if not holdout_rows and fallback_rows:
        holdout_rows.append(fallback_rows[0])
    write_manifest(output_paths["local_holdout"], holdout_rows)
    summary["local_holdout"] = len(holdout_rows)
    return summary
