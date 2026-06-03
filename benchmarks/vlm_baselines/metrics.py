"""Task-aware deterministic metrics for agriculture VLM benchmarks."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from statistics import mean
from typing import Any, Callable

from prediction_parsing import (
    detect_forbidden_claims,
    detect_overconfidence,
    extract_answer_field,
    extract_structured_sections,
    normalize_label,
    normalize_text,
    normalize_yes_no,
    parse_numeric_answer,
    parse_prediction_output,
    repetition_stats,
)


def exact_match(reference: str, prediction: str) -> bool:
    return normalize_text(reference) == normalize_text(prediction)


def best_exact_match(references: list[str], prediction: str) -> bool:
    return any(exact_match(reference, prediction) for reference in references)


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = normalize_text(reference).split()
    pred_tokens = normalize_text(prediction).split()
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum((ref_counts & pred_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / float(len(pred_tokens))
    recall = overlap / float(len(ref_tokens))
    return 2 * precision * recall / (precision + recall)


def best_token_f1(references: list[str], prediction: str) -> float:
    return max([token_f1(reference, prediction) for reference in references] + [0.0])


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def normalize_prediction(
    *,
    raw_output: str,
    task_type: str,
    verifier_mode: str,
    label_space: list[str],
) -> tuple[str, bool]:
    """Compatibility wrapper used by older callers."""
    parsed = parse_prediction_output(
        raw_output=raw_output,
        task_type=task_type,
        verifier_mode=verifier_mode,
        label_space=label_space,
    )
    return parsed.normalized_prediction, parsed.invalid_prediction


def parse_prediction_for_metrics(
    *,
    raw_output: str,
    task_type: str,
    verifier_mode: str,
    label_space: list[str],
) -> dict[str, Any]:
    parsed = parse_prediction_output(
        raw_output=raw_output,
        task_type=task_type,
        verifier_mode=verifier_mode,
        label_space=label_space,
    )
    return {
        "parsed_prediction": parsed.parsed_prediction,
        "normalized_prediction": parsed.normalized_prediction,
        "parse_status": parsed.parse_status,
        "invalid_prediction": parsed.invalid_prediction,
        **parsed.extra,
    }


def _references(record: dict[str, Any]) -> list[str]:
    refs = [str(ref) for ref in (record.get("references") or []) if str(ref).strip()]
    gt = str(record.get("ground_truth") or "").strip()
    if gt:
        refs.append(gt)
    deduped: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        key = normalize_text(ref)
        if key and key not in seen:
            seen.add(key)
            deduped.append(ref)
    return deduped


def _prediction(record: dict[str, Any]) -> str:
    if record.get("parsed_prediction") is not None:
        return str(record.get("parsed_prediction") or "")
    if record.get("normalized_prediction") is not None:
        return str(record.get("normalized_prediction") or "")
    answer, _ = extract_answer_field(str(record.get("raw_output") or ""))
    return answer


def _normalized_prediction(record: dict[str, Any]) -> str:
    if record.get("normalized_prediction") is not None:
        return str(record.get("normalized_prediction") or "")
    return normalize_text(_prediction(record))


def classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_examples": 0}
    refs = [normalize_label(record.get("ground_truth")) for record in records]
    preds = [normalize_label(_normalized_prediction(record)) for record in records]
    accepted_refs = []
    for record, ref in zip(records, refs):
        aliases = {ref} if ref else set()
        aliases.update(normalize_label(value) for value in _references(record) if normalize_label(value))
        accepted_refs.append(aliases)
    known_labels = sorted({label for aliases in accepted_refs for label in aliases if label})
    known_label_set = set(known_labels)
    for record in records:
        metadata = record.get("metadata") or {}
        label_space = metadata.get("classification_label_space") or metadata.get("allowed_classification_labels") or []
        if isinstance(label_space, list):
            known_label_set.update(normalize_label(value) for value in label_space if normalize_label(value))
    known_labels = sorted(known_label_set)
    ref_labels = sorted(label for label in set(refs) if label)
    pred_labels = sorted(label for label in set(preds) if label)
    all_labels = sorted(set(ref_labels) | set(pred_labels) | {"<invalid>"})

    correct = sum(1 for ref, pred in zip(refs, preds) if ref and ref == pred)
    accepted_correct = sum(1 for aliases, pred in zip(accepted_refs, preds) if pred and pred in aliases)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ref, pred, record in zip(refs, preds, records):
        pred_label = pred if pred and not record.get("invalid_prediction") else "<invalid>"
        confusion[ref or "<missing_ground_truth>"][pred_label] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    macro_f1_values = []
    recall_values = []
    weighted_f1_sum = 0.0
    total_support = 0
    for label in ref_labels:
        tp = sum(1 for ref, pred in zip(refs, preds) if ref == label and pred == label)
        fp = sum(1 for ref, pred in zip(refs, preds) if ref != label and pred == label)
        fn = sum(1 for ref, pred in zip(refs, preds) if ref == label and pred != label)
        support = sum(1 for ref in refs if ref == label)
        prf = precision_recall_f1(tp, fp, fn)
        per_class[label] = {**prf, "support": support}
        macro_f1_values.append(prf["f1"])
        recall_values.append(prf["recall"])
        weighted_f1_sum += prf["f1"] * support
        total_support += support

    invalid = sum(1 for record in records if record.get("invalid_prediction"))
    missing = sum(1 for record in records if not _normalized_prediction(record))
    out_of_space = sum(
        1
        for record, pred in zip(records, preds)
        if pred
        and (
            record.get("out_of_label_space")
            or record.get("parse_status") == "out_of_label_space"
            or (pred not in known_labels and not record.get("invalid_prediction"))
        )
    )
    parse_status_counts = Counter(str(record.get("parse_status") or "missing") for record in records)
    return {
        "num_examples": len(records),
        "top1_accuracy": correct / float(len(records)),
        "accuracy": correct / float(len(records)),
        "accepted_label_accuracy": accepted_correct / float(len(records)),
        "semantic_alias_accuracy": accepted_correct / float(len(records)),
        "macro_f1": mean(macro_f1_values) if macro_f1_values else 0.0,
        "weighted_f1": weighted_f1_sum / float(total_support) if total_support else 0.0,
        "balanced_accuracy": mean(recall_values) if recall_values else 0.0,
        "invalid_output_rate": invalid / float(len(records)),
        "missing_answer_rate": missing / float(len(records)),
        "out_of_label_space_rate": out_of_space / float(len(records)),
        "parse_status_counts": dict(parse_status_counts),
        "support_per_class": {label: int(metrics["support"]) for label, metrics in per_class.items()},
        "per_class_precision": {label: metrics["precision"] for label, metrics in per_class.items()},
        "per_class_recall": {label: metrics["recall"] for label, metrics in per_class.items()},
        "per_class_f1": {label: metrics["f1"] for label, metrics in per_class.items()},
        "per_class": per_class,
        "confusion_matrix": {ref: dict(preds_by_label) for ref, preds_by_label in confusion.items()},
        "labels": all_labels,
    }


def _is_yes_no_refs(refs: list[str]) -> bool:
    normalized = {normalize_text(ref) for ref in refs if normalize_text(ref)}
    return bool(normalized) and normalized.issubset({"yes", "no"})


def _best_numeric_reference(refs: list[str]) -> float | None:
    for ref in refs:
        value = parse_numeric_answer(ref)
        if value is not None:
            return value
    return None


def _numeric_match(ref_value: float, pred_value: float, *, rel_tol: float = 0.05, abs_tol: float = 0.5) -> bool:
    return abs(ref_value - pred_value) <= max(abs_tol, abs(ref_value) * rel_tol)


def _contradicts_yes_no(prediction: str, refs: list[str]) -> bool:
    if not _is_yes_no_refs(refs):
        return False
    normalized_ref = normalize_text(refs[0])
    pred_yes_no, status = normalize_yes_no(prediction)
    return status == "ambiguous" or (pred_yes_no in {"yes", "no"} and normalized_ref != pred_yes_no)


def vqa_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_examples": 0}
    exact = 0
    normalized_exact = 0
    relaxed = 0
    token_f1_values = []
    containment_values = []
    yes_no_total = 0
    yes_no_correct = 0
    numeric_total = 0
    numeric_correct = 0
    invalid = 0
    missing = 0
    for record in records:
        refs = _references(record)
        pred = _prediction(record)
        normalized_pred = normalize_text(pred)
        invalid += 1 if record.get("invalid_prediction") else 0
        missing += 1 if not normalized_pred else 0
        exact += 1 if any(str(ref).strip() == pred.strip() for ref in refs) else 0
        normalized_exact += 1 if best_exact_match(refs, pred) else 0
        token_f1_values.append(best_token_f1(refs, pred))
        ref_norms = [normalize_text(ref) for ref in refs]
        containment_values.append(1.0 if normalized_pred and any(ref in normalized_pred for ref in ref_norms) else 0.0)

        ref_numeric = _best_numeric_reference(refs)
        pred_numeric = parse_numeric_answer(pred)
        if ref_numeric is not None:
            numeric_total += 1
            if pred_numeric is not None and _numeric_match(ref_numeric, pred_numeric):
                numeric_correct += 1

        if _is_yes_no_refs(refs):
            yes_no_total += 1
            pred_yes_no, status = normalize_yes_no(pred)
            if status != "ambiguous" and pred_yes_no and pred_yes_no in {normalize_text(ref) for ref in refs}:
                yes_no_correct += 1

        relaxed_ok = best_exact_match(refs, pred)
        if ref_numeric is not None and pred_numeric is not None:
            relaxed_ok = relaxed_ok or _numeric_match(ref_numeric, pred_numeric)
        if _is_yes_no_refs(refs):
            pred_yes_no, status = normalize_yes_no(pred)
            relaxed_ok = status != "ambiguous" and pred_yes_no in {normalize_text(ref) for ref in refs}
        if _contradicts_yes_no(pred, refs):
            relaxed_ok = False
        relaxed += 1 if relaxed_ok else 0

    return {
        "num_examples": len(records),
        "exact_match": exact / float(len(records)),
        "normalized_exact_match": normalized_exact / float(len(records)),
        "relaxed_accuracy": relaxed / float(len(records)),
        "token_f1": mean(token_f1_values) if token_f1_values else 0.0,
        "yes_no_accuracy": yes_no_correct / float(yes_no_total) if yes_no_total else None,
        "yes_no_count": yes_no_total,
        "numeric_relaxed_accuracy": numeric_correct / float(numeric_total) if numeric_total else None,
        "numeric_count": numeric_total,
        "answer_containment_score": mean(containment_values) if containment_values else 0.0,
        "invalid_output_rate": invalid / float(len(records)),
        "missing_answer_rate": missing / float(len(records)),
    }


def clarify_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_examples": 0}
    refs = [normalize_text(record.get("ground_truth")) for record in records]
    preds = [_normalized_prediction(record) for record in records]
    labels = ["clarify", "respond"]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ref, pred in zip(refs, preds):
        confusion[ref or "<missing_ground_truth>"][pred if pred in labels else "<invalid>"] += 1

    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(1 for ref, pred in zip(refs, preds) if ref == label and pred == label)
        fp = sum(1 for ref, pred in zip(refs, preds) if ref != label and pred == label)
        fn = sum(1 for ref, pred in zip(refs, preds) if ref == label and pred != label)
        support = sum(1 for ref in refs if ref == label)
        per_label[label] = {**precision_recall_f1(tp, fp, fn), "support": support}

    correct = sum(1 for ref, pred in zip(refs, preds) if ref in labels and ref == pred)
    expected_respond = sum(1 for ref in refs if ref == "respond")
    expected_clarify = sum(1 for ref in refs if ref == "clarify")
    over_clarification = sum(1 for ref, pred in zip(refs, preds) if ref == "respond" and pred == "clarify")
    under_clarification = sum(1 for ref, pred in zip(refs, preds) if ref == "clarify" and pred == "respond")
    invalid = sum(1 for record, pred in zip(records, preds) if record.get("invalid_prediction") or pred not in labels)
    empty_clarifying_question = 0
    empty_answer = 0
    for record, pred in zip(records, preds):
        raw = str(record.get("raw_output") or "")
        answer, _ = extract_answer_field(raw)
        if pred == "clarify" and not answer.strip():
            empty_clarifying_question += 1
        if pred == "respond" and not answer.strip():
            empty_answer += 1

    return {
        "num_examples": len(records),
        "decision_accuracy": correct / float(len(records)),
        "clarify_precision": per_label["clarify"]["precision"],
        "clarify_recall": per_label["clarify"]["recall"],
        "clarify_f1": per_label["clarify"]["f1"],
        "respond_precision": per_label["respond"]["precision"],
        "respond_recall": per_label["respond"]["recall"],
        "respond_f1": per_label["respond"]["f1"],
        "macro_f1": mean([per_label["clarify"]["f1"], per_label["respond"]["f1"]]),
        "confusion_matrix": {ref: dict(preds_by_label) for ref, preds_by_label in confusion.items()},
        "over_clarification_rate": over_clarification / float(expected_respond) if expected_respond else 0.0,
        "under_clarification_rate": under_clarification / float(expected_clarify) if expected_clarify else 0.0,
        "invalid_decision_rate": invalid / float(len(records)),
        "empty_clarifying_question_rate": empty_clarifying_question / float(len(records)),
        "empty_answer_rate": empty_answer / float(len(records)),
        "per_decision": per_label,
    }


def consultation_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_examples": 0}
    section_hits: Counter[str] = Counter()
    required_hits: Counter[str] = Counter()
    required_totals: Counter[str] = Counter()
    structured_scores = []
    required_scores = []
    management_scores = []
    forbidden = 0
    overconfident = 0
    uncertainty_compliant = 0
    uncertainty_total = 0
    followup_present = 0
    followup_expected = 0
    answer_lengths = []
    repetition_rates = []
    max_trigrams = []
    token_f1_values = []

    for record in records:
        raw = str(record.get("raw_output") or record.get("parsed_prediction") or "")
        verifier = record.get("verifier") or {}
        if isinstance(verifier, str):
            verifier = {}
        required_sections = [str(section) for section in verifier.get("required_sections") or []]
        if not required_sections:
            required_sections = ["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"]
        sections = record.get("sections")
        if not isinstance(sections, dict):
            sections = extract_structured_sections(raw)
        normalized_sections = {normalize_text(key).replace(" ", "-"): value for key, value in sections.items()}
        present_required = 0
        for section in required_sections:
            key = normalize_text(section).replace(" ", "-")
            required_totals[section] += 1
            if normalized_sections.get(key, "").strip():
                required_hits[section] += 1
                present_required += 1
        for section in ["diagnosis", "evidence", "uncertainty", "management", "follow-up"]:
            if normalized_sections.get(section, "").strip():
                section_hits[section] += 1
        structured_scores.append(len([value for value in normalized_sections.values() if str(value).strip()]) / 5.0)
        required_scores.append(present_required / float(len(required_sections)) if required_sections else 0.0)

        keywords = [str(value) for value in verifier.get("management_keywords") or [] if str(value).strip()]
        if keywords:
            normalized_raw = normalize_text(raw)
            hits = sum(1 for keyword in keywords if normalize_text(keyword) in normalized_raw)
            management_scores.append(min(hits / float(len(keywords)), 1.0))
        else:
            management_scores.append(1.0 if normalized_sections.get("management", "").strip() else 0.0)

        forbidden_claims = detect_forbidden_claims(raw, verifier.get("forbidden_claims") or [])
        forbidden += 1 if forbidden_claims else 0
        overconfidence_markers = detect_overconfidence(raw)
        overconfident += 1 if overconfidence_markers else 0

        uncertainty_required = bool(verifier.get("uncertainty_required"))
        if uncertainty_required:
            uncertainty_total += 1
            uncertainty_text = normalize_text(normalized_sections.get("uncertainty", "") + " " + raw)
            cautious = any(
                marker in uncertainty_text
                for marker in ["may", "might", "possible", "uncertain", "cannot confirm", "confirm", "inspect", "sample"]
            )
            if cautious and not overconfidence_markers:
                uncertainty_compliant += 1
        if uncertainty_required or record.get("task_type") == "consultation":
            followup_expected += 1
            if normalized_sections.get("follow-up", "").strip() or "?" in raw:
                followup_present += 1

        rep = repetition_stats(raw)
        answer_lengths.append(int(rep["token_count"]))
        repetition_rates.append(float(rep["repetition_rate"]))
        max_trigrams.append(int(rep["max_trigram_count"]))
        refs = _references(record)
        if refs:
            token_f1_values.append(best_token_f1(refs, raw))

    return {
        "num_examples": len(records),
        "structured_section_compliance": mean(structured_scores) if structured_scores else 0.0,
        "required_section_compliance": mean(required_scores) if required_scores else 0.0,
        "required_section_compliance_by_section": {
            section: required_hits[section] / float(required_totals[section]) if required_totals[section] else 0.0
            for section in sorted(required_totals)
        },
        "section_presence_rate": {section: section_hits[section] / float(len(records)) for section in section_hits},
        "management_keyword_coverage": mean(management_scores) if management_scores else 0.0,
        "forbidden_claim_rate": forbidden / float(len(records)),
        "unsafe_or_overconfident_claim_rate": overconfident / float(len(records)),
        "uncertainty_compliance": uncertainty_compliant / float(uncertainty_total) if uncertainty_total else None,
        "uncertainty_required_count": uncertainty_total,
        "followup_question_presence": followup_present / float(followup_expected) if followup_expected else None,
        "answer_length_stats": {
            "min": min(answer_lengths) if answer_lengths else 0,
            "max": max(answer_lengths) if answer_lengths else 0,
            "mean": mean(answer_lengths) if answer_lengths else 0.0,
        },
        "repetition_rate": mean(repetition_rates) if repetition_rates else 0.0,
        "max_trigram_count_mean": mean(max_trigrams) if max_trigrams else 0.0,
        "token_f1_diagnostic": mean(token_f1_values) if token_f1_values else None,
        "limitations": "Deterministic consultation metrics check format, keyword coverage, uncertainty, and safety proxies; they do not fully verify agronomic correctness.",
    }


def _task_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "classification": [
            record
            for record in records
            if record.get("verifier_mode") == "label"
            or record.get("task_type") in {"classification", "label_diagnosis"}
        ],
        "short_vqa": [
            record
            for record in records
            if record.get("task_type") == "vqa"
            or record.get("verifier_mode") in {"exact_match", "synonym"}
        ],
        "clarify_or_respond": [
            record
            for record in records
            if record.get("task_type") == "clarify_or_respond" or record.get("verifier_mode") == "clarify"
        ],
        "consultation": [
            record
            for record in records
            if record.get("task_type") == "consultation" or record.get("verifier_mode") == "structured"
        ],
    }


def _primary_score(task_name: str, payload: dict[str, Any]) -> float | None:
    if not payload or not payload.get("num_examples"):
        return None
    if task_name == "classification":
        return float(payload.get("macro_f1", 0.0))
    if task_name == "short_vqa":
        return float(payload.get("relaxed_accuracy", 0.0))
    if task_name == "clarify_or_respond":
        return float(payload.get("macro_f1", 0.0))
    if task_name == "consultation":
        return float(payload.get("structured_section_compliance", 0.0))
    return None


def _group_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(record.get(key) or "missing") for record in records))


def _crop_disease_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    crops: Counter[str] = Counter()
    diseases: Counter[str] = Counter()
    for record in records:
        metadata = record.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        crops[str(record.get("crop") or metadata.get("crop") or "missing")] += 1
        diseases[str(record.get("disease") or metadata.get("disease") or "missing")] += 1
    return {"crop": dict(crops), "disease": dict(diseases)}


def _per_source_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("source_dataset") or "missing")].append(record)
    for source, source_records in sorted(grouped.items()):
        families = _task_records([record for record in source_records if not record.get("error_message")])
        task_payloads = {
            "classification": classification_metrics(families["classification"]),
            "short_vqa": vqa_metrics(families["short_vqa"]),
            "clarify_or_respond": clarify_metrics(families["clarify_or_respond"]),
            "consultation": consultation_metrics(families["consultation"]),
        }
        scores = [
            score
            for task_name, payload in task_payloads.items()
            for score in [_primary_score(task_name, payload)]
            if score is not None
        ]
        output[source] = {
            "num_examples": len(source_records),
            "failure_rate": sum(1 for record in source_records if record.get("error_message")) / float(len(source_records)),
            "invalid_prediction_rate": sum(1 for record in source_records if record.get("invalid_prediction"))
            / float(len(source_records)),
            "by_task_type": _group_counts(source_records, "task_type"),
            "task_macro_average": mean(scores) if scores else None,
            "tasks": task_payloads,
        }
    return output


def _bootstrap_ci(
    records: list[dict[str, Any]],
    scorer: Callable[[list[dict[str, Any]]], float | None],
    *,
    samples: int,
    seed: int,
) -> dict[str, float] | None:
    if not records or samples <= 0:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sample = [records[rng.randrange(len(records))] for _ in range(len(records))]
        value = scorer(sample)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    if not values:
        return None
    values.sort()
    low_index = int(0.025 * (len(values) - 1))
    high_index = int(0.975 * (len(values) - 1))
    return {"low": values[low_index], "high": values[high_index], "samples": len(values)}


def evaluate_prediction_records(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    total = len(records)
    failed = [record for record in records if record.get("error_message")]
    valid_for_metrics = [record for record in records if not record.get("error_message")]
    invalid = [record for record in records if record.get("invalid_prediction")]
    by_phase = _group_counts(records, "phase")
    by_split = _group_counts(records, "split")
    families = _task_records(valid_for_metrics)
    task_metrics = {
        "classification": classification_metrics(families["classification"]),
        "short_vqa": vqa_metrics(families["short_vqa"]),
        "vqa": vqa_metrics(families["short_vqa"]),
        "clarify_or_respond": clarify_metrics(families["clarify_or_respond"]),
        "consultation": consultation_metrics(families["consultation"]),
    }
    primary_scores = [
        score
        for task_name in ["classification", "short_vqa", "clarify_or_respond", "consultation"]
        for score in [_primary_score(task_name, task_metrics[task_name])]
        if score is not None
    ]
    metrics = {
        "num_examples": total,
        "overall_num_examples": total,
        "num_failed": len(failed),
        "failure_rate": len(failed) / float(total) if total else math.nan,
        "num_invalid_predictions": len(invalid),
        "invalid_prediction_rate": len(invalid) / float(total) if total else math.nan,
        "by_phase": by_phase,
        "by_split": by_split,
        "by_task_type": _group_counts(records, "task_type"),
        "by_source_dataset": _group_counts(records, "source_dataset"),
        "by_verifier_mode": _group_counts(records, "verifier_mode"),
        "by_crop_disease": _crop_disease_counts(records),
        "task_macro_average": mean(primary_scores) if primary_scores else None,
        "task_micro_average_note": "Micro metrics are reported inside each task family; the aggregate macro average prevents majority tasks from dominating.",
        "per_task": task_metrics,
        "classification": task_metrics["classification"],
        "vqa": task_metrics["short_vqa"],
        "short_vqa": task_metrics["short_vqa"],
        "clarify_or_respond": task_metrics["clarify_or_respond"],
        "consultation": task_metrics["consultation"],
        "per_source_dataset": _per_source_summary(records),
        "per_phase": {},
    }
    for phase in sorted(by_phase):
        phase_records = [record for record in records if str(record.get("phase") or "missing") == phase]
        if len(phase_records) == len(records):
            metrics["per_phase"][phase] = {
                "num_examples": len(phase_records),
                "failure_rate": metrics["failure_rate"],
                "invalid_prediction_rate": metrics["invalid_prediction_rate"],
                "task_macro_average": metrics["task_macro_average"],
            }
        else:
            phase_payload = evaluate_prediction_records(phase_records, bootstrap_samples=0)
            metrics["per_phase"][phase] = {
                "num_examples": phase_payload["num_examples"],
                "failure_rate": phase_payload["failure_rate"],
                "invalid_prediction_rate": phase_payload["invalid_prediction_rate"],
                "task_macro_average": phase_payload["task_macro_average"],
            }

    if bootstrap_samples:
        metrics["confidence_intervals"] = {
            "task_macro_average": _bootstrap_ci(
                records,
                lambda rows: evaluate_prediction_records(rows, bootstrap_samples=0).get("task_macro_average"),
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        }
    return metrics
