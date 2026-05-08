"""Lightweight benchmark metrics with no external evaluator dependency."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any

from utils import normalize_text


def strip_leading_class_number(text: str) -> str:
    return re.sub(r"^\s*\d+\s+", "", normalize_text(text)).strip()


def metric_label(text: str) -> str:
    normalized = normalize_text(text)
    stripped = strip_leading_class_number(normalized)
    return stripped or normalized


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


def infer_decision(prediction: str) -> str:
    normalized = normalize_text(prediction)
    if not normalized:
        return ""
    first = normalized.split()[0]
    if first in {"clarify", "respond"}:
        return first
    clarify_markers = [
        "need more information",
        "more information",
        "could you",
        "please provide",
        "please describe",
        "need additional",
        "cannot determine",
        "not enough information",
    ]
    if any(marker in normalized for marker in clarify_markers):
        return "clarify"
    return "respond"


def extract_label_prediction(raw_output: str, label_space: list[str]) -> tuple[str, bool]:
    normalized_output = normalize_text(raw_output)
    if not normalized_output:
        return "", True
    by_metric_label: dict[str, str] = {}
    for label in label_space:
        key = metric_label(label)
        if key:
            by_metric_label.setdefault(key, label)

    matches: list[tuple[int, str]] = []
    output_metric = metric_label(normalized_output)
    for key, label in by_metric_label.items():
        if output_metric == key or normalized_output == normalize_text(label):
            matches.append((len(key), label))
        elif key and re.search(r"(^|\s)%s($|\s)" % re.escape(key), normalized_output):
            matches.append((len(key), label))
    if not matches:
        return normalized_output, True
    matches.sort(reverse=True)
    longest = matches[0][0]
    winners = [label for length, label in matches if length == longest]
    unique = sorted(set(winners))
    if len(unique) != 1:
        return normalized_output, True
    return unique[0], False


def normalize_prediction(
    *,
    raw_output: str,
    task_type: str,
    verifier_mode: str,
    label_space: list[str],
) -> tuple[str, bool]:
    if verifier_mode == "label":
        return extract_label_prediction(raw_output, label_space)
    if verifier_mode == "clarify" or task_type == "clarify_or_respond":
        decision = infer_decision(raw_output)
        return decision, not bool(decision)
    return (raw_output or "").strip(), not bool((raw_output or "").strip())


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    refs = [metric_label(record.get("ground_truth", "")) for record in records]
    preds = [metric_label(record.get("normalized_prediction", "")) for record in records]
    labels = sorted(set(refs) | set(preds))
    if not records:
        return {"num_examples": 0}

    correct = sum(1 for ref, pred in zip(refs, preds) if ref and ref == pred)
    per_class: dict[str, dict[str, float | int]] = {}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ref, pred in zip(refs, preds):
        confusion[ref][pred] += 1

    macro_f1_values = []
    weighted_f1_sum = 0.0
    total_support = 0
    for label in labels:
        tp = sum(1 for ref, pred in zip(refs, preds) if ref == label and pred == label)
        fp = sum(1 for ref, pred in zip(refs, preds) if ref != label and pred == label)
        fn = sum(1 for ref, pred in zip(refs, preds) if ref == label and pred != label)
        support = sum(1 for ref in refs if ref == label)
        prf = precision_recall_f1(tp, fp, fn)
        per_class[label] = {**prf, "support": support}
        macro_f1_values.append(prf["f1"])
        weighted_f1_sum += prf["f1"] * support
        total_support += support

    invalid = sum(1 for record in records if record.get("invalid_prediction"))
    return {
        "num_examples": len(records),
        "accuracy": correct / float(len(records)),
        "macro_f1": sum(macro_f1_values) / float(len(macro_f1_values)) if macro_f1_values else 0.0,
        "weighted_f1": weighted_f1_sum / float(total_support) if total_support else 0.0,
        "invalid_output_rate": invalid / float(len(records)),
        "per_class": per_class,
        "confusion_matrix": {ref: dict(preds_by_label) for ref, preds_by_label in confusion.items()},
    }


def vqa_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_examples": 0}
    exact = 0
    relaxed = 0
    f1_values = []
    invalid = 0
    for record in records:
        refs = record.get("references") or [record.get("ground_truth", "")]
        pred = record.get("normalized_prediction") or record.get("raw_output") or ""
        exact += 1 if best_exact_match(refs, pred) else 0
        normalized_refs = {normalize_text(ref) for ref in refs}
        normalized_pred = normalize_text(pred)
        if normalized_refs.issubset({"yes", "no"}) and normalized_pred.split()[:1]:
            relaxed += 1 if normalized_pred.split()[0] in normalized_refs else 0
        else:
            relaxed += 1 if best_exact_match(refs, pred) else 0
        f1_values.append(best_token_f1(refs, pred))
        invalid += 1 if record.get("invalid_prediction") else 0
    return {
        "num_examples": len(records),
        "exact_match": exact / float(len(records)),
        "relaxed_accuracy": relaxed / float(len(records)),
        "token_f1": sum(f1_values) / float(len(f1_values)) if f1_values else 0.0,
        "invalid_or_missing_answer_rate": invalid / float(len(records)),
    }


def clarify_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_examples": 0}
    refs = [normalize_text(record.get("ground_truth", "")) for record in records]
    preds = [normalize_text(record.get("normalized_prediction", "")) for record in records]
    tp = fp = fn = tn = 0
    for ref, pred in zip(refs, preds):
        if ref == "clarify" and pred == "clarify":
            tp += 1
        elif ref != "clarify" and pred == "clarify":
            fp += 1
        elif ref == "clarify" and pred != "clarify":
            fn += 1
        else:
            tn += 1
    total = float(len(records))
    prf = precision_recall_f1(tp, fp, fn)
    expected_respond = fp + tn
    expected_clarify = tp + fn
    return {
        "num_examples": len(records),
        "clarify_accuracy": (tp + tn) / total,
        "clarify_precision": prf["precision"],
        "clarify_recall": prf["recall"],
        "clarify_f1": prf["f1"],
        "unnecessary_clarification_rate": fp / float(expected_respond) if expected_respond else 0.0,
        "premature_answer_rate": fn / float(expected_clarify) if expected_clarify else 0.0,
        "predicted_clarify_rate": (tp + fp) / total,
        "expected_clarify_rate": expected_clarify / total,
    }


def evaluate_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    failed = [record for record in records if record.get("error_message")]
    invalid = [record for record in records if record.get("invalid_prediction")]
    by_task = Counter(record.get("task_type") or "missing" for record in records)
    by_dataset = Counter(record.get("source_dataset") or "missing" for record in records)
    by_mode = Counter(record.get("verifier_mode") or "missing" for record in records)

    classification = [
        record
        for record in records
        if record.get("verifier_mode") == "label" and not record.get("error_message")
    ]
    vqa = [
        record
        for record in records
        if record.get("verifier_mode") == "exact_match" and not record.get("error_message")
    ]
    clarify = [
        record
        for record in records
        if record.get("verifier_mode") == "clarify" and not record.get("error_message")
    ]

    metrics = {
        "num_examples": total,
        "num_failed": len(failed),
        "failure_rate": len(failed) / float(total) if total else math.nan,
        "num_invalid_predictions": len(invalid),
        "invalid_prediction_rate": len(invalid) / float(total) if total else math.nan,
        "by_task_type": dict(by_task),
        "by_source_dataset": dict(by_dataset),
        "by_verifier_mode": dict(by_mode),
        "classification": classification_metrics(classification),
        "vqa": vqa_metrics(vqa),
        "clarify_or_respond": clarify_metrics(clarify),
    }
    return metrics
