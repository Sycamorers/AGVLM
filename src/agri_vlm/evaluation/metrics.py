"""Evaluation metrics."""

from typing import Dict, Sequence

from agri_vlm.rewards.clarify_decision import infer_decision
from agri_vlm.utils.text import best_exact_match, normalize_label


def accuracy(references: Sequence[str], predictions: Sequence[str]) -> float:
    if not references:
        return 0.0
    hits = 0
    for reference, prediction in zip(references, predictions):
        if normalize_label(reference) == normalize_label(prediction):
            hits += 1
    return hits / float(len(references))


def macro_f1(references: Sequence[str], predictions: Sequence[str]) -> float:
    labels = sorted({normalize_label(item) for item in list(references) + list(predictions)})
    if not labels:
        return 0.0
    f1_values = []
    normalized_refs = [normalize_label(item) for item in references]
    normalized_preds = [normalize_label(item) for item in predictions]
    for label in labels:
        tp = fp = fn = 0
        for ref, pred in zip(normalized_refs, normalized_preds):
            if pred == label and ref == label:
                tp += 1
            elif pred == label and ref != label:
                fp += 1
            elif pred != label and ref == label:
                fn += 1
        precision = tp / float(tp + fp) if (tp + fp) else 0.0
        recall = tp / float(tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))
    return sum(f1_values) / float(len(f1_values))


def exact_match_rate(references: Sequence[Sequence[str]], predictions: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    return sum(best_exact_match(reference, prediction) for reference, prediction in zip(references, predictions)) / float(len(predictions))


def clarify_accuracy(references: Sequence[str], predictions: Sequence[str]) -> float:
    if not references:
        return 0.0
    return sum(
        1.0 if infer_decision(prediction) == reference else 0.0
        for reference, prediction in zip(references, predictions)
    ) / float(len(references))


def clarify_decision_metrics(references: Sequence[str], predictions: Sequence[str]) -> Dict[str, float]:
    """Compute decision metrics for the respond-vs-clarify action space."""
    if not references:
        return {
            "clarify_accuracy": 0.0,
            "clarify_precision": 0.0,
            "clarify_recall": 0.0,
            "unnecessary_clarification_rate": 0.0,
            "premature_answer_rate": 0.0,
            "predicted_clarify_rate": 0.0,
            "expected_clarify_rate": 0.0,
        }

    inferred = [infer_decision(prediction) for prediction in predictions]
    total = float(len(references))
    tp = fp = fn = tn = 0
    for reference, prediction in zip(references, inferred):
        if reference == "clarify" and prediction == "clarify":
            tp += 1
        elif reference != "clarify" and prediction == "clarify":
            fp += 1
        elif reference == "clarify" and prediction != "clarify":
            fn += 1
        else:
            tn += 1
    clarify_precision = tp / float(tp + fp) if tp + fp else 0.0
    clarify_recall = tp / float(tp + fn) if tp + fn else 0.0
    expected_respond = fp + tn
    expected_clarify = tp + fn
    return {
        "clarify_accuracy": (tp + tn) / total,
        "clarify_precision": clarify_precision,
        "clarify_recall": clarify_recall,
        "unnecessary_clarification_rate": fp / float(expected_respond) if expected_respond else 0.0,
        "premature_answer_rate": fn / float(expected_clarify) if expected_clarify else 0.0,
        "predicted_clarify_rate": (tp + fp) / total,
        "expected_clarify_rate": expected_clarify / total,
    }
