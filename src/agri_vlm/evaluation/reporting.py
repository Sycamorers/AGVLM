"""Helpers for serializing evaluation predictions."""

from typing import Any, Iterable, List

from agri_vlm.data.conversation_format import target_to_text


def build_prediction_rows(rows: Iterable[Any], predictions: Iterable[str]) -> List[dict]:
    payload = []
    for row, prediction in zip(rows, predictions):
        payload.append(
            {
                "sample_id": row.sample_id,
                "source_dataset": row.source_dataset,
                "task_type": row.task_type,
                "split": row.split,
                "images": list(row.images),
                "prediction": prediction,
                "reference_text": target_to_text(row),
                "answer_text": row.target.answer_text,
                "canonical_label": row.target.canonical_label,
                "acceptable_answers": list(row.target.acceptable_answers),
                "decision": row.target.decision,
                "metadata": row.metadata,
            }
        )
    return payload
