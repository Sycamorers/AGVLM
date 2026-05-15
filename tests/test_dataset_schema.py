from agri_vlm.schemas.dataset_schema import UnifiedSample
import pytest


def valid_sample() -> dict:
    return {
        "sample_id": "sample-1",
        "source_dataset": "plantdoc",
        "task_type": "classification",
        "split": "train",
        "images": ["data/raw/_smoke/leaf.png"],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an agricultural assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data/raw/_smoke/leaf.png"},
                    {"type": "text", "text": "Identify the disease."},
                ],
            },
        ],
        "target": {"answer_text": "tomato early blight", "canonical_label": "tomato early blight"},
        "metadata": {"crop": "tomato"},
        "verifier": {"mode": "label", "accepted_labels": ["tomato early blight"]},
        "reward_meta": {"weights": {"normalized_label": 1.0}},
    }


def test_dataset_schema_accepts_valid_sample() -> None:
    sample = UnifiedSample.model_validate(valid_sample())
    assert sample.target.canonical_label == "tomato early blight"


def test_dataset_schema_accepts_optional_preference_annotation() -> None:
    payload = valid_sample()
    payload["preference"] = {
        "preference_score": 0.8,
        "preference_rationale": "Expert preferred the calibrated answer.",
        "chosen_response": "Answer: tomato early blight",
        "rejected_response": "Answer: potato late blight",
        "expert_quality_score": 0.9,
        "agronomic_correctness_score": 0.9,
        "management_usefulness_score": 0.7,
        "uncertainty_calibration_score": 0.8,
        "safety_score": 1.0,
    }
    sample = UnifiedSample.model_validate(payload)
    assert sample.preference is not None
    assert sample.preference.chosen_response == "Answer: tomato early blight"


def test_dataset_schema_rejects_missing_images() -> None:
    payload = valid_sample()
    payload["images"] = []
    with pytest.raises(Exception):
        UnifiedSample.model_validate(payload)
