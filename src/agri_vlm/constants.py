"""Project-wide constants."""

APP_NAME = "agri-vlm-v1"

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
OPTIONAL_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

TASK_TYPE_CLASSIFICATION = "classification"
TASK_TYPE_VQA = "vqa"
TASK_TYPE_SYMPTOM_EXPLANATION = "symptom_explanation"
TASK_TYPE_CONSULTATION = "consultation"
TASK_TYPE_CLARIFY = "clarify_or_respond"

SUPPORTED_TASK_TYPES = [
    TASK_TYPE_CLASSIFICATION,
    TASK_TYPE_VQA,
    TASK_TYPE_SYMPTOM_EXPLANATION,
    TASK_TYPE_CONSULTATION,
    TASK_TYPE_CLARIFY,
]

SUPPORTED_SPLITS = ["train", "validation", "test", "holdout", "dev"]

DEFAULT_CONSULTATION_SECTIONS = [
    "Diagnosis",
    "Evidence",
    "Uncertainty",
    "Management",
    "Follow-up",
]

MANUAL_DATASET_PLACEHOLDER_PREFIX = "PLACEHOLDER_"
