"""Normalization helpers for agricultural labels and prompts."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agri_vlm.constants import DEFAULT_CONSULTATION_SECTIONS
from agri_vlm.utils.text import normalize_label, normalize_whitespace


def slugify(text: str) -> str:
    return normalize_label(text).replace(" ", "-")


def normalize_split_name(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = normalize_label(value)
    if normalized in {"val", "valid"}:
        return "validation"
    if normalized == "dev":
        return "dev"
    return normalized


def detect_split_from_path(path: Path) -> Optional[str]:
    for part in path.parts:
        normalized = normalize_split_name(part)
        if normalized in {"train", "validation", "test", "dev"}:
            return normalized
    return None


def parse_plant_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    if "___" in label:
        crop, disease = label.split("___", 1)
        return normalize_whitespace(crop.replace("_", " ")), normalize_whitespace(
            disease.replace("_", " ")
        )
    if "__" in label:
        crop, disease = label.split("__", 1)
        return normalize_whitespace(crop.replace("_", " ")), normalize_whitespace(
            disease.replace("_", " ")
        )
    return None, normalize_whitespace(label.replace("_", " "))


def parse_ip102_label(label: str) -> str:
    return normalize_whitespace(label.replace("_", " ").replace("-", " "))


def default_system_prompt() -> str:
    return (
        "You are an agricultural vision-language assistant focused on ground-level RGB crop "
        "disease, pest, symptom, and consultation tasks."
    )


def build_user_message_text(task_type: str, question: Optional[str] = None) -> str:
    if question:
        return normalize_whitespace(question)
    if task_type == "classification":
        return "Identify the crop issue or pest in this agricultural image."
    if task_type == "symptom_explanation":
        return "Explain the visible symptoms in this agricultural image."
    if task_type == "clarify_or_respond":
        return "Decide whether to clarify first or respond directly, then act accordingly."
    return "Answer the agricultural question using only the evidence available in the image."


def build_structured_consultation_answer(
    diagnosis: str, management_steps: List[str], uncertainty: str
) -> str:
    management_text = "; ".join([step.strip() for step in management_steps if step.strip()])
    lines = [
        "%s: %s" % (DEFAULT_CONSULTATION_SECTIONS[0], diagnosis),
        "%s: Visual evidence should be checked against crop-specific symptom patterns."
        % DEFAULT_CONSULTATION_SECTIONS[1],
        "%s: %s" % (DEFAULT_CONSULTATION_SECTIONS[2], uncertainty),
        "%s: %s" % (DEFAULT_CONSULTATION_SECTIONS[3], management_text or "No management steps provided."),
        "%s: Request clearer close-up images and field context if confidence is limited."
        % DEFAULT_CONSULTATION_SECTIONS[4],
    ]
    return "\n".join(lines)


def relative_posix_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path, base_dir).replace(os.sep, "/")


def metadata_with_license(base: Optional[Dict[str, Any]] = None, license_name: Optional[str] = None) -> Dict[str, Any]:
    payload = dict(base or {})
    if license_name:
        payload["license"] = license_name
    return payload
