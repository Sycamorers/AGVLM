"""Model/checkpoint config loading and validation for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from prediction_parsing import PLACEHOLDER_PATH_MARKERS, normalize_text
from utils import REPO_ROOT, load_yaml, model_slug


RAW_PHI4_REASONING = "microsoft/Phi-4-reasoning-vision-15B"
EXTERNAL_CHECKPOINT_TYPES = {"external_baseline", "base"}
PROJECT_CHECKPOINT_TYPES = {"sft", "rl"}
ADAPTER_TYPE_TO_CHECKPOINT_TYPE = {
    "hf_base": "base",
    "hf_lora": "sft",
    "agvlm_sft": "sft",
    "agvlm_rl": "rl",
    "merged_checkpoint": "sft",
    "external_baseline": "external_baseline",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def _is_placeholder(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = normalize_text(text)
    return lowered in PLACEHOLDER_PATH_MARKERS or "todo" in lowered or "change me" in lowered or "/path/to/" in text


def _resolve_path(path_value: Any) -> Path | None:
    if _is_placeholder(path_value):
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _adapter_tensor_summary(path: Path) -> tuple[int, int, list[str]]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            non_empty = sum(1 for key in keys if int(handle.get_tensor(key).numel()) > 0)
        return len(keys), non_empty, keys
    if path.suffix == ".bin":
        import torch

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            state_dict = payload.get("state_dict") if isinstance(payload.get("state_dict"), dict) else payload
        else:
            state_dict = {}
        tensor_items = [(key, value) for key, value in state_dict.items() if hasattr(value, "numel")]
        non_empty = sum(1 for _, value in tensor_items if int(value.numel()) > 0)
        return len(tensor_items), non_empty, [key for key, _ in tensor_items]
    return 0, 0, []


def _validate_peft_adapter_path(path: Path, *, model_key: str) -> list[str]:
    errors: list[str] = []
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        return ["Model %s adapter path is missing adapter_config.json: %s" % (model_key, path)]
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["Model %s adapter_config.json is not valid JSON: %s" % (model_key, config_path)]

    tensor_path = None
    for name in ["adapter_model.safetensors", "adapter_model.bin"]:
        candidate = path / name
        if candidate.is_file():
            tensor_path = candidate
            break
    if tensor_path is None:
        return ["Model %s adapter path is missing adapter_model.safetensors or adapter_model.bin: %s" % (model_key, path)]
    if tensor_path.stat().st_size <= 0:
        return ["Model %s adapter tensor file is empty: %s" % (model_key, tensor_path)]
    try:
        num_tensors, non_empty_tensors, tensor_names = _adapter_tensor_summary(tensor_path)
    except Exception as exc:
        return ["Model %s adapter tensor file could not be inspected: %s (%s)" % (model_key, tensor_path, exc)]
    if num_tensors <= 0:
        errors.append("Model %s adapter tensor file contains no tensors: %s" % (model_key, tensor_path))
    if non_empty_tensors <= 0:
        errors.append("Model %s adapter tensor file contains no non-empty tensors: %s" % (model_key, tensor_path))
    if str(adapter_config.get("peft_type") or "").lower() == "lora" and not any("lora_" in name for name in tensor_names):
        errors.append("Model %s LoRA adapter tensor file contains no lora_* tensors: %s" % (model_key, tensor_path))
    return errors


def _path_has_model_artifacts(path: Path, *, model_key: str) -> list[str]:
    if path.is_file():
        return [] if path.stat().st_size > 0 else ["Model %s checkpoint file is empty: %s" % (model_key, path)]
    if not path.is_dir():
        return ["Model %s checkpoint/adapter path is not a directory or file: %s" % (model_key, path)]
    if (path / "adapter_config.json").exists():
        return _validate_peft_adapter_path(path, model_key=model_key)
    if (path / "config.json").exists():
        has_weights = any(
            candidate.is_file() and candidate.stat().st_size > 0
            for pattern in ["model.safetensors", "pytorch_model.bin", "model-*.safetensors", "pytorch_model-*.bin"]
            for candidate in path.glob(pattern)
        )
        if has_weights or (path / "model.safetensors.index.json").exists() or (path / "pytorch_model.bin.index.json").exists():
            return []
    return ["Model %s checkpoint/adapter path has no model artifacts: %s" % (model_key, path)]


def _entry_from_baseline(raw: dict[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name") or raw.get("model_name") or "")
    key = str(raw.get("key") or raw.get("model_key") or model_slug(name))
    return {
        **raw,
        "model_key": key,
        "model_name": name,
        "checkpoint_type": "external_baseline",
        "adapter_type": "external_baseline",
        "base_model_name_or_path": name,
        "checkpoint_path": raw.get("checkpoint_path") or "",
        "adapter_path": raw.get("adapter_path") or "",
        "processor_name_or_path": raw.get("processor_name_or_path") or name,
        "model_family": raw.get("model_family") or "hf_vlm",
        "max_images": raw.get("max_images"),
        "image_policy": raw.get("image_policy") or "",
    }


def _entry_from_checkpoint(raw: dict[str, Any]) -> dict[str, Any]:
    key = str(raw.get("model_key") or raw.get("key") or "")
    name = str(raw.get("model_name") or raw.get("name") or key)
    adapter_type = str(raw.get("adapter_type") or "")
    checkpoint_type = str(raw.get("checkpoint_type") or ADAPTER_TYPE_TO_CHECKPOINT_TYPE.get(adapter_type, adapter_type))
    return {
        **raw,
        "model_key": key,
        "model_name": name,
        "checkpoint_type": checkpoint_type,
        "adapter_type": adapter_type,
        "base_model_name_or_path": raw.get("base_model_name_or_path") or raw.get("base_model") or "",
        "adapter_path": raw.get("adapter_path") or "",
        "checkpoint_path": raw.get("checkpoint_path") or "",
        "processor_name_or_path": raw.get("processor_name_or_path") or raw.get("base_model_name_or_path") or "",
        "model_family": raw.get("model_family") or "phi4_reasoning_vision",
        "max_images": raw.get("max_images", 3),
        "image_policy": raw.get("image_policy") or "all_images",
    }


def load_model_configurations(
    *,
    model_config_path: Path | None,
    checkpoint_config_path: Path | None,
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if model_config_path and model_config_path.exists():
        payload = load_yaml(model_config_path)
        for raw in payload.get("models", []):
            entry = _entry_from_baseline(raw)
            entries[entry["model_key"]] = entry
            entries[entry["model_name"]] = entry
    if checkpoint_config_path and checkpoint_config_path.exists():
        payload = load_yaml(checkpoint_config_path)
        raw_models = payload.get("models", [])
        if isinstance(raw_models, dict):
            raw_models = [{**value, "model_key": key} for key, value in raw_models.items()]
        for raw in raw_models:
            entry = _entry_from_checkpoint(raw)
            entries[entry["model_key"]] = entry
            entries[entry["model_name"]] = entry
    return entries


def validate_model_entry(
    entry: dict[str, Any],
    *,
    phase: str,
    require_runnable: bool,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    checkpoint_type = str(entry.get("checkpoint_type") or "")
    adapter_type = str(entry.get("adapter_type") or "")
    model_key = str(entry.get("model_key") or entry.get("model_name") or "")
    base_model = str(entry.get("base_model_name_or_path") or "")
    checkpoint_path = entry.get("checkpoint_path") or ""
    adapter_path = entry.get("adapter_path") or ""

    if checkpoint_type not in EXTERNAL_CHECKPOINT_TYPES | PROJECT_CHECKPOINT_TYPES:
        errors.append("Model %s has unsupported checkpoint_type=%r." % (model_key, checkpoint_type))
    if adapter_type and adapter_type not in ADAPTER_TYPE_TO_CHECKPOINT_TYPE:
        errors.append("Model %s has unsupported adapter_type=%r." % (model_key, adapter_type))

    if checkpoint_type in PROJECT_CHECKPOINT_TYPES:
        path_value = adapter_path if adapter_path else checkpoint_path
        resolved = _resolve_path(path_value)
        if resolved is None:
            message = (
                "Model %s has a placeholder or empty checkpoint/adapter path. Replace it with a completed checkpoint before running."
                % model_key
            )
            (errors if require_runnable else warnings).append(message)
        elif require_runnable and not resolved.exists():
            errors.append("Model %s checkpoint/adapter path does not exist: %s" % (model_key, resolved))
        elif require_runnable:
            errors.extend(_path_has_model_artifacts(resolved, model_key=model_key))

    if checkpoint_type == "sft":
        if normalize_text(base_model) == normalize_text(RAW_PHI4_REASONING) and _is_placeholder(adapter_path) and _is_placeholder(checkpoint_path):
            message = "SFT model %s cannot point only to raw %s; provide a completed SFT checkpoint or mark it as base." % (model_key, RAW_PHI4_REASONING)
            (errors if require_runnable else warnings).append(message)
    if checkpoint_type == "rl":
        if normalize_text(str(checkpoint_path or adapter_path or base_model)) == normalize_text(RAW_PHI4_REASONING):
            errors.append("RL model %s cannot point directly to raw %s." % (model_key, RAW_PHI4_REASONING))
        initialized_from = entry.get("initialized_from_sft_checkpoint") or ""
        if _is_placeholder(initialized_from):
            message = "RL model %s must record initialized_from_sft_checkpoint." % model_key
            (errors if require_runnable else warnings).append(message)
        elif require_runnable:
            resolved = _resolve_path(initialized_from)
            if resolved is not None and not resolved.exists():
                errors.append("RL model %s initialized_from_sft_checkpoint does not exist: %s" % (model_key, resolved))

    if phase == "rl" and checkpoint_type == "rl":
        initialized_from = entry.get("initialized_from_sft_checkpoint") or ""
        if _is_placeholder(initialized_from):
            message = "RL benchmark entry %s must identify the completed SFT checkpoint used to initialize RL." % model_key
            (errors if require_runnable else warnings).append(message)
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def resolve_model_entry(
    *,
    model_key: str | None,
    model_name: str | None,
    model_config_path: Path | None,
    checkpoint_config_path: Path | None,
) -> dict[str, Any]:
    entries = load_model_configurations(
        model_config_path=model_config_path,
        checkpoint_config_path=checkpoint_config_path,
    )
    requested = model_key or model_name
    if requested and requested in entries:
        return dict(entries[requested])
    if model_name:
        return _entry_from_baseline({"name": model_name})
    available = sorted(key for key in entries if "/" not in key)
    raise KeyError("Unknown model. Requested %r. Available model keys: %s" % (requested, ", ".join(available)))


def validate_all_checkpoint_entries(checkpoint_config_path: Path | None, *, phase: str = "both") -> dict[str, Any]:
    if checkpoint_config_path is None or not checkpoint_config_path.exists():
        return {"exists": False, "entries": {}, "warnings": ["checkpoint config not found"], "errors": []}
    entries = load_model_configurations(model_config_path=None, checkpoint_config_path=checkpoint_config_path)
    seen = {}
    warnings: list[str] = []
    errors: list[str] = []
    for key, entry in entries.items():
        if key != entry.get("model_key"):
            continue
        validation = validate_model_entry(entry, phase=phase, require_runnable=False)
        seen[key] = {
            "checkpoint_type": entry.get("checkpoint_type"),
            "adapter_type": entry.get("adapter_type"),
            "base_model_name_or_path": entry.get("base_model_name_or_path"),
            "checkpoint_path": entry.get("checkpoint_path"),
            "adapter_path": entry.get("adapter_path"),
            "warnings": validation.warnings,
            "errors": validation.errors,
        }
        warnings.extend(validation.warnings)
        errors.extend(validation.errors)
    return {"exists": True, "entries": seen, "warnings": warnings, "errors": errors}
