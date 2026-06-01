"""Checkpoint discovery and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _checkpoint_step(path: Path) -> Optional[int]:
    prefix = "checkpoint-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    suffix = path.name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _has_deepspeed_resume_state(path: Path) -> bool:
    latest_path = path / "latest"
    if not latest_path.is_file():
        return False
    tag = latest_path.read_text(encoding="utf-8").strip()
    if not tag:
        return False
    tag_dir = path / tag
    if not tag_dir.is_dir():
        return False
    model_states = list(tag_dir.glob("*_model_states.pt"))
    optim_states = list(tag_dir.glob("*_optim_states.pt"))
    return bool(model_states) and len(model_states) == len(optim_states)


def _has_standard_resume_state(path: Path) -> bool:
    model_files = (
        "pytorch_model.bin",
        "model.safetensors",
        "adapter_model.safetensors",
    )
    return (
        (path / "optimizer.pt").is_file()
        and (path / "scheduler.pt").is_file()
        and any((path / name).is_file() for name in model_files)
    )


def _as_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ["state_dict", "model", "module"]:
            nested = payload.get(key)
            if isinstance(nested, dict):
                return nested
        return payload
    return {}


def _load_safetensors_summary(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - safetensors is expected in training envs
        raise RuntimeError("safetensors is required to validate %s: %s" % (path, exc)) from exc

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        non_empty_tensors = 0
        first_tensor_name = ""
        first_tensor_shape: list[int] = []
        first_tensor_dtype = ""
        for key in keys:
            tensor = handle.get_tensor(key)
            if not first_tensor_name:
                first_tensor_name = key
                first_tensor_shape = [int(dim) for dim in tensor.shape]
                first_tensor_dtype = str(tensor.dtype)
            if int(tensor.numel()) > 0:
                non_empty_tensors += 1
    return {
        "format": "safetensors",
        "path": str(path),
        "num_tensors": len(keys),
        "non_empty_tensors": non_empty_tensors,
        "tensor_names": keys,
        "first_tensor": first_tensor_name,
        "first_tensor_shape": first_tensor_shape,
        "first_tensor_dtype": first_tensor_dtype,
    }


def _load_torch_state_summary(path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is optional for lightweight tooling
        raise RuntimeError("torch is required to validate %s: %s" % (path, exc)) from exc

    payload = torch.load(path, map_location="cpu")
    state_dict = _as_state_dict(payload)
    tensor_items = [(name, value) for name, value in state_dict.items() if hasattr(value, "numel")]
    non_empty_tensors = sum(1 for _, tensor in tensor_items if int(tensor.numel()) > 0)
    first_name = tensor_items[0][0] if tensor_items else ""
    first_tensor = tensor_items[0][1] if tensor_items else None
    return {
        "format": "torch",
        "path": str(path),
        "num_tensors": len(tensor_items),
        "non_empty_tensors": non_empty_tensors,
        "tensor_names": [name for name, _ in tensor_items],
        "first_tensor": first_name,
        "first_tensor_shape": [int(dim) for dim in getattr(first_tensor, "shape", [])] if first_tensor is not None else [],
        "first_tensor_dtype": str(getattr(first_tensor, "dtype", "")) if first_tensor is not None else "",
    }


def _tensor_file_summary(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        return _load_safetensors_summary(path)
    if path.suffix == ".bin":
        return _load_torch_state_summary(path)
    raise ValueError("Unsupported tensor checkpoint file: %s" % path)


def _read_adapter_config(path: Path) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("PEFT adapter config is not valid JSON: %s" % config_path) from exc
    if not isinstance(payload, dict):
        raise ValueError("PEFT adapter config must decode to an object: %s" % config_path)
    return payload


def validate_peft_adapter_checkpoint(path: Path | str) -> dict[str, Any]:
    """Validate that a PEFT adapter directory contains usable tensor artifacts."""
    adapter_dir = Path(path).expanduser()
    if not adapter_dir.is_dir():
        raise FileNotFoundError("PEFT adapter checkpoint is not a directory: %s" % adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("PEFT adapter checkpoint is missing adapter_config.json: %s" % adapter_dir)

    tensor_path = None
    for name in ["adapter_model.safetensors", "adapter_model.bin"]:
        candidate = adapter_dir / name
        if candidate.is_file():
            tensor_path = candidate
            break
    if tensor_path is None:
        raise FileNotFoundError("PEFT adapter checkpoint is missing adapter_model.safetensors or adapter_model.bin: %s" % adapter_dir)
    if tensor_path.stat().st_size <= 0:
        raise ValueError("PEFT adapter tensor file is empty: %s" % tensor_path)

    adapter_config = _read_adapter_config(adapter_dir)
    tensor_summary = _tensor_file_summary(tensor_path)
    if int(tensor_summary["num_tensors"]) <= 0:
        raise ValueError("PEFT adapter tensor file contains no tensors: %s" % tensor_path)
    if int(tensor_summary["non_empty_tensors"]) <= 0:
        raise ValueError("PEFT adapter tensor file contains no non-empty tensors: %s" % tensor_path)
    peft_type = str(adapter_config.get("peft_type") or "").lower()
    if peft_type == "lora" and not any("lora_" in name for name in tensor_summary["tensor_names"]):
        raise ValueError("LoRA adapter tensor file contains no lora_* tensors: %s" % tensor_path)
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_config_path": str(config_path),
        "adapter_model_path": str(tensor_path),
        "peft_type": adapter_config.get("peft_type"),
        "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        **{key: value for key, value in tensor_summary.items() if key != "tensor_names"},
    }


def checkpoint_has_valid_model_artifacts(path: Path | str) -> bool:
    """Return True when a checkpoint path has loadable model or adapter artifacts."""
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_file():
        return checkpoint_path.stat().st_size > 0
    if not checkpoint_path.is_dir():
        return False
    if (checkpoint_path / "adapter_config.json").exists():
        try:
            validate_peft_adapter_checkpoint(checkpoint_path)
        except (FileNotFoundError, RuntimeError, ValueError):
            return False
        return True
    if (checkpoint_path / "config.json").exists() and any(
        candidate.is_file() and candidate.stat().st_size > 0
        for pattern in ["model.safetensors", "pytorch_model.bin", "model-*.safetensors", "pytorch_model-*.bin"]
        for candidate in checkpoint_path.glob(pattern)
    ):
        return True
    if (checkpoint_path / "model.safetensors.index.json").exists() or (checkpoint_path / "pytorch_model.bin.index.json").exists():
        return True
    return False


def _is_complete_checkpoint(path: Path) -> bool:
    return (
        (path / "trainer_state.json").is_file()
        and (path / "training_args.bin").is_file()
        and (_has_deepspeed_resume_state(path) or _has_standard_resume_state(path))
    )


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = [
        (step, path)
        for path in output_dir.glob("checkpoint-*")
        if (step := _checkpoint_step(path)) is not None and _is_complete_checkpoint(path)
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resolve_resume_checkpoint(output_dir: Path, requested: Optional[str]) -> Optional[Path]:
    if not requested:
        return None
    if requested == "auto":
        return find_latest_checkpoint(output_dir)
    path = Path(requested)
    return path if path.exists() else None
