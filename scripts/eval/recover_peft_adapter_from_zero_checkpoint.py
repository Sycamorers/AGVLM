#!/usr/bin/env python
"""Recover a PEFT LoRA adapter from a DeepSpeed ZeRO checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _load_zero_to_fp32(checkpoint_dir: Path) -> ModuleType:
    script_path = checkpoint_dir / "zero_to_fp32.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing DeepSpeed converter: {script_path}")
    spec = importlib.util.spec_from_file_location("zero_to_fp32_recovery", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import DeepSpeed converter: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalise_peft_key(key: str) -> str:
    return re.sub(r"\.lora_([AB])\.default\.weight$", r".lora_\1.weight", key)


def _parse_dtype(value: str) -> torch.dtype:
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _load_reference_shapes(reference_adapter_model: Path | None) -> dict[str, tuple[int, ...]]:
    if reference_adapter_model is None:
        return {}
    with safe_open(reference_adapter_model, framework="pt", device="cpu") as handle:
        return {key: tuple(handle.get_tensor(key).shape) for key in handle.keys()}


def recover_adapter(
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    dtype: torch.dtype,
    reference_adapter_config: Path | None,
    reference_adapter_model: Path | None,
) -> dict[str, object]:
    checkpoint_dir = checkpoint_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    zero_to_fp32 = _load_zero_to_fp32(checkpoint_dir)
    state_dict = zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
        str(checkpoint_dir),
        tag=None,
        exclude_frozen_parameters=False,
        lazy_mode=False,
    )

    adapter_state: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if ".lora_" not in key:
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.numel() == 0:
            continue
        adapter_state[_normalise_peft_key(key)] = tensor.detach().cpu().to(dtype=dtype).contiguous()

    if not adapter_state:
        raise RuntimeError(f"No non-empty LoRA tensors recovered from {checkpoint_dir}")

    reference_shapes = _load_reference_shapes(reference_adapter_model)
    if reference_shapes:
        recovered_shapes = {key: tuple(tensor.shape) for key, tensor in adapter_state.items()}
        missing = sorted(set(reference_shapes) - set(recovered_shapes))
        extra = sorted(set(recovered_shapes) - set(reference_shapes))
        mismatched = sorted(
            key
            for key in set(reference_shapes) & set(recovered_shapes)
            if reference_shapes[key] != recovered_shapes[key]
        )
        if missing or extra or mismatched:
            raise RuntimeError(
                "Recovered adapter shape mismatch: "
                f"missing={missing[:5]} extra={extra[:5]} mismatched={mismatched[:5]}"
            )

    config_source = checkpoint_dir / "adapter_config.json"
    if not config_source.is_file():
        if reference_adapter_config is None:
            raise FileNotFoundError(f"Missing adapter_config.json under {checkpoint_dir}")
        config_source = reference_adapter_config
    shutil.copyfile(config_source, output_dir / "adapter_config.json")
    readme_source = checkpoint_dir / "README.md"
    if readme_source.is_file():
        shutil.copyfile(readme_source, output_dir / "README.md")

    adapter_model_path = output_dir / "adapter_model.safetensors"
    save_file(adapter_state, adapter_model_path, metadata={"format": "pt"})

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "adapter_model_path": str(adapter_model_path),
        "dtype": str(dtype).replace("torch.", ""),
        "num_tensors": len(adapter_state),
        "num_parameters": int(sum(tensor.numel() for tensor in adapter_state.values())),
        "adapter_bytes": adapter_model_path.stat().st_size,
    }
    (output_dir / "recovery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--reference-adapter-config", type=Path)
    parser.add_argument("--reference-adapter-model", type=Path)
    args = parser.parse_args()

    summary = recover_adapter(
        args.checkpoint_dir,
        args.output_dir,
        dtype=_parse_dtype(args.dtype),
        reference_adapter_config=args.reference_adapter_config,
        reference_adapter_model=args.reference_adapter_model,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
