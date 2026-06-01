#!/usr/bin/env python3
"""Run one inference-only VLM benchmark model on a phase split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

from build_phase_splits import RL_PHASE, SFT_PHASE, build_phase_splits
from checkpoint_config import resolve_model_entry, validate_model_entry
from dataset_adapter import BenchmarkSample, load_benchmark_samples
from evaluate_predictions import build_summary_table, evaluate_file
from metrics import parse_prediction_for_metrics
from model_adapters import HuggingFaceVLMAdapter, MODEL_SPECS, is_oom_error
from utils import (
    BENCHMARK_ROOT,
    collect_environment_info,
    configure_inference_environment,
    ensure_dir,
    git_value,
    maybe_cuda_memory,
    model_slug,
    set_seed,
    utc_now,
    write_json,
)


class RestartWithQuantization(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=None, help="HF model name or legacy selector.")
    parser.add_argument("--model-key", default=None, help="Model key from model/checkpoint config.")
    parser.add_argument("--model-config", default=str(BENCHMARK_ROOT / "baseline_models.yaml"))
    parser.add_argument("--checkpoint-config", default=str(BENCHMARK_ROOT / "agvlm_checkpoint_models.yaml"))
    parser.add_argument("--phase", choices=["sft", "rl"], default="sft")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 uses task-aware defaults.")
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT / "results"))
    parser.add_argument("--split-dir", default=str(BENCHMARK_ROOT / "splits"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--disable-oom-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and samples without loading a model.")
    parser.add_argument("--skip-model-load", action="store_true", help="Alias for --dry-run.")
    parser.add_argument("--allow-fallback-split", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    return parser.parse_args()


def _phase_name(phase: str) -> str:
    return SFT_PHASE if phase == "sft" else RL_PHASE


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_path:
        return Path(args.manifest_path)
    return Path(args.split_dir) / ("%s_%s_manifest.jsonl" % (args.phase, args.split))


def _ground_truth_for_metrics(sample: BenchmarkSample) -> str:
    target = sample.row.get("target") or {}
    verifier = sample.row.get("verifier") or {}
    if sample.verifier_mode == "clarify" or sample.task_type == "clarify_or_respond":
        return str(verifier.get("expected_decision") or target.get("decision") or sample.expected_answer)
    if sample.verifier_mode == "label" and target.get("canonical_label"):
        return str(target["canonical_label"])
    return sample.expected_answer


def _max_new_tokens_for_sample(sample: BenchmarkSample, requested: int) -> int:
    if requested:
        return requested
    if sample.verifier_mode == "label" or sample.task_type == "classification":
        return 64
    if sample.task_type == "consultation" or sample.verifier_mode == "structured":
        return 256
    if sample.task_type == "clarify_or_respond" or sample.verifier_mode == "clarify":
        return 128
    return 128


def _generation_config(args: argparse.Namespace, sample: BenchmarkSample) -> dict[str, Any]:
    config = {
        "max_new_tokens": _max_new_tokens_for_sample(sample, args.max_new_tokens),
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_beams": 1,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    if args.min_new_tokens:
        config["min_new_tokens"] = args.min_new_tokens
    return config


def _base_record(
    sample: BenchmarkSample,
    *,
    args: argparse.Namespace,
    model_entry: dict[str, Any],
    generation_config: dict[str, Any],
    quantization: str,
) -> dict[str, Any]:
    return {
        "phase": _phase_name(args.phase),
        "split": args.split,
        "model_name": model_entry.get("model_name") or args.model_name or args.model_key,
        "model_key": model_entry.get("model_key") or model_slug(str(model_entry.get("model_name") or "")),
        "checkpoint_type": model_entry.get("checkpoint_type") or "external_baseline",
        "base_model_name_or_path": model_entry.get("base_model_name_or_path") or model_entry.get("model_name"),
        "adapter_path": model_entry.get("adapter_path") or "",
        "checkpoint_path": model_entry.get("checkpoint_path") or "",
        "sample_id": sample.sample_id,
        "source_dataset": sample.row.get("source_dataset"),
        "task_type": sample.task_type,
        "verifier_mode": sample.verifier_mode,
        "metadata": sample.row.get("metadata") or {},
        "image_paths": sample.image_paths,
        "image_count": len(sample.image_paths),
        "image_policy": model_entry.get("image_policy") or "",
        "prompt": sample.prompt,
        "system_prompt": sample.system_prompt,
        "ground_truth": _ground_truth_for_metrics(sample),
        "references": sample.references,
        "verifier": sample.row.get("verifier") or {},
        "generation_config": generation_config,
        "dtype": args.dtype,
        "inference_dtype": args.dtype,
        "quantization": quantization,
        "benchmark_manifest_path": str(_manifest_path(args)),
        "git_commit": git_value("rev-parse", "HEAD"),
        "created_at_utc": utc_now(),
    }


def _record_for_error(
    sample: BenchmarkSample,
    *,
    args: argparse.Namespace,
    model_entry: dict[str, Any],
    generation_config: dict[str, Any],
    quantization: str,
    runtime_seconds: float,
    error_message: str,
    error_traceback: str | None = None,
) -> dict[str, Any]:
    record = _base_record(
        sample,
        args=args,
        model_entry=model_entry,
        generation_config=generation_config,
        quantization=quantization,
    )
    record.update(
        {
            "raw_output": "",
            "parsed_prediction": "",
            "normalized_prediction": "",
            "parse_status": "failed",
            "invalid_prediction": True,
            "runtime_seconds": runtime_seconds,
            "error_message": error_message,
            "error_traceback": error_traceback,
        }
    )
    return record


def _prediction_record(
    sample: BenchmarkSample,
    *,
    args: argparse.Namespace,
    model_entry: dict[str, Any],
    result: dict[str, Any],
    generation_config: dict[str, Any],
    quantization: str,
    runtime_seconds: float,
    model_revision: str | None,
) -> dict[str, Any]:
    raw_output = result.get("raw_output") or ""
    parsed = parse_prediction_for_metrics(
        raw_output=raw_output,
        task_type=sample.task_type,
        verifier_mode=sample.verifier_mode,
        label_space=sample.label_space,
    )
    record = _base_record(
        sample,
        args=args,
        model_entry=model_entry,
        generation_config=generation_config,
        quantization=quantization,
    )
    image_policy = result.get("image_policy") or record["image_policy"] or "all_images"
    record.update(
        {
            "prompt": result.get("prompt") or sample.prompt,
            "raw_output": raw_output,
            "parsed_prediction": parsed.get("parsed_prediction", ""),
            "normalized_prediction": parsed.get("normalized_prediction", ""),
            "parse_status": parsed.get("parse_status", "missing"),
            "invalid_prediction": bool(parsed.get("invalid_prediction")),
            "sections": parsed.get("sections"),
            "label_mentions": parsed.get("label_mentions"),
            "out_of_label_space": bool(parsed.get("out_of_label_space")),
            "format_retry_used": bool(result.get("format_retry_used")),
            "raw_output_before_format_retry": result.get("raw_output_before_format_retry"),
            "model_revision": model_revision,
            "runtime_seconds": runtime_seconds,
            "images_used": result.get("images_used"),
            "image_policy": image_policy,
            "error_message": None,
        }
    )
    return record


def _run_once(
    *,
    args: argparse.Namespace,
    model_entry: dict[str, Any],
    samples: list[BenchmarkSample],
    output_path: Path,
    quantization: str,
    oom_fallback_used: bool,
    oom_fallback_reason: str | None = None,
) -> dict[str, Any]:
    adapter = HuggingFaceVLMAdapter(
        str(model_entry.get("model_name") or args.model_name or args.model_key),
        device=args.device,
        dtype=args.dtype,
        quantization=quantization,
        attn_implementation=args.attn_implementation,
        model_entry=model_entry,
    )
    try:
        try:
            adapter.load_model()
        except Exception as exc:
            if quantization == "none" and is_oom_error(exc) and not args.disable_oom_fallback:
                raise RestartWithQuantization(str(exc)) from exc
            raise

        model_revision = adapter.load_metadata.get("model_commit_hash")
        ensure_dir(output_path.parent)
        with output_path.open("w", encoding="utf-8") as handle:
            for index, sample in enumerate(samples, start=1):
                generation_config = _generation_config(args, sample)
                start = time.perf_counter()
                try:
                    result = adapter.generate(sample, generation_config)
                    runtime_seconds = time.perf_counter() - start
                    record = _prediction_record(
                        sample,
                        args=args,
                        model_entry=model_entry,
                        result=result,
                        generation_config=generation_config,
                        quantization=quantization,
                        runtime_seconds=runtime_seconds,
                        model_revision=model_revision,
                    )
                except Exception as exc:
                    runtime_seconds = time.perf_counter() - start
                    if quantization == "none" and is_oom_error(exc) and not args.disable_oom_fallback:
                        raise RestartWithQuantization(str(exc)) from exc
                    record = _record_for_error(
                        sample,
                        args=args,
                        model_entry=model_entry,
                        generation_config=generation_config,
                        quantization=quantization,
                        runtime_seconds=runtime_seconds,
                        error_message="%s: %s" % (type(exc).__name__, exc),
                        error_traceback=traceback.format_exc(limit=8),
                    )
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                if index % 25 == 0:
                    print("completed %s/%s samples; cuda=%s" % (index, len(samples), maybe_cuda_memory(args.device)), flush=True)

        return {
            "adapter_load_metadata": adapter.load_metadata,
            "quantization": quantization,
            "oom_fallback_used": oom_fallback_used,
            "oom_fallback_reason": oom_fallback_reason,
        }
    finally:
        adapter.unload_model()


def _dry_run_payload(args: argparse.Namespace, model_entry: dict[str, Any], samples: list[BenchmarkSample]) -> dict[str, Any]:
    return {
        "dry_run": True,
        "phase": _phase_name(args.phase),
        "split": args.split,
        "manifest_path": str(_manifest_path(args)),
        "num_selected_samples": len(samples),
        "model_key": model_entry.get("model_key"),
        "model_name": model_entry.get("model_name"),
        "checkpoint_type": model_entry.get("checkpoint_type"),
        "base_model_name_or_path": model_entry.get("base_model_name_or_path"),
        "adapter_path": model_entry.get("adapter_path"),
        "checkpoint_path": model_entry.get("checkpoint_path"),
        "sample_preview": [
            {
                "sample_id": sample.sample_id,
                "task_type": sample.task_type,
                "source_dataset": sample.row.get("source_dataset"),
                "image_count": len(sample.image_paths),
                "generation_config": _generation_config(args, sample),
            }
            for sample in samples[:5]
        ],
    }


def main() -> int:
    args = parse_args()
    if args.skip_model_load:
        args.dry_run = True
    configure_inference_environment()
    set_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError("This benchmark runner is intentionally sequential and currently supports --batch-size 1 only.")
    if args.smoke_test and not args.max_samples:
        args.max_samples = 5

    model_entry = resolve_model_entry(
        model_key=args.model_key,
        model_name=args.model_name,
        model_config_path=Path(args.model_config) if args.model_config else None,
        checkpoint_config_path=Path(args.checkpoint_config) if args.checkpoint_config else None,
    )
    validation = validate_model_entry(model_entry, phase=args.phase, require_runnable=True)
    if validation.warnings:
        for warning in validation.warnings:
            print("WARNING: %s" % warning, file=sys.stderr)
    if not validation.ok:
        raise ValueError("Invalid model configuration:\n- " + "\n- ".join(validation.errors))

    split_dir = Path(args.split_dir)
    build_phase_splits(
        phase=args.phase,
        output_dir=split_dir,
        seed=args.seed,
        force=False,
        allow_fallback_split=args.allow_fallback_split,
        write_report=True,
    )
    manifest_path = _manifest_path(args)
    if not manifest_path.exists():
        raise FileNotFoundError("Missing split manifest: %s" % manifest_path)

    samples = load_benchmark_samples(manifest_path, args.split)
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No samples selected from %s" % manifest_path)

    if args.dry_run:
        payload = _dry_run_payload(args, model_entry, samples)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    output_dir = Path(args.output_dir)
    predictions_dir = ensure_dir(output_dir / "predictions")
    metrics_dir = ensure_dir(output_dir / "metrics")
    metadata_dir = ensure_dir(output_dir / "metadata")
    key = str(model_entry.get("model_key") or model_slug(str(model_entry.get("model_name") or args.model_name)))
    slug = model_slug("%s_%s_%s" % (_phase_name(args.phase), key, args.split))
    predictions_path = predictions_dir / ("%s.jsonl" % slug)
    metadata_path = metadata_dir / ("%s_run.json" % slug)

    quantization = args.quantization
    fallback_reason = None
    oom_fallback_used = False
    try:
        run_metadata = _run_once(
            args=args,
            model_entry=model_entry,
            samples=samples,
            output_path=predictions_path,
            quantization=quantization,
            oom_fallback_used=False,
        )
    except RestartWithQuantization as exc:
        spec = MODEL_SPECS.get(str(model_entry.get("model_name") or ""))
        fallback = spec.fallback_quantization if spec is not None else "4bit"
        if fallback != "4bit" or args.quantization != "none" or args.disable_oom_fallback:
            raise
        fallback_reason = str(exc)
        oom_fallback_used = True
        print("OOM detected; restarting this model from sample 1 with explicit 4-bit quantization.", flush=True)
        quantization = "4bit"
        run_metadata = _run_once(
            args=args,
            model_entry=model_entry,
            samples=samples,
            output_path=predictions_path,
            quantization=quantization,
            oom_fallback_used=True,
            oom_fallback_reason=fallback_reason,
        )

    metrics = evaluate_file(
        predictions_path,
        model_name=str(model_entry.get("model_name") or args.model_name),
        model_key=key,
        phase=_phase_name(args.phase),
        split=args.split,
        output_dir=metrics_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    summary_path = metrics_dir / "summary_table.csv"
    build_summary_table(metrics_dir, summary_path)
    metadata = {
        "args": vars(args),
        "model_entry": model_entry,
        "split_manifest": str(manifest_path),
        "num_samples": len(samples),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_dir / ("%s_metrics.json" % slug)),
        "summary_table": str(summary_path),
        "environment": collect_environment_info(args.device),
        "run_metadata": run_metadata,
        "oom_fallback_used": oom_fallback_used,
        "oom_fallback_reason": fallback_reason,
    }
    write_json(metadata_path, metadata)
    print(json.dumps({"metrics": metrics, "metadata_path": str(metadata_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
