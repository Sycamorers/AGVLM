#!/usr/bin/env python3
"""Run one inference-only VLM baseline on an isolated benchmark split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any

from dataset_adapter import BenchmarkSample, load_benchmark_samples
from evaluate_predictions import build_summary_table, evaluate_file
from metrics import normalize_prediction
from model_adapters import HuggingFaceVLMAdapter, MODEL_SPECS, is_oom_error
from split_dataset import ensure_split_manifests
from utils import (
    BENCHMARK_ROOT,
    collect_environment_info,
    configure_inference_environment,
    ensure_dir,
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
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
    return parser.parse_args()


def _ground_truth_for_metrics(sample: BenchmarkSample) -> str:
    target = sample.row.get("target") or {}
    if sample.verifier_mode == "clarify" or sample.task_type == "clarify_or_respond":
        return str(target.get("decision") or sample.expected_answer)
    if sample.verifier_mode == "label" and target.get("canonical_label"):
        return str(target["canonical_label"])
    return sample.expected_answer


def _record_for_error(
    sample: BenchmarkSample,
    *,
    args: argparse.Namespace,
    generation_config: dict[str, Any],
    quantization: str,
    runtime_seconds: float,
    error_message: str,
    error_traceback: str | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "source_dataset": sample.row.get("source_dataset"),
        "split": args.split,
        "original_split": sample.row.get("split"),
        "task_type": sample.task_type,
        "verifier_mode": sample.verifier_mode,
        "image_paths": sample.image_paths,
        "image_id": (sample.row.get("metadata") or {}).get("source_image_id"),
        "prompt": sample.prompt,
        "raw_output": "",
        "normalized_prediction": "",
        "invalid_prediction": True,
        "ground_truth": _ground_truth_for_metrics(sample),
        "references": sample.references,
        "model_name": args.model_name,
        "generation_config": generation_config,
        "inference_dtype": args.dtype,
        "quantization": quantization,
        "runtime_seconds": runtime_seconds,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "created_at_utc": utc_now(),
    }


def _prediction_record(
    sample: BenchmarkSample,
    *,
    args: argparse.Namespace,
    result: dict[str, Any],
    generation_config: dict[str, Any],
    quantization: str,
    runtime_seconds: float,
    model_revision: str | None,
) -> dict[str, Any]:
    raw_output = result.get("raw_output") or ""
    normalized, invalid = normalize_prediction(
        raw_output=raw_output,
        task_type=sample.task_type,
        verifier_mode=sample.verifier_mode,
        label_space=sample.label_space,
    )
    return {
        "sample_id": sample.sample_id,
        "source_dataset": sample.row.get("source_dataset"),
        "split": args.split,
        "original_split": sample.row.get("split"),
        "task_type": sample.task_type,
        "verifier_mode": sample.verifier_mode,
        "image_paths": sample.image_paths,
        "image_id": (sample.row.get("metadata") or {}).get("source_image_id"),
        "prompt": result.get("prompt") or sample.prompt,
        "raw_output": raw_output,
        "normalized_prediction": normalized,
        "invalid_prediction": invalid,
        "ground_truth": _ground_truth_for_metrics(sample),
        "references": sample.references,
        "model_name": args.model_name,
        "model_revision": model_revision,
        "generation_config": generation_config,
        "inference_dtype": args.dtype,
        "quantization": quantization,
        "runtime_seconds": runtime_seconds,
        "images_used": result.get("images_used"),
        "image_policy": result.get("image_policy"),
        "error_message": None,
        "created_at_utc": utc_now(),
    }


def _run_once(
    *,
    args: argparse.Namespace,
    samples: list[BenchmarkSample],
    output_path: Path,
    quantization: str,
    oom_fallback_used: bool,
    oom_fallback_reason: str | None = None,
) -> dict[str, Any]:
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_beams": 1,
        "batch_size": args.batch_size,
    }
    adapter = HuggingFaceVLMAdapter(
        args.model_name,
        device=args.device,
        dtype=args.dtype,
        quantization=quantization,
        attn_implementation=args.attn_implementation,
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
                start = time.perf_counter()
                try:
                    result = adapter.generate(sample, generation_config)
                    runtime_seconds = time.perf_counter() - start
                    record = _prediction_record(
                        sample,
                        args=args,
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
            "generation_config": generation_config,
            "quantization": quantization,
            "oom_fallback_used": oom_fallback_used,
            "oom_fallback_reason": oom_fallback_reason,
        }
    finally:
        adapter.unload_model()


def main() -> int:
    args = parse_args()
    configure_inference_environment()
    set_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError("This benchmark runner is intentionally sequential and currently supports --batch-size 1 only.")
    if args.smoke_test and not args.max_samples:
        args.max_samples = 5

    split_dir = Path(args.split_dir)
    ensure_split_manifests(split_dir, seed=args.seed)
    manifest_path = split_dir / ("%s_manifest.jsonl" % args.split)
    if not manifest_path.exists():
        raise FileNotFoundError("Missing split manifest: %s" % manifest_path)

    samples = load_benchmark_samples(manifest_path, args.split)
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No samples selected from %s" % manifest_path)

    output_dir = Path(args.output_dir)
    predictions_dir = ensure_dir(output_dir / "predictions")
    metrics_dir = ensure_dir(output_dir / "metrics")
    metadata_dir = ensure_dir(output_dir / "metadata")
    slug = model_slug(args.model_name)
    predictions_path = predictions_dir / ("%s_%s.jsonl" % (slug, args.split))
    metadata_path = metadata_dir / ("%s_%s_run.json" % (slug, args.split))

    quantization = args.quantization
    fallback_reason = None
    oom_fallback_used = False
    try:
        run_metadata = _run_once(
            args=args,
            samples=samples,
            output_path=predictions_path,
            quantization=quantization,
            oom_fallback_used=False,
        )
    except RestartWithQuantization as exc:
        spec = MODEL_SPECS.get(args.model_name)
        fallback = spec.fallback_quantization if spec is not None else "4bit"
        if fallback != "4bit" or args.quantization != "none" or args.disable_oom_fallback:
            raise
        fallback_reason = str(exc)
        oom_fallback_used = True
        print("OOM detected; restarting this model from sample 1 with explicit 4-bit quantization.", flush=True)
        quantization = "4bit"
        run_metadata = _run_once(
            args=args,
            samples=samples,
            output_path=predictions_path,
            quantization=quantization,
            oom_fallback_used=True,
            oom_fallback_reason=fallback_reason,
        )

    metrics = evaluate_file(
        predictions_path,
        model_name=args.model_name,
        split=args.split,
        output_dir=metrics_dir,
    )
    summary_path = metrics_dir / "summary_table.csv"
    build_summary_table(metrics_dir, summary_path)
    metadata = {
        "args": vars(args),
        "model_spec": MODEL_SPECS.get(args.model_name).__dict__ if args.model_name in MODEL_SPECS else None,
        "split_manifest": str(manifest_path),
        "num_samples": len(samples),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_dir / ("%s_%s_metrics.json" % (slug, args.split))),
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
