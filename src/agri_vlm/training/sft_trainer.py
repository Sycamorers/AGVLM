"""Supervised fine-tuning entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agri_vlm.data.manifest_io import read_manifest, summarize_manifest
from agri_vlm.evaluation.inference import generate_predictions_from_loaded_model
from agri_vlm.evaluation.local_eval import score_local_predictions
from agri_vlm.evaluation.reporting import build_prediction_rows
from agri_vlm.logging_utils import configure_logging
from agri_vlm.modeling.freezing import apply_freezing
from agri_vlm.modeling.model_factory import load_model, load_sft_checkpoint_model
from agri_vlm.modeling.peft_setup import maybe_wrap_with_peft
from agri_vlm.modeling.processor_factory import load_processor
from agri_vlm.training.callbacks import JsonlMetricsCallback
from agri_vlm.training.collators import build_sft_data_collator
from agri_vlm.training.run_artifacts import prepare_run_artifacts, write_training_artifact_manifest
from agri_vlm.utils.checkpointing import resolve_resume_checkpoint
from agri_vlm.utils.distributed import (
    configure_torch_runtime,
    destroy_distributed_process_group,
    get_distributed_context,
)
from agri_vlm.utils.io import ensure_dir, write_jsonl


class ManifestListDataset:
    """A tiny dataset wrapper around validated manifest rows."""

    def __init__(self, rows: List[Any]) -> None:
        self.rows = [row.model_dump(mode="json") for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


def _image_count_histogram(rows: List[Any]) -> Dict[str, int]:
    histogram: Dict[str, int] = {}
    for row in rows:
        key = str(len(row.images))
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def _filter_rows_by_max_images(rows: List[Any], *, max_images_per_sample: int | None) -> List[Any]:
    if max_images_per_sample is None:
        return rows
    return [row for row in rows if len(row.images) <= max_images_per_sample]


def _sample_group_key(row: Any) -> str:
    source_image_id = row.metadata.get("source_image_id") or row.images[0]
    return "%s::%s" % (row.source_dataset, source_image_id)


def _assert_no_train_eval_overlap(train_rows: List[Any], eval_rows: List[Any]) -> None:
    if not eval_rows:
        return
    train_ids = {row.sample_id for row in train_rows}
    eval_ids = {row.sample_id for row in eval_rows}
    train_group_keys = {_sample_group_key(row) for row in train_rows}
    eval_group_keys = {_sample_group_key(row) for row in eval_rows}
    exact_overlap = train_ids.intersection(eval_ids)
    group_overlap = train_group_keys.intersection(eval_group_keys)
    if exact_overlap or group_overlap:
        raise ValueError(
            "Train/eval manifest overlap detected: exact_sample_id=%s group_key=%s. "
            "Build non-overlapping train/eval manifests before launching SFT."
            % (len(exact_overlap), len(group_overlap))
        )


def _chunked_causal_lm_loss(
    logits: Any,
    labels: Any,
    *,
    chunk_size: int,
    ignore_index: int = -100,
) -> Any:
    """Compute shifted causal-LM loss without a full fp32 logits copy."""
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].to(logits.device)
    valid_items = shift_labels.ne(ignore_index).sum()
    if valid_items.item() == 0:
        return shift_logits.sum() * 0.0

    seq_len = shift_labels.shape[1]
    vocab_size = shift_logits.shape[-1]
    total_loss = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_labels = shift_labels[:, start:end].reshape(-1)
        if chunk_labels.ne(ignore_index).sum().item() == 0:
            continue
        chunk_logits = shift_logits[:, start:end, :].reshape(-1, vocab_size).float()
        chunk_loss = F.cross_entropy(
            chunk_logits,
            chunk_labels,
            ignore_index=ignore_index,
            reduction="sum",
        )
        total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss
    return total_loss / valid_items


def _torch_dist_is_initialized() -> bool:
    try:
        import torch.distributed as dist
    except Exception:  # pragma: no cover - torch is optional for dry-run tooling
        return False
    return dist.is_available() and dist.is_initialized()


def _distributed_max_count(count: int, device: Optional[Any]) -> int:
    if not _torch_dist_is_initialized():
        return count
    import torch
    import torch.distributed as dist

    backend = dist.get_backend()
    tensor_device = device
    if tensor_device is None:
        tensor_device = "cuda" if backend == "nccl" and torch.cuda.is_available() else "cpu"
    tensor = torch.tensor([count], device=tensor_device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _all_gather_objects(payload: Any) -> List[Any]:
    if not _torch_dist_is_initialized():
        return [payload]
    import torch.distributed as dist

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)
    return gathered


def _broadcast_object_from_zero(payload: Any) -> Any:
    if not _torch_dist_is_initialized():
        return payload
    import torch.distributed as dist

    values = [payload]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _run_validation_generation_performance(
    *,
    model: Any,
    processor: Any,
    eval_rows: List[Any],
    max_examples: int,
    batch_size: int,
    max_new_tokens: int,
    metric_prefix: str,
    step: int,
    output_dir: Path,
    save_predictions: bool,
    device: Optional[Any],
) -> Dict[str, Any]:
    if not eval_rows:
        raise ValueError("Validation generation metrics require a non-empty eval manifest.")

    distributed_context = get_distributed_context()
    selected_rows = eval_rows[:max_examples] if max_examples else list(eval_rows)
    indexed_rows = list(enumerate(selected_rows))
    local_items = indexed_rows[distributed_context.rank :: distributed_context.world_size]
    real_local_count = len(local_items)

    max_local_count = _distributed_max_count(real_local_count, device=device)
    if max_local_count > real_local_count:
        pad_item = local_items[-1] if local_items else indexed_rows[0]
        local_items = [*local_items, *([pad_item] * (max_local_count - real_local_count))]

    local_predictions = generate_predictions_from_loaded_model(
        [row for _, row in local_items],
        model=model,
        processor=processor,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
        synced_gpus=distributed_context.is_distributed and _torch_dist_is_initialized(),
    )
    local_payload = [
        {
            "index": index,
            "sample": row.model_dump(mode="json"),
            "prediction": prediction,
        }
        for (index, row), prediction in zip(local_items[:real_local_count], local_predictions[:real_local_count])
    ]
    gathered_payloads = _all_gather_objects(local_payload)

    metrics = None
    if distributed_context.is_main_process:
        gathered = []
        for payload in gathered_payloads:
            gathered.extend(payload or [])
        gathered = sorted(gathered, key=lambda item: item["index"])
        scored_rows = [type(eval_rows[0]).model_validate(item["sample"]) for item in gathered]
        predictions = [item["prediction"] for item in gathered]
        raw_metrics = score_local_predictions(scored_rows, predictions)
        metrics = {
            "%s_%s" % (metric_prefix, key): value
            for key, value in raw_metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if save_predictions:
            predictions_path = output_dir / "validation_predictions" / ("step-%s.jsonl" % step)
            write_jsonl(predictions_path, build_prediction_rows(scored_rows, predictions))

    broadcast_metrics = _broadcast_object_from_zero(metrics)
    return broadcast_metrics or {}


def _build_sft_trainer_class(
    loss_chunk_size: int,
    validation_generation_config: Optional[Dict[str, Any]] = None,
) -> Any:
    from transformers import Trainer

    validation_generation_config = validation_generation_config or {}
    validation_generation_enabled = bool(validation_generation_config.get("enabled", False))

    if loss_chunk_size <= 0 and not validation_generation_enabled:
        return Trainer

    class AgriSFTTrainer(Trainer):
        if loss_chunk_size > 0:

            def compute_loss(
                self,
                model: Any,
                inputs: Dict[str, Any],
                return_outputs: bool = False,
                num_items_in_batch: Any = None,
            ) -> Any:
                labels = inputs["labels"]
                model_inputs = dict(inputs)
                model_inputs.pop("labels")
                outputs = model(**model_inputs)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                loss = _chunked_causal_lm_loss(
                    logits,
                    labels,
                    chunk_size=loss_chunk_size,
                )
                return (loss, outputs) if return_outputs else loss

        if validation_generation_enabled:

            def evaluate(
                self,
                eval_dataset: Optional[Any] = None,
                ignore_keys: Optional[List[str]] = None,
                metric_key_prefix: str = "eval",
            ) -> Dict[str, Any]:
                metrics = super().evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                )
                performance_metrics = _run_validation_generation_performance(
                    model=getattr(self, "model_wrapped", None) or self.model,
                    processor=validation_generation_config["processor"],
                    eval_rows=validation_generation_config["eval_rows"],
                    max_examples=validation_generation_config["max_examples"],
                    batch_size=validation_generation_config["batch_size"],
                    max_new_tokens=validation_generation_config["max_new_tokens"],
                    metric_prefix="%s_performance" % metric_key_prefix,
                    step=self.state.global_step,
                    output_dir=validation_generation_config["output_dir"],
                    save_predictions=validation_generation_config["save_predictions"],
                    device=self.args.device,
                )
                if performance_metrics:
                    self.log(performance_metrics)
                    metrics.update(performance_metrics)
                return metrics

    return AgriSFTTrainer


def _build_dry_run_summary(train_rows: List[Any], eval_rows: List[Any], output_dir: Path) -> Dict[str, Any]:
    distributed_context = get_distributed_context()
    summary = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_summary": summarize_manifest(train_rows),
        "eval_summary": summarize_manifest(eval_rows),
        "train_image_count_histogram": _image_count_histogram(train_rows),
        "eval_image_count_histogram": _image_count_histogram(eval_rows),
        "distributed": distributed_context.as_dict(),
    }
    ensure_dir(output_dir)
    (output_dir / "dry_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _resolve_deepspeed_config_path(config_value: str | None) -> str | None:
    if not config_value:
        return None
    config_path = Path(config_value).expanduser()
    if not config_path.exists():
        raise FileNotFoundError("DeepSpeed config path does not exist: %s" % config_path)
    return str(config_path)


def _is_deepspeed_zero_param(parameter: Any) -> bool:
    return hasattr(parameter, "ds_id")


def _collect_peft_raw_state_dict_for_save(model: Any, *, should_save: bool) -> Dict[str, Any]:
    """Gather raw LoRA tensors from ZeRO-3 partitions so PEFT can save an adapter."""
    raw_state_dict: Dict[str, Any] = {}
    lora_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_" in name
    ]
    if not lora_parameters:
        return raw_state_dict

    try:
        import deepspeed
    except Exception:  # pragma: no cover - deepspeed is optional outside large runs
        deepspeed = None

    for name, parameter in lora_parameters:
        if deepspeed is not None and _is_deepspeed_zero_param(parameter):
            with deepspeed.zero.GatheredParameters([parameter], modifier_rank=None):
                if should_save:
                    raw_state_dict[name] = parameter.detach().cpu().clone()
        elif should_save:
            raw_state_dict[name] = parameter.detach().cpu().clone()

    return raw_state_dict if should_save else {}


def _save_peft_adapter_model(model: Any, output_dir: Path, *, should_save: bool) -> None:
    raw_state_dict = _collect_peft_raw_state_dict_for_save(model, should_save=should_save)
    if should_save:
        from peft import get_peft_model_state_dict

        peft_state_dict = get_peft_model_state_dict(model, state_dict=raw_state_dict)
        if not peft_state_dict:
            raise RuntimeError("PEFT adapter state dict is empty; refusing to save an unusable adapter.")
        model.save_pretrained(
            output_dir,
            state_dict=raw_state_dict,
            safe_serialization=True,
            is_main_process=True,
        )


def _save_trained_model(trainer: Any, train_config: Any, output_dir: Path) -> None:
    if train_config.use_peft and train_config.deepspeed:
        _save_peft_adapter_model(
            trainer.model,
            output_dir,
            should_save=trainer.is_world_process_zero(),
        )
        return
    trainer.save_model()


def run_sft(model_config: Any, train_config: Any) -> Dict[str, Any]:
    """Run SFT or validate the setup in dry-run mode."""
    distributed_context = get_distributed_context(set_device=True)
    logger = configure_logging(logger_name="agri_vlm.training.sft")

    train_rows = read_manifest(Path(train_config.manifest_path))
    eval_rows = []
    if train_config.eval_manifest_path:
        eval_manifest_path = Path(train_config.eval_manifest_path)
        if not eval_manifest_path.exists():
            raise FileNotFoundError("SFT eval manifest path does not exist: %s" % eval_manifest_path)
        eval_rows = read_manifest(eval_manifest_path)

    original_train_rows = len(train_rows)
    original_eval_rows = len(eval_rows)
    train_rows = _filter_rows_by_max_images(
        train_rows,
        max_images_per_sample=train_config.max_images_per_sample,
    )
    eval_rows = _filter_rows_by_max_images(
        eval_rows,
        max_images_per_sample=train_config.max_images_per_sample,
    )

    output_dir = Path(train_config.output_dir)
    if train_config.smoke_max_samples:
        train_rows = train_rows[: train_config.smoke_max_samples]
        eval_rows = eval_rows[: train_config.smoke_max_samples]
    if train_config.fail_on_train_eval_overlap:
        _assert_no_train_eval_overlap(train_rows, eval_rows)

    run_artifacts = prepare_run_artifacts(
        stage="sft",
        model_config=model_config,
        train_config=train_config,
        distributed_context=distributed_context,
        dry_run=train_config.dry_run,
    )
    checkpoint_output_dir = run_artifacts.checkpoint_output_dir

    if train_config.dry_run:
        return _build_dry_run_summary(train_rows, eval_rows, output_dir)

    from transformers import TrainingArguments, set_seed

    try:
        configure_torch_runtime(tf32=train_config.tf32)
        ensure_dir(output_dir)
        ensure_dir(checkpoint_output_dir)
        set_seed(train_config.seed)
        logger.info("Starting SFT with distributed context: %s", distributed_context.as_dict())
        if train_config.max_images_per_sample is not None:
            logger.info(
                "Applied max_images_per_sample=%s filter: train %s -> %s rows, eval %s -> %s rows",
                train_config.max_images_per_sample,
                original_train_rows,
                len(train_rows),
                original_eval_rows,
                len(eval_rows),
            )
        if not train_rows:
            raise ValueError("No SFT training rows remain after applying the configured filters.")

        training_args = TrainingArguments(
            output_dir=str(checkpoint_output_dir),
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            per_device_eval_batch_size=train_config.per_device_eval_batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            num_train_epochs=train_config.num_train_epochs,
            max_steps=train_config.max_steps,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            warmup_ratio=train_config.warmup_ratio,
            max_grad_norm=train_config.max_grad_norm,
            logging_steps=train_config.logging_steps,
            logging_dir=str(run_artifacts.tensorboard_dir),
            save_steps=train_config.save_steps,
            save_strategy=train_config.save_strategy,
            eval_steps=train_config.eval_steps,
            save_total_limit=train_config.save_total_limit,
            prediction_loss_only=train_config.prediction_loss_only,
            bf16=train_config.bf16,
            fp16=train_config.fp16,
            tf32=train_config.tf32,
            report_to=run_artifacts.report_to,
            run_name=run_artifacts.run_name,
            eval_strategy="steps" if eval_rows else "no",
            remove_unused_columns=False,
            gradient_checkpointing=train_config.gradient_checkpointing,
            seed=train_config.seed,
            data_seed=train_config.seed,
            dataloader_num_workers=train_config.dataloader_num_workers,
            dataloader_pin_memory=train_config.dataloader_pin_memory,
            dataloader_persistent_workers=train_config.dataloader_persistent_workers,
            ddp_find_unused_parameters=train_config.ddp_find_unused_parameters,
            ddp_timeout=train_config.ddp_timeout,
            log_on_each_node=train_config.log_on_each_node,
            save_on_each_node=train_config.save_on_each_node,
            full_determinism=train_config.full_determinism,
            disable_tqdm=not distributed_context.is_main_process,
            deepspeed=_resolve_deepspeed_config_path(train_config.deepspeed),
        )

        processor = load_processor(model_config)
        if train_config.sft_checkpoint_path:
            model = load_sft_checkpoint_model(
                model_config=model_config,
                checkpoint_path=train_config.sft_checkpoint_path,
                distributed_context=distributed_context,
                is_trainable=True,
            )
            freeze_stats = apply_freezing(model, train_config.freeze)
        else:
            model = load_model(
                model_config.model_name_or_path,
                model_config=model_config,
                distributed_context=distributed_context,
            )
            freeze_stats = apply_freezing(model, train_config.freeze)
            model = maybe_wrap_with_peft(model, train_config=train_config)

        validation_generation_config = None
        if train_config.eval_generation_metrics:
            if not eval_rows:
                raise ValueError("eval_generation_metrics is enabled but no validation rows are available.")
            validation_generation_config = {
                "enabled": True,
                "processor": processor,
                "eval_rows": eval_rows,
                "max_examples": train_config.eval_generation_max_examples,
                "batch_size": train_config.eval_generation_batch_size,
                "max_new_tokens": train_config.eval_generation_max_new_tokens,
                "output_dir": output_dir,
                "save_predictions": train_config.eval_generation_save_predictions,
            }

        trainer_class = _build_sft_trainer_class(
            train_config.loss_chunk_size,
            validation_generation_config=validation_generation_config,
        )
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=ManifestListDataset(train_rows),
            eval_dataset=ManifestListDataset(eval_rows) if eval_rows else None,
            data_collator=build_sft_data_collator(model_config=model_config, processor=processor),
            callbacks=[
                JsonlMetricsCallback(
                    run_artifacts.metrics_jsonl_path,
                    mirror_paths=[run_artifacts.legacy_metrics_jsonl_path],
                )
            ],
        )

        resume_path = resolve_resume_checkpoint(checkpoint_output_dir, train_config.resume_from_checkpoint)
        trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
        _save_trained_model(trainer, train_config=train_config, output_dir=checkpoint_output_dir)
        if trainer.is_world_process_zero():
            processor.save_pretrained(checkpoint_output_dir)
        summary = _build_dry_run_summary(train_rows, eval_rows, output_dir)
        summary["checkpoint_output_dir"] = str(checkpoint_output_dir)
        summary["freeze_stats"] = freeze_stats
        if trainer.is_world_process_zero():
            write_training_artifact_manifest(run_artifacts, extra={"freeze_stats": freeze_stats})
        logger.info("Finished SFT. Freeze stats: %s", freeze_stats)
        return summary
    finally:
        destroy_distributed_process_group()
