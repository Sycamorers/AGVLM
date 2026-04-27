"""Supervised fine-tuning entrypoints."""

import json
from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.manifest_io import read_manifest, summarize_manifest
from agri_vlm.logging_utils import configure_logging
from agri_vlm.modeling.freezing import apply_freezing
from agri_vlm.modeling.model_factory import load_model
from agri_vlm.modeling.peft_setup import maybe_wrap_with_peft
from agri_vlm.modeling.processor_factory import load_processor
from agri_vlm.training.callbacks import JsonlMetricsCallback
from agri_vlm.training.collators import QwenVLChatCollator
from agri_vlm.training.run_artifacts import prepare_run_artifacts, write_training_artifact_manifest
from agri_vlm.utils.checkpointing import resolve_resume_checkpoint
from agri_vlm.utils.distributed import configure_torch_runtime, get_distributed_context
from agri_vlm.utils.io import ensure_dir


class ManifestListDataset:
    """A tiny dataset wrapper around validated manifest rows."""

    def __init__(self, rows: List[Any]) -> None:
        self.rows = [row.model_dump(mode="json") for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


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
        chunk_logits = shift_logits[:, start:end, :].reshape(-1, vocab_size).float()
        chunk_labels = shift_labels[:, start:end].reshape(-1)
        chunk_loss = F.cross_entropy(
            chunk_logits,
            chunk_labels,
            ignore_index=ignore_index,
            reduction="sum",
        )
        total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss
    return total_loss / valid_items


def _build_sft_trainer_class(loss_chunk_size: int) -> Any:
    from transformers import Trainer

    if loss_chunk_size <= 0:
        return Trainer

    class ChunkedLossTrainer(Trainer):
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

    return ChunkedLossTrainer


def _build_dry_run_summary(train_rows: List[Any], eval_rows: List[Any], output_dir: Path) -> Dict[str, Any]:
    distributed_context = get_distributed_context()
    summary = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_summary": summarize_manifest(train_rows),
        "eval_summary": summarize_manifest(eval_rows),
        "distributed": distributed_context.as_dict(),
    }
    ensure_dir(output_dir)
    (output_dir / "dry_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_sft(model_config: Any, train_config: Any) -> Dict[str, Any]:
    """Run SFT or validate the setup in dry-run mode."""
    distributed_context = get_distributed_context(set_device=True)
    logger = configure_logging(logger_name="agri_vlm.training.sft")

    train_rows = read_manifest(Path(train_config.manifest_path))
    eval_rows = []
    if train_config.eval_manifest_path and Path(train_config.eval_manifest_path).exists():
        eval_rows = read_manifest(Path(train_config.eval_manifest_path))

    output_dir = Path(train_config.output_dir)
    if train_config.smoke_max_samples:
        train_rows = train_rows[: train_config.smoke_max_samples]
        eval_rows = eval_rows[: train_config.smoke_max_samples]

    run_artifacts = prepare_run_artifacts(
        stage="sft",
        model_config=model_config,
        train_config=train_config,
        distributed_context=distributed_context,
        dry_run=train_config.dry_run,
    )

    if train_config.dry_run:
        return _build_dry_run_summary(train_rows, eval_rows, output_dir)

    from transformers import TrainingArguments, set_seed

    configure_torch_runtime(tf32=train_config.tf32)
    ensure_dir(output_dir)
    set_seed(train_config.seed)
    logger.info("Starting SFT with distributed context: %s", distributed_context.as_dict())

    processor = load_processor(model_config)
    model = load_model(
        model_config.model_name_or_path,
        model_config=model_config,
        distributed_context=distributed_context,
    )
    freeze_stats = apply_freezing(model, train_config.freeze)
    model = maybe_wrap_with_peft(model, train_config=train_config)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        num_train_epochs=train_config.num_train_epochs,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        max_grad_norm=train_config.max_grad_norm,
        logging_steps=train_config.logging_steps,
        logging_dir=str(run_artifacts.tensorboard_dir),
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        save_total_limit=train_config.save_total_limit,
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
    )

    trainer_class = _build_sft_trainer_class(train_config.loss_chunk_size)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=ManifestListDataset(train_rows),
        eval_dataset=ManifestListDataset(eval_rows) if eval_rows else None,
        data_collator=QwenVLChatCollator(processor=processor),
        callbacks=[
            JsonlMetricsCallback(
                run_artifacts.metrics_jsonl_path,
                mirror_paths=[run_artifacts.legacy_metrics_jsonl_path],
            )
        ],
    )

    resume_path = resolve_resume_checkpoint(output_dir, train_config.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    trainer.save_model()
    if trainer.is_world_process_zero():
        processor.save_pretrained(output_dir)
    summary = _build_dry_run_summary(train_rows, eval_rows, output_dir)
    summary["freeze_stats"] = freeze_stats
    if trainer.is_world_process_zero():
        write_training_artifact_manifest(run_artifacts, extra={"freeze_stats": freeze_stats})
    logger.info("Finished SFT. Freeze stats: %s", freeze_stats)
    return summary
