"""Supervised fine-tuning entrypoints."""

import json
from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.manifest_io import read_manifest, summarize_manifest
from agri_vlm.modeling.freezing import apply_freezing
from agri_vlm.modeling.model_factory import load_model
from agri_vlm.modeling.peft_setup import maybe_wrap_with_peft
from agri_vlm.modeling.processor_factory import load_processor
from agri_vlm.training.callbacks import JsonlMetricsCallback
from agri_vlm.training.collators import QwenVLChatCollator
from agri_vlm.utils.checkpointing import find_latest_checkpoint
from agri_vlm.utils.io import ensure_dir


class ManifestListDataset:
    """A tiny dataset wrapper around validated manifest rows."""

    def __init__(self, rows: List[Any]) -> None:
        self.rows = [row.model_dump(mode="json") for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


def _build_dry_run_summary(train_rows: List[Any], eval_rows: List[Any], output_dir: Path) -> Dict[str, Any]:
    summary = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_summary": summarize_manifest(train_rows),
        "eval_summary": summarize_manifest(eval_rows),
    }
    ensure_dir(output_dir)
    (output_dir / "dry_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_sft(model_config: Any, train_config: Any) -> Dict[str, Any]:
    """Run SFT or validate the setup in dry-run mode."""
    train_rows = read_manifest(Path(train_config.manifest_path))
    eval_rows = []
    if train_config.eval_manifest_path and Path(train_config.eval_manifest_path).exists():
        eval_rows = read_manifest(Path(train_config.eval_manifest_path))

    output_dir = Path(train_config.output_dir)
    if train_config.smoke_max_samples:
        train_rows = train_rows[: train_config.smoke_max_samples]
        eval_rows = eval_rows[: train_config.smoke_max_samples]

    if train_config.dry_run:
        return _build_dry_run_summary(train_rows, eval_rows, output_dir)

    from transformers import Trainer, TrainingArguments, set_seed

    ensure_dir(output_dir)
    set_seed(train_config.seed)

    processor = load_processor(model_config)
    model = load_model(model_config.model_name_or_path, model_config=model_config)
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
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        save_total_limit=train_config.save_total_limit,
        bf16=train_config.bf16,
        fp16=train_config.fp16,
        report_to=train_config.report_to,
        evaluation_strategy="steps" if eval_rows else "no",
        remove_unused_columns=False,
        gradient_checkpointing=train_config.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ManifestListDataset(train_rows),
        eval_dataset=ManifestListDataset(eval_rows) if eval_rows else None,
        data_collator=QwenVLChatCollator(processor=processor),
        callbacks=[JsonlMetricsCallback(output_dir / "metrics.jsonl")],
    )

    resume_path = find_latest_checkpoint(output_dir)
    trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    trainer.save_model()
    processor.save_pretrained(output_dir)
    summary = _build_dry_run_summary(train_rows, eval_rows, output_dir)
    summary["freeze_stats"] = freeze_stats
    return summary
