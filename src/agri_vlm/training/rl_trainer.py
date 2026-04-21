"""GRPO post-training entrypoints."""

import json
from pathlib import Path
from typing import Any, Dict, List

from transformers.processing_utils import ProcessorMixin

from agri_vlm.data.conversation_format import sample_to_prompt_messages
from agri_vlm.data.manifest_io import read_manifest, summarize_manifest
from agri_vlm.logging_utils import configure_logging
from agri_vlm.modeling.model_factory import load_sft_checkpoint_model, torch_dtype_from_name
from agri_vlm.modeling.processor_factory import load_processor
from agri_vlm.rewards.composite import make_trl_reward_function
from agri_vlm.utils.checkpointing import resolve_resume_checkpoint
from agri_vlm.utils.distributed import configure_torch_runtime, get_distributed_context
from agri_vlm.utils.image import open_image
from agri_vlm.utils.io import ensure_dir


def _build_rl_dry_run_summary(rows: List[Any], output_dir: Path) -> Dict[str, Any]:
    distributed_context = get_distributed_context()
    summary = {
        "train_rows": len(rows),
        "train_summary": summarize_manifest(rows),
        "distributed": distributed_context.as_dict(),
    }
    ensure_dir(output_dir)
    (output_dir / "dry_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _drop_none_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none_fields(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none_fields(item) for item in value]
    return value


def _cast_vision_modules(model: Any, dtype: Any) -> None:
    stack = [model]
    seen = set()
    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        visual = getattr(current, "visual", None)
        if visual is not None and hasattr(visual, "to"):
            visual.to(dtype=dtype)
        for child_name in ("base_model", "model", "module"):
            child = getattr(current, child_name, None)
            if child is not None:
                stack.append(child)


def _wrap_generate_with_autocast(model: Any, dtype: Any) -> None:
    import torch

    if dtype not in (torch.bfloat16, torch.float16):
        return
    original_generate = model.generate

    def generate_with_autocast(*args: Any, **kwargs: Any) -> Any:
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=dtype):
                return original_generate(*args, **kwargs)
        return original_generate(*args, **kwargs)

    model.generate = generate_with_autocast


class ProcessorDTypeAdapter(ProcessorMixin):
    """Cast processor multimodal tensors to the dtype used by the loaded model."""

    def __init__(self, processor: Any, pixel_dtype: Any) -> None:
        self.processor = processor
        self.pixel_dtype = pixel_dtype

    def _cast_batch(self, batch: Any) -> Any:
        for key in ("pixel_values", "pixel_values_videos"):
            try:
                value = batch[key]
            except (KeyError, TypeError):
                continue
            if hasattr(value, "to"):
                batch[key] = value.to(dtype=self.pixel_dtype)
        return batch

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._cast_batch(self.processor(*args, **kwargs))

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self._cast_batch(self.processor.apply_chat_template(*args, **kwargs))

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        return self.processor.save_pretrained(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.processor, name)


def run_rl_grpo(model_config: Any, train_config: Any) -> Dict[str, Any]:
    """Run GRPO training on top of the SFT checkpoint."""
    distributed_context = get_distributed_context(set_device=True)
    logger = configure_logging(logger_name="agri_vlm.training.rl")
    rows = read_manifest(Path(train_config.manifest_path))
    if train_config.smoke_max_samples:
        rows = rows[: train_config.smoke_max_samples]

    output_dir = Path(train_config.output_dir)
    if train_config.dry_run:
        return _build_rl_dry_run_summary(rows, output_dir)

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    configure_torch_runtime(tf32=train_config.tf32)
    ensure_dir(output_dir)
    logger.info("Starting GRPO with distributed context: %s", distributed_context.as_dict())
    processor = ProcessorDTypeAdapter(
        load_processor(model_config),
        pixel_dtype=torch_dtype_from_name(model_config.torch_dtype),
    )
    model = load_sft_checkpoint_model(
        model_config=model_config,
        checkpoint_path=train_config.sft_checkpoint_path,
        distributed_context=distributed_context,
    )
    _cast_vision_modules(model, torch_dtype_from_name(model_config.torch_dtype))
    _wrap_generate_with_autocast(model, torch_dtype_from_name(model_config.torch_dtype))
    records = []
    for row in rows:
        records.append(
            {
                "prompt": sample_to_prompt_messages(row),
                "image_paths": row.images,
                "task_type": row.task_type,
                "sample_id": row.sample_id,
                "target_json": json.dumps(row.target.model_dump(mode="json"), ensure_ascii=False),
                "verifier_json": json.dumps(row.verifier.model_dump(mode="json"), ensure_ascii=False),
                "reward_meta_json": json.dumps(row.reward_meta.model_dump(mode="json"), ensure_ascii=False),
                "metadata_json": json.dumps(row.metadata, ensure_ascii=False),
            }
        )

    dataset = Dataset.from_list(records)

    def transform(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        batch["prompt"] = [_drop_none_fields(prompt) for prompt in batch["prompt"]]
        batch["images"] = [
            [open_image(Path(path)) for path in image_paths] for image_paths in batch["image_paths"]
        ]
        return batch

    dataset.set_transform(transform)

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        num_train_epochs=train_config.num_train_epochs,
        max_grad_norm=train_config.max_grad_norm,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        bf16=train_config.bf16,
        tf32=train_config.tf32,
        gradient_checkpointing=train_config.gradient_checkpointing,
        max_prompt_length=train_config.max_prompt_length,
        max_completion_length=train_config.max_completion_length,
        num_generations=train_config.num_generations,
        beta=train_config.beta,
        loss_type=train_config.loss_type,
        scale_rewards=train_config.scale_rewards,
        use_vllm=train_config.use_vllm,
        vllm_mode=train_config.vllm_mode,
        report_to=train_config.report_to,
        remove_unused_columns=False,
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

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=processor,
        reward_funcs=[
            make_trl_reward_function(
                reward_modules=train_config.reward_modules,
                reward_weights=train_config.reward_weights,
            )
        ],
    )
    resume_path = resolve_resume_checkpoint(output_dir, train_config.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    trainer.save_model()
    if trainer.is_world_process_zero():
        processor.save_pretrained(output_dir)
    logger.info("Finished GRPO run with %s training rows.", len(rows))
    return _build_rl_dry_run_summary(rows, output_dir)
