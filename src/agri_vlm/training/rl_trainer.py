"""GRPO post-training entrypoints."""

import json
from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.conversation_format import sample_to_prompt_messages
from agri_vlm.data.manifest_io import read_manifest, summarize_manifest
from agri_vlm.logging_utils import configure_logging
from agri_vlm.modeling.model_factory import load_sft_checkpoint_model
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
    processor = load_processor(model_config)
    model = load_sft_checkpoint_model(
        model_config=model_config,
        checkpoint_path=train_config.sft_checkpoint_path,
        distributed_context=distributed_context,
    )
    records = []
    for row in rows:
        prompt = processor.apply_chat_template(
            sample_to_prompt_messages(row),
            tokenize=False,
            add_generation_prompt=True,
        )
        records.append(
            {
                "prompt": prompt,
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
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        bf16=train_config.bf16,
        tf32=train_config.tf32,
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
