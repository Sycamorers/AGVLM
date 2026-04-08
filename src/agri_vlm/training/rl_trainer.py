"""GRPO post-training entrypoints."""

import json
from pathlib import Path
from typing import Any, Dict, List

from agri_vlm.data.conversation_format import sample_to_prompt_messages
from agri_vlm.data.manifest_io import read_manifest, summarize_manifest
from agri_vlm.modeling.peft_setup import build_lora_config
from agri_vlm.modeling.processor_factory import load_processor
from agri_vlm.rewards.composite import make_trl_reward_function
from agri_vlm.utils.image import open_image
from agri_vlm.utils.io import ensure_dir


def _build_rl_dry_run_summary(rows: List[Any], output_dir: Path) -> Dict[str, Any]:
    summary = {
        "train_rows": len(rows),
        "train_summary": summarize_manifest(rows),
    }
    ensure_dir(output_dir)
    (output_dir / "dry_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_rl_grpo(model_config: Any, train_config: Any) -> Dict[str, Any]:
    """Run GRPO training on top of the SFT checkpoint."""
    rows = read_manifest(Path(train_config.manifest_path))
    if train_config.smoke_max_samples:
        rows = rows[: train_config.smoke_max_samples]

    output_dir = Path(train_config.output_dir)
    if train_config.dry_run:
        return _build_rl_dry_run_summary(rows, output_dir)

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    ensure_dir(output_dir)
    processor = load_processor(model_config)
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
    )

    trainer = GRPOTrainer(
        model=train_config.sft_checkpoint_path,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=processor,
        reward_funcs=[
            make_trl_reward_function(
                reward_modules=train_config.reward_modules,
                reward_weights=train_config.reward_weights,
            )
        ],
        peft_config=build_lora_config(train_config) if train_config.use_peft else None,
    )
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_dir)
    return _build_rl_dry_run_summary(rows, output_dir)
