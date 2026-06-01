"""GRPO post-training entrypoints."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from transformers.processing_utils import ProcessorMixin
except ImportError:  # pragma: no cover - transformers is optional for dry-run/safety tests
    class ProcessorMixin:  # type: ignore[no-redef]
        pass

from agri_vlm.data.conversation_format import sample_to_prompt_messages
from agri_vlm.data.manifest_io import read_manifest, summarize_manifest
from agri_vlm.logging_utils import configure_logging
from agri_vlm.modeling.model_factory import load_model, load_sft_checkpoint_model, torch_dtype_from_name
from agri_vlm.modeling.processor_factory import load_processor
from agri_vlm.rewards.composite import make_trl_reward_function
from agri_vlm.training.callbacks import JsonlMetricsCallback
from agri_vlm.training.run_artifacts import prepare_run_artifacts, write_training_artifact_manifest
from agri_vlm.utils.checkpointing import checkpoint_has_valid_model_artifacts, resolve_resume_checkpoint
from agri_vlm.utils.distributed import configure_torch_runtime, get_distributed_context
from agri_vlm.utils.image import open_image
from agri_vlm.utils.io import ensure_dir


SFT_CHECKPOINT_PLACEHOLDER_TOKENS = (
    "<",
    ">",
    "final_sft_checkpoint",
    "final-sft-checkpoint",
    "checkpoint_or_adapter",
    "placeholder",
    "replace_me",
    "todo",
)


def _build_rl_dry_run_summary(rows: List[Any], output_dir: Path) -> Dict[str, Any]:
    distributed_context = get_distributed_context()
    transformed_sample_check = verify_grpo_transformed_sample(rows[:1]) if rows else {"ok": False, "reason": "no rows"}
    summary = {
        "train_rows": len(rows),
        "train_summary": summarize_manifest(rows),
        "distributed": distributed_context.as_dict(),
        "transformed_sample_check": transformed_sample_check,
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


def build_grpo_records(rows: List[Any]) -> List[Dict[str, Any]]:
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
    return records


def transform_grpo_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    batch["prompt"] = [_drop_none_fields(prompt) for prompt in batch["prompt"]]
    batch["images"] = [
        [open_image(Path(path)) for path in image_paths] for image_paths in batch["image_paths"]
    ]
    return batch


def verify_grpo_transformed_sample(rows: List[Any]) -> Dict[str, Any]:
    if not rows:
        return {"ok": False, "reason": "no rows"}
    records = build_grpo_records(rows[:1])
    batch = {key: [value] for key, value in records[0].items()}
    transformed = transform_grpo_batch(batch)
    images = transformed.get("images") or []
    first_images = images[0] if images else []
    return {
        "ok": bool(first_images),
        "columns": sorted(transformed.keys()),
        "image_count": len(first_images),
        "image_modes": [getattr(image, "mode", "") for image in first_images],
    }


def _cast_vision_modules(model: Any, dtype: Any) -> None:
    if _is_quantized_model(model):
        return
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


def _is_quantized_model(model: Any) -> bool:
    stack = [model]
    seen = set()
    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        if bool(getattr(current, "is_loaded_in_4bit", False) or getattr(current, "is_loaded_in_8bit", False)):
            return True
        config = getattr(current, "config", None)
        if getattr(config, "quantization_config", None) is not None:
            return True
        for child_name in ("base_model", "model", "module"):
            child = getattr(current, child_name, None)
            if child is not None:
                stack.append(child)
    return False


def _is_quantized_model_config(model_config: Any) -> bool:
    return bool(getattr(model_config, "load_in_4bit", False) or getattr(model_config, "load_in_8bit", False))


def _is_quantized_cast_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "bitsandbytes" in message and ("cannot cast" in message or "new dtype" in message)


def _load_with_quantized_cast_guard(loader: Any) -> Any:
    """Ignore remote-code dtype casts that are invalid for bitsandbytes models."""
    from transformers import modeling_utils

    original_to = modeling_utils.PreTrainedModel.to

    def guarded_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original_to(self, *args, **kwargs)
        except ValueError as exc:
            if _is_quantized_cast_error(exc) and _is_quantized_model(self):
                return self
            raise

    modeling_utils.PreTrainedModel.to = guarded_to
    try:
        return loader()
    finally:
        modeling_utils.PreTrainedModel.to = original_to


def _prepare_quantized_model_for_training(model: Any, model_config: Any) -> Any:
    from peft import prepare_model_for_kbit_training

    try:
        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(getattr(model_config, "gradient_checkpointing", False)),
        )
    except TypeError:
        return prepare_model_for_kbit_training(model)


def _load_sft_checkpoint_model_for_rl(
    model_config: Any,
    checkpoint_path: str,
    distributed_context: Any,
) -> Any:
    if not _is_quantized_model_config(model_config):
        return load_sft_checkpoint_model(
            model_config=model_config,
            checkpoint_path=checkpoint_path,
            distributed_context=distributed_context,
        )

    checkpoint_dir = Path(checkpoint_path)
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        return _load_with_quantized_cast_guard(
            lambda: load_sft_checkpoint_model(
                model_config=model_config,
                checkpoint_path=checkpoint_path,
                distributed_context=distributed_context,
            )
        )

    from peft import PeftModel

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model_name_or_path = adapter_config.get("base_model_name_or_path") or model_config.model_name_or_path
    model = _load_with_quantized_cast_guard(
        lambda: load_model(
            model_name_or_path=base_model_name_or_path,
            model_config=model_config,
            distributed_context=distributed_context,
        )
    )
    model = _prepare_quantized_model_for_training(model, model_config)
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
    if hasattr(model, "config"):
        model.config.use_cache = model_config.use_cache
    return model


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


def _resolve_deepspeed_config_path(config_value: Optional[str]) -> Optional[str]:
    if not config_value:
        return None
    path = Path(config_value).expanduser()
    if path.exists():
        return str(path)
    repo_relative = Path.cwd() / config_value
    return str(repo_relative) if repo_relative.exists() else config_value


def _is_placeholder_checkpoint_path(checkpoint_path: str) -> bool:
    normalized = checkpoint_path.strip().lower()
    return any(token in normalized for token in SFT_CHECKPOINT_PLACEHOLDER_TOKENS)


def _is_base_model_checkpoint_path(checkpoint_path: str, model_config: Any) -> bool:
    candidates = {
        str(getattr(model_config, "model_name_or_path", "") or "").strip(),
        str(getattr(model_config, "processor_name_or_path", "") or "").strip(),
        str(getattr(model_config, "name", "") or "").strip(),
    }
    normalized_checkpoint = checkpoint_path.strip()
    if normalized_checkpoint in candidates:
        return True
    path = Path(normalized_checkpoint).expanduser()
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        if path == candidate_path:
            return True
        if path.exists() and candidate_path.exists() and path.resolve() == candidate_path.resolve():
            return True
    return False


def _checkpoint_has_model_artifacts(path: Path) -> bool:
    return checkpoint_has_valid_model_artifacts(path)


def validate_rl_sft_checkpoint_path(model_config: Any, train_config: Any) -> Path:
    """Validate the mandatory completed-SFT checkpoint for non-dry-run GRPO."""
    checkpoint_path = str(getattr(train_config, "sft_checkpoint_path", "") or "").strip()
    if not checkpoint_path:
        raise ValueError("Non-dry-run GRPO requires `sft_checkpoint_path` to point to a completed SFT checkpoint.")
    if _is_placeholder_checkpoint_path(checkpoint_path):
        raise ValueError(
            "Non-dry-run GRPO cannot use placeholder `sft_checkpoint_path`: %s" % checkpoint_path
        )
    if _is_base_model_checkpoint_path(checkpoint_path, model_config):
        raise ValueError(
            "Non-dry-run GRPO must start from a completed SFT checkpoint, not the raw/base model: %s"
            % checkpoint_path
        )
    resolved_path = Path(checkpoint_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(
            "Configured `sft_checkpoint_path` does not exist. Wait for SFT completion or set a real adapter/checkpoint path: %s"
            % checkpoint_path
        )
    if not _checkpoint_has_model_artifacts(resolved_path):
        raise FileNotFoundError(
            "`sft_checkpoint_path` exists but does not look like a completed SFT model or adapter checkpoint: %s"
            % checkpoint_path
        )
    return resolved_path


def _is_phi4_reasoning_model_config(model_config: Any) -> bool:
    model_name = "%s %s" % (
        getattr(model_config, "name", ""),
        getattr(model_config, "model_name_or_path", ""),
    )
    lower_name = model_name.lower()
    return "phi-4-reasoning-vision" in lower_name or "phi4_reasoning_vision" in lower_name


def _as_conversation_batch(conversation: Any) -> Any:
    if isinstance(conversation, list) and conversation and isinstance(conversation[0], list):
        return conversation, True
    return [conversation], False


def _render_phi4_reasoning_conversation(conversation: List[Dict[str, Any]]) -> Any:
    rendered = []
    images = []
    for message in conversation:
        parts = []
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            for block in content:
                block_type = block.get("type")
                if block_type == "image":
                    parts.append("<image>")
                    image = block.get("image")
                    if image is not None:
                        images.append(image)
                elif block_type == "text":
                    parts.append(block.get("text", ""))
        rendered.append({"role": message.get("role", "user"), "content": "".join(parts)})
    return rendered, images


def _as_python_token_ids(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, list):
        return value.tolist()
    return value


def _decode_replacement_token_id(tokenizer: Any) -> int:
    for attr in ("pad_token_id", "eos_token_id", "unk_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _max_decode_token_id(tokenizer: Any) -> Optional[int]:
    try:
        return int(len(tokenizer))
    except TypeError:
        value = getattr(tokenizer, "vocab_size", None)
        return int(value) if isinstance(value, int) else None


def _sanitize_token_ids_for_decode(value: Any, tokenizer: Any) -> Any:
    value = _as_python_token_ids(value)
    replacement = _decode_replacement_token_id(tokenizer)
    max_token_id = _max_decode_token_id(tokenizer)

    def sanitize(item: Any) -> Any:
        item = _as_python_token_ids(item)
        if isinstance(item, list):
            return [sanitize(child) for child in item]
        if isinstance(item, int):
            if item < 0:
                return replacement
            if max_token_id is not None and item >= max_token_id:
                return replacement
        return item

    return sanitize(value)


def _looks_like_phi4_spatial_shapes(value: Any) -> bool:
    return bool(hasattr(value, "ndim") and value.ndim == 2 and int(value.shape[-1]) == 2)


class ProcessorDTypeAdapter(ProcessorMixin):
    """Cast processor multimodal tensors to the dtype used by the loaded model."""

    def __init__(self, processor: Any, pixel_dtype: Any, model_config: Any) -> None:
        self.processor = processor
        self.pixel_dtype = pixel_dtype
        self.model_config = model_config

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
        if _is_phi4_reasoning_model_config(self.model_config) and not hasattr(self.processor, "chat_template"):
            return self._apply_phi4_reasoning_chat_template(*args, **kwargs)
        return self._cast_batch(self.processor.apply_chat_template(*args, **kwargs))

    def batch_decode(self, sequences: Any, *args: Any, **kwargs: Any) -> Any:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return self.processor.batch_decode(sequences, *args, **kwargs)
        sanitized = _sanitize_token_ids_for_decode(sequences, tokenizer)
        return self.processor.batch_decode(sanitized, *args, **kwargs)

    def decode(self, token_ids: Any, *args: Any, **kwargs: Any) -> Any:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return self.processor.decode(token_ids, *args, **kwargs)
        sanitized = _sanitize_token_ids_for_decode(token_ids, tokenizer)
        return self.processor.decode(sanitized, *args, **kwargs)

    def _apply_phi4_reasoning_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        conversation = kwargs.pop("conversation", None)
        if conversation is None:
            if not args:
                raise TypeError("apply_chat_template requires a conversation.")
            conversation = args[0]
            args = args[1:]
        if args:
            raise TypeError("Unexpected positional arguments for Phi-4 RL chat template fallback.")

        tokenize = bool(kwargs.pop("tokenize", False))
        return_dict = bool(kwargs.pop("return_dict", False))
        add_generation_prompt = bool(kwargs.pop("add_generation_prompt", False))
        padding_side = kwargs.pop("padding_side", None)
        kwargs.pop("tools", None)
        kwargs.pop("documents", None)
        kwargs.pop("continue_final_message", None)

        conversations, was_batched = _as_conversation_batch(conversation)
        rendered_conversations = []
        image_batches = []
        for item in conversations:
            rendered, images = _render_phi4_reasoning_conversation(item)
            rendered_conversations.append(rendered)
            image_batches.append(images)

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise AttributeError("Phi-4 RL chat template fallback requires a processor tokenizer chat template.")

        texts = [
            tokenizer.apply_chat_template(
                rendered,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            for rendered in rendered_conversations
        ]
        if not tokenize:
            return texts if was_batched else texts[0]

        previous_padding_side = getattr(tokenizer, "padding_side", None)
        if padding_side and previous_padding_side is not None:
            tokenizer.padding_side = padding_side
        try:
            processor_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key
                in {
                    "add_special_tokens",
                    "max_length",
                    "padding",
                    "return_tensors",
                    "truncation",
                }
            }
            flat_images = [image for images in image_batches for image in images]
            batch = self.processor(
                text=texts,
                images=flat_images or None,
                **processor_kwargs,
            )
        finally:
            if padding_side and previous_padding_side is not None:
                tokenizer.padding_side = previous_padding_side

        if not return_dict and hasattr(batch, "input_ids"):
            return batch.input_ids
        return self._cast_batch(batch)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        if _is_phi4_reasoning_model_config(self.model_config):
            for optional_attribute in ("chat_template", "audio_tokenizer"):
                if not hasattr(self.processor, optional_attribute):
                    setattr(self.processor, optional_attribute, None)
        return self.processor.save_pretrained(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.processor, name)


def run_rl_grpo(model_config: Any, train_config: Any) -> Dict[str, Any]:
    """Run GRPO training on top of the SFT checkpoint."""
    distributed_context = get_distributed_context(set_device=True)
    logger = configure_logging(logger_name="agri_vlm.training.rl")
    if not train_config.dry_run:
        validate_rl_sft_checkpoint_path(model_config=model_config, train_config=train_config)
    rows = read_manifest(Path(train_config.manifest_path))
    if train_config.smoke_max_samples:
        rows = rows[: train_config.smoke_max_samples]

    output_dir = Path(train_config.output_dir)
    run_artifacts = prepare_run_artifacts(
        stage="rl_grpo",
        model_config=model_config,
        train_config=train_config,
        distributed_context=distributed_context,
        dry_run=train_config.dry_run,
    )
    checkpoint_output_dir = run_artifacts.checkpoint_output_dir
    if train_config.dry_run:
        return _build_rl_dry_run_summary(rows, output_dir)

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from trl.data_utils import apply_chat_template, prepare_multimodal_messages
    from trl.trainer.grpo_trainer import entropy_from_logits, selective_log_softmax
    from trl.trainer.utils import (
        shuffle_sequence_dict,
        split_pixel_values_by_grid,
        split_tensor_dict,
        unsplit_pixel_values_by_grid,
    )
    from transformers import Trainer
    import torch

    configure_torch_runtime(tf32=train_config.tf32)
    ensure_dir(output_dir)
    ensure_dir(checkpoint_output_dir)
    logger.info("Starting GRPO with distributed context: %s", distributed_context.as_dict())
    processor = ProcessorDTypeAdapter(
        load_processor(model_config),
        pixel_dtype=torch_dtype_from_name(model_config.torch_dtype),
        model_config=model_config,
    )
    model = _load_sft_checkpoint_model_for_rl(
        model_config=model_config,
        checkpoint_path=train_config.sft_checkpoint_path,
        distributed_context=distributed_context,
    )
    _cast_vision_modules(model, torch_dtype_from_name(model_config.torch_dtype))
    _wrap_generate_with_autocast(model, torch_dtype_from_name(model_config.torch_dtype))
    records = build_grpo_records(rows)

    dataset = Dataset.from_list(records)
    dataset.set_transform(transform_grpo_batch)

    grpo_kwargs = {
        "output_dir": str(checkpoint_output_dir),
        "learning_rate": train_config.learning_rate,
        "weight_decay": train_config.weight_decay,
        "warmup_ratio": train_config.warmup_ratio,
        "num_train_epochs": train_config.num_train_epochs,
        "max_steps": train_config.max_steps,
        "max_grad_norm": train_config.max_grad_norm,
        "logging_steps": train_config.logging_steps,
        "logging_dir": str(run_artifacts.tensorboard_dir),
        "save_steps": train_config.save_steps,
        "save_total_limit": train_config.save_total_limit,
        "per_device_train_batch_size": train_config.per_device_train_batch_size,
        "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
        "bf16": train_config.bf16,
        "tf32": train_config.tf32,
        "gradient_checkpointing": train_config.gradient_checkpointing,
        "deepspeed": _resolve_deepspeed_config_path(train_config.deepspeed),
        "max_prompt_length": train_config.max_prompt_length,
        "max_completion_length": train_config.max_completion_length,
        "num_generations": train_config.num_generations,
        "beta": train_config.beta,
        "loss_type": train_config.loss_type,
        "scale_rewards": train_config.scale_rewards,
        "use_vllm": train_config.use_vllm,
        "vllm_mode": train_config.vllm_mode,
        "report_to": run_artifacts.report_to,
        "run_name": run_artifacts.run_name,
        "remove_unused_columns": False,
        "seed": train_config.seed,
        "data_seed": train_config.seed,
        "dataloader_num_workers": train_config.dataloader_num_workers,
        "dataloader_pin_memory": train_config.dataloader_pin_memory,
        "dataloader_persistent_workers": train_config.dataloader_persistent_workers,
        "ddp_find_unused_parameters": train_config.ddp_find_unused_parameters,
        "ddp_timeout": train_config.ddp_timeout,
        "log_on_each_node": train_config.log_on_each_node,
        "save_on_each_node": train_config.save_on_each_node,
        "full_determinism": train_config.full_determinism,
        "disable_tqdm": not distributed_context.is_main_process,
    }
    if train_config.optim:
        grpo_kwargs["optim"] = train_config.optim
    if train_config.optim_args:
        grpo_kwargs["optim_args"] = train_config.optim_args
    grpo_args = GRPOConfig(**grpo_kwargs)

    class AgriPhi4GRPOTrainer(GRPOTrainer):
        """GRPOTrainer compatibility shim for Phi-4 Reasoning Vision processor fields."""

        def _is_phi4_spatial_generation_batch(self, batch: Dict[str, Any]) -> bool:
            return (
                _is_phi4_reasoning_model_config(model_config)
                and _looks_like_phi4_spatial_shapes(batch.get("image_grid_thw"))
                and batch.get("pixel_values") is not None
            )

        def _phi4_forward_kwargs_from_inputs(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not _is_phi4_reasoning_model_config(model_config) or not inputs:
                return {}
            if "images" in inputs[0]:
                images = [example.get("images") for example in inputs]
            elif "image" in inputs[0]:
                images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
            else:
                return {}
            if images is None or all(image_list == [] for image_list in images):
                return {}
            prompts = [
                prepare_multimodal_messages(example["prompt"], image_list)
                for example, image_list in zip(inputs, images)
            ]
            prompts_text = [
                apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            return {key: value for key, value in prompt_inputs.items() if key not in ["input_ids", "attention_mask"]}

        def _generate_and_score_completions(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
            output = super()._generate_and_score_completions(inputs)
            if _is_phi4_reasoning_model_config(model_config) and "image_grid_thw" not in output:
                forward_kwargs = self._phi4_forward_kwargs_from_inputs(inputs)
                spatial_shapes = forward_kwargs.get("spatial_shapes")
                if spatial_shapes is not None:
                    output["image_grid_thw"] = spatial_shapes
            return output

        def _prepare_inputs(self, generation_batch: Dict[str, Any]) -> Dict[str, Any]:
            mode = "train" if self.model.training else "eval"
            if mode != "train":
                return self._generate_and_score_completions(generation_batch)

            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                is_phi4_spatial = self._is_phi4_spatial_generation_batch(generation_batch)
                if not is_phi4_spatial:
                    generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = (
                    generation_batches
                    if is_phi4_spatial
                    else [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
                )
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
            return inputs

        def _get_per_token_logps_and_entropies(
            self,
            model: Any,
            input_ids: Any,
            attention_mask: Any,
            logits_to_keep: int,
            batch_size: Optional[int] = None,
            compute_entropy: bool = False,
            pixel_values: Any = None,
            image_grid_thw: Any = None,
            num_images: Any = None,
            pixel_attention_mask: Any = None,
            image_sizes: Any = None,
            token_type_ids: Any = None,
            spatial_shapes: Any = None,
        ) -> Any:
            if image_grid_thw is None and spatial_shapes is not None and _is_phi4_reasoning_model_config(model_config):
                image_grid_thw = spatial_shapes
            if not _looks_like_phi4_spatial_shapes(image_grid_thw):
                return super()._get_per_token_logps_and_entropies(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=logits_to_keep,
                    batch_size=batch_size,
                    compute_entropy=compute_entropy,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    num_images=num_images,
                    pixel_attention_mask=pixel_attention_mask,
                    image_sizes=image_sizes,
                    token_type_ids=token_type_ids,
                )

            batch_size = batch_size or input_ids.size(0)
            all_logps = []
            all_entropies = []
            for start in range(0, input_ids.size(0), batch_size):
                input_ids_batch = input_ids[start : start + batch_size]
                attention_mask_batch = attention_mask[start : start + batch_size]
                model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

                image_start = start
                image_end = start + batch_size
                if num_images is not None:
                    counts = [int(item) for item in num_images]
                    if len(counts) >= start + batch_size:
                        cum_images = torch.tensor([0] + counts, device=input_ids.device).cumsum(0)
                        image_start = int(cum_images[start].item())
                        image_end = int(cum_images[start + batch_size].item())
                if pixel_values is not None:
                    model_inputs["pixel_values"] = pixel_values[image_start:image_end]
                if pixel_attention_mask is not None:
                    model_inputs["pixel_attention_mask"] = pixel_attention_mask[image_start:image_end]
                model_inputs["spatial_shapes"] = image_grid_thw[image_start:image_end]
                if image_sizes is not None:
                    model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
                if token_type_ids is not None:
                    model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]
                if "logits_to_keep" in self.model_kwarg_keys:
                    model_inputs["logits_to_keep"] = logits_to_keep + 1
                model_inputs["use_cache"] = False

                logits = model(**model_inputs).logits
                logits = logits[:, :-1, :]
                logits = logits[:, -logits_to_keep:, :]
                logits = logits / self.temperature
                completion_ids = input_ids_batch[:, -logits_to_keep:]
                logps = selective_log_softmax(logits, completion_ids)
                all_logps.append(logps)

                if compute_entropy:
                    with torch.no_grad():
                        all_entropies.append(entropy_from_logits(logits))

            logps = torch.cat(all_logps, dim=0)
            entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
            return logps, entropies

    trainer = AgriPhi4GRPOTrainer(
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
        callbacks=[
            JsonlMetricsCallback(
                run_artifacts.metrics_jsonl_path,
                mirror_paths=[run_artifacts.legacy_metrics_jsonl_path],
            )
        ],
    )
    resume_path = resolve_resume_checkpoint(checkpoint_output_dir, train_config.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    trainer.save_model()
    if trainer.is_world_process_zero():
        processor.save_pretrained(checkpoint_output_dir)
        write_training_artifact_manifest(
            run_artifacts,
            extra={
                "reward_modules": train_config.reward_modules,
                "reward_weights": train_config.reward_weights,
            },
        )
    logger.info("Finished GRPO run with %s training rows.", len(rows))
    return _build_rl_dry_run_summary(rows, output_dir)
