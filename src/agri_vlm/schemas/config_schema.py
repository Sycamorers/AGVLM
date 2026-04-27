"""Configuration schemas."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agri_vlm.utils.io import load_yaml


class LoRAConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = Field(default_factory=list)


class FreezeConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    freeze_vision_encoder: bool = True
    freeze_projector: bool = False
    trainable_text_modules: List[str] = Field(default_factory=list)


class ModelConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model_name_or_path: str
    processor_name_or_path: Optional[str] = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    use_flash_attention_2: bool = True
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    trust_remote_code: bool = False
    device_map: Optional[str] = "auto"
    distributed_device_map: str = "local_process"
    gradient_checkpointing: bool = True
    low_cpu_mem_usage: bool = True
    use_cache: bool = False
    max_pixels: Optional[int] = None
    min_pixels: Optional[int] = None


class TrainConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: str
    output_dir: str
    eval_manifest_path: Optional[str] = None
    seed: int = 17
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    loss_chunk_size: int = Field(default=0, ge=0)
    use_peft: bool = True
    report_to: List[str] = Field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    logging_dir: Optional[str] = None
    artifact_dir: Optional[str] = None
    save_run_metadata: bool = True
    dry_run: bool = False
    smoke_max_samples: int = 8
    max_images_per_sample: Optional[int] = Field(default=None, ge=1)
    resume_from_checkpoint: Optional[str] = "auto"
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    ddp_find_unused_parameters: Optional[bool] = False
    ddp_timeout: int = 1800
    log_on_each_node: bool = False
    save_on_each_node: bool = False
    full_determinism: bool = False
    tf32: bool = True
    freeze: FreezeConfigSchema = Field(default_factory=FreezeConfigSchema)
    lora: LoRAConfigSchema = Field(default_factory=LoRAConfigSchema)


class RLTrainConfigSchema(TrainConfigSchema):
    model_config = ConfigDict(extra="forbid")

    sft_checkpoint_path: str
    beta: float = 0.0
    num_generations: int = 4
    max_prompt_length: int = 2048
    max_completion_length: int = 256
    loss_type: str = "grpo"
    scale_rewards: str = "group"
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    reward_modules: List[str] = Field(default_factory=list)
    reward_weights: Dict[str, float] = Field(default_factory=dict)


class EvalConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: str
    output_path: str
    batch_size: int = 1
    max_new_tokens: int = 128
    prediction_mode: str = "oracle"
    checkpoint_path: Optional[str] = None
    predictions_path: Optional[str] = None
    max_examples: int = 0


class ManifestBuildConfigSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    sources: Any
    output_path: Optional[str] = None


def load_config(path: Path, schema_class: Any) -> Any:
    payload = load_yaml(path)
    return schema_class.model_validate(payload)


def resolve_path(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()
