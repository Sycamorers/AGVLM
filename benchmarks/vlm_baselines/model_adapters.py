"""Inference-only Hugging Face VLM adapters.

All heavy ML imports are lazy so split creation and metric aggregation can run
in lightweight environments.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
import gc
from pathlib import Path
import re
import sys
import traceback
from typing import Any

from dataset_adapter import BenchmarkSample, build_chat_messages, build_plain_prompt
from prediction_parsing import extract_answer_field, normalize_text
from utils import REPO_ROOT, maybe_cuda_memory


@dataclass
class AdapterSpec:
    model_name: str
    loader_classes: list[str]
    prompt_style: str
    trust_remote_code: bool = False
    supports_multi_image: bool = True
    single_image_policy: str = "error"
    default_dtype: str = "bf16"
    default_quantization: str = "none"
    fallback_quantization: str = "4bit"
    default_attn_implementation: str | None = None
    processor_kwargs: dict[str, Any] | None = None
    notes: str = ""


MODEL_SPECS: dict[str, AdapterSpec] = {
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct": AdapterSpec(
        model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        loader_classes=["AutoModelForImageTextToText"],
        prompt_style="chat_prompt_images",
        notes="Uses Transformers image-text-to-text chat template with PIL images.",
    ),
    "google/paligemma2-3b-mix-448": AdapterSpec(
        model_name="google/paligemma2-3b-mix-448",
        loader_classes=["PaliGemmaForConditionalGeneration", "AutoModelForImageTextToText"],
        prompt_style="paligemma",
        supports_multi_image=False,
        single_image_policy="first_and_log",
        notes="PaliGemma is treated as single-image; multi-image samples use the first image and record that policy.",
    ),
    "microsoft/Phi-4-multimodal-instruct": AdapterSpec(
        model_name="microsoft/Phi-4-multimodal-instruct",
        loader_classes=["Phi4MMForCausalLM", "AutoModelForCausalLM"],
        prompt_style="phi4",
        trust_remote_code=True,
        default_attn_implementation="eager",
        processor_kwargs={"dynamic_hd": 4},
        notes="Uses Phi-4 multimodal image placeholder prompt format with capped dynamic image crops for L4 inference.",
    ),
    "allenai/Molmo2-4B": AdapterSpec(
        model_name="allenai/Molmo2-4B",
        loader_classes=["AutoModelForImageTextToText"],
        prompt_style="chat_tokenized_paths",
        trust_remote_code=True,
        notes="Uses Molmo2 remote-code processor chat template.",
    ),
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": AdapterSpec(
        model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        loader_classes=["LlavaOnevisionForConditionalGeneration", "AutoModelForImageTextToText"],
        prompt_style="chat_prompt_images",
        default_dtype="bf16",
        fallback_quantization="4bit",
        notes="OneVision ov checkpoint supports multi-image prompts; batch size remains 1.",
    ),
    "Qwen/Qwen2.5-VL-3B-Instruct": AdapterSpec(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        loader_classes=["Qwen2_5_VLForConditionalGeneration", "AutoModelForImageTextToText"],
        prompt_style="qwen_vl",
        processor_kwargs={"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28},
        notes="Uses qwen-vl-utils when installed; processor pixel budget is capped by default.",
    ),
    "microsoft/Phi-4-reasoning-vision-15B": AdapterSpec(
        model_name="microsoft/Phi-4-reasoning-vision-15B",
        loader_classes=["Phi4ForCausalLMV", "AutoModelForCausalLM", "AutoModelForImageTextToText"],
        prompt_style="phi4",
        trust_remote_code=True,
        default_attn_implementation="eager",
        notes="Project base model for SFT/RL checkpoint benchmarking.",
    ),
}


def patch_phi4mm_base_generation_hook(model_class: Any) -> bool:
    """Patch Phi-4-MM remote code for newer PEFT versions.

    The model's remote code applies LoRA adapters to the inner Phi4MMModel
    during construction. PEFT 0.18 expects CAUSAL_LM bases to expose
    prepare_inputs_for_generation, but that hook only exists on the outer
    Phi4MMForCausalLM class. Adding a minimal pass-through hook to the inner
    class restores construction without changing generated outputs.
    """

    module = sys.modules.get(getattr(model_class, "__module__", ""))
    base_class = getattr(module, "Phi4MMModel", None) if module is not None else None
    if base_class is None or hasattr(base_class, "prepare_inputs_for_generation"):
        return False

    def prepare_inputs_for_generation(self: Any, input_ids: Any, **kwargs: Any) -> dict[str, Any]:
        model_inputs = {"input_ids": input_ids}
        model_inputs.update({key: value for key, value in kwargs.items() if value is not None})
        return model_inputs

    setattr(base_class, "prepare_inputs_for_generation", prepare_inputs_for_generation)
    return True


def patch_dynamic_cache_usable_length() -> bool:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return False
    if hasattr(DynamicCache, "get_usable_length"):
        return False

    def get_usable_length(self: Any, new_seq_length: int | None = None, layer_idx: int = 0) -> int:
        del new_seq_length
        try:
            return int(self.get_seq_length(layer_idx))
        except TypeError:
            return int(self.get_seq_length())

    setattr(DynamicCache, "get_usable_length", get_usable_length)
    return True


def patch_phi4mm_num_logits_default(model_class: Any) -> bool:
    original_forward = getattr(model_class, "forward", None)
    if original_forward is None or getattr(original_forward, "_agri_vlm_num_logits_patch", False):
        return False

    def forward(self: Any, *args: Any, num_logits_to_keep: int | None = 0, **kwargs: Any) -> Any:
        if num_logits_to_keep is None:
            num_logits_to_keep = 0
        return original_forward(self, *args, num_logits_to_keep=num_logits_to_keep, **kwargs)

    setattr(forward, "_agri_vlm_num_logits_patch", True)
    setattr(model_class, "forward", forward)
    return True


def patch_phi4mm_quantized_lora_disable(model_class: Any) -> bool:
    original_set = getattr(model_class, "set_lora_adapter", None)
    original_unset = getattr(model_class, "unset_lora_adapter", None)
    if (
        original_set is None
        or original_unset is None
        or getattr(original_set, "_agri_vlm_quantized_lora_patch", False)
        or getattr(original_unset, "_agri_vlm_quantized_lora_patch", False)
    ):
        return False

    def iter_lora_layers(self: Any) -> Any:
        from peft.tuners.lora.layer import LoraLayer

        for module in self.modules():
            if isinstance(module, LoraLayer):
                yield module

    def ensure_unmerged(module: Any) -> None:
        if getattr(module, "merged", False):
            import warnings

            warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
            module.unmerge()

    def freeze_float_parameters(module: Any) -> None:
        for layer_name in getattr(module, "adapter_layer_names", []):
            layer = getattr(module, layer_name)
            parameters = layer.parameters() if hasattr(layer, "parameters") else []
            for parameter in parameters:
                if parameter.is_floating_point() or parameter.is_complex():
                    parameter.requires_grad_(False)

    def set_lora_adapter(self: Any, adapter_name: str) -> None:
        for module in iter_lora_layers(self):
            ensure_unmerged(module)
            freeze_float_parameters(module)
            module._active_adapter = [adapter_name]
            module._disable_adapters = False

    def unset_lora_adapter(self: Any) -> None:
        for module in iter_lora_layers(self):
            ensure_unmerged(module)
            freeze_float_parameters(module)
            module._disable_adapters = True

    setattr(set_lora_adapter, "_agri_vlm_quantized_lora_patch", True)
    setattr(unset_lora_adapter, "_agri_vlm_quantized_lora_patch", True)
    setattr(model_class, "set_lora_adapter", set_lora_adapter)
    setattr(model_class, "unset_lora_adapter", unset_lora_adapter)
    return True


def patch_phi4_reasoning_quantized_dtype_sync(model_class: Any) -> bool:
    """Skip Phi-4 reasoning vision's invalid post-load dtype cast for 4-bit models."""

    original_to = getattr(model_class, "to", None)
    if original_to is None or getattr(original_to, "_agri_vlm_quantized_dtype_sync_patch", False):
        return False

    def to(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original_to(self, *args, **kwargs)
        except ValueError as exc:
            message = str(exc)
            if "bitsandbytes model" in message and "dtype" in message:
                return self
            raise

    setattr(to, "_agri_vlm_quantized_dtype_sync_patch", True)
    setattr(model_class, "to", to)
    return True


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message or "cublas_status_alloc_failed" in message


class HuggingFaceVLMAdapter:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bf16",
        quantization: str = "none",
        attn_implementation: str | None = None,
        model_entry: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_entry = dict(model_entry or {})
        self.base_model_name_or_path = str(
            self.model_entry.get("base_model_name_or_path") or self.model_entry.get("model_name") or model_name
        )
        self.checkpoint_path = str(self.model_entry.get("checkpoint_path") or "")
        self.adapter_path = str(self.model_entry.get("adapter_path") or "")
        self.processor_name_or_path = str(
            self.model_entry.get("processor_name_or_path")
            or self.checkpoint_path
            or self.base_model_name_or_path
            or model_name
        )
        spec = MODEL_SPECS.get(
            model_name,
            MODEL_SPECS.get(
                self.base_model_name_or_path,
                AdapterSpec(
                model_name=model_name,
                loader_classes=["AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"],
                prompt_style="chat_prompt_images",
                trust_remote_code=bool(self.model_entry.get("trust_remote_code", True)),
                notes="Generic fallback adapter for image-text-to-text models.",
                ),
            ),
        )
        self.spec = copy.copy(spec)
        if self.model_entry.get("trust_remote_code") is not None:
            self.spec.trust_remote_code = bool(self.model_entry.get("trust_remote_code"))
        if self.model_entry.get("max_images") == 1 or self.model_entry.get("image_policy") == "first_image":
            self.spec.supports_multi_image = False
            self.spec.single_image_policy = "first_and_log"
        self.device = device
        self.dtype_name = dtype
        self.quantization = quantization
        self.attn_implementation = attn_implementation or self.spec.default_attn_implementation
        self.processor: Any = None
        self.model: Any = None
        self.torch_dtype: Any = None
        self.load_metadata: dict[str, Any] = {}

    def load_model(self) -> None:
        import torch
        import transformers

        self.torch_dtype = self._resolve_dtype(torch)
        processor_kwargs = dict(self.spec.processor_kwargs or {})
        if self.spec.trust_remote_code:
            processor_kwargs["trust_remote_code"] = True
        self.processor = transformers.AutoProcessor.from_pretrained(self.processor_name_or_path, **processor_kwargs)

        model_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
        }
        if self.spec.trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        if self.attn_implementation:
            config = transformers.AutoConfig.from_pretrained(
                self._model_load_source(),
                trust_remote_code=self.spec.trust_remote_code,
            )
            for attr_name in ("_attn_implementation", "_attn_implementation_internal"):
                if hasattr(config, attr_name):
                    setattr(config, attr_name, self.attn_implementation)
            model_kwargs["config"] = config
            model_kwargs["attn_implementation"] = self.attn_implementation
        if self.quantization == "4bit":
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = self._device_map()
            model_kwargs["dtype"] = self.torch_dtype
        elif self.device == "auto":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = self.torch_dtype
        elif self.device.startswith("cuda"):
            model_kwargs["device_map"] = self._device_map()
            model_kwargs["torch_dtype"] = self.torch_dtype
        else:
            model_kwargs["torch_dtype"] = self.torch_dtype

        errors: list[str] = []
        for class_name in self.spec.loader_classes:
            model_class = self._resolve_model_class(transformers, class_name)
            if model_class is None:
                errors.append("%s is not available in transformers" % class_name)
                continue
            try:
                self.model = self._from_pretrained_with_retries(model_class, model_kwargs)
                break
            except Exception as exc:
                errors.append("%s: %s: %s\n%s" % (class_name, type(exc).__name__, exc, traceback.format_exc()))
                if is_oom_error(exc):
                    raise
        if self.model is None:
            raise RuntimeError("Could not load %s. Attempts: %s" % (self.model_name, errors))

        if "device_map" not in model_kwargs and self.device != "cpu" and hasattr(self.model, "to"):
            self.model.to(self.device)
        self.model.eval()
        model_config = getattr(self.model, "config", None)
        self.load_metadata = {
            "model_name": self.model_name,
            "model_key": self.model_entry.get("model_key"),
            "checkpoint_type": self.model_entry.get("checkpoint_type"),
            "adapter_type": self.model_entry.get("adapter_type"),
            "base_model_name_or_path": self.base_model_name_or_path,
            "checkpoint_path": self.checkpoint_path,
            "adapter_path": self.adapter_path,
            "processor_name_or_path": self.processor_name_or_path,
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "dtype": self.dtype_name,
            "torch_dtype": str(self.torch_dtype),
            "quantization": self.quantization,
            "device": self.device,
            "attn_implementation": self.attn_implementation,
            "model_commit_hash": getattr(model_config, "_commit_hash", None),
            "memory_after_load": maybe_cuda_memory(self.device),
        }

    def _resolve_model_class(self, transformers: Any, class_name: str) -> Any:
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            return model_class
        if class_name == "Phi4MMForCausalLM":
            return self._load_phi4mm_causal_lm_class()
        if class_name == "Phi4ForCausalLMV":
            return self._load_phi4_reasoning_causal_lm_class()
        return None

    def _load_phi4mm_causal_lm_class(self) -> Any:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_class = get_class_from_dynamic_module(
            "modeling_phi4mm.Phi4MMForCausalLM",
            self._model_load_source(),
            trust_remote_code=True,
        )
        patch_phi4mm_base_generation_hook(model_class)
        patch_phi4mm_num_logits_default(model_class)
        patch_phi4mm_quantized_lora_disable(model_class)
        patch_dynamic_cache_usable_length()
        return model_class

    def _load_phi4_reasoning_causal_lm_class(self) -> Any:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_class = get_class_from_dynamic_module(
            "modeling_phi4_visionr.Phi4ForCausalLMV",
            self._model_load_source(),
            trust_remote_code=True,
        )
        patch_phi4_reasoning_quantized_dtype_sync(model_class)
        patch_dynamic_cache_usable_length()
        return model_class

    def _from_pretrained_with_retries(self, model_class: Any, model_kwargs: dict[str, Any]) -> Any:
        kwargs = dict(model_kwargs)
        source = self._model_load_source()
        try:
            model = model_class.from_pretrained(source, **kwargs)
            return self._maybe_attach_adapter(model)
        except TypeError as first:
            retry_kwargs = dict(kwargs)
            if "torch_dtype" in retry_kwargs:
                retry_kwargs["dtype"] = retry_kwargs.pop("torch_dtype")
            try:
                model = model_class.from_pretrained(source, **retry_kwargs)
                return self._maybe_attach_adapter(model)
            except TypeError:
                reduced_kwargs = dict(retry_kwargs)
                reduced_kwargs.pop("attn_implementation", None)
                model = model_class.from_pretrained(source, **reduced_kwargs)
                return self._maybe_attach_adapter(model)
            except Exception:
                raise first

    def _model_load_source(self) -> str:
        if self.adapter_path:
            return self.base_model_name_or_path
        return self.checkpoint_path or self.base_model_name_or_path or self.model_name

    def _maybe_attach_adapter(self, model: Any) -> Any:
        if not self.adapter_path:
            return model
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("peft is required to load adapter_path=%s: %s" % (self.adapter_path, exc)) from exc
        kwargs: dict[str, Any] = {"is_trainable": False}
        if self.quantization == "4bit":
            kwargs["autocast_adapter_dtype"] = False
        try:
            return PeftModel.from_pretrained(model, self.adapter_path, **kwargs)
        except TypeError:
            kwargs.pop("autocast_adapter_dtype", None)
            return PeftModel.from_pretrained(model, self.adapter_path, **kwargs)

    def _resolve_dtype(self, torch: Any) -> Any:
        requested = (self.dtype_name or "bf16").lower()
        if requested in {"auto", "none"}:
            return "auto"
        if requested in {"bf16", "bfloat16"}:
            if self.device.startswith("cuda") and hasattr(torch.cuda, "is_bf16_supported"):
                if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                    return torch.float16
            return torch.bfloat16
        if requested in {"fp16", "float16", "half"}:
            return torch.float16
        if requested in {"fp32", "float32"}:
            return torch.float32
        raise ValueError("Unsupported dtype: %s" % self.dtype_name)

    def _device_map(self) -> dict[str, int | str] | str:
        if self.device == "auto":
            return "auto"
        if self.device.startswith("cuda:"):
            return {"": int(self.device.split(":", 1)[1])}
        if self.device == "cuda":
            return {"": 0}
        return {"": self.device}

    def build_prompt(self, sample: BenchmarkSample) -> str:
        return sample.prompt

    def preprocess(self, sample: BenchmarkSample) -> dict[str, Any]:
        from PIL import Image

        image_paths = self._select_image_paths(sample)
        pil_images = []
        for image_path in image_paths:
            resolved = Path(image_path)
            if not resolved.is_absolute():
                resolved = REPO_ROOT / resolved
            if not resolved.exists():
                raise FileNotFoundError("Missing image for sample %s: %s" % (sample.sample_id, image_path))
            pil_images.append(Image.open(resolved).convert("RGB"))
        prompt = self.build_prompt(sample)
        if self.spec.prompt_style == "paligemma":
            inputs = self.processor(text=prompt, images=pil_images[0], return_tensors="pt")
        elif self.spec.prompt_style == "phi4":
            phi_prompt = build_plain_prompt(sample.row, image_count=len(pil_images), label_space=sample.label_space)
            inputs = self.processor(text=phi_prompt, images=pil_images, return_tensors="pt")
            prompt = phi_prompt
        elif self.spec.prompt_style == "qwen_vl":
            inputs, prompt = self._preprocess_qwen(sample, image_paths, pil_images)
        elif self.spec.prompt_style == "chat_tokenized_paths":
            messages = self._build_path_chat_messages(sample, image_paths)
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            messages = build_chat_messages(
                sample.row,
                image_paths=image_paths,
                label_space=sample.label_space,
                include_image_paths=False,
            )
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=[pil_images], padding=True, return_tensors="pt")
        inputs = self._move_inputs(inputs)
        return {
            "inputs": inputs,
            "prompt": prompt,
            "images_used": image_paths,
            "image_policy": self._image_policy_note(sample, image_paths),
        }

    def _build_path_chat_messages(
        self,
        sample: BenchmarkSample,
        image_paths: list[str],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for image_path in image_paths:
            resolved = Path(image_path)
            if not resolved.is_absolute():
                resolved = REPO_ROOT / resolved
            content.append({"type": "image", "image": str(resolved)})
        content.append({"type": "text", "text": sample.prompt})
        return [{"role": "user", "content": content}]

    def _preprocess_qwen(
        self,
        sample: BenchmarkSample,
        image_paths: list[str],
        pil_images: list[Any],
    ) -> tuple[Any, str]:
        messages = build_chat_messages(
            sample.row,
            image_paths=image_paths,
            label_space=sample.label_space,
            include_image_paths=True,
        )
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            inputs = self.processor(text=[prompt], images=[pil_images], padding=True, return_tensors="pt")
        return inputs, prompt

    def _move_inputs(self, inputs: Any) -> Any:
        import torch

        target_device = self._generation_device()
        moved = {}
        if hasattr(inputs, "items"):
            items = inputs.items()
        elif hasattr(inputs, "to"):
            try:
                return inputs.to(target_device)
            except TypeError:
                return inputs.to(device=target_device)
        else:
            items = dict(inputs).items()
        for key, value in items:
            if hasattr(value, "to"):
                if torch.is_tensor(value) and torch.is_floating_point(value) and self.torch_dtype != "auto":
                    moved[key] = value.to(device=target_device, dtype=self.torch_dtype)
                else:
                    moved[key] = value.to(target_device)
            else:
                moved[key] = value
        return moved

    def _generation_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except Exception:
            return getattr(self.model, "device", self.device)

    def _select_image_paths(self, sample: BenchmarkSample) -> list[str]:
        image_paths = sample.image_paths
        if not image_paths:
            raise ValueError("Sample %s has no image paths" % sample.sample_id)
        if len(image_paths) > 1 and not self.spec.supports_multi_image:
            if self.spec.single_image_policy == "first_and_log":
                return image_paths[:1]
            raise ValueError("%s does not support multi-image samples" % self.model_name)
        return image_paths

    def _image_policy_note(self, sample: BenchmarkSample, image_paths: list[str]) -> str:
        total = len(sample.image_paths)
        used = len(image_paths)
        if total == used:
            return "all_images"
        return "used_%s_of_%s_images_%s" % (used, total, self.spec.single_image_policy)

    def generate(self, sample: BenchmarkSample, generation_config: dict[str, Any]) -> dict[str, Any]:
        import torch

        prepared = self.preprocess(sample)
        inputs = prepared["inputs"]
        generate_kwargs = {
            "max_new_tokens": int(generation_config.get("max_new_tokens", 128)),
            "do_sample": bool(generation_config.get("do_sample", False)),
            "num_beams": int(generation_config.get("num_beams", 1)),
            "use_cache": True,
        }
        min_new_tokens = int(generation_config.get("min_new_tokens") or 0)
        if min_new_tokens:
            generate_kwargs["min_new_tokens"] = min_new_tokens
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = float(generation_config.get("temperature", 1.0))
            generate_kwargs["top_p"] = float(generation_config.get("top_p", 1.0))
        raw_output = self._generate_text(inputs, prepared["prompt"], generate_kwargs, torch)
        postprocessed = self.postprocess(self._strip_prompt_echo(raw_output, prepared["prompt"]))
        format_retry_used = False
        raw_output_before_retry = None
        if self._needs_format_retry(sample, postprocessed):
            retry_kwargs = dict(generate_kwargs)
            retry_kwargs["min_new_tokens"] = min(
                int(retry_kwargs.get("max_new_tokens", 128)),
                max(int(retry_kwargs.get("min_new_tokens") or 0), 8),
            )
            raw_output_before_retry = postprocessed
            retry_raw_output = self._generate_text(inputs, prepared["prompt"], retry_kwargs, torch)
            retry_postprocessed = self.postprocess(self._strip_prompt_echo(retry_raw_output, prepared["prompt"]))
            if retry_postprocessed.strip():
                postprocessed = retry_postprocessed
                format_retry_used = True
        return {
            "raw_output": postprocessed,
            "prompt": prepared["prompt"],
            "images_used": prepared["images_used"],
            "image_policy": prepared["image_policy"],
            "format_retry_used": format_retry_used,
            "raw_output_before_format_retry": raw_output_before_retry,
        }

    def _generate_text(self, inputs: Any, prompt: str, generate_kwargs: dict[str, Any], torch: Any) -> str:
        with torch.no_grad(), self._generation_autocast_context(torch):
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        input_length = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        decoded_ids = output_ids
        sliced = False
        if input_length and getattr(output_ids, "shape", [0, 0])[-1] > input_length:
            decoded_ids = output_ids[:, input_length:]
            sliced = True
        raw_output = self._decode(decoded_ids)
        if not raw_output.strip() and sliced:
            raw_output = self._strip_prompt_echo(self._decode(output_ids), prompt)
        return raw_output

    def _needs_format_retry(self, sample: BenchmarkSample, raw_output: str) -> bool:
        task_type = sample.task_type
        verifier_mode = sample.verifier_mode
        if task_type not in {"classification", "label_diagnosis", "vqa", "clarify_or_respond"} and verifier_mode not in {
            "label",
            "exact_match",
            "synonym",
            "clarify",
        }:
            return False
        text = (raw_output or "").strip()
        if not text:
            return True
        normalized = normalize_text(text)
        if normalized in {"answer", "answer:", "final answer", "final answer:", "decision", "decision:"}:
            return True
        answer, answer_status = extract_answer_field(text)
        if answer_status == "failed" and re.search(r"(?im)^\s*(?:final\s+)?answer\s*:\s*$", text):
            return True
        if (task_type == "clarify_or_respond" or verifier_mode == "clarify") and "decision:" not in text.lower():
            return True
        return False

    def _generation_autocast_context(self, torch: Any) -> Any:
        if self.torch_dtype not in (getattr(torch, "bfloat16", None), getattr(torch, "float16", None)):
            return nullcontext()
        device = self._generation_device()
        device_type = getattr(device, "type", None)
        if device_type is None and str(device).startswith("cuda"):
            device_type = "cuda"
        if device_type == "cuda" and hasattr(torch, "autocast"):
            return torch.autocast(device_type="cuda", dtype=self.torch_dtype)
        return nullcontext()

    def _decode(self, output_ids: Any) -> str:
        decoder = getattr(self.processor, "batch_decode", None)
        if decoder is not None:
            return decoder(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _strip_prompt_echo(self, raw_output: str, prompt: str) -> str:
        output = (raw_output or "").strip()
        prompt_text = (prompt or "").strip()
        if prompt_text and output.startswith(prompt_text):
            return output[len(prompt_text) :].strip()
        return raw_output

    def postprocess(self, raw_output: str) -> str:
        return (raw_output or "").strip()

    def unload_model(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def load_model(model_name: str, device: str, dtype: str, quantization: str | None = None) -> HuggingFaceVLMAdapter:
    adapter = HuggingFaceVLMAdapter(
        model_name=model_name,
        device=device,
        dtype=dtype,
        quantization=quantization or "none",
    )
    adapter.load_model()
    return adapter
