"""Inference-only Hugging Face VLM adapters.

All heavy ML imports are lazy so split creation and metric aggregation can run
in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any

from dataset_adapter import BenchmarkSample, build_chat_messages, build_plain_prompt
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
        loader_classes=["AutoModelForCausalLM"],
        prompt_style="phi4",
        trust_remote_code=True,
        notes="Uses Phi-4 multimodal image placeholder prompt format.",
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
}


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
    ) -> None:
        self.model_name = model_name
        self.spec = MODEL_SPECS.get(
            model_name,
            AdapterSpec(
                model_name=model_name,
                loader_classes=["AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"],
                prompt_style="chat_prompt_images",
                trust_remote_code=True,
                notes="Generic fallback adapter for image-text-to-text models.",
            ),
        )
        self.device = device
        self.dtype_name = dtype
        self.quantization = quantization
        self.attn_implementation = attn_implementation
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
        self.processor = transformers.AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)

        model_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
        }
        if self.spec.trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        if self.attn_implementation:
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
        elif self.device.startswith("cuda"):
            model_kwargs["device_map"] = self._device_map()
            model_kwargs["torch_dtype"] = self.torch_dtype
        else:
            model_kwargs["torch_dtype"] = self.torch_dtype

        errors: list[str] = []
        for class_name in self.spec.loader_classes:
            model_class = getattr(transformers, class_name, None)
            if model_class is None:
                errors.append("%s is not available in transformers" % class_name)
                continue
            try:
                self.model = self._from_pretrained_with_retries(model_class, model_kwargs)
                break
            except Exception as exc:
                errors.append("%s: %s: %s" % (class_name, type(exc).__name__, exc))
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

    def _from_pretrained_with_retries(self, model_class: Any, model_kwargs: dict[str, Any]) -> Any:
        kwargs = dict(model_kwargs)
        try:
            return model_class.from_pretrained(self.model_name, **kwargs)
        except TypeError as first:
            retry_kwargs = dict(kwargs)
            if "torch_dtype" in retry_kwargs:
                retry_kwargs["dtype"] = retry_kwargs.pop("torch_dtype")
            try:
                return model_class.from_pretrained(self.model_name, **retry_kwargs)
            except TypeError:
                reduced_kwargs = dict(retry_kwargs)
                reduced_kwargs.pop("attn_implementation", None)
                return model_class.from_pretrained(self.model_name, **reduced_kwargs)
            except Exception:
                raise first

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
            phi_prompt = build_plain_prompt(sample.row, image_count=len(pil_images))
            inputs = self.processor(text=phi_prompt, images=pil_images, return_tensors="pt")
            prompt = phi_prompt
        elif self.spec.prompt_style == "qwen_vl":
            inputs, prompt = self._preprocess_qwen(sample, image_paths, pil_images)
        elif self.spec.prompt_style == "chat_tokenized_paths":
            messages = build_chat_messages(sample.row, image_paths=image_paths, include_image_paths=True, include_system=False)
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            messages = build_chat_messages(sample.row, image_paths=image_paths, include_image_paths=False)
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=[pil_images], padding=True, return_tensors="pt")
        inputs = self._move_inputs(inputs)
        return {
            "inputs": inputs,
            "prompt": prompt,
            "images_used": image_paths,
            "image_policy": self._image_policy_note(sample, image_paths),
        }

    def _preprocess_qwen(
        self,
        sample: BenchmarkSample,
        image_paths: list[str],
        pil_images: list[Any],
    ) -> tuple[Any, str]:
        messages = build_chat_messages(sample.row, image_paths=image_paths, include_image_paths=True)
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
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = float(generation_config.get("temperature", 1.0))
            generate_kwargs["top_p"] = float(generation_config.get("top_p", 1.0))
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        input_length = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        if input_length and getattr(output_ids, "shape", [0, 0])[-1] > input_length:
            decoded_ids = output_ids[:, input_length:]
        else:
            decoded_ids = output_ids
        raw_output = self._decode(decoded_ids)
        if not raw_output.strip() and decoded_ids is not output_ids:
            raw_output = self._decode(output_ids)
        return {
            "raw_output": self.postprocess(raw_output),
            "prompt": prepared["prompt"],
            "images_used": prepared["images_used"],
            "image_policy": prepared["image_policy"],
        }

    def _decode(self, output_ids: Any) -> str:
        decoder = getattr(self.processor, "batch_decode", None)
        if decoder is not None:
            return decoder(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
