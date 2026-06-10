from types import ModuleType, SimpleNamespace
import sys


ROOT_MODULE = "fake_phi4mm_remote"


def test_patch_phi4mm_base_generation_hook_adds_inner_model_method():
    sys.path.insert(0, "benchmarks/vlm_baselines")
    from model_adapters import patch_phi4mm_base_generation_hook

    module = ModuleType(ROOT_MODULE)

    class Phi4MMModel:
        pass

    module.Phi4MMModel = Phi4MMModel
    sys.modules[ROOT_MODULE] = module
    model_class = SimpleNamespace(__module__=ROOT_MODULE)

    try:
        patched = patch_phi4mm_base_generation_hook(model_class)
        assert patched is True
        assert hasattr(Phi4MMModel, "prepare_inputs_for_generation")
        inputs = Phi4MMModel().prepare_inputs_for_generation("ids", attention_mask="mask", unused=None)
        assert inputs == {"input_ids": "ids", "attention_mask": "mask"}
        assert patch_phi4mm_base_generation_hook(model_class) is False
    finally:
        sys.modules.pop(ROOT_MODULE, None)


def test_patch_dynamic_cache_usable_length_adds_removed_cache_alias(monkeypatch):
    sys.path.insert(0, "benchmarks/vlm_baselines")
    from model_adapters import patch_dynamic_cache_usable_length

    transformers_module = ModuleType("transformers")
    cache_utils_module = ModuleType("transformers.cache_utils")

    class DynamicCache:
        def get_seq_length(self, layer_idx=0):
            return layer_idx + 7

    cache_utils_module.DynamicCache = DynamicCache
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", cache_utils_module)

    assert patch_dynamic_cache_usable_length() is True
    assert DynamicCache().get_usable_length(layer_idx=3) == 10
    assert patch_dynamic_cache_usable_length() is False


def test_patch_phi4mm_num_logits_default_converts_none_to_zero():
    sys.path.insert(0, "benchmarks/vlm_baselines")
    from model_adapters import patch_phi4mm_num_logits_default

    class FakePhi4MMForCausalLM:
        def forward(self, *, num_logits_to_keep=0):
            return num_logits_to_keep

    assert patch_phi4mm_num_logits_default(FakePhi4MMForCausalLM) is True
    assert FakePhi4MMForCausalLM().forward(num_logits_to_keep=None) == 0
    assert FakePhi4MMForCausalLM().forward(num_logits_to_keep=4) == 4
    assert patch_phi4mm_num_logits_default(FakePhi4MMForCausalLM) is False


def test_patch_phi4mm_quantized_lora_disable_skips_integer_parameters(monkeypatch):
    sys.path.insert(0, "benchmarks/vlm_baselines")
    from model_adapters import patch_phi4mm_quantized_lora_disable

    peft_module = ModuleType("peft")
    tuners_module = ModuleType("peft.tuners")
    lora_module = ModuleType("peft.tuners.lora")
    layer_module = ModuleType("peft.tuners.lora.layer")

    class FakeParameter:
        def __init__(self, floating):
            self.floating = floating
            self.requires_grad_disabled = False

        def is_floating_point(self):
            return self.floating

        def is_complex(self):
            return False

        def requires_grad_(self, value):
            if not self.floating:
                raise RuntimeError("only Tensors of floating point dtype can require gradients")
            self.requires_grad_disabled = value is False

    class FakeLayer:
        def __init__(self):
            self.float_param = FakeParameter(True)
            self.int_param = FakeParameter(False)

        def parameters(self):
            return [self.float_param, self.int_param]

    class LoraLayer:
        adapter_layer_names = ["default_layer"]

        def __init__(self):
            self.default_layer = FakeLayer()
            self.merged = False
            self._disable_adapters = False

    layer_module.LoraLayer = LoraLayer
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "peft.tuners", tuners_module)
    monkeypatch.setitem(sys.modules, "peft.tuners.lora", lora_module)
    monkeypatch.setitem(sys.modules, "peft.tuners.lora.layer", layer_module)

    class FakePhi4MMForCausalLM:
        def __init__(self):
            self.lora = LoraLayer()

        def modules(self):
            return [self, self.lora]

        def set_lora_adapter(self, adapter_name):
            raise AssertionError("original should be replaced")

        def unset_lora_adapter(self):
            raise AssertionError("original should be replaced")

    assert patch_phi4mm_quantized_lora_disable(FakePhi4MMForCausalLM) is True
    model = FakePhi4MMForCausalLM()
    model.set_lora_adapter("vision")
    assert model.lora._active_adapter == ["vision"]
    assert model.lora._disable_adapters is False
    model.unset_lora_adapter()
    assert model.lora.default_layer.float_param.requires_grad_disabled is True
    assert model.lora._disable_adapters is True
    assert patch_phi4mm_quantized_lora_disable(FakePhi4MMForCausalLM) is False


def test_quantized_adapter_attach_disables_autocast(monkeypatch):
    sys.path.insert(0, "benchmarks/vlm_baselines")
    from model_adapters import HuggingFaceVLMAdapter

    calls = []
    peft_module = ModuleType("peft")

    class PeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path, **kwargs):
            calls.append({"model": model, "adapter_path": adapter_path, "kwargs": kwargs})
            return {"wrapped": model}

    peft_module.PeftModel = PeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    adapter = HuggingFaceVLMAdapter(
        "agvlm_phi4_sft_completed",
        quantization="4bit",
        model_entry={
            "base_model_name_or_path": "microsoft/Phi-4-reasoning-vision-15B",
            "adapter_path": "/tmp/adapter",
        },
    )

    assert adapter._maybe_attach_adapter("model") == {"wrapped": "model"}
    assert calls[0]["kwargs"]["is_trainable"] is False
    assert calls[0]["kwargs"]["autocast_adapter_dtype"] is False


def test_patch_phi4_reasoning_quantized_dtype_sync_ignores_bnb_dtype_cast():
    sys.path.insert(0, "benchmarks/vlm_baselines")
    from model_adapters import patch_phi4_reasoning_quantized_dtype_sync

    class FakePhi4ReasoningForCausalLM:
        def __init__(self):
            self.to_calls = 0

        def to(self, *args, **kwargs):
            self.to_calls += 1
            raise ValueError("You cannot cast a bitsandbytes model in a new `dtype`.")

    assert patch_phi4_reasoning_quantized_dtype_sync(FakePhi4ReasoningForCausalLM) is True
    model = FakePhi4ReasoningForCausalLM()
    assert model.to("bf16") is model
    assert model.to_calls == 1
    assert patch_phi4_reasoning_quantized_dtype_sync(FakePhi4ReasoningForCausalLM) is False


def test_quantized_load_passes_dtype_to_from_pretrained(monkeypatch):
    sys.path.insert(0, "benchmarks/vlm_baselines")
    import model_adapters
    from model_adapters import HuggingFaceVLMAdapter

    captured = {}

    class FakeTorch:
        bfloat16 = "bf16"
        float16 = "fp16"
        float32 = "fp32"

        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_bf16_supported():
                return True

    class FakeModel:
        config = SimpleNamespace()

        def eval(self):
            return None

    class FakeModelClass:
        @staticmethod
        def from_pretrained(source, **kwargs):
            captured["source"] = source
            captured["kwargs"] = kwargs
            return FakeModel()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class FakeBnbConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeConfig:
        _attn_implementation = None

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeConfig()

    fake_transformers = SimpleNamespace(
        AutoConfig=FakeConfig,
        AutoProcessor=FakeProcessor,
        BitsAndBytesConfig=FakeBnbConfig,
        AutoModelForCausalLM=FakeModelClass,
    )
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(model_adapters, "maybe_cuda_memory", lambda device: {})

    adapter = HuggingFaceVLMAdapter(
        "microsoft/Phi-4-reasoning-vision-15B",
        quantization="4bit",
        model_entry={"base_model_name_or_path": "microsoft/Phi-4-reasoning-vision-15B"},
    )
    monkeypatch.setattr(adapter, "_load_phi4_reasoning_causal_lm_class", lambda: FakeModelClass)
    adapter.load_model()

    assert captured["kwargs"]["dtype"] == "bf16"
    assert "torch_dtype" not in captured["kwargs"]
    assert captured["kwargs"]["quantization_config"].kwargs["bnb_4bit_compute_dtype"] == "bf16"


def test_generation_autocast_context_uses_cuda_for_bf16(monkeypatch):
    sys.path.insert(0, "benchmarks/vlm_baselines")
    from model_adapters import HuggingFaceVLMAdapter

    calls = []

    class FakeAutocast:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")

    class FakeTorch:
        bfloat16 = "bf16"
        float16 = "fp16"

        @staticmethod
        def autocast(**kwargs):
            calls.append(kwargs)
            return FakeAutocast()

    adapter = HuggingFaceVLMAdapter("fake/model")
    adapter.torch_dtype = "bf16"
    monkeypatch.setattr(adapter, "_generation_device", lambda: SimpleNamespace(type="cuda"))

    with adapter._generation_autocast_context(FakeTorch):
        calls.append("body")

    assert calls == [{"device_type": "cuda", "dtype": "bf16"}, "enter", "body", "exit"]


def test_strip_prompt_echo_removes_decoded_prompt():
    sys.path.insert(0, "benchmarks/vlm_baselines")
    from model_adapters import HuggingFaceVLMAdapter

    adapter = HuggingFaceVLMAdapter("fake/model")
    prompt = "<|user|>Identify the pest<|end|><|assistant|>"

    assert adapter._strip_prompt_echo(prompt, prompt) == ""
    assert adapter._strip_prompt_echo(prompt + "Answer: aphids", prompt) == "Answer: aphids"
    assert adapter._strip_prompt_echo("Answer: aphids", prompt) == "Answer: aphids"


def test_short_answer_format_retry_detects_empty_answer_prefix():
    sys.path.insert(0, "benchmarks/vlm_baselines")
    from model_adapters import HuggingFaceVLMAdapter

    adapter = HuggingFaceVLMAdapter("fake/model")
    sample = SimpleNamespace(task_type="vqa", verifier_mode="exact_match")

    assert adapter._needs_format_retry(sample, "Answer:") is True
    assert adapter._needs_format_retry(sample, "Answer: tomato early blight") is False


def test_constrained_label_token_options_allow_prefixes_and_eos():
    sys.path.insert(0, "benchmarks/vlm_baselines")
    from model_adapters import _next_allowed_label_tokens

    label_token_sequences = [[10], [10, 20], [30, 40]]
    eos_token_ids = [2]

    assert _next_allowed_label_tokens([], label_token_sequences, eos_token_ids) == [10, 30]
    assert _next_allowed_label_tokens([10], label_token_sequences, eos_token_ids) == [2, 20]
    assert _next_allowed_label_tokens([10, 20], label_token_sequences, eos_token_ids) == [2]
    assert _next_allowed_label_tokens([999], label_token_sequences, eos_token_ids) == [2]
