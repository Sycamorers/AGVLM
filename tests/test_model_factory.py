from types import SimpleNamespace

from agri_vlm.modeling import model_factory
from agri_vlm.modeling.model_factory import (
    _patch_phi4_vision_projector_for_zero3,
    _prepare_phi4_multimodal_vision_only,
    build_model_init_kwargs,
    load_model,
)
from agri_vlm.utils.distributed import DistributedContext


def test_bf16_distributed_model_init_omits_device_map_and_quantizer() -> None:
    model_config = SimpleNamespace(
        attn_implementation="sdpa",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        device_map="auto",
        distributed_device_map="none",
        load_in_4bit=False,
        low_cpu_mem_usage=True,
        torch_dtype="bfloat16",
        trust_remote_code=False,
    )
    distributed_context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=16,
        device="cuda:0",
        backend="nccl",
    )

    kwargs = build_model_init_kwargs(
        model_config,
        distributed_context=distributed_context,
    )

    assert "device_map" not in kwargs
    assert "quantization_config" not in kwargs
    assert kwargs["attn_implementation"] == "sdpa"


def test_prepare_phi4_vision_only_removes_speech_adapter_and_trains_image_embedding() -> None:
    image_parameter = SimpleNamespace(requires_grad=False)

    class FakeImageEmbed:
        def parameters(self):
            return [image_parameter]

    class FakeModule:
        def __init__(self):
            self.lora_A = {"vision": object(), "speech": object()}
            self.lora_B = {"vision": object(), "speech": object()}

    lora_module = FakeModule()

    class FakeModel:
        def __init__(self):
            self.active_adapter = None
            self.model = SimpleNamespace(
                embed_tokens_extend=SimpleNamespace(
                    audio_embed=object(),
                    image_embed=FakeImageEmbed(),
                )
            )

        def set_lora_adapter(self, adapter_name):
            self.active_adapter = adapter_name

        def modules(self):
            return [self, lora_module]

    model = FakeModel()
    model_config = SimpleNamespace(phi4_vision_only=True, phi4_train_image_embedding=True)

    _prepare_phi4_multimodal_vision_only(model, model_config)

    assert model.active_adapter == "vision"
    assert "speech" not in lora_module.lora_A
    assert "speech" not in lora_module.lora_B
    assert not hasattr(model.model.embed_tokens_extend, "audio_embed")
    assert image_parameter.requires_grad is True


def test_phi4_projector_patch_vectorizes_image_feature_list() -> None:
    import torch

    calls = []

    class FakeProjector:
        def __call__(self, features):
            calls.append(tuple(features.shape))
            return features + 1

    class FakeVisionTower:
        def __call__(self, _images):
            return [
                torch.zeros(2, 3),
                torch.ones(4, 3),
                torch.full((1, 3), 2.0),
            ]

    class FakeInnerModel:
        def __init__(self):
            self.mm_projector = FakeProjector()
            self.vision_tower = FakeVisionTower()

        def get_vision_tower(self):
            return self.vision_tower

    class FakeModel:
        def __init__(self):
            self.inner_model = FakeInnerModel()

        def get_model(self):
            return self.inner_model

        def encode_images(self, _images):
            raise AssertionError("unpatched encode_images should be replaced")

    model = FakeModel()
    model_config = SimpleNamespace(
        name="phi4_reasoning_vision_15b",
        model_name_or_path="microsoft/Phi-4-reasoning-vision-15B",
    )

    _patch_phi4_vision_projector_for_zero3(model, model_config)
    outputs = model.encode_images("images")

    assert calls == [(7, 3)]
    assert [tuple(output.shape) for output in outputs] == [(2, 3), (4, 3), (1, 3)]
    assert torch.equal(outputs[0], torch.ones(2, 3))
    assert torch.equal(outputs[1], torch.full((4, 3), 2.0))
    assert torch.equal(outputs[2], torch.full((1, 3), 3.0))


def test_load_model_uses_non_reentrant_gradient_checkpointing(monkeypatch) -> None:
    class FakeModel:
        def __init__(self):
            self.gradient_checkpointing_kwargs = None
            self.config = SimpleNamespace(use_cache=True)

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs=None):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    fake_model = FakeModel()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, _model_name_or_path, **_kwargs):
            return fake_model

    monkeypatch.setattr(model_factory, "_resolve_model_class", lambda _model_config: FakeModelClass)

    model_config = SimpleNamespace(
        attn_implementation="sdpa",
        attn_implementation_kwarg="attn_implementation",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        device_map=None,
        distributed_device_map="none",
        gradient_checkpointing=True,
        load_in_4bit=False,
        low_cpu_mem_usage=True,
        phi4_vision_only=False,
        torch_dtype="bfloat16",
        trust_remote_code=False,
        use_cache=False,
    )

    model = load_model("fake-model", model_config=model_config)

    assert model is fake_model
    assert fake_model.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert fake_model.config.use_cache is False
