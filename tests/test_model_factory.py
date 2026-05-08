from types import SimpleNamespace

from agri_vlm.modeling.model_factory import build_model_init_kwargs, _prepare_phi4_multimodal_vision_only
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
