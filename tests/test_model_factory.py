from types import SimpleNamespace

from agri_vlm.modeling.model_factory import build_model_init_kwargs
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
