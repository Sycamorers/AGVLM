import sys
from types import SimpleNamespace

from agri_vlm.modeling.processor_factory import load_processor


def test_load_processor_forwards_pixel_limits(monkeypatch) -> None:
    calls = {}

    class DummyAutoProcessor:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["name"] = name
            calls["kwargs"] = kwargs
            tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
            return SimpleNamespace(tokenizer=tokenizer)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=DummyAutoProcessor))
    model_config = SimpleNamespace(
        model_name_or_path="base-model",
        processor_name_or_path=None,
        trust_remote_code=False,
        min_pixels=3136,
        max_pixels=1003520,
    )

    processor = load_processor(model_config)

    assert calls["name"] == "base-model"
    assert calls["kwargs"] == {
        "trust_remote_code": False,
        "min_pixels": 3136,
        "max_pixels": 1003520,
    }
    assert processor.tokenizer.pad_token == "<eos>"
