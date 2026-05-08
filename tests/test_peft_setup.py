import sys
import types
from types import SimpleNamespace

from agri_vlm.modeling.peft_setup import maybe_wrap_with_peft


def test_peft_setup_enables_input_grads_for_gradient_checkpointing(monkeypatch) -> None:
    calls = {"get_peft_model": False, "prepare_kbit": False}

    class FakeModel:
        is_loaded_in_4bit = False

        def __init__(self):
            self.input_grads_enabled = False

        def enable_input_require_grads(self):
            self.input_grads_enabled = True

    def fake_get_peft_model(model, _config):
        calls["get_peft_model"] = True
        return model

    def fake_prepare_model_for_kbit_training(model, **_kwargs):
        calls["prepare_kbit"] = True
        return model

    fake_peft = types.SimpleNamespace(
        LoraConfig=lambda **kwargs: kwargs,
        TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
        get_peft_model=fake_get_peft_model,
        prepare_model_for_kbit_training=fake_prepare_model_for_kbit_training,
    )
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    train_config = SimpleNamespace(
        gradient_checkpointing=True,
        use_peft=True,
        lora=SimpleNamespace(
            r=8,
            alpha=16,
            dropout=0.0,
            bias="none",
            target_modules=["qkv_proj"],
        ),
    )
    model = FakeModel()

    wrapped = maybe_wrap_with_peft(model, train_config=train_config)

    assert wrapped is model
    assert model.input_grads_enabled is True
    assert calls == {"get_peft_model": True, "prepare_kbit": False}
