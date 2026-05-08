import pytest


def _sample() -> dict:
    return {
        "sample_id": "sample-1",
        "source_dataset": "plantvillage",
        "task_type": "classification",
        "split": "train",
        "images": ["image.png"],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Agricultural RGB consultation only."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image.png"},
                    {"type": "text", "text": "Identify the crop issue."},
                ],
            },
        ],
        "target": {"answer_text": "apple scab"},
    }


def test_sft_collator_masks_prompt_and_padding_tokens(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    import agri_vlm.training.collators as collators

    class FakeTokenizer:
        pad_token_id = 0
        padding_side = "right"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            if add_generation_prompt:
                assert messages[-1]["role"] == "user"
                return "prompt"
            assert messages[-1]["role"] == "assistant"
            return "full"

        def __call__(self, *, text, images, padding, return_tensors):
            assert images == [["opened:image.png"]]
            assert padding is True
            assert return_tensors == "pt"
            if text == ["prompt"]:
                return {
                    "input_ids": torch.tensor([[10, 11, 12, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                }
            if text == ["full"]:
                return {
                    "input_ids": torch.tensor([[10, 11, 12, 20, 21, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
                    "token_type_ids": torch.tensor([[0, 0, 0, 0, 0, 0]]),
                }
            raise AssertionError("unexpected text batch: %r" % (text,))

    monkeypatch.setattr(collators, "open_image", lambda path: "opened:%s" % path)

    batch = collators.VisionLanguageChatCollator(processor=FakeProcessor())([_sample()])

    assert "token_type_ids" not in batch
    assert batch["labels"].tolist() == [[-100, -100, -100, 20, 21, -100]]


def test_sft_collator_masks_prompt_with_left_padding(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    import agri_vlm.training.collators as collators

    class FakeTokenizer:
        pad_token_id = 0
        padding_side = "left"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return "prompt" if add_generation_prompt else "full"

        def __call__(self, *, text, images, padding, return_tensors):
            if text == ["prompt"]:
                return {
                    "input_ids": torch.tensor([[0, 10, 11, 12]]),
                    "attention_mask": torch.tensor([[0, 1, 1, 1]]),
                }
            return {
                "input_ids": torch.tensor([[0, 10, 11, 12, 20, 21]]),
                "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1]]),
            }

    monkeypatch.setattr(collators, "open_image", lambda path: "opened:%s" % path)

    batch = collators.VisionLanguageChatCollator(processor=FakeProcessor())([_sample()])

    assert batch["labels"].tolist() == [[-100, -100, -100, -100, 20, 21]]


def test_phi4_collator_uses_flat_images_and_phi_chat_template(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    import agri_vlm.training.collators as collators

    rendered_messages = []

    class FakeTokenizer:
        pad_token_id = 0
        padding_side = "right"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            rendered_messages.append(messages)
            assert tokenize is False
            if add_generation_prompt:
                assert messages[-1]["role"] == "user"
                return "prompt"
            assert messages[-1]["role"] == "assistant"
            return "full"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def __call__(self, *, text, images, padding, return_tensors):
            assert images == ["opened:image.png"]
            assert padding is True
            assert return_tensors == "pt"
            if text == ["prompt"]:
                return {
                    "input_ids": torch.tensor([[10, 11, 12, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                    "input_mode": torch.tensor([1]),
                }
            if text == ["full"]:
                return {
                    "input_ids": torch.tensor([[10, 11, 12, 20, 21, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
                    "input_mode": torch.tensor([1]),
                }
            raise AssertionError("unexpected text batch: %r" % (text,))

    monkeypatch.setattr(collators, "open_image", lambda path: "opened:%s" % path)

    batch = collators.Phi4MultimodalVisionCollator(processor=FakeProcessor())([_sample()])

    assert rendered_messages[0][-1]["content"] == "<|image_1|>Identify the crop issue."
    assert rendered_messages[1][-1]["content"] == "apple scab"
    assert batch["labels"].tolist() == [[-100, -100, -100, 20, 21, -100]]


def test_phi4_reasoning_vision_collator_uses_image_tokens(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    import agri_vlm.training.collators as collators

    rendered_messages = []

    class FakeTokenizer:
        pad_token_id = 0
        padding_side = "right"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            rendered_messages.append(messages)
            assert tokenize is False
            if add_generation_prompt:
                return "prompt"
            return "full"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def __call__(self, *, text, images, padding, return_tensors):
            assert images == ["opened:image.png"]
            assert padding is True
            assert return_tensors == "pt"
            if text == ["prompt"]:
                return {
                    "input_ids": torch.tensor([[10, 11, 12, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                }
            if text == ["full"]:
                return {
                    "input_ids": torch.tensor([[10, 11, 12, 20, 21, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
                }
            raise AssertionError("unexpected text batch: %r" % (text,))

    monkeypatch.setattr(collators, "open_image", lambda path: "opened:%s" % path)

    batch = collators.Phi4ReasoningVisionCollator(processor=FakeProcessor())([_sample()])

    assert rendered_messages[0][-1]["content"] == "<image>Identify the crop issue."
    assert rendered_messages[1][-1]["content"] == "apple scab"
    assert batch["labels"].tolist() == [[-100, -100, -100, 20, 21, -100]]
