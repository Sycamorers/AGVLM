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


def test_instructional_sft_format_adds_prompt_contract_and_answer_target() -> None:
    from agri_vlm.data.conversation_format import sample_to_prompt_messages, sample_to_training_messages
    from agri_vlm.schemas.dataset_schema import UnifiedSample

    sample = UnifiedSample.model_validate(_sample())

    prompt_messages = sample_to_prompt_messages(sample, prompt_format="instructional")
    training_messages = sample_to_training_messages(
        sample,
        prompt_format="instructional",
        target_format="instructional",
    )

    prompt_text = prompt_messages[-1]["content"][-1]["text"]
    assert "Respond in this format:" in prompt_text
    assert "Answer: <canonical agricultural label>" in prompt_text
    assert "Evidence: <brief visible symptom evidence>" in prompt_text
    assert "Do not leave Answer blank" in prompt_text
    assert training_messages[-1]["content"][0]["text"] == (
        "Answer: apple scab\n"
        "Evidence: Visible agricultural symptoms or pest features support this label."
    )


def test_instructional_sft_format_renders_clarify_decision_target() -> None:
    from agri_vlm.data.conversation_format import sample_to_training_messages
    from agri_vlm.schemas.dataset_schema import UnifiedSample

    payload = _sample()
    payload["task_type"] = "clarify_or_respond"
    payload["target"] = {
        "decision": "clarify",
        "answer_text": "Could you share a close-up of the underside of the leaf?",
    }
    payload["verifier"] = {"mode": "clarify", "expected_decision": "clarify"}
    sample = UnifiedSample.model_validate(payload)

    training_messages = sample_to_training_messages(
        sample,
        prompt_format="instructional",
        target_format="instructional",
    )

    prompt_text = training_messages[-2]["content"][-1]["text"]
    assert "Decision: clarify" in prompt_text
    assert training_messages[-1]["content"][0]["text"] == (
        "Decision: clarify\n"
        "Clarifying question: Could you share a close-up of the underside of the leaf?"
    )


def test_instructional_sft_format_renders_structured_consultation_target() -> None:
    from agri_vlm.data.conversation_format import sample_to_training_messages
    from agri_vlm.schemas.dataset_schema import UnifiedSample

    payload = _sample()
    payload["task_type"] = "consultation"
    payload["target"] = {
        "answer_text": "Species: tomato\nDiagnosis: early blight\nSymptoms: concentric leaf lesions\nManagement: remove infected leaves",
        "canonical_label": "early blight",
        "structured": {
            "diagnosis": "early blight",
            "management_steps": ["remove infected leaves"],
        },
    }
    payload["verifier"] = {
        "mode": "structured",
        "required_sections": ["Diagnosis", "Evidence", "Uncertainty", "Management", "Follow-up"],
    }
    sample = UnifiedSample.model_validate(payload)

    training_messages = sample_to_training_messages(
        sample,
        prompt_format="instructional",
        target_format="instructional",
    )

    prompt_text = training_messages[-2]["content"][-1]["text"]
    target_text = training_messages[-1]["content"][0]["text"]
    assert "Respond using these line-start section headers exactly once:" in prompt_text
    assert target_text.startswith("Diagnosis: early blight\nEvidence: concentric leaf lesions")
    assert "\nManagement: remove infected leaves\n" in target_text
