import pytest

from agri_vlm.training.collators import _assistant_suffix_start


def test_assistant_suffix_start_returns_suffix_index() -> None:
    full_input_ids = [10, 20, 30, 40, 50]
    assistant_input_ids = [40, 50]

    assert _assistant_suffix_start(full_input_ids, assistant_input_ids) == 3


def test_assistant_suffix_start_rejects_misaligned_suffix() -> None:
    full_input_ids = [10, 20, 30, 40, 50]
    assistant_input_ids = [30, 50]

    with pytest.raises(ValueError, match="not aligned"):
        _assistant_suffix_start(full_input_ids, assistant_input_ids)


def test_assistant_suffix_start_rejects_empty_suffix() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        _assistant_suffix_start([10, 20], [])
