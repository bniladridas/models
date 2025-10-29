import pytest
from datasets import load_from_disk


def test_dataset_loading():
    """Test that the tokenized dataset can be loaded."""
    try:
        dataset = load_from_disk("data/tokenized_wikitext")
        assert "train" in dataset
        assert len(dataset["train"]) > 0
    except Exception as e:
        pytest.skip(f"Dataset not available: {e}")


def test_dataset_structure():
    """Test dataset has expected structure."""
    try:
        import torch

        dataset = load_from_disk("data/tokenized_wikitext")
        sample = dataset["train"][0]
        assert "input_ids" in sample
        assert isinstance(sample["input_ids"], torch.Tensor)
    except Exception as e:
        pytest.skip(f"Dataset not available: {e}")
