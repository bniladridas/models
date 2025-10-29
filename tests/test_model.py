import torch
from transformers import GPT2Config, GPT2LMHeadModel


def test_model_config():
    """Test GPT-2 config creation."""
    config = GPT2Config(
        vocab_size=50257,
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    assert config.vocab_size == 50257
    assert config.n_embd == 768


def test_model_initialization():
    """Test model can be initialized."""
    config = GPT2Config(vocab_size=1000, n_positions=128, n_embd=64, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config)
    assert model is not None
    assert sum(p.numel() for p in model.parameters()) > 0  # Has parameters


def test_model_forward():
    """Test model forward pass."""
    config = GPT2Config(vocab_size=1000, n_positions=128, n_embd=64, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config)
    input_ids = torch.randint(0, 1000, (1, 10))
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert "loss" in outputs
    assert outputs.loss.item() > 0
