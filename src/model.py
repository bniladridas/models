import torch
from transformers import GPT2Config, GPT2LMHeadModel

# Custom config for ~500M parameters (adjustable to 1B if memory allows)
# Approximate: n_embd=1024, n_layer=24, n_head=16 -> ~500M params
config = GPT2Config(
    vocab_size=50257,  # GPT-2 vocab
    n_positions=512,  # Shorter for memory
    n_embd=1024,
    n_layer=24,
    n_head=16,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    gradient_checkpointing=False,  # Enable later for memory
    use_cache=False,  # For training
)

model = GPT2LMHeadModel(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # Should be ~1B

# Move to MPS if available
if torch.backends.mps.is_available():
    model = model.to("mps")
    print("Model moved to MPS")
else:
    print("MPS not available")
