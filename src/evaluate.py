import os

import torch
from transformers import GPT2Tokenizer

# Match model type from environment
MODEL_TYPE = os.environ.get("MODEL_TYPE", "gpt2")

# Check if model exists
if not os.path.exists("models/trained_model"):
    print(
        "No trained model found in 'models/trained_model/' directory. Please train a model first."
    )
    exit(1)

if MODEL_TYPE == "gpt2":
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("models/trained_model", local_files_only=True)
elif MODEL_TYPE == "gpt-neo":
    from transformers import GPTNeoForCausalLM

    model = GPTNeoForCausalLM.from_pretrained("models/trained_model", local_files_only=True)
else:
    raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Move to device
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# Generate text
model.eval()
input_text = "The quick brown fox"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)
