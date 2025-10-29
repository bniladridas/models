from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load dataset (small subset for testing)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Select small subset for training
tokenized_datasets["train"] = tokenized_datasets["train"].select(range(1000))

# Convert to PyTorch tensors
tokenized_datasets.set_format("torch")

print("Dataset prepared:")
print(f"Train size: {len(tokenized_datasets['train'])}")
print(f"Validation size: {len(tokenized_datasets['validation'])}")

# Save for later use
tokenized_datasets.save_to_disk("data/tokenized_wikitext")
