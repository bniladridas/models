import os

import torch
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

# Choose model type: "gpt2" or "gpt-neo"
MODEL_TYPE = os.environ.get("MODEL_TYPE", "gpt2")

if __name__ == "__main__":
    # Load dataset (assuming prepared)
    try:
        dataset_path = "data/tokenized_wikitext"
        print(f"Loading dataset from {dataset_path}")
        tokenized_datasets = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Dataset not found ({e}), using dummy data")
        # Dummy data for testing
        input_ids = torch.randint(0, 50257, (100, 512))
        dataset = [{"input_ids": input_ids[i]} for i in range(100)]
        train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        train_dataset = tokenized_datasets["train"]
        train_dataloader = DataLoader(
            train_dataset, batch_size=1, shuffle=True
        )  # Small batch for memory

    # Create model based on type
    if MODEL_TYPE == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=50257,
            n_positions=512,
            n_embd=768,
            n_layer=12,
            n_head=12,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=False,
        )
        model = GPT2LMHeadModel(config)
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    elif MODEL_TYPE == "gpt-neo":
        from transformers import GPTNeoConfig, GPTNeoForCausalLM

        config = GPTNeoConfig(
            vocab_size=50257,
            max_position_embeddings=512,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            activation_function="gelu_new",
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=False,
        )
        model = GPTNeoForCausalLM(config)
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    # Move to MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Scheduler
    num_epochs = int(os.environ.get("NUM_EPOCHS", 1))
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    step = 0
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = (
                {k: v.to(device) for k, v in batch.items()}
                if isinstance(batch, dict)
                else batch[0].to(device)
            )
            labels = batch["input_ids"] if isinstance(batch, dict) else batch

            outputs = model(**batch, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            step += 1
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Save model
    model.save_pretrained("models/trained_model")
    print("Training completed")
