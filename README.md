# gpt-2 training project

train gpt-2 or gpt-neo models on the wikitext dataset using pytorch and transformers.

## quick start

run:

```bash
./main.sh
```

this script provides an interactive setup for training, testing, and evaluation.

## usage

* **train** — choose model (`gpt-2` / `gpt-neo`), number of epochs, and execution method (`local` / `docker`)
* **test** — run unit tests:

  ```bash
  pytest tests/
  ```
* **evaluate** — generate text using a trained model.

## project structure

| directory | description                                                  |
| --------- | ------------------------------------------------------------ |
| `src/`    | core code — `data.py`, `train.py`, `evaluate.py`, `model.py` |
| `models/` | saved model checkpoints                                      |
| `data/`   | tokenized datasets (generate via `python src/data.py`)       |
| `docker/` | docker configuration files                                   |

## dataset configuration

edit `src/data.py` to customize dataset and preprocessing:

```python
# load dataset (change to your preferred dataset)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# load tokenizer (change for different models)
tokenizer = gpt2tokenizer.from_pretrained("gpt2")

# tokenize function (adjust max_length for sequence length)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=true, padding="max_length", max_length=512)

# select subset for training (change range for data size)
tokenized_datasets["train"] = tokenized_datasets["train"].select(range(1000))
```

## notes

* default training runs on **cpu** inside docker; update configurations for **gpu** usage.
* trained models are automatically saved in `models/` with timestamped filenames.
* model available on hugging face: https://huggingface.co/harpertoken/harpertokenGPT2

## conventional commits

this project enforces conventional commit standards for consistency and clarity.

### setup

enable the `commit-msg` hook:

```bash
cp scripts/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```

### guidelines

commit messages must:

* start with a type: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`
* follow with a concise lowercase description
* keep the first line ≤ 30 characters

example:

```bash
feat: add new training feature
```

### history cleanup

rewrite existing commit messages:

```bash
./scripts/rewrite_msg.sh
git push --force-with-lease
```
