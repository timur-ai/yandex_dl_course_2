## Text Autocomplete (Word-level LSTM + GPT‑2 Baseline)

A compact, instructional NLP project for next‑token prediction and inline text suggestions. You will build a word‑level tokenizer and vocabulary, train a small LSTM language model, and evaluate it with ROUGE‑1/2 against reference suffixes. A DistilGPT‑2 baseline is included for comparison.

### What you’ll learn

- Tokenization/vocabulary building and special tokens design
- Preparing supervised tensors for next‑token prediction (shifted targets + masks)
- Training an LSTM LM and generating greedy continuations
- Evaluating completions with ROUGE‑1/2 and comparing to a GPT‑2 baseline

---

## Repository structure

- `configs/config.yaml`: Project configuration (paths, tokenizer, training, evaluation).
- `solution.ipynb`: Guided, end‑to‑end notebook (recommended to start here).
- `src/`
  - `config_utils.py`: Typed Pydantic config models and loader.
  - `data_utils.py`: Normalization, preprocessing, dataset IO, train/val/test split.
  - `tokenizer_vocab.py`: Regex word tokenizer and `WordVocab` with encode/decode.
  - `next_token_dataset.py`: Tensor dataset + dataloaders for next‑token prediction.
  - `lstm_model.py`: `LSTMLanguageModel` and greedy `generate`.
  - `lstm_train.py`: Training/validation loops, optimizer/scheduler, checkpoint save.
  - `eval_lstm.py`: ROUGE metrics, prefix/suffix split, evaluation loop.
  - `eval_transformer_pipeline.py`: DistilGPT‑2 baseline evaluation with ROUGE in course vocab space.
  - `project_utils.py`: Device/AMP/seed/runtime helpers.
  - `summary.py`: Pretty printing and qualitative example display.

Runtime deps are in `pyproject.toml`. Python version is pinned in `.python-version`.

---

## Requirements

- Python 3.10.x (e.g., 3.10.12)
- Optional CUDA GPU for faster training

On Windows, dataloader workers default to 0 for stability; CUDA is supported if available.

---

## Setup

### Option A: uv (recommended)

```powershell
# Install uv (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Create and sync a 3.10 virtual environment
uv venv --python 3.10
uv sync --group dev

# Run commands in the venv
uv run python -V
```

### Option B: pip/venv

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
# (Optional dev tools)
pip install ruff pytest hypothesis pyyaml
```

---

## Data preparation

Place a UTF‑8 text file with one example per line at the path specified by `data.raw_data_path` (default: `data/tweets.txt`). Then preprocess and split via the notebook or the snippet below.

Key config paths (editable in `configs/config.yaml`):

- `data.raw_data_path`: input `.txt`
- `data.processed_path`: cleaned CSV with a single column `text`
- `data.train_path`, `data.val_path`, `data.test_path`: CSV splits

---

## End‑to‑end: train LSTM and evaluate

Run this as a standalone script or paste into a cell. It uses only the provided modules.

```python
from pathlib import Path

import torch

from src.config_utils import load_config_model, LSTMConfig
from src.data_utils import preprocess_dataset, load_processed_texts, split_and_persist_texts
from src.tokenizer_vocab import WordVocab, build_supervised_tensors
from src.next_token_dataset import create_next_token_dataloaders
from src.lstm_model import LSTMLanguageModel
from src.lstm_train import create_optimizer_scheduler, train_epoch, valid_epoch, save_checkpoint
from src.eval_lstm import evaluate_on_texts
from src.summary import print_scores, show_examples
from src.project_utils import initialize_runtime


def main() -> None:
    cfg = load_config_model("configs/config.yaml")
    rt = initialize_runtime(seed=cfg.split.seed)
    device = rt.device

    # 1) Preprocess and split
    Path(cfg.data.train_path).parent.mkdir(parents=True, exist_ok=True)
    preprocess_dataset(
        raw_data_path=cfg.data.raw_data_path,
        out_path=cfg.data.processed_path,
        dev_limit=cfg.data.dev_limit,
        min_chars=cfg.preprocessing.min_chars,
        max_tokens=cfg.preprocessing.max_tokens,
    )
    texts = load_processed_texts(cfg.data.processed_path, limit=cfg.data.dev_limit)
    train_texts, val_texts, test_texts, _paths = split_and_persist_texts(
        texts,
        cfg.data.train_path,
        cfg.data.val_path,
        cfg.data.test_path,
        train_frac=cfg.split.train_frac,
        val_frac=cfg.split.val_frac,
        seed=cfg.split.seed,
        dev_limit=cfg.data.dev_limit,
    )

    # 2) Build vocabulary
    vocab = WordVocab.build(
        train_texts,
        special_tokens=cfg.special_tokens.model_dump(),
        max_vocab_size=cfg.tokenizer.max_vocab_size,
        token_pattern=cfg.tokenizer.token_pattern,
        min_freq=cfg.tokenizer.min_freq,
    )

    # 3) Supervised tensors and dataloaders
    train_tensors = build_supervised_tensors(
        train_texts, vocab, max_len=cfg.sequence.max_length, device=device
    )
    val_tensors = build_supervised_tensors(
        val_texts, vocab, max_len=cfg.sequence.max_length, device=device
    )
    test_tensors = build_supervised_tensors(
        test_texts, vocab, max_len=cfg.sequence.max_length, device=device
    )
    train_loader, val_loader, test_loader = create_next_token_dataloaders(
        train_tensors, val_tensors, test_tensors, cfg, device
    )

    # 4) Model, optimizer, scheduler
    model_cfg = LSTMConfig(
        num_embeddings=len(vocab.id_to_token),
        embedding_dim=cfg.model.lstm.embedding_dim,
        hidden_size=cfg.model.lstm.hidden_size,
        num_layers=cfg.model.lstm.num_layers,
        dropout=cfg.model.lstm.dropout,
        pad_token_id=vocab.pad_id,
    )
    model = LSTMLanguageModel(model_cfg).to(device)
    optimizer, scheduler = create_optimizer_scheduler(
        model,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        use_scheduler=cfg.scheduler.use,
        step_size=cfg.scheduler.step_size,
        gamma=cfg.scheduler.gamma,
    )

    # 5) Train
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            max_grad_norm=cfg.training.max_grad_norm,
            progress_desc=f"train[{epoch}/{cfg.training.epochs}]",
        )
        val_loss = valid_epoch(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        print({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)})

    # 6) Evaluate with ROUGE on test
    scores_test = evaluate_on_texts(
        model,
        texts=test_texts,
        vocab=vocab,
        device=device,
        max_new_tokens=cfg.generation.max_new_tokens,
        max_samples=cfg.evaluation.test_samples,
    )
    print_scores("LSTM@test", scores_test)

    # 7) Save minimal checkpoint (weights + vocab)
    ckpt_dir = Path(cfg.checkpoint.directory)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        vocab,
        state_path=str(ckpt_dir / "lstm_state_dict.pt"),
        vocab_path=str(ckpt_dir / "vocab.pt"),
    )

    # 8) Qualitative samples
    show_examples(model, test_texts, vocab, device, max_new_tokens=cfg.generation.max_new_tokens, k=5)


if __name__ == "__main__":
    main()
```

Run with uv:

```powershell
uv run python run_lstm.py  # if you save the snippet as run_lstm.py
```

---

## DistilGPT‑2 baseline evaluation

This evaluates Hugging Face `distilgpt2` on the same prefix/suffix protocol, then retokenizes the generated tail with the course vocabulary to compute ROUGE in the same space.

```python
import torch
from src.config_utils import load_config_model
from src.data_utils import load_processed_texts
from src.tokenizer_vocab import WordVocab
from src.eval_transformer_pipeline import evaluate_gpt2_on_texts
from src.summary import print_scores

cfg = load_config_model()
test_texts = load_processed_texts(cfg.data.test_path, limit=cfg.evaluation.test_samples)

# Reuse the vocab you built for LSTM (recommended). For demo, build from train split:
train_texts = load_processed_texts(cfg.data.train_path)
vocab = WordVocab.build(
    train_texts,
    special_tokens=cfg.special_tokens.model_dump(),
    max_vocab_size=cfg.tokenizer.max_vocab_size,
    token_pattern=cfg.tokenizer.token_pattern,
    min_freq=cfg.tokenizer.min_freq,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scores_gpt2 = evaluate_gpt2_on_texts(
    texts=test_texts,
    vocab=vocab,
    device=device,
    model_name=cfg.gpt2.model_name,
    max_new_tokens=cfg.generation.max_new_tokens,
    max_samples=cfg.gpt2.eval_samples,
    token_pattern=vocab.token_pattern,
)
print_scores("DistilGPT2@test", scores_gpt2)
```

Note: Downloading a HF model requires internet and may be slow on first run.

---

## Configuration cheatsheet (`configs/config.yaml`)

- **data**: paths to raw/processed/split files; `dev_limit` for quick local runs
- **preprocessing**: `min_chars`, `max_tokens` (pre‑tokenization filters)
- **tokenizer**: `max_vocab_size`, `token_pattern` (regex), `min_freq`
- **split**: `seed`, `train_frac`, `val_frac`
- **sequence**: `max_length` for input/target after shift
- **special_tokens**: strings for `pad`, `unk`, `bos`, `eos`
- **model.lstm**: `embedding_dim`, `hidden_size`, `num_layers`, `dropout`
- **training**: `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `max_grad_norm`
- **scheduler**: `use`, `step_size`, `gamma`
- **dataloader**: worker/pinning knobs (auto‑tuned for device in code)
- **generation**: `max_new_tokens`
- **evaluation**: optional sample caps for faster eval
- **gpt2**: `model_name` and sample cap for baseline
- **checkpoint**: output directory for `*.pt` artifacts

---

## Notebook workflow

Open `solution.ipynb` and run cells top‑to‑bottom. It mirrors the code above and is the easiest way to iterate on config and observe metrics/qualitative examples.

```powershell
uv run jupyter lab  # or: uv run jupyter notebook
```

---

## Testing, linting, formatting

```powershell
# Lint/format (ruff)
uv run ruff check .
uv run ruff format .

# Run tests (if/when present)
uv run pytest -q
```

---

## Troubleshooting

- **Windows + CPU only**: `num_workers` is 0 by default for stability. Training is slower; reduce `training.batch_size` if you hit RAM limits.
- **CUDA OOM**: Lower `training.batch_size` and/or `sequence.max_length`. Mixed precision is enabled automatically on CUDA.
- **No text after preprocessing**: Loosen `preprocessing.min_chars` / `max_tokens`, or check regex normalization in `configs/config.yaml`.
- **Slow or non‑interactive tqdm bars**: This is normal in some terminals/IDEs. Computation still proceeds.
- **HF model download errors**: Ensure internet access and retry; optionally pre‑download `distilgpt2` via `transformers` cache.

---

## License

Educational use. Add your preferred open‑source license if you plan to distribute.
