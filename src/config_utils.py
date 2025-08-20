"""Typed configuration models using Pydantic.

These classes mirror the YAML structure under `configs/config.yaml` and provide:

- Field validation and defaults
- Editor/IDE autocompletion and type hints
- A single, canonical place to document configuration semantics

You typically call ``load_config_model()`` to parse the YAML into an ``AppConfig``
instance and then thread that object through the rest of the pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import yaml
from pydantic import BaseModel, Field


class NormalizationConfig(BaseModel):
    """Regex patterns and flags for text normalization.

    These settings drive URL, @mention, and emoji removal in the preprocessing
    pipeline (see `src/data_utils.py:get_normalize_fn`). Flags are specified by
    their names as strings (e.g., "IGNORECASE", "UNICODE"). At runtime they are
    mapped to `re.<FLAG>` values; unknown names are ignored safely.

    Notes
    - The defaults aim to be broadly effective across social/text corpora.
    - If you need inline comments or whitespace in patterns, also include the
      "VERBOSE" flag in the corresponding `*_flags` list.
    - Emoji handling uses inclusive Unicode ranges and a few single code points
      for common symbols. Tune to your task if you want to retain some symbols.
    """

    # URL detection: matches http/https schemes and bare "www." prefixes.
    url_pattern: str = r"(https?://\S+|www\.\S+)"
    # Apply case-insensitive matching by default to catch mixed-case domains.
    url_flags: list[str] = Field(default_factory=lambda: ["IGNORECASE"])

    # Mentions like "@user" consisting of word characters (letters/digits/_).
    mention_pattern: str = r"@\w+"
    # Typically no flags are needed unless extending beyond ASCII word chars.
    mention_flags: list[str] = Field(default_factory=list)

    # Emoji/pictograph coverage via Unicode code point ranges:
    # - U+1F600–U+1F64F: Emoticons
    # - U+1F300–U+1F5FF: Misc Symbols and Pictographs
    # - U+1F680–U+1F6FF: Transport & Map
    # - U+1F1E6–U+1F1FF: Regional Indicator Symbols (flags)
    # - U+2702–U+27B0: Dingbats subset
    # - U+24C2–U+1F251: Enclosed characters and additional symbols
    # - U+1F900–U+1F9FF: Supplemental Symbols and Pictographs
    # - U+1FA70–U+1FAFF: Symbols for legacy computing, chess, etc.
    # - U+2600–U+26FF: Miscellaneous Symbols (weather, etc.)
    # - U+2B50: White medium star
    # - U+2B06–U+2B07: Up/down arrows
    # - U+2B05: Leftwards black arrow
    # - U+2B55: Heavy large circle
    # The trailing '+' collapses consecutive emojis into one match.
    emoji_pattern: str = (
        "[\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e6-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f900-\U0001f9ff"
        "\U0001fa70-\U0001faff"
        "\U00002600-\U000026ff"
        "\U00002b50"
        "\U00002b06-\U00002b07"
        "\U00002b05"
        "\U00002b55"
        "]+"
    )
    # Use UNICODE so character classes like \w and ranges behave on full Unicode.
    # Add VERBOSE if you switch to a commented/whitespace pattern form.
    emoji_flags: list[str] = Field(default_factory=lambda: ["UNICODE"])


class DataConfig(BaseModel):
    """Paths and limits for dataset files used across the project.

    All paths are relative to the repository root unless made absolute by the
    user. The processed dataset is a single-column CSV with header ``text``.
    """

    raw_data_path: str = "data/tweets.txt"
    processed_path: str = "dataset_processed.csv"
    train_path: str = "data/train.csv"
    val_path: str = "data/val.csv"
    test_path: str = "data/test.csv"
    dev_limit: int | None = None


class PreprocessingConfig(BaseModel):
    """Pre-tokenization cleaning constraints.

    - ``min_chars``: discard texts shorter than this after normalization
    - ``max_tokens``: discard texts with more whitespace-separated tokens
      (used as a coarse length filter before model tokenization)
    """

    min_chars: int = 5
    max_tokens: int = 60


class TokenizerConfig(BaseModel):
    """Tokenizer vocabulary and tokenization behavior.

    ``token_pattern`` is a regex used by the simple word tokenizer. The default
    captures words and standalone punctuation, which is often sufficient for
    small projects and introductory assignments.
    """

    max_vocab_size: int = 20000
    token_pattern: str = r"\w+|[^\w\s]"
    min_freq: int = 1


class SplitConfig(BaseModel):
    """Random split parameters for train/val/test partitions.

    The remaining portion after ``train_frac`` and ``val_frac`` goes to test.
    Splits are deterministic given the seed.
    """

    seed: int = 42
    train_frac: float = 0.80
    val_frac: float = 0.10


class SequenceConfig(BaseModel):
    """Sequence length constraints for model inputs.

    ``max_length`` applies to both inputs and targets after adding special
    tokens and performing the one-step shift for next-token supervision.
    """

    max_length: int = 64


class SpecialTokens(BaseModel):
    """Special marker tokens used by the tokenizer and models.

    These are reserved at the start of the vocabulary with fixed indices.
    """

    pad: str = "<pad>"
    unk: str = "<unk>"
    bos: str = "<bos>"
    eos: str = "<eos>"


class ModelLSTMConfig(BaseModel):
    """Static LSTM architecture hyperparameters.

    Notes
    - ``dropout`` inside ``nn.LSTM`` is active only when ``num_layers > 1``.
    """

    embedding_dim: int = 256
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


class LSTMConfig(BaseModel):
    """Runtime LSTM config including vocabulary-dependent fields.

    This mirrors defaults from `model.lstm` while adding fields that are only
    known at runtime (e.g., `num_embeddings`, `pad_token_id`).
    """

    num_embeddings: int
    embedding_dim: int = 256
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    pad_token_id: int = 0


class ModelConfig(BaseModel):
    """Model family configurations (currently LSTM)."""

    lstm: ModelLSTMConfig


class TrainingConfig(BaseModel):
    """Core training loop hyperparameters.

    ``log_every`` is accepted for compatibility and may be ignored by some
    training utilities in this repository.
    """

    epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    log_every: int = 100


class SchedulerConfig(BaseModel):
    """LR scheduler parameters for step-based decay.

    When ``use`` is false, downstream code will skip creating a scheduler.
    """

    use: bool = True
    step_size: int = 1
    gamma: float = 0.95


class DataloaderConfig(BaseModel):
    """DataLoader knobs, tuned per device automatically when None.

    The ``*_cuda`` fields are used when the active device is CUDA; otherwise
    the CPU-specific values are used.
    """

    num_workers_cpu: int = 0
    num_workers_cuda: int | None = None
    pin_memory: bool | None = None
    persistent_workers: bool | None = None


class GenerationConfig(BaseModel):
    """Text generation parameters for inference/evaluation.

    Applies both to the LSTM generator and the GPT-2 baseline evaluation.
    """

    max_new_tokens: int = 20


class EvaluationConfig(BaseModel):
    """Numbers of samples to use during validation and testing.

    ``None`` means use the full split without subsampling.
    """

    val_samples: int | None = 500
    test_samples: int | None = 500


class ProfilingConfig(BaseModel):
    """Micro-benchmark parameters for latency and throughput profiling.

    Not all projects will use these fields; they are provided for completeness
    and future extensions.
    """

    warmup_iters: int = 10
    measure_iters: int = 100
    prefix_len: int = 12


class TargetsConfig(BaseModel):
    """Performance targets to keep the system within product budgets.

    These are illustrative constraints to guide mobile/edge deployment.
    """

    latency_p50_ms: float = 50
    latency_p95_ms: float = 150
    memory_mb: float = 50


class UXConfig(BaseModel):
    """User-experience tuning for suggestions display/behavior.

    These settings can drive front-end behavior when integrating the model.
    """

    suggest_tail_new_tokens: int = 12


class GPT2Config(BaseModel):
    """Configuration for the GPT-2 baseline used in evaluation.

    ``eval_samples`` controls the number of examples to sample for speed.
    """

    eval_samples: int | None = 200
    model_name: str = "distilgpt2"


class CheckpointConfig(BaseModel):
    """Where model checkpoints are stored/loaded from.

    Only minimal artifacts are saved in this project: model state and vocab.
    """

    directory: str = "data"


class AppConfig(BaseModel):
    """Top-level application configuration composed of typed sections.

    Access fields via attributes (e.g., ``cfg.training.batch_size``). You can
    convert to a plain mapping via ``cfg.model_dump()`` when needed.
    """

    data: DataConfig
    preprocessing: PreprocessingConfig
    tokenizer: TokenizerConfig
    split: SplitConfig
    sequence: SequenceConfig
    special_tokens: SpecialTokens
    model: ModelConfig
    training: TrainingConfig
    scheduler: SchedulerConfig
    dataloader: DataloaderConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    profiling: ProfilingConfig
    targets: TargetsConfig
    ux: UXConfig
    gpt2: GPT2Config
    normalization: NormalizationConfig = Field(default_factory=lambda: NormalizationConfig())
    checkpoint: CheckpointConfig = Field(default_factory=lambda: CheckpointConfig())


def load_config_model(path: str = "configs/config.yaml") -> AppConfig:
    """Load YAML into a validated ``AppConfig`` using Pydantic.

    Raises a ``RuntimeError`` if PyYAML is missing.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs/config.yaml. Install 'pyyaml'.")

    with open(path, encoding="utf-8") as f:
        raw: object = yaml.safe_load(f) or {}

    # Minimal skeleton ensures validation succeeds even if the YAML is not a
    # mapping or omits sections. User-provided sections override these.
    skeleton: dict[str, object] = {
        "data": {},
        "preprocessing": {},
        "tokenizer": {},
        "split": {},
        "sequence": {},
        "special_tokens": {},
        "model": {"lstm": {}},
        "training": {},
        "scheduler": {},
        "dataloader": {},
        "generation": {},
        "evaluation": {},
        "profiling": {},
        "targets": {},
        "ux": {},
        "gpt2": {},
        "normalization": {},
    }

    if isinstance(raw, Mapping):
        raw_map = cast(Mapping[object, object], raw)
        provided: dict[str, object] = {k: v for k, v in raw_map.items() if isinstance(k, str)}  # type: ignore[dict-item]
        merged = skeleton | provided
    else:
        merged = skeleton

    return AppConfig.model_validate(merged)


__all__ = [
    "AppConfig",
    "NormalizationConfig",
    "DataConfig",
    "PreprocessingConfig",
    "TokenizerConfig",
    "SplitConfig",
    "SequenceConfig",
    "SpecialTokens",
    "LSTMConfig",
    "ModelLSTMConfig",
    "ModelConfig",
    "TrainingConfig",
    "SchedulerConfig",
    "DataloaderConfig",
    "GenerationConfig",
    "EvaluationConfig",
    "ProfilingConfig",
    "TargetsConfig",
    "UXConfig",
    "GPT2Config",
    "CheckpointConfig",
    "load_config_model",
]
