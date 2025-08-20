"""Data preprocessing utilities for text autocompletion projects.

This module implements a light-weight pipeline aligned with the course lesson:

- Read a line-delimited raw text file (one text per line)
- Normalize text: Unicode NFKC, lowercase, drop URLs/@mentions/emojis, trim
- Filter by minimal character count and maximal whitespace token length
- Persist a single-column CSV with header ``text`` for downstream steps

Design notes for learners
- Normalization patterns are configurable via ``NormalizationConfig`` to make
  the cleaning stage explicit and reproducible.
- We implement our own tiny tokenizer for transparency before using any large
  frameworks. This reinforces the relationship between text, tokens, and the
  supervision signal (next-token prediction with a 1-step shift).
"""

from __future__ import annotations

import csv
import logging
import os
import random
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from src.config_utils import AppConfig, NormalizationConfig

# Module-level logger for library-friendly diagnostics (no direct prints)
logger = logging.getLogger(__name__)


def _flags_from_names(names: Iterable[str]) -> int:
    """Translate a list of regex flag names into the combined `re` flags bitmask.

    Unknown names are ignored safely. This allows the YAML config to specify
    flags like "IGNORECASE" or "UNICODE" without importing `re` at config time.
    """
    value = 0
    for n in names:
        try:
            flag = getattr(re, str(n), 0)
        except Exception:  # noqa: BLE001
            flag = 0
        if isinstance(flag, int):
            value |= flag
    return value


def _as_str_list(value: object, default: Sequence[str]) -> list[str]:
    """Return `value` as a list[str], or a copy of `default` if incompatible."""
    if isinstance(value, list | tuple):
        seq: Sequence[object] = cast(Sequence[object], value)
        return [str(x) for x in seq]
    return list(default)


def _sanitize_flag_names(names: Iterable[str]) -> list[str]:
    """Deduplicate and clean friendly flag aliases (e.g., U/UNICODE, A/ASCII).

    We also suppress LOCALE because it is rarely desirable for NLP workloads.
    When both ASCII and UNICODE are present, UNICODE wins.
    """
    seen: set[str] = set()
    cleaned: list[str] = []

    def is_ascii(n: str) -> bool:
        return n.upper() in {"A", "ASCII"}

    def is_unicode(n: str) -> bool:
        return n.upper() in {"U", "UNICODE"}

    for raw in names:
        name = str(raw)
        up = name.upper()
        if up in {"L", "LOCALE"}:
            continue
        if up in seen:
            continue
        seen.add(up)
        cleaned.append(name)

    if any(is_ascii(x) for x in cleaned) and any(is_unicode(x) for x in cleaned):
        cleaned = [x for x in cleaned if not is_ascii(x)]
    return cleaned


def _extract_normalization_mapping(
    cfg_or_norm: Mapping[str, object] | AppConfig | NormalizationConfig | None,
) -> dict[str, object]:
    """Extract a normalization section as a plain mapping from various inputs."""
    if cfg_or_norm is None:
        return {}
    if isinstance(cfg_or_norm, NormalizationConfig):
        return cast(dict[str, object], cfg_or_norm.model_dump())
    if isinstance(cfg_or_norm, AppConfig):
        dumped: Mapping[str, object] = cast(Mapping[str, object], cfg_or_norm.model_dump())
        norm_section: object = dumped.get("normalization")
        return cast(dict[str, object], norm_section) if isinstance(norm_section, dict) else {}
    assert isinstance(cfg_or_norm, Mapping)
    mapping: Mapping[str, object] = cfg_or_norm
    norm_section2: object = mapping.get("normalization")
    return cast(dict[str, object], norm_section2) if isinstance(norm_section2, dict) else dict(mapping)


def _compile_patterns(
    cfg_or_norm: Mapping[str, object] | AppConfig | NormalizationConfig | None = None,
) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    """Compile URL, mention, and emoji regex patterns from config.

    Falls back to sensible defaults if some fields are missing or invalid.
    """
    norm = _extract_normalization_mapping(cfg_or_norm)

    url_pat = str(norm.get("url_pattern", r"(https?://\S+|www\.\S+)"))
    mention_pat = str(norm.get("mention_pattern", r"@\w+"))
    emoji_pat = str(norm.get("emoji_pattern", r"[\U0001F300-\U0001FAFF]+"))

    url_flag_names = _sanitize_flag_names(_as_str_list(norm.get("url_flags"), ["IGNORECASE"]))
    mention_flag_names = _sanitize_flag_names(_as_str_list(norm.get("mention_flags"), []))
    emoji_flag_names = _sanitize_flag_names(_as_str_list(norm.get("emoji_flags"), ["UNICODE"]))

    def _compile(pat: str, names: list[str]) -> re.Pattern[str]:
        try:
            return re.compile(pat, flags=_flags_from_names(names))
        except Exception:  # noqa: BLE001
            return re.compile(pat)

    return (
        _compile(url_pat, url_flag_names),
        _compile(mention_pat, mention_flag_names),
        _compile(emoji_pat, emoji_flag_names),
    )


def get_normalize_fn(
    cfg_or_norm: Mapping[str, object] | AppConfig | NormalizationConfig | None = None,
) -> Callable[[str], str]:
    """Return a pure function that normalizes raw text according to config."""
    url_rx, mention_rx, emoji_rx = _compile_patterns(cfg_or_norm)

    def normalize(text: str) -> str:
        s = unicodedata.normalize("NFKC", text).lower()
        s = url_rx.sub("", s)
        s = mention_rx.sub("", s)
        s = emoji_rx.sub("", s)
        return re.sub(r"\s+", " ", s).strip()

    return normalize


def _iter_raw_texts(path: str, limit: int | None = None) -> Iterable[str]:
    """Yield raw text from a line-delimited text file (.txt), one text per line."""
    p = Path(path)
    yielded = 0
    with p.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            s = line.rstrip("\n\r")
            if not s:
                continue
            yield s
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def _token_count(s: str) -> int:
    """Return a simple whitespace token count."""
    return len(s.split())


def preprocess_dataset(
    raw_data_path: str,
    out_path: str = "dataset_processed.csv",
    dev_limit: int | None = None,
    min_chars: int = 5,
    max_tokens: int = 60,
    normalization: object | None = None,
) -> str:
    """Normalize/filter text and persist a single-column CSV named ``text``.

    Returns the output file path as a string. This function intentionally keeps
    I/O simple and explicit for teaching purposes.
    """
    raw_path = Path(raw_data_path)
    if not raw_path.exists():
        # Convenience fallback: look under a local "data/" directory by filename.
        alt = Path("data") / raw_path.name
        if alt.exists():
            raw_path = alt
        else:
            raise FileNotFoundError(f"Dataset not found at {raw_data_path}")

    normalize = get_normalize_fn(None)
    written = 0

    out_p = Path(out_path)
    try:
        from tqdm.auto import tqdm as _tqdm  # type: ignore

        # Disable progress bar when stderr is not a TTY or opt-out via env.
        disable = (not sys.stderr.isatty()) or os.environ.get("DISABLE_TQDM") == "1"
    except Exception:  # noqa: BLE001
        _tqdm = None  # type: ignore
        disable = True  # type: ignore

    with out_p.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["text"])  # header
        iterator = _iter_raw_texts(str(raw_path), limit=dev_limit)
        if _tqdm is not None:
            iterator = _tqdm(
                iterator,
                total=dev_limit,
                desc="preprocess",
                disable=disable,
                leave=False,
            )
        for raw in iterator:  # type: ignore[assignment]
            s = normalize(raw)
            if not s or len(s) < min_chars:
                continue
            if _token_count(s) > max_tokens:
                continue
            writer.writerow([s])
            written += 1
            if dev_limit is not None and written >= dev_limit:
                break

    if written == 0:
        # Always write the header so downstream readers see a valid CSV.
        with out_p.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["text"])
    return str(out_p)


def load_processed_texts(path: str, limit: int | None = None) -> list[str]:
    """Load texts from a processed CSV written by `preprocess_dataset`."""
    p = Path(path)
    if not p.exists():
        return []
    texts: list[str] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # If a "text" column exists, use it; otherwise read the last column.
        idx = (
            0
            if not header
            else (header.index("text") if "text" in header else len(header) - 1)
        )
        for i, row in enumerate(reader):
            if not row:
                continue
            texts.append(str(row[idx]))
            if limit is not None and (i + 1) >= limit:
                break
    return texts


def split_and_persist_texts(
    texts: list[str],
    train_path: str,
    val_path: str,
    test_path: str,
    train_frac: float,
    val_frac: float,
    seed: int,
    dev_limit: int | None = None,
) -> tuple[list[str], list[str], list[str], tuple[str, str, str]]:
    """Deterministically split texts into train/val/test, persist, and reload.

    Steps
    - Shuffle indices with the provided seed (deterministic Random)
    - Split by ``train_frac`` and ``val_frac``; remainder becomes test
    - Write three single-column CSV files with header ``text``
    - Reload splits via ``load_processed_texts`` to verify artifacts

    Returns
    - ``(train_texts, val_texts, test_texts, (train_path, val_path, test_path))``
    """
    # Deterministic shuffle of indices based on the provided seed.
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    n = len(indices)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    parts = (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )
    train_texts, val_texts, test_texts = ([texts[i] for i in idxs] for idxs in parts)

    paths = tuple(map(Path, (train_path, val_path, test_path)))
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    for p, rows in zip(paths, (train_texts, val_texts, test_texts), strict=False):
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text"])  # single-column CSV
            w.writerows([t] for t in rows)

    names = ("train", "val", "test")
    logger.info(
        "Wrote splits: %s %s",
        dict(zip(names, map(str, paths), strict=False)),
        {"sizes": dict(zip(names, map(len, (train_texts, val_texts, test_texts)), strict=False))},
    )

    TRAIN_PATH, VAL_PATH, TEST_PATH = map(str, paths)
    train_texts_loaded, val_texts_loaded, test_texts_loaded = (
        load_processed_texts(p, limit=dev_limit)
        for p in (TRAIN_PATH, VAL_PATH, TEST_PATH)
    )
    logger.info(
        "Loaded splits: %d %d %d",
        len(train_texts_loaded),
        len(val_texts_loaded),
        len(test_texts_loaded),
    )

    return (
        train_texts_loaded,
        val_texts_loaded,
        test_texts_loaded,
        (TRAIN_PATH, VAL_PATH, TEST_PATH),
    )


__all__ = [
    "get_normalize_fn",
    "preprocess_dataset",
    "load_processed_texts",
    "split_and_persist_texts",
]
