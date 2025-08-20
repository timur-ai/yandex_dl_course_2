from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from itertools import islice
from math import floor
from typing import Protocol

import torch
import torch.nn as nn
from tqdm.auto import tqdm


class VocabProtocol(Protocol):
    """Minimal vocabulary contract needed by evaluation utilities."""
    pad_id: int
    bos_id: int
    eos_id: int

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


def _strip_special(
    ids: Sequence[int], pad_id: int, bos_id: int, eos_id: int
) -> list[int]:
    """Remove special tokens and everything after the first ``eos_id``.

    This keeps evaluation focused on real content tokens.
    """
    out: list[int] = []
    for tok in ids:
        if tok == pad_id:
            continue
        if tok == bos_id:
            continue
        if tok == eos_id:
            break
        out.append(tok)
    return out


def _ngram_counts(tokens: Sequence[int], n: int) -> dict[tuple[int, ...], int]:
    """Count occurrences of n-grams represented as integer token tuples."""
    if n <= 0 or len(tokens) < n:
        return {}
    counts: dict[tuple[int, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _rouge_n(
    pred: Sequence[int], ref: Sequence[int], n: int
) -> tuple[float, float, float]:
    """Compute ROUGE-N precision/recall/F1 over integer token sequences."""
    pred_counts = _ngram_counts(pred, n)
    ref_counts = _ngram_counts(ref, n)
    if not pred_counts:
        precision = 0.0
    else:
        overlap = sum(min(pred_counts.get(k, 0), v) for k, v in ref_counts.items())
        precision = overlap / max(1, sum(pred_counts.values()))
    if not ref_counts:
        recall = 0.0
    else:
        overlap = sum(min(pred_counts.get(k, 0), v) for k, v in ref_counts.items())
        recall = overlap / max(1, sum(ref_counts.values()))
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def rouge_1_2_metrics(
    pred_ids: Sequence[int],
    ref_ids: Sequence[int],
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> dict[str, float]:
    """Compute ROUGE-1 and ROUGE-2 metrics for predicted vs reference tails."""
    pred_clean = _strip_special(pred_ids, pad_id, bos_id, eos_id)
    ref_clean = _strip_special(ref_ids, pad_id, bos_id, eos_id)
    p1, r1, f1_1 = _rouge_n(pred_clean, ref_clean, 1)
    p2, r2, f1_2 = _rouge_n(pred_clean, ref_clean, 2)
    return {
        "rouge1_p": p1,
        "rouge1_r": r1,
        "rouge1_f1": f1_1,
        "rouge2_p": p2,
        "rouge2_r": r2,
        "rouge2_f1": f1_2,
    }


def aggregate_rouge(results: Sequence[dict[str, float]]) -> dict[str, float]:
    """Average a list of ROUGE metric dicts; return NaNs when empty."""
    if not results:
        return {
            k: float("nan")
            for k in (
                "rouge1_p",
                "rouge1_r",
                "rouge1_f1",
                "rouge2_p",
                "rouge2_r",
                "rouge2_f1",
            )
        }
    totals: dict[str, float] = {k: 0.0 for k in results[0].keys()}
    for r in results:
        for k, v in r.items():
            totals[k] += float(v)
    n = float(len(results))
    return {k: totals[k] / n for k in totals}


def build_prefix_and_suffix(
    ids: list[int], _bos_id: int, _eos_id: int
) -> tuple[list[int], list[int]]:
    """Split a tokenized text into 3/4 prefix and 1/4 suffix (assignment rule)."""
    if len(ids) < 4:
        return ids, []
    # Implements the assignment rule: use first 75% tokens as input (prefix)
    # and the remaining 25% as the completion target (suffix).
    split_idx = max(1, floor(len(ids) * 0.75))
    prefix = ids[:split_idx]
    suffix = ids[split_idx:]
    return prefix, suffix


class GenerateModelProtocol(Protocol):
    """Subset of model methods used by the evaluation loop."""
    def generate(
        self,
        prefix_ids: torch.Tensor,
        max_new_tokens: int = 20,
        eos_token_id: int | None = None,
    ) -> torch.Tensor: ...

    def eval(self) -> nn.Module: ...

@torch.inference_mode()
def evaluate_on_texts(
    model: GenerateModelProtocol,
    texts: Iterable[str],
    vocab: VocabProtocol,
    device: torch.device,
    max_new_tokens: int = 20,
    max_samples: int | None = 500,
) -> dict[str, float]:
    """Evaluate a next-token generator on texts using ROUGE-1/2 metrics.

    The function applies the 75/25 split rule, generates up to ``max_new_tokens``
    for each example, and averages token-level ROUGE metrics across samples.
    """
    _ = model.eval()
    results: list[dict[str, float]] = []
    iter_texts = texts if max_samples is None else islice(texts, max_samples)
    pbar = tqdm(iter_texts, total=(None if max_samples is None else max_samples), desc="eval", file=sys.stdout)
    for text in pbar:
        ids = vocab.encode(text, add_special_tokens=True)
        # Apply the 3/4 (prefix) vs 1/4 (suffix) split for evaluation
        prefix, suffix = build_prefix_and_suffix(ids, vocab.bos_id, vocab.eos_id)
        if not suffix:
            continue
        # Evaluate generation only against up to max_new_tokens from the suffix
        ref_tail = suffix[:max_new_tokens]
        prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=device)
        gen_out = model.generate(
            prefix_tensor, max_new_tokens=max_new_tokens, eos_token_id=vocab.eos_id
        )
        pred_tail = gen_out[0].tolist()[len(prefix) :]
        metrics = rouge_1_2_metrics(
            pred_tail,
            ref_tail,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
        )
        results.append(metrics)
    pbar.close()
    return aggregate_rouge(results)


strip_special = _strip_special


__all__ = [
    "VocabProtocol",
    "strip_special",
    "rouge_1_2_metrics",
    "aggregate_rouge",
    "build_prefix_and_suffix",
    "evaluate_on_texts",
]
