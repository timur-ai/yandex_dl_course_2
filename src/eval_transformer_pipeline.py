from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import Any, Protocol

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval_lstm import (
    aggregate_rouge,
    build_prefix_and_suffix,
    rouge_1_2_metrics,
)
from src.tokenizer_vocab import word_tokenize


class VocabProtocol(Protocol):
    """Vocabulary contract needed to compare GPT-2 outputs in ROUGE space."""
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int
    token_to_id: dict[str, int]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str: ...


@torch.inference_mode()
def gpt2_generate_tail(
    tok: Any, model: Any, device: torch.device, prefix_text: str, max_new_tokens: int
) -> str:
    """Generate a continuation with a Hugging Face causal LM given a prefix.

    We decode only the tail (excluding the prompt) to align with the LSTM eval.
    """
    enc = tok(prefix_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    gen_tail_ids = out[0][input_ids.shape[1] :]
    tail_text = tok.decode(gen_tail_ids, skip_special_tokens=True)
    return tail_text


@torch.inference_mode()
def evaluate_gpt2_on_texts(
    texts: Iterable[str],
    vocab: VocabProtocol,
    device: torch.device,
    *,
    model_name: str,
    max_new_tokens: int,
    max_samples: int | None,
    token_pattern: str,
) -> dict[str, float]:
    """Evaluate DistilGPT2 on the same texts and ROUGE protocol as the LSTM.

    The model operates in its own tokenizer space, so we re-tokenize the
    generated tail with our course vocabulary to compute ROUGE in the same space.
    """
    tok_gpt2: Any = AutoTokenizer.from_pretrained(model_name)
    gpt2: Any = AutoModelForCausalLM.from_pretrained(model_name)
    gpt2.to(device)
    gpt2.eval()

    results: list[dict[str, float]] = []
    iter_texts = texts if max_samples is None else islice(texts, max_samples)
    pbar = tqdm(iter_texts, total=(None if max_samples is None else max_samples), desc="gpt2-eval")
    for text in pbar:
        ids = vocab.encode(text, add_special_tokens=True)
        # Apply the assignment's 3/4 prefix and 1/4 completion split
        prefix_ids, suffix_ids = build_prefix_and_suffix(ids, vocab.bos_id, vocab.eos_id)
        if not suffix_ids:
            continue
        # Compare only up to max_new_tokens from the 1/4 suffix
        ref_tail_ids = suffix_ids[:max_new_tokens]
        prefix_text = vocab.decode(prefix_ids, skip_special_tokens=True)
        pred_tail_text = gpt2_generate_tail(
            tok_gpt2, gpt2, device, prefix_text, max_new_tokens
        )
        pred_tail_ids = [
            vocab.token_to_id.get(t, vocab.unk_id)
            for t in word_tokenize(pred_tail_text, token_pattern)
        ]
        m = rouge_1_2_metrics(
            pred_tail_ids,
            ref_tail_ids,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
        )
        results.append(m)
    pbar.close()

    return aggregate_rouge(results)


__all__ = [
    "gpt2_generate_tail",
    "evaluate_gpt2_on_texts",
]
