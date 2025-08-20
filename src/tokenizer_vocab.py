from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch


def word_tokenize(text: str, token_pattern: str) -> list[str]:
    """Regex-based word tokenizer used throughout the project.

    The default pattern splits on words and standalone punctuation. For
    instructional clarity we keep it minimal and deterministic.
    """
    if not text:
        return []
    regex = re.compile(token_pattern, flags=re.UNICODE)
    return regex.findall(text)


@dataclass(slots=True)
class WordVocab:
    """Simple word-level vocabulary with special tokens and helpers.

    Methods
    - ``build``: construct vocabulary from an iterable of texts
    - ``encode``: convert a text into token IDs (with optional special tokens)
    - ``decode``: convert token IDs back into a space-joined string
    """
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int
    token_pattern: str

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        *,
        special_tokens: dict[str, str],
        max_vocab_size: int,
        token_pattern: str,
        min_freq: int = 1,
    ) -> WordVocab:
        """Build a vocabulary by counting token frequencies over texts.

        Special tokens are inserted first with fixed indices in order:
        ``[pad, unk, bos, eos]``. The remainder is filled by most common tokens
        that meet ``min_freq`` until ``max_vocab_size`` is reached.
        """
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(word_tokenize(t, token_pattern))
        pad = special_tokens["pad"]
        unk = special_tokens["unk"]
        bos = special_tokens["bos"]
        eos = special_tokens["eos"]
        id_to_token = [pad, unk, bos, eos]
        token_to_id = {tok: idx for idx, tok in enumerate(id_to_token)}
        for tok, freq in counts.most_common():
            if freq < min_freq:
                continue
            if tok in token_to_id:
                continue
            id_to_token.append(tok)
            token_to_id[tok] = len(id_to_token) - 1
            # Stop when we reach the requested vocabulary size (including specials)
            if len(id_to_token) >= max_vocab_size:
                break
        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            pad_id=token_to_id[pad],
            unk_id=token_to_id[unk],
            bos_id=token_to_id[bos],
            eos_id=token_to_id[eos],
            token_pattern=token_pattern,
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Turn text into a list of token IDs using ``unk_id`` for OOV tokens."""
        toks = word_tokenize(text, self.token_pattern)
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in toks]
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Turn token IDs back into a space-joined text string.

        Invalid IDs are skipped; special tokens can be omitted when requested.
        """
        pieces: list[str] = []
        for i in ids:
            if i < 0 or i >= len(self.id_to_token):
                continue
            tok = self.id_to_token[i]
            if skip_special_tokens and tok in (
                self.id_to_token[self.pad_id],
                self.id_to_token[self.unk_id],
                self.id_to_token[self.bos_id],
                self.id_to_token[self.eos_id],
            ):
                continue
            pieces.append(tok)
        return " ".join(pieces)


def _pad_sequence(seq: Sequence[int], pad_id: int, max_len: int) -> list[int]:
    """Right-pad or truncate a token ID sequence to ``max_len``."""
    if len(seq) >= max_len:
        return list(seq[:max_len])
    out = list(seq)
    out.extend([pad_id] * (max_len - len(seq)))
    return out


def build_supervised_tensors(
    texts_in: list[str],
    vocab: WordVocab,
    *,
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct model-ready (X, Y, mask) tensors for next-token prediction.

    For each text we add special tokens, shift the sequence by 1 to form input
    and target pairs, then pad/truncate to ``max_len``. The attention mask is 1
    on non-pad positions of the input.
    """
    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for t in texts_in:
        ids = vocab.encode(t, add_special_tokens=True)
        if len(ids) < 2:
            continue
        x = ids[:-1]
        y = ids[1:]
        x = x[:max_len]
        y = y[:max_len]
        x_p = _pad_sequence(x, vocab.pad_id, max_len)
        y_p = _pad_sequence(y, vocab.pad_id, max_len)
        m = [1 if tok != vocab.pad_id else 0 for tok in x_p]
        input_rows.append(x_p)
        target_rows.append(y_p)
        mask_rows.append(m)
    # Build CPU tensors; device transfer happens in the training loop for clarity
    X = torch.tensor(input_rows, dtype=torch.long)
    Y = torch.tensor(target_rows, dtype=torch.long)
    M = torch.tensor(mask_rows, dtype=torch.long)
    return X, Y, M


__all__ = [
    "word_tokenize",
    "WordVocab",
    "build_supervised_tensors",
]
