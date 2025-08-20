from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config_utils import LSTMConfig


class LSTMLanguageModel(nn.Module):
    """Compact LSTM language model for next-token prediction and generation.

    Educational highlights
    - Word embeddings feed into a multi-layer `nn.LSTM` with optional dropout
    - A linear projection maps hidden states to vocabulary logits
    - The forward method can return logits and an optional masked loss
    - The `generate` method performs greedy decoding for up to `max_new_tokens`
    """
    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=(config.dropout if config.num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.hidden_size, config.num_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with optional masked cross-entropy loss.

        Args
        - ``input_ids``: Long tensor (batch, time)
        - ``targets``: Optional next-token targets (batch, time)
        - ``attention_mask``: Optional 1/0 mask (batch, time) to exclude padding

        Returns
        - ``(logits, loss)`` where logits is (batch, time, vocab) and ``loss``
          is a scalar or ``None`` if ``targets`` is not provided
        """
        # input_ids: (B, T)
        emb = self.embedding(input_ids)  # (B, T, E)
        out, _hidden = self.lstm(emb)  # (B, T, H)
        out = self.dropout(out)
        logits = self.proj(out)  # (B, T, V)
        if targets is None:
            return logits, None
        if attention_mask is None:
            # Compute token-level cross-entropy with padding ignored
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
            return logits, loss
        # Masked average loss for educational clarity
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.config.pad_token_id,
            reduction="none",
        )
        mask = attention_mask.reshape(-1).float()
        loss = (per_tok * mask).sum() / mask.sum().clamp_min(1)
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        prefix_ids: torch.Tensor,
        max_new_tokens: int = 20,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy generation continuing from a given prefix.

        This method appends one token per step, always choosing the argmax over
        the last-step logits, and stops early if ``eos_token_id`` is encountered.
        """
        # Ensure 2D (B, T) and move to model's device
        if prefix_ids.dim() == 1:
            prefix_ids = prefix_ids.unsqueeze(0)
        model_device = next(self.parameters()).device
        prefix_ids = prefix_ids.to(model_device)
        generated = prefix_ids
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=model_device)
        for _step in range(max_new_tokens):
            logits, _loss_unused = self(generated)
            next_token_logits = logits[:, -1, :]
            next_ids = torch.argmax(next_token_logits, dim=-1)
            if eos_token_id is not None:
                # If already finished, keep padding instead of generating more
                next_ids = torch.where(
                    finished,
                    torch.full_like(next_ids, self.config.pad_token_id),
                    next_ids,
                )
                finished = finished | (next_ids == eos_token_id)
            generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
            if eos_token_id is not None and bool(torch.all(finished)):
                break
        return generated
