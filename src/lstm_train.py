from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler as _SchedulerBase
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .tokenizer_vocab import WordVocab


def _move_batch_to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Copy a batch of small tensors to the target device.

    Using ``non_blocking=True`` enables asynchronous copies when pinned memory
    and CUDA streams are in use, but is harmless on CPU.
    """
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def create_optimizer_scheduler(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    use_scheduler: bool,
    step_size: int,
    gamma: float,
) -> tuple[torch.optim.Optimizer, _SchedulerBase | None]:
    """Create Adam optimizer and an optional StepLR scheduler.

    StepLR decays the learning rate by ``gamma`` every ``step_size`` epochs when
    ``use_scheduler`` is true; otherwise no scheduler is returned.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        if use_scheduler
        else None
    )
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_grad_norm: float = 1.0,
    progress_desc: str | None = None,
) -> float:
    """One full pass over the training loader with gradient updates.

    Returns the mean training loss across steps (masked cross-entropy).
    """
    model.train()
    total, steps = 0.0, 0
    pbar = tqdm(loader, desc=(progress_desc or "train"))
    for batch in pbar:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(
            batch["input_ids"],
            targets=batch["targets"],
            attention_mask=batch["attention_mask"],
        )
        assert loss is not None
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        val = float(loss.item())
        total += val
        steps += 1
        try:
            pbar.set_postfix({"loss": f"{val:.4f}"}, refresh=False)
        except Exception:
            pass
    return total / max(1, steps)


@torch.inference_mode()
def valid_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    """Evaluate mean loss over a validation loader without gradient updates."""
    model.eval()
    total, steps = 0.0, 0
    pbar = tqdm(loader, desc="valid")
    for batch in pbar:
        batch = _move_batch_to_device(batch, device)
        _logits, loss = model(
            batch["input_ids"],
            targets=batch["targets"],
            attention_mask=batch["attention_mask"],
        )
        assert loss is not None
        val = float(loss.item())
        total += val
        steps += 1
        try:
            pbar.set_postfix({"loss": f"{val:.4f}"}, refresh=False)
        except Exception:
            pass
    return total / max(1, steps)


def save_checkpoint(
    model: nn.Module,
    vocab: WordVocab,
    *,
    state_path: str,
    vocab_path: str,
) -> None:
    """Persist a minimal checkpoint: model ``state_dict`` and vocab mapping.

    Keeping artifacts small encourages iteration and is sufficient to resume
    inference with the exact same tokenizer and model weights.
    """
    torch.save(model.state_dict(), state_path)
    torch.save(
        {
            "token_to_id": vocab.token_to_id,
            "id_to_token": vocab.id_to_token,
            "pad_id": vocab.pad_id,
            "unk_id": vocab.unk_id,
            "bos_id": vocab.bos_id,
            "eos_id": vocab.eos_id,
            "token_pattern": vocab.token_pattern,
        },
        vocab_path,
    )


__all__ = [
    "create_optimizer_scheduler",
    "train_one_epoch",
    "train_epoch",
    "valid_epoch",
    "save_checkpoint",
]


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    autocast: Any,
    *,
    max_grad_norm: float = 1.0,
    log_every: int = 100,
    progress_desc: str | None = None,
) -> float:
    """Compatibility wrapper: single-epoch training loop.

    Ignores AMP scaler on CPU and uses an explicit attention mask for loss.
    """
    _ = log_every  # kept for signature compatibility
    model.train()
    total, steps = 0.0, 0
    pbar = tqdm(train_loader, desc=(progress_desc or "train"))
    for batch in pbar:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            _logits, loss = model(
                batch["input_ids"],
                targets=batch["targets"],
                attention_mask=batch.get("attention_mask"),
            )
        assert loss is not None
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += float(loss.item())
        steps += 1
    return total / max(1, steps)
