from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from src.config_utils import AppConfig


class NextTokenTensorDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset wrapping input/target/mask tensors for next-token prediction.

    Each item is a dict with keys ``input_ids``, ``targets``, and
    ``attention_mask``; all tensors are 1D slices from the prebuilt matrices.
    """

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    attention_mask: torch.Tensor

    def __init__(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        assert input_ids.shape == target_ids.shape == attention_mask.shape
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return self.input_ids.size(0)

    @override
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "targets": self.target_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def _resolve_loader_params(
    cfg: AppConfig, device: torch.device
) -> tuple[bool, int, bool]:
    """Derive DataLoader pinning and worker settings from config and device."""
    is_cuda = device.type == "cuda"
    pin_memory = (
        True if is_cuda else False
        if cfg.dataloader.pin_memory is None
        else cfg.dataloader.pin_memory
    )
    num_workers = (
        (cfg.dataloader.num_workers_cuda or 0) if is_cuda else cfg.dataloader.num_workers_cpu
    )
    persistent_workers = (
        (num_workers > 0)
        if cfg.dataloader.persistent_workers is None
        else cfg.dataloader.persistent_workers
    )
    return pin_memory, num_workers, persistent_workers


def create_next_token_dataloaders(
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: AppConfig,
    device: torch.device,
) -> tuple[
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]],
]:
    """Build DataLoaders for next-token prediction from prebuilt tensors.

    Returns ``(train_loader, val_loader, test_loader)``.
    """
    is_cuda = device.type == "cuda"
    common = dict(
        batch_size=cfg.training.batch_size,
        pin_memory=is_cuda,
        num_workers=0,
        persistent_workers=False,
    )

    train_ds = NextTokenTensorDataset(*train_tensors)
    val_ds = NextTokenTensorDataset(*val_tensors)
    test_ds = NextTokenTensorDataset(*test_tensors)

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader
