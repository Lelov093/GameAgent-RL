"""Categorical policy distributions used by Own PPO."""

from __future__ import annotations

import torch
from torch.distributions import Categorical


class MaskedCategorical(Categorical):
    """Categorical distribution restricted to the boolean legal-action support."""

    def __init__(self, logits: torch.Tensor, action_masks: torch.Tensor):
        if action_masks.dtype is not torch.bool:
            raise TypeError("action_masks must use bool dtype")
        if action_masks.shape != logits.shape:
            raise ValueError("action_masks must have the same shape as logits")
        if not torch.all(action_masks.any(dim=-1)):
            raise ValueError("each distribution row must contain a legal action")
        self.action_masks = action_masks
        negative_sentinel = torch.finfo(logits.dtype).min
        masked_logits = torch.where(action_masks, logits, negative_sentinel)
        super().__init__(logits=masked_logits)
