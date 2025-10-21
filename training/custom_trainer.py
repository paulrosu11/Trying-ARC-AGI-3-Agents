"""Custom trainer for supervised fine-tuning runs."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers import Trainer

IGNORE_TOKEN_ID = -100


class ArkAGISFTTrainer(Trainer):
    """Trainer that supports optional loss masks for agentic SFT."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        loss_mask: Optional[torch.Tensor] = inputs.pop("loss_mask", None)
        labels: Optional[torch.Tensor] = inputs.get("labels")

        if loss_mask is not None and labels is not None:
            masked_labels = labels.clone()
            masked_labels = masked_labels.masked_fill(loss_mask == 0, IGNORE_TOKEN_ID)
            inputs["labels"] = masked_labels

        return super().compute_loss(model, inputs, return_outputs)
