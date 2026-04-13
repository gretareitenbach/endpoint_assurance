from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAsymmetricLoss(nn.Module):
    def __init__(
        self,
        fp_weight: float = 50.0,
        tamper_positive_weight: float = 1.0,
        wear_weight: float = 0.5,
    ):
        super().__init__()
        self.fp_weight = fp_weight
        self.tamper_positive_weight = tamper_positive_weight
        self.wear_weight = wear_weight

    def forward(
        self,
        tamper_logits: torch.Tensor,
        wear_logits: torch.Tensor,
        tamper_targets: torch.Tensor,
        wear_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tamper_prob = torch.sigmoid(tamper_logits)

        # Strongly penalize false positives by weighting the y=0 term.
        pos_loss = -tamper_targets * torch.log(torch.clamp(tamper_prob, 1e-6, 1.0))
        neg_loss = -(1.0 - tamper_targets) * torch.log(torch.clamp(1.0 - tamper_prob, 1e-6, 1.0))
        tamper_loss = (self.tamper_positive_weight * pos_loss + self.fp_weight * neg_loss).mean()

        wear_loss = F.binary_cross_entropy_with_logits(wear_logits, wear_targets)
        total_loss = tamper_loss + self.wear_weight * wear_loss

        return {
            "loss": total_loss,
            "tamper_loss": tamper_loss,
            "wear_loss": wear_loss,
        }
