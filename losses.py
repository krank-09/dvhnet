"""
losses.py
=========
Combined DVHnet training objective:

    L_total = L_MSE + lambda1 * L_clinical + lambda2 * L_mono

- L_MSE:      point-wise fidelity (Smooth L1 is offered as a more robust
              alternative to raw MSE, since it's less dominated by the
              curve's steep transition region).
- L_clinical: extra weight on clinically-critical summary points, D2% and
              D50%, since those two indices drive plan acceptance decisions.
- L_mono:     penalizes any place where the *predicted* curve increases
              with dose (physically impossible for a cumulative DVH).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _index_for_percent(num_bins: int, percent: float) -> int:
    """
    Cumulative DVH y[k] = P(dose >= D_k). D_x% is the dose at which the curve
    crosses volume fraction x/100. We approximate the *bin index* nearest that
    volume level for a lightweight, differentiable proxy loss (rather than
    inverting the curve, which is done properly at eval time in metrics.py).
    """
    # This is a fixed index into the *volume axis* only used to pick which
    # bins to upweight; true dose-metric inversion happens in metrics.py.
    return max(0, min(num_bins - 1, round((1.0 - percent / 100.0) * (num_bins - 1))))


class DVHLoss(nn.Module):
    def __init__(self, num_bins: int = 256,
                 lambda_clinical: float = 0.5,
                 lambda_mono: float = 0.1,
                 clinical_percents=(2.0, 50.0),
                 clinical_weight: float = 5.0,
                 use_smooth_l1: bool = True):
        """
        Args:
            num_bins: length of the DVH vector (dose axis resolution).
            lambda_clinical: weight (lambda_1) on the clinical-point penalty.
            lambda_mono: weight (lambda_2) on the monotonicity penalty.
            clinical_percents: which D_x% volume points to upweight, e.g. (2, 50)
                                for D2% (near-max/hot-spot proxy) and D50% (median).
            clinical_weight: multiplier applied to squared error at those bins,
                              on top of the base point-wise loss.
            use_smooth_l1: Smooth L1 (Huber) instead of raw MSE for the base
                            point-wise term; more robust to the DVH's steep
                            fall-off region than plain squared error.
        """
        super().__init__()
        self.num_bins = num_bins
        self.lambda_clinical = lambda_clinical
        self.lambda_mono = lambda_mono
        self.clinical_weight = clinical_weight
        self.use_smooth_l1 = use_smooth_l1

        clinical_idx = [_index_for_percent(num_bins, p) for p in clinical_percents]
        self.register_buffer("clinical_idx", torch.tensor(clinical_idx, dtype=torch.long))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        pred, target: [B, num_bins], both in [0, 1].
        Returns a dict with the total loss and each component (for logging).
        """
        # 1) Point-wise fidelity
        if self.use_smooth_l1:
            l_fidelity = F.smooth_l1_loss(pred, target, reduction="mean")
        else:
            l_fidelity = F.mse_loss(pred, target, reduction="mean")

        # 2) Clinical metric penalization: extra squared-error weight at D2%/D50% bins
        pred_clin = pred.index_select(dim=1, index=self.clinical_idx)
        target_clin = target.index_select(dim=1, index=self.clinical_idx)
        l_clinical = F.mse_loss(pred_clin, target_clin, reduction="mean") * self.clinical_weight

        # 3) Monotonicity: penalize any increase from bin k to bin k+1
        diffs = pred[:, 1:] - pred[:, :-1]                 # [B, num_bins-1]
        l_mono = torch.clamp(diffs, min=0.0).pow(2).mean()  # squared hinge, smoother gradient than raw max(0,.)

        total = l_fidelity + self.lambda_clinical * l_clinical + self.lambda_mono * l_mono

        return {
            "loss": total,
            "loss_fidelity": l_fidelity.detach(),
            "loss_clinical": l_clinical.detach(),
            "loss_mono": l_mono.detach(),
        }
