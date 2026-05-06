from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_snr


def compute_masked_diffusion_mse_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    noise_scheduler,
    timesteps: torch.Tensor,
    snr_gamma: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    latent_mask = F.interpolate(valid_mask, size=model_pred.shape[-2:], mode='nearest')
    latent_mask = latent_mask.to(dtype=model_pred.dtype)
    mse = (model_pred - target) ** 2
    mse = mse * latent_mask
    denom = latent_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    loss_per_sample = mse.sum(dim=(1, 2, 3)) / denom

    if snr_gamma is not None and snr_gamma > 0:
        snr = compute_snr(noise_scheduler, timesteps)
        if noise_scheduler.config.prediction_type == 'v_prediction':
            weights = snr / (snr + 1)
        else:
            weights = torch.minimum(snr, torch.full_like(snr, snr_gamma)) / snr
        loss_per_sample = loss_per_sample * weights

    return loss_per_sample.mean(), latent_mask


def masked_l1(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask = mask.repeat(1, x.shape[1], 1, 1)
    num = (mask * (x - y).abs()).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den


def bg_clean_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, threshold: float = 0.92) -> torch.Tensor:
    line_mask = (target.mean(dim=1, keepdim=True) < threshold).float()
    bg_mask = valid_mask * (1.0 - line_mask)
    return masked_l1(pred, target, bg_mask)


def fg_detail_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, threshold: float = 0.92) -> torch.Tensor:
    line_mask = (target.mean(dim=1, keepdim=True) < threshold).float()
    fg_mask = valid_mask * line_mask
    return masked_l1(pred, target, fg_mask)


def gradient_consistency_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    def _grad(x: torch.Tensor):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    if valid_mask.shape[1] == 1 and pred.shape[1] != 1:
        valid_mask_c = valid_mask.repeat(1, pred.shape[1], 1, 1)
    else:
        valid_mask_c = valid_mask

    pred_gx, pred_gy = _grad(pred)
    tgt_gx, tgt_gy = _grad(target)

    mask_x = valid_mask_c[:, :, :, 1:] * valid_mask_c[:, :, :, :-1]
    mask_y = valid_mask_c[:, :, 1:, :] * valid_mask_c[:, :, :-1, :]

    loss_x = masked_l1(pred_gx, tgt_gx, mask_x)
    loss_y = masked_l1(pred_gy, tgt_gy, mask_y)
    return 0.5 * (loss_x + loss_y)
