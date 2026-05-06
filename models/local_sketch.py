from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resampler import PerceiverResampler


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(max(1, out_ch // 16), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalSketchQueryEncoder(nn.Module):
    """Single-view coarse-anchor sketch query encoder.

    Design goal:
    - use the *same raw sketch* as input,
    - internally suppress unreliable high-frequency strokes,
    - extract bbox-localized *coarse layout anchors* instead of raw local details,
    - estimate a confidence scalar for each role query and use it as a soft gate.

    Compared with the previous full-resolution design, this version first builds a
    coarse representation by average-pooling the sketch/mask, then encodes that
    representation into a low-frequency feature map. Role queries are pooled from
    this coarse map using the provided normalized bboxes. A deterministic
    confidence term is computed from:
      1) valid-mask coverage inside the bbox,
      2) residual energy between raw sketch and a coarse upsampled sketch,
      3) (optional) diffusion timestep (higher timestep => stronger trust).
    """

    def __init__(
        self,
        out_dim: int,
        num_queries: int = 4,
        roi_size: int = 12,
        hidden_dims: tuple[int, int, int] = (32, 64, 128),
        num_global_queries: Optional[int] = None,
        input_downsample: int = 4,
        min_confidence: float = 0.25,
        residual_weight: float = 4.0,
        use_timestep_confidence: bool = True,
    ):
        super().__init__()
        self.roi_size = int(roi_size)
        self.num_queries = int(num_queries)
        self.num_global_queries = int(num_global_queries or max(6, num_queries))
        self.input_downsample = int(max(1, input_downsample))
        self.min_confidence = float(min_confidence)
        self.residual_weight = float(residual_weight)
        self.use_timestep_confidence = bool(use_timestep_confidence)

        self.backbone = nn.Sequential(
            ConvBNAct(2, hidden_dims[0], stride=2),
            ConvBNAct(hidden_dims[0], hidden_dims[1], stride=2),
            ConvBNAct(hidden_dims[1], hidden_dims[2], stride=2),
        )
        self.token_proj = nn.Linear(hidden_dims[-1], out_dim)
        self.global_resampler = PerceiverResampler(
            dim=out_dim,
            depth=2,
            dim_head=64,
            heads=8,
            num_queries=self.num_global_queries,
            embedding_dim=out_dim,
            output_dim=out_dim,
            apply_pos_emb=False,
        )
        self.role_resampler = PerceiverResampler(
            dim=out_dim,
            depth=2,
            dim_head=64,
            heads=8,
            num_queries=num_queries,
            embedding_dim=out_dim,
            output_dim=out_dim,
            apply_pos_emb=False,
        )
        self.role_bbox_proj = nn.Sequential(
            nn.Linear(4, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.global_query_gate = nn.Parameter(torch.tensor(2.0))

    def _module_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        p = next(self.backbone.parameters())
        return p.device, p.dtype

    @staticmethod
    def _build_roi_grid(
        boxes: torch.Tensor,
        roi_h: int,
        roi_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x0, y0, x1, y1 = boxes.unbind(dim=-1)
        x0 = x0.clamp(0.0, 1.0)
        y0 = y0.clamp(0.0, 1.0)
        x1 = x1.clamp(0.0, 1.0)
        y1 = y1.clamp(0.0, 1.0)
        x1 = torch.maximum(x1, x0 + 1e-4)
        y1 = torch.maximum(y1, y0 + 1e-4)

        xs = (torch.arange(roi_w, device=device, dtype=dtype) + 0.5) / roi_w
        ys = (torch.arange(roi_h, device=device, dtype=dtype) + 0.5) / roi_h
        gx = x0[:, None] + (x1 - x0)[:, None] * xs[None, :]
        gy = y0[:, None] + (y1 - y0)[:, None] * ys[None, :]
        gx = gx[:, None, :].expand(boxes.shape[0], roi_h, roi_w)
        gy = gy[:, :, None].expand(boxes.shape[0], roi_h, roi_w)
        return torch.stack([gx * 2.0 - 1.0, gy * 2.0 - 1.0], dim=-1)

    def _build_coarse_inputs(self, sketch: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Low-pass abstraction from the *same* raw sketch.
        if self.input_downsample > 1:
            coarse_sketch = F.avg_pool2d(sketch * valid_mask, kernel_size=self.input_downsample, stride=self.input_downsample, ceil_mode=True)
            coarse_valid = F.avg_pool2d(valid_mask, kernel_size=self.input_downsample, stride=self.input_downsample, ceil_mode=True)
            coarse_valid = coarse_valid.clamp_(0.0, 1.0)
        else:
            coarse_sketch = sketch * valid_mask
            coarse_valid = valid_mask

        coarse_sketch = torch.where(coarse_valid > 1e-3, coarse_sketch / coarse_valid.clamp_min(1e-3), torch.zeros_like(coarse_sketch))

        # Reconstruct a low-frequency version at raw resolution for confidence estimation.
        up_coarse = F.interpolate(coarse_sketch, size=sketch.shape[-2:], mode='bilinear', align_corners=False)
        return coarse_sketch, coarse_valid, up_coarse

    def _pool_scalar_map(self, scalar_map: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        # scalar_map: [B,1,H,W], boxes: [B,N,4] -> [B,N]
        b, n, _ = boxes.shape
        if n == 0:
            return scalar_map.new_zeros((b, 0))
        boxes_flat = boxes.reshape(b * n, 4).to(device=scalar_map.device, dtype=scalar_map.dtype)
        grid = self._build_roi_grid(boxes_flat, self.roi_size, self.roi_size, device=scalar_map.device, dtype=scalar_map.dtype)
        feat = scalar_map[:, None].expand(b, n, 1, scalar_map.shape[-2], scalar_map.shape[-1]).reshape(b * n, 1, scalar_map.shape[-2], scalar_map.shape[-1])
        roi = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return roi.mean(dim=(1, 2, 3)).reshape(b, n)

    def _compute_role_confidence(
        self,
        sketch: torch.Tensor,
        valid_mask: torch.Tensor,
        up_coarse_sketch: torch.Tensor,
        role_bboxes: torch.Tensor,
        role_valid_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        num_train_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        # Residual energy: high residual => unreliable fine details.
        residual = (sketch - up_coarse_sketch).abs() * valid_mask
        residual_mean = self._pool_scalar_map(residual, role_bboxes)
        valid_mean = self._pool_scalar_map(valid_mask, role_bboxes)

        conf = valid_mean * torch.exp(-self.residual_weight * residual_mean)

        if self.use_timestep_confidence and timesteps is not None:
            denom = float(max(1, int(num_train_timesteps) - 1)) if num_train_timesteps is not None else float(max(1, int(timesteps.max().item()) + 1))
            t_norm = timesteps.to(dtype=conf.dtype) / denom
            # High diffusion timestep (noisier latent / earlier denoising) => trust layout anchors more.
            time_gain = 0.35 + 0.65 * t_norm
            conf = conf * time_gain[:, None]

        conf = conf.clamp(min=self.min_confidence, max=1.0)
        if role_valid_mask is not None:
            conf = conf * role_valid_mask.to(device=conf.device, dtype=conf.dtype)
        return conf

    def encode_full_sketch(
        self,
        sketch: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        module_device, module_dtype = self._module_device_dtype()
        sketch = sketch.to(device=module_device, dtype=module_dtype)
        valid_mask = valid_mask.to(device=module_device, dtype=module_dtype)
        coarse_sketch, coarse_valid, _ = self._build_coarse_inputs(sketch, valid_mask)
        x = torch.cat([coarse_sketch, coarse_valid], dim=1)
        feat = self.backbone(x)
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        global_queries = self.global_resampler(tokens)
        global_queries = torch.sigmoid(self.global_query_gate).to(global_queries.dtype) * global_queries
        return feat, global_queries

    def encode_global_queries(
        self,
        sketch: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, global_queries = self.encode_full_sketch(sketch, valid_mask)
        return global_queries

    def forward(
        self,
        sketch: torch.Tensor,
        valid_mask: torch.Tensor,
        role_bboxes: Optional[torch.Tensor],
        role_valid_mask: Optional[torch.Tensor] = None,
        global_sketch_queries: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        num_train_timesteps: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        del global_sketch_queries  # kept only for interface compatibility
        if role_bboxes is None:
            return None
        if role_bboxes.ndim != 3 or role_bboxes.shape[-1] != 4:
            raise ValueError('role_bboxes must have shape [B, N_role, 4] with normalized xyxy boxes.')

        b, n_role, _ = role_bboxes.shape
        if n_role == 0:
            return None

        module_device, module_dtype = self._module_device_dtype()
        sketch = sketch.to(device=module_device, dtype=module_dtype)
        valid_mask = valid_mask.to(device=module_device, dtype=module_dtype)

        coarse_sketch, coarse_valid, up_coarse_sketch = self._build_coarse_inputs(sketch, valid_mask)
        x = torch.cat([coarse_sketch, coarse_valid], dim=1)
        feat_map = self.backbone(x)

        box_dtype = feat_map.dtype
        boxes = role_bboxes.reshape(b * n_role, 4).to(device=feat_map.device, dtype=box_dtype)
        grid = self._build_roi_grid(boxes, self.roi_size, self.roi_size, device=feat_map.device, dtype=box_dtype)

        feat_map_expanded = feat_map[:, None].expand(b, n_role, feat_map.shape[1], feat_map.shape[2], feat_map.shape[3])
        feat_map_expanded = feat_map_expanded.reshape(b * n_role, feat_map.shape[1], feat_map.shape[2], feat_map.shape[3])
        roi_feat = F.grid_sample(feat_map_expanded, grid, mode='bilinear', padding_mode='border', align_corners=False)
        roi_tokens = roi_feat.flatten(2).transpose(1, 2)
        roi_tokens = self.token_proj(roi_tokens)
        role_queries = self.role_resampler(roi_tokens)

        bbox_embed = self.role_bbox_proj(boxes.to(dtype=self.role_bbox_proj[0].weight.dtype)).unsqueeze(1)
        bbox_embed = bbox_embed.to(dtype=role_queries.dtype)
        role_queries = role_queries + bbox_embed

        role_queries = role_queries.reshape(b, n_role, role_queries.shape[1], role_queries.shape[2])
        role_conf = self._compute_role_confidence(
            sketch=sketch,
            valid_mask=valid_mask,
            up_coarse_sketch=up_coarse_sketch,
            role_bboxes=role_bboxes.to(device=sketch.device, dtype=sketch.dtype),
            role_valid_mask=role_valid_mask,
            timesteps=timesteps,
            num_train_timesteps=num_train_timesteps,
        )
        role_queries = role_queries * role_conf[:, :, None, None].to(device=role_queries.device, dtype=role_queries.dtype)

        if role_valid_mask is not None:
            role_queries = role_queries * role_valid_mask[:, :, None, None].to(device=role_queries.device, dtype=role_queries.dtype)
        return role_queries
