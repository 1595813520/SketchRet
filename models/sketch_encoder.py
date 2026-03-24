from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .resampler import Resampler


# -------------------------
# utils
# -------------------------
def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# -------------------------
# output container
# -------------------------
@dataclass
class SketchEncoderOutput:
    sem_tokens: torch.Tensor                    # [B, K, cross_attn_dim]
    spatial_feats: Dict[str, torch.Tensor]     # {"mid": ..., "high": ...}
    patch_tokens: torch.Tensor                 # [B, N, C]
    patch_map: torch.Tensor                    # [B, C, Hp, Wp]
    role_tokens: Optional[torch.Tensor] = None # [B, N_role, K_role, cross_attn_dim]


# -------------------------
# DINOv2 backbone wrapper
# -------------------------
class DinoV2Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        freeze: bool = False,
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.patch_size = self._get_patch_size()
        self.embed_dim = self._get_embed_dim()

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def _get_patch_size(self) -> int:
        ps = getattr(self.model.patch_embed, "patch_size", 14)
        if isinstance(ps, tuple):
            return ps[0]
        return ps

    def _get_embed_dim(self) -> int:
        return getattr(self.model, "embed_dim", 768)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.pixel_mean) / self.pixel_std

    def _resize_to_patch_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = x.shape[-2:]
        new_h = math.ceil(h / self.patch_size) * self.patch_size
        new_w = math.ceil(w / self.patch_size) * self.patch_size
        if new_h == h and new_w == w:
            return x, (h, w)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x, (new_h, new_w)

    def _extract_tokens(self, feats):
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                patch_tokens = feats["x_norm_patchtokens"]
            elif "x_prenorm" in feats:
                x = feats["x_prenorm"]
                patch_tokens = x[:, 1:, :] if x.dim() == 3 else x
            else:
                raise ValueError(f"Unsupported DINOv2 feature dict keys: {list(feats.keys())}")

            if "x_norm_clstoken" in feats:
                cls_token = feats["x_norm_clstoken"]
            elif "x_prenorm" in feats:
                x = feats["x_prenorm"]
                cls_token = x[:, 0, :] if x.dim() == 3 else patch_tokens.mean(dim=1)
            else:
                cls_token = patch_tokens.mean(dim=1)
        else:
            if feats.dim() == 3 and feats.shape[1] > 1:
                cls_token = feats[:, 0, :]
                patch_tokens = feats[:, 1:, :]
            else:
                patch_tokens = feats
                cls_token = patch_tokens.mean(dim=1)
        return cls_token, patch_tokens

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._normalize(x)
        x, (rh, rw) = self._resize_to_patch_multiple(x)

        feats = self.model.forward_features(x)
        cls_token, patch_tokens = self._extract_tokens(feats)

        b, n, c = patch_tokens.shape
        gh, gw = rh // self.patch_size, rw // self.patch_size
        assert gh * gw == n, f"Patch token count mismatch: gh*gw={gh*gw}, N={n}"

        patch_map = patch_tokens.view(b, gh, gw, c).permute(0, 3, 1, 2).contiguous()
        return cls_token, patch_tokens, patch_map


# -------------------------
# semantic head
# -------------------------
class SketchSemanticHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        cross_attn_dim: int = 768,
        num_queries: int = 8,
        resampler_dim: int = 1024,
        resampler_depth: int = 4,
        resampler_heads: int = 16,
        resampler_dim_head: int = 64,
        apply_pos_emb: bool = True,
        gate_init: float = 2.0,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, cross_attn_dim)
        self.resampler = Resampler(
            dim=resampler_dim,
            depth=resampler_depth,
            dim_head=resampler_dim_head,
            heads=resampler_heads,
            num_queries=num_queries,
            embedding_dim=cross_attn_dim,
            output_dim=cross_attn_dim,
            apply_pos_emb=apply_pos_emb,
        )
        # keep learnable gate, but do NOT start fully closed.
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))

    def forward(self, patch_tokens: torch.Tensor, cls_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.proj(self.pre_norm(patch_tokens))
        if cls_token is not None:
            cls = self.proj(self.pre_norm(cls_token)).unsqueeze(1)
            x = torch.cat([cls, x], dim=1)
        sem_tokens = self.resampler(x)
        return torch.sigmoid(self.gate_logit) * sem_tokens


class RoleSketchTokenHead(nn.Module):
    """
    Extract local sketch tokens from role bboxes on top of the same sketch backbone.
    This is intentionally lightweight: no new backbone, only ROI sampling + a small resampler.
    """
    def __init__(
        self,
        in_dim: int = 768,
        cross_attn_dim: int = 768,
        num_queries: int = 4,
        roi_size: int = 4,
        resampler_dim: int = 1024,
        resampler_depth: int = 2,
        resampler_heads: int = 8,
        resampler_dim_head: int = 64,
        gate_init: float = 2.0,
    ):
        super().__init__()
        self.roi_size = roi_size
        self.pre_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, cross_attn_dim)
        self.resampler = Resampler(
            dim=resampler_dim,
            depth=resampler_depth,
            dim_head=resampler_dim_head,
            heads=resampler_heads,
            num_queries=num_queries,
            embedding_dim=cross_attn_dim,
            output_dim=cross_attn_dim,
            apply_pos_emb=True,
        )
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))

    @staticmethod
    def _build_roi_grid(
        boxes: torch.Tensor,
        roi_h: int,
        roi_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # boxes: [M, 4], normalized xyxy in [0, 1]
        m = boxes.shape[0]
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

        gx = gx[:, None, :].expand(m, roi_h, roi_w)
        gy = gy[:, :, None].expand(m, roi_h, roi_w)

        grid = torch.stack([gx * 2.0 - 1.0, gy * 2.0 - 1.0], dim=-1)  # [M, roi_h, roi_w, 2]
        return grid

    def forward(
        self,
        patch_map: torch.Tensor,
        role_bboxes: Optional[torch.Tensor],
        role_valid_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if role_bboxes is None:
            return None

        if role_bboxes.ndim != 3 or role_bboxes.shape[-1] != 4:
            raise ValueError("role_bboxes must be [B, N_role, 4] normalized xyxy.")

        b, n_role, _ = role_bboxes.shape
        if n_role == 0:
            return None

        c = patch_map.shape[1]
        roi_size = self.roi_size
        dtype = patch_map.dtype
        device = patch_map.device

        feat = patch_map[:, None, :, :, :].expand(b, n_role, c, patch_map.shape[-2], patch_map.shape[-1])
        feat = feat.reshape(b * n_role, c, patch_map.shape[-2], patch_map.shape[-1])

        boxes = role_bboxes.reshape(b * n_role, 4).to(device=device, dtype=dtype)
        grid = self._build_roi_grid(boxes, roi_size, roi_size, device=device, dtype=dtype)
        sampled = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.reshape(b * n_role, c, roi_size * roi_size).transpose(1, 2)  # [B*N, S, C]

        sampled = self.proj(self.pre_norm(sampled))
        role_tokens = self.resampler(sampled)  # [B*N, K_role, D]
        role_tokens = role_tokens.reshape(b, n_role, role_tokens.shape[1], role_tokens.shape[2])

        if role_valid_mask is not None:
            role_tokens = role_tokens * role_valid_mask[:, :, None, None].to(role_tokens.dtype)

        return torch.sigmoid(self.gate_logit) * role_tokens


# -------------------------
# weak spatial head
# -------------------------
class SketchSpatialHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        stem_dim: int = 256,
        out_dim_mid: int = 640,
        out_dim_high: int = 320,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, stem_dim, kernel_size=1),
            LayerNorm2d(stem_dim),
            nn.SiLU(),
        )

        self.high_proj = zero_module(nn.Conv2d(stem_dim, out_dim_high, kernel_size=1))
        self.gamma_high = nn.Parameter(torch.tensor(0.0))

        self.mid_proj = zero_module(nn.Conv2d(stem_dim, out_dim_mid, kernel_size=1))
        self.gamma_mid = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        patch_map: torch.Tensor,
        high_size: Tuple[int, int],
        mid_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        x = self.stem(patch_map)
        feat_high = F.interpolate(x, size=high_size, mode="bilinear", align_corners=False)
        feat_mid = F.interpolate(x, size=mid_size, mode="bilinear", align_corners=False)
        feat_high = self.gamma_high.tanh() * self.high_proj(feat_high)
        feat_mid = self.gamma_mid.tanh() * self.mid_proj(feat_mid)
        return {"high": feat_high, "mid": feat_mid}


# -------------------------
# full sketch encoder
# -------------------------
class SketchEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2.lvd142m",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        cross_attn_dim: int = 768,
        num_sem_queries: int = 8,
        num_role_queries: int = 4,
        spatial_mid_dim: int = 640,
        spatial_high_dim: int = 320,
        enable_role_tokens: bool = True,
        role_roi_size: int = 4,
    ):
        super().__init__()
        self.backbone = DinoV2Backbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
            freeze=freeze_backbone,
        )

        embed_dim = self.backbone.embed_dim
        self.semantic_head = SketchSemanticHead(
            in_dim=embed_dim,
            cross_attn_dim=cross_attn_dim,
            num_queries=num_sem_queries,
        )
        self.spatial_head = SketchSpatialHead(
            in_dim=embed_dim,
            out_dim_mid=spatial_mid_dim,
            out_dim_high=spatial_high_dim,
        )

        self.enable_role_tokens = enable_role_tokens
        self.role_head = None
        if enable_role_tokens:
            self.role_head = RoleSketchTokenHead(
                in_dim=embed_dim,
                cross_attn_dim=cross_attn_dim,
                num_queries=num_role_queries,
                roi_size=role_roi_size,
            )

    @torch.no_grad()
    def infer_patch_grid(self, h: int, w: int) -> Tuple[int, int]:
        ph = math.ceil(h / self.backbone.patch_size)
        pw = math.ceil(w / self.backbone.patch_size)
        return ph, pw

    def forward(
        self,
        sketch: torch.Tensor,
        high_size: Tuple[int, int],
        mid_size: Tuple[int, int],
        role_bboxes: Optional[torch.Tensor] = None,
        role_valid_mask: Optional[torch.Tensor] = None,
    ) -> SketchEncoderOutput:
        cls_token, patch_tokens, patch_map = self.backbone(sketch)
        sem_tokens = self.semantic_head(patch_tokens, cls_token)
        spatial_feats = self.spatial_head(
            patch_map=patch_map,
            high_size=high_size,
            mid_size=mid_size,
        )

        role_tokens = None
        if self.role_head is not None and role_bboxes is not None:
            role_tokens = self.role_head(
                patch_map=patch_map,
                role_bboxes=role_bboxes,
                role_valid_mask=role_valid_mask,
            )

        return SketchEncoderOutput(
            sem_tokens=sem_tokens,
            spatial_feats=spatial_feats,
            patch_tokens=patch_tokens,
            patch_map=patch_map,
            role_tokens=role_tokens,
        )
