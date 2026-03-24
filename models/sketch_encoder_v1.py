# models/sketch_encoder.py
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
        # x: [B,C,H,W]
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


# -------------------------
# output container
# -------------------------
@dataclass
class SketchEncoderOutput:
    sem_tokens: torch.Tensor                  # [B, K, cross_attn_dim]
    spatial_feats: Dict[str, torch.Tensor]   # {"mid": [B,Cm,Hm,Wm], "high": [B,Ch,Hh,Wh]}
    patch_tokens: torch.Tensor               # [B, N, C]
    patch_map: torch.Tensor                  # [B, C, Hp, Wp]


# -------------------------
# DINOv2 backbone wrapper
# -------------------------
class DinoV2Backbone(nn.Module):
    """
    timm DINOv2 wrapper.
    Returns patch tokens and patch map.
    """
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

        # DINOv2 / ViT normalization (ImageNet style)
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
        """
        Handle timm ViT / DINOv2 feature output.
        """
        if isinstance(feats, dict):
            # timm DINOv2 usually returns dict with patch tokens
            if "x_norm_patchtokens" in feats:
                patch_tokens = feats["x_norm_patchtokens"]              # [B, N, C]
            elif "x_prenorm" in feats:
                x = feats["x_prenorm"]
                patch_tokens = x[:, 1:, :] if x.dim() == 3 else x
            else:
                raise ValueError(f"Unsupported DINOv2 feature dict keys: {list(feats.keys())}")

            if "x_norm_clstoken" in feats:
                cls_token = feats["x_norm_clstoken"]                   # [B, C]
            elif "x_prenorm" in feats:
                x = feats["x_prenorm"]
                cls_token = x[:, 0, :] if x.dim() == 3 else patch_tokens.mean(dim=1)
            else:
                cls_token = patch_tokens.mean(dim=1)
        else:
            # fallback: tensor [B, 1+N, C] or [B, N, C]
            if feats.dim() == 3 and feats.shape[1] > 1:
                cls_token = feats[:, 0, :]
                patch_tokens = feats[:, 1:, :]
            else:
                patch_tokens = feats
                cls_token = patch_tokens.mean(dim=1)
        return cls_token, patch_tokens

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B,1/3,H,W]
        x = self._normalize(x)
        x, (rh, rw) = self._resize_to_patch_multiple(x)

        feats = self.model.forward_features(x)
        cls_token, patch_tokens = self._extract_tokens(feats)  # [B,C], [B,N,C]

        B, N, C = patch_tokens.shape
        gh, gw = rh // self.patch_size, rw // self.patch_size
        assert gh * gw == N, f"Patch token count mismatch: gh*gw={gh*gw}, N={N}"

        patch_map = patch_tokens.view(B, gh, gw, C).permute(0, 3, 1, 2).contiguous()  # [B,C,Hg,Wg]
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

        # semantic gate: start tiny to avoid sketch tokens overpowering text at the beginning
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, patch_tokens: torch.Tensor, cls_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        # patch_tokens: [B,N,C]
        x = self.proj(self.pre_norm(patch_tokens))

        if cls_token is not None:
            cls = self.proj(self.pre_norm(cls_token)).unsqueeze(1)  # [B,1,D]
            x = torch.cat([cls, x], dim=1)

        sem_tokens = self.resampler(x)   # [B,K,D]
        return self.gate.tanh() * sem_tokens


# -------------------------
# weak spatial head
# -------------------------
class SketchSpatialHead(nn.Module):
    """
    Very light spatial hint head.
    Only two scales: mid / high.
    """
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

        # high resolution branch
        self.high_proj = zero_module(nn.Conv2d(stem_dim, out_dim_high, kernel_size=1))
        self.gamma_high = nn.Parameter(torch.tensor(0.0))

        # mid resolution branch
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

        return {
            "high": feat_high,   # usually injected to high-res block
            "mid": feat_mid,     # usually injected to mid / lower-res block
        }


# -------------------------
# full sketch encoder
# -------------------------
class SketchEncoder(nn.Module):
    """
    Final recommended sketch encoder:
      - one backbone
      - semantic head -> sketch semantic tokens
      - weak spatial head -> two light spatial hints
    """
    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2.lvd142m",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        cross_attn_dim: int = 768,
        num_sem_queries: int = 8,
        spatial_mid_dim: int = 640,
        spatial_high_dim: int = 320,
    ):
        super().__init__()
        self.backbone = DinoV2Backbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
            freeze=freeze_backbone,  # default False: no separate frozen DINO branch
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
    ) -> SketchEncoderOutput:
        """
        Args:
            sketch: [B,1/3,H,W]
            high_size: target feature size for high-res hint, e.g. (64, 64)
            mid_size: target feature size for mid-res hint, e.g. (32, 32)

        Returns:
            SketchEncoderOutput
        """
        cls_token, patch_tokens, patch_map = self.backbone(sketch)

        sem_tokens = self.semantic_head(patch_tokens, cls_token)
        spatial_feats = self.spatial_head(
            patch_map=patch_map,
            high_size=high_size,
            mid_size=mid_size,
        )

        return SketchEncoderOutput(
            sem_tokens=sem_tokens,
            spatial_feats=spatial_feats,
            patch_tokens=patch_tokens,
            patch_map=patch_map,
        )
        
        
'''
if __name__ == "__main__":
    B = 2
    sketch = torch.randn(B, 1, 512, 512)

    model = SketchEncoder()
    out = model(sketch, high_size=(64, 64), mid_size=(32, 32))

    print("sem_tokens:", out.sem_tokens.shape)              # [B, K, 768]
    print("mid:", out.spatial_feats["mid"].shape)          # [B, 640, 32, 32]
    print("high:", out.spatial_feats["high"].shape)        # [B, 320, 64, 64]
'''
