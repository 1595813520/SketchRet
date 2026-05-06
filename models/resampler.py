from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, nq, _ = q_in.shape
        nk = kv_in.shape[1]
        q = self.to_q(self.norm_q(q_in))
        k = self.to_k(self.norm_kv(kv_in))
        v = self.to_v(self.norm_kv(kv_in))

        q = q.view(b, nq, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, nk, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, nk, self.heads, self.dim_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if key_padding_mask is not None:
            # key_padding_mask: [B, Nk], True means valid
            valid = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            attn = attn.masked_fill(~valid, -1e4)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, nq, self.inner_dim)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        dim_head: int = 64,
        heads: int = 8,
        num_queries: int = 8,
        embedding_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        apply_pos_emb: bool = False,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        embedding_dim = embedding_dim or dim
        output_dim = output_dim or dim
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / math.sqrt(dim))
        self.proj_in = nn.Identity() if embedding_dim == dim else nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Identity() if output_dim == dim else nn.Linear(dim, output_dim)
        self.apply_pos_emb = apply_pos_emb
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) / math.sqrt(dim)) if apply_pos_emb else None
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    CrossAttention(dim=dim, heads=heads, dim_head=dim_head),
                    FeedForward(dim=dim),
                ])
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.proj_in(x)
        if self.apply_pos_emb and self.pos_emb is not None:
            x = x + self.pos_emb[:, : x.shape[1], :]
        latents = self.latents.expand(x.shape[0], -1, -1)
        for attn, ff in self.layers:
            latents = latents + attn(latents, x, key_padding_mask=key_padding_mask)
            latents = latents + ff(latents)
        return self.proj_out(self.norm(latents))
